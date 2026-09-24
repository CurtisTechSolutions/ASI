//! The HTTP transport: a small HTTP/1.1 server, written out like everything
//! else here because the crate has no dependencies.
//!
//! It is the same shape as the Python and Go servers - a request comes in, a
//! route hands back JSON, and anything that is not `/api/...` is served from the
//! prebuilt frontend - and it speaks the same JSON contract, so
//! `frontend/dist` runs against this server unchanged
//! (`../../DESIGN.md` §12).
//!
//! What it deliberately does *not* do: TLS, HTTP/2, chunked request bodies,
//! keep-alive.  It answers one request per connection and closes, which is what
//! a localhost model server needs and nothing more.  Every connection is a
//! thread; the service behind it takes a lock, so the model is never touched by
//! two requests at once.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use crate::json::{parse, Json};
use crate::log::Level;

/// How much of a request body the server will read (16 MiB): an uploaded corpus
/// arrives this way, and the cap is what keeps a bad `Content-Length` from
/// asking for the machine's memory.
pub const MAX_BODY: usize = 16 * 1024 * 1024;

/// One request, parsed.
pub struct Request {
    pub method: String,
    pub path: String,
    /// The query string as `[(key, value)]`, percent-decoded.
    pub query: Vec<(String, String)>,
    /// The body parsed as JSON (`Json::Null` when there is none, or when it
    /// is not JSON - an upload arrives as multipart or as raw bytes).
    pub body: Json,
    /// The body as it arrived.
    pub raw: Vec<u8>,
    /// The headers, names lower-cased, in the order sent.
    pub headers: Vec<(String, String)>,
}

impl Request {
    /// A request with a JSON body and nothing else - what a route's own tests
    /// build.
    pub fn json(method: &str, path: &str, body: Json) -> Request {
        Request {
            method: method.to_string(),
            path: path.to_string(),
            query: Vec::new(),
            raw: body.render(0).into_bytes(),
            body,
            headers: vec![("content-type".to_string(), "application/json".to_string())],
        }
    }

    /// A header's value (the name is matched without regard to case).
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers.iter().find(|(k, _)| *k == name).map(|(_, v)| v.as_str())
    }

    /// The media type the body was sent as, without its parameters.
    pub fn content_type(&self) -> String {
        self.header("content-type")
            .unwrap_or("")
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase()
    }

    /// A query parameter, last one wins.
    pub fn query(&self, key: &str) -> Option<&str> {
        self.query.iter().rev().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    /// A query parameter as a number, with a default and a clear error.
    pub fn query_usize(&self, key: &str, fallback: usize) -> Result<usize, ApiError> {
        match self.query(key) {
            None => Ok(fallback),
            Some(raw) => raw.parse().map_err(|_| {
                ApiError::bad_request(format!("query parameter {key:?} must be an integer (got {raw:?})"))
            }),
        }
    }

    /// A body field as text.
    pub fn text(&self, key: &str, fallback: &str) -> String {
        self.body.at(key).as_str().unwrap_or(fallback).to_string()
    }

    /// A body field as a number.
    pub fn number(&self, key: &str, fallback: f64) -> Result<f64, ApiError> {
        match self.body.get(key) {
            None | Some(Json::Null) => Ok(fallback),
            Some(value) => value
                .as_f64()
                .ok_or_else(|| ApiError::bad_request(format!("{key:?} must be a number"))),
        }
    }

    /// A body field as a count.
    pub fn usize(&self, key: &str, fallback: usize) -> Result<usize, ApiError> {
        let value = self.number(key, fallback as f64)?;
        if value < 0.0 {
            return Err(ApiError::bad_request(format!("{key:?} must be >= 0")));
        }
        Ok(value as usize)
    }

    /// A body field as a flag.
    pub fn flag(&self, key: &str, fallback: bool) -> bool {
        self.body.at(key).as_bool().unwrap_or(fallback)
    }

    /// A body field as a list of texts: `{"texts": [...]}`, or one `{"text": "..."}`.
    pub fn texts(&self, key: &str) -> Vec<String> {
        match self.body.get(key) {
            Some(Json::Arr(items)) => items.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect(),
            Some(Json::Str(one)) => one
                .lines()
                .map(|l| l.to_string())
                .filter(|l| !l.trim().is_empty())
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// An error with the status code it should be answered with.
#[derive(Debug)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
}

impl ApiError {
    pub fn bad_request<S: Into<String>>(message: S) -> ApiError {
        ApiError {
            status: 400,
            message: message.into(),
        }
    }
    pub fn not_found<S: Into<String>>(message: S) -> ApiError {
        ApiError {
            status: 404,
            message: message.into(),
        }
    }
    pub fn conflict<S: Into<String>>(message: S) -> ApiError {
        ApiError {
            status: 409,
            message: message.into(),
        }
    }
    /// An error with any status: 502 for an upstream (an LLM) that failed,
    /// 503 for one that is not there, 500 for the server's own fault.
    pub fn with_status<S: Into<String>>(status: u16, message: S) -> ApiError {
        ApiError {
            status,
            message: message.into(),
        }
    }
}

impl From<String> for ApiError {
    /// Every model error is a bad request until something says otherwise.
    fn from(message: String) -> ApiError {
        ApiError::bad_request(message)
    }
}

/// What the server's own lines are filed under.
const LOG: &str = "http";

/// What a route hands back.
///
/// A route that answers 200 returns the document alone; one that answers with
/// another success code (202 for a job that was accepted and is still running,
/// as the Python and Go servers answer) wraps it with [`accepted`].
pub type Answer = Result<Json, ApiError>;

/// A successful answer that is not a plain 200: the document and its status.
///
/// Only the two job routes use it, and only to say 202 - the code the Python
/// and Go servers answer `train` and `2nrl` with, because the work has been
/// accepted rather than finished.
pub fn accepted(doc: Json) -> Json {
    Json::Obj(vec![
        (STATUS_KEY.to_string(), Json::Int(202)),
        ("body".to_string(), doc),
    ])
}

/// The key [`accepted`] marks a status with; no API document has a key like it.
const STATUS_KEY: &str = "__status";

/// Splits a route's answer into the status and the document it should be sent with.
fn status_of(doc: Json) -> (u16, Json) {
    if let Json::Obj(pairs) = &doc {
        if pairs.len() == 2 && pairs[0].0 == STATUS_KEY {
            if let (Some(status), body) = (pairs[0].1.as_i64(), pairs[1].1.clone()) {
                return (status as u16, body);
            }
        }
    }
    (200, doc)
}

/// What answers one route: a plain function of the server's state and the request.
///
/// The state arrives as the `Arc` the server holds rather than a borrow of it,
/// so a route that starts background work can clone it and hand it to a thread
/// (which is how `train` and `2nrl` answer before the work is done).
pub type Handler<S> = fn(&Arc<S>, &Request) -> Answer;

/// What answers a route that streams: it writes its answer into a [`Sink`],
/// one JSON object per line, as it happens.  A request it refuses before
/// anything was sent gets that error as an ordinary JSON answer; a failure
/// after the first line becomes the stream's last event, `{"event": "error"}`,
/// because the status line has already gone.
pub type StreamHandler<S> = fn(&Arc<S>, &Request, &mut Sink) -> Result<(), ApiError>;

/// Where a streaming route writes: chunked `application/x-ndjson`, one JSON
/// object per line, each flushed as it is sent, so a client reads the events
/// while the model is still talking.  The headers wait for the first event.
/// A client that goes away is remembered ([`Sink::failed`]) and everything
/// after that is dropped rather than reported, since the conversation behind
/// the stream finishes either way.
pub struct Sink<'a> {
    out: &'a mut dyn Write,
    started: bool,
    failed: bool,
}

impl<'a> Sink<'a> {
    pub fn new(out: &'a mut dyn Write) -> Sink<'a> {
        Sink {
            out,
            started: false,
            failed: false,
        }
    }

    /// Whether the headers have gone (after which the status cannot change).
    pub fn started(&self) -> bool {
        self.started
    }

    /// Whether a write failed: the client went away.
    pub fn failed(&self) -> bool {
        self.failed
    }

    /// Sends one event as its own line.
    pub fn send(&mut self, doc: &Json) {
        if self.failed {
            return;
        }
        let line = doc.render(0) + "\n";
        let mut chunk = String::new();
        if !self.started {
            self.started = true;
            chunk.push_str(
                "HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson; charset=utf-8\r\n\
                 Transfer-Encoding: chunked\r\nCache-Control: no-store\r\n\
                 Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
            );
        }
        chunk.push_str(&format!("{:x}\r\n{line}\r\n", line.len()));
        if self
            .out
            .write_all(chunk.as_bytes())
            .and_then(|_| self.out.flush())
            .is_err()
        {
            self.failed = true;
        }
    }

    /// Ends the stream (the empty chunk), once something was sent.
    pub fn finish(&mut self) {
        if self.started && !self.failed && self.out.write_all(b"0\r\n\r\n").and_then(|_| self.out.flush()).is_err() {
            self.failed = true;
        }
    }
}

/// A server: the routes, and where the prebuilt frontend lives.
pub struct Server<S> {
    state: Arc<S>,
    routes: Vec<(&'static str, &'static str, Handler<S>)>,
    stream_routes: Vec<(&'static str, &'static str, StreamHandler<S>)>,
    frontend: Option<PathBuf>,
}

impl<S: Send + Sync + 'static> Server<S> {
    pub fn new(state: Arc<S>) -> Server<S> {
        Server {
            state,
            routes: Vec::new(),
            stream_routes: Vec::new(),
            frontend: None,
        }
    }

    /// Adds one route; the first match wins.
    pub fn route(&mut self, method: &'static str, path: &'static str, handler: Handler<S>) {
        self.routes.push((method, path, handler));
    }

    /// Adds one route that streams its answer ([`StreamHandler`]).
    pub fn stream_route(&mut self, method: &'static str, path: &'static str, handler: StreamHandler<S>) {
        self.stream_routes.push((method, path, handler));
    }

    /// Every route, as `"METHOD /path"`, in the order added (the streaming ones last).
    pub fn routes(&self) -> Vec<String> {
        self.routes
            .iter()
            .map(|(m, p, _)| format!("{m} {p}"))
            .chain(self.stream_routes.iter().map(|(m, p, _)| format!("{m} {p}")))
            .collect()
    }

    /// The state the routes are answered from.
    pub fn state(&self) -> &Arc<S> {
        &self.state
    }

    /// The directory the static files are served from (the built frontend).
    pub fn frontend<P: Into<PathBuf>>(&mut self, dir: P) {
        let dir = dir.into();
        if dir.is_dir() {
            self.frontend = Some(dir);
        }
    }

    /// Serves until the process is stopped; returns the address it bound to.
    pub fn serve(self, host: &str, port: u16) -> Result<(), String> {
        let listener = TcpListener::bind((host, port)).map_err(|err| format!("cannot bind {host}:{port}: {err}"))?;
        crate::log_info!(LOG, "listening on http://{host}:{port} ({} routes)", self.routes.len());
        let shared = Arc::new(self);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let server = Arc::clone(&shared);
                    std::thread::spawn(move || {
                        if let Err(err) = server.handle(stream) {
                            // the client hung up, or the socket went away mid-answer
                            crate::log_debug!(LOG, "connection ended: {err}");
                        }
                    });
                }
                // one refused connection is not the end of the server
                Err(err) => {
                    crate::log_warn!(LOG, "refused connection: {err}");
                    continue;
                }
            }
        }
        Ok(())
    }

    fn handle(&self, mut stream: TcpStream) -> std::io::Result<()> {
        let started = std::time::Instant::now();
        let request = match read_request(&mut stream) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(message) => {
                crate::log_warn!(LOG, "400 (unreadable request): {message}");
                return write_json(&mut stream, 400, &error_doc(&message));
            }
        };
        // an API route, the prebuilt frontend, or a 404 that says which
        if let Some(handler) = self.match_route(&request) {
            let answer = handler(&self.state, &request);
            let (status, doc) = match answer {
                Ok(doc) => status_of(doc),
                Err(err) => (err.status, error_doc(&err.message)),
            };
            // one line per request, with what it cost: a 4xx or 5xx is worth a
            // warning because it is the server refusing, and a 2xx is the
            // ordinary traffic that only a debug run wants to see
            let level = if status >= 400 { Level::Warn } else { Level::Debug };
            crate::log_at!(
                LOG,
                level,
                "{} {} -> {status} in {:.1}ms{}",
                request.method,
                request.path,
                started.elapsed().as_secs_f64() * 1000.0,
                match doc.at("error").as_str() {
                    Some(why) => format!(": {why}"),
                    None => String::new(),
                }
            );
            return write_json(&mut stream, status, &doc);
        }
        if let Some(handler) = self.match_stream_route(&request) {
            let mut sink = Sink::new(&mut stream);
            let outcome = handler(&self.state, &request, &mut sink);
            let (level, status, why) = match &outcome {
                Ok(()) => (Level::Debug, 200, String::new()),
                Err(err) if !sink.started() => (Level::Warn, err.status, format!(": {}", err.message)),
                Err(err) => (
                    Level::Warn,
                    200,
                    format!(" (the stream ended in an error: {})", err.message),
                ),
            };
            crate::log_at!(
                LOG,
                level,
                "{} {} -> {status} in {:.1}ms{why}",
                request.method,
                request.path,
                started.elapsed().as_secs_f64() * 1000.0
            );
            match outcome {
                Err(err) if !sink.started() => return write_json(&mut stream, err.status, &error_doc(&err.message)),
                Err(err) => sink.send(&Json::obj([
                    ("event", Json::str("error")),
                    ("error", Json::str(err.message)),
                ])),
                Ok(()) => {}
            }
            if !sink.started() {
                // a stream that had nothing to say is still a stream: an empty body
                sink.send(&Json::obj([("event", Json::str("done"))]));
            }
            sink.finish();
            return Ok(());
        }
        if request.path.starts_with("/api/") {
            crate::log_warn!(LOG, "404 no route {} {}", request.method, request.path);
            return write_json(
                &mut stream,
                404,
                &error_doc(&format!("no route {} {}", request.method, request.path)),
            );
        }
        crate::log_trace!(LOG, "{} {} (static)", request.method, request.path);
        self.serve_static(&mut stream, &request)
    }

    fn match_route(&self, request: &Request) -> Option<Handler<S>> {
        self.routes
            .iter()
            .find(|(method, path, _)| *path == request.path && *method == request.method)
            .map(|(_, _, handler)| *handler)
    }

    fn match_stream_route(&self, request: &Request) -> Option<StreamHandler<S>> {
        self.stream_routes
            .iter()
            .find(|(method, path, _)| *path == request.path && *method == request.method)
            .map(|(_, _, handler)| *handler)
    }

    fn serve_static(&self, stream: &mut TcpStream, request: &Request) -> std::io::Result<()> {
        let Some(root) = &self.frontend else {
            return write_json(stream, 404, &error_doc("no frontend directory is being served"));
        };
        if request.method != "GET" && request.method != "HEAD" {
            return write_json(stream, 405, &error_doc("method not allowed"));
        }
        let relative = request.path.trim_start_matches('/');
        let mut file = safe_join(root, relative);
        if file.is_none() || file.as_ref().is_some_and(|p| p.is_dir()) {
            file = safe_join(root, "index.html");
        }
        match file.and_then(|path| std::fs::read(&path).ok().map(|bytes| (path, bytes))) {
            // a single-page app: an unknown path is one of its routes, not a missing file
            None => match std::fs::read(root.join("index.html")) {
                Ok(bytes) => write_bytes(stream, 200, "text/html; charset=utf-8", &bytes),
                Err(_) => write_json(stream, 404, &error_doc("not found")),
            },
            Some((path, bytes)) => write_bytes(stream, 200, content_type(&path), &bytes),
        }
    }
}

/// `root/relative`, or `None` when the path tries to leave the directory.
fn safe_join(root: &Path, relative: &str) -> Option<PathBuf> {
    if relative.is_empty() {
        return Some(root.join("index.html"));
    }
    let candidate = Path::new(relative);
    for part in candidate.components() {
        match part {
            Component::Normal(_) => {}
            // "..", a root, a prefix: anything that could climb out
            _ => return None,
        }
    }
    Some(root.join(candidate))
}

fn content_type(path: &Path) -> &'static str {
    match path.extension().and_then(|e| e.to_str()).unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "text/javascript; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "svg" => "image/svg+xml",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "ico" => "image/x-icon",
        "woff2" => "font/woff2",
        "map" => "application/json; charset=utf-8",
        _ => "application/octet-stream",
    }
}

/// Reads one request; `Ok(None)` when the connection closed before sending one.
fn read_request(stream: &mut TcpStream) -> Result<Option<Request>, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);
    let mut line = String::new();
    if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
        return Ok(None);
    }
    let mut parts = line.trim_end().split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    if method.is_empty() {
        return Err("empty request line".to_string());
    }
    let mut length = 0usize;
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().unwrap_or(0);
            }
            headers.push((name.trim().to_ascii_lowercase(), value.trim().to_string()));
        }
    }
    if length > MAX_BODY {
        return Err(format!("request body larger than {MAX_BODY} bytes"));
    }
    let mut body = vec![0u8; length];
    if length > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), parse_query(query)),
        None => (target, Vec::new()),
    };
    let parsed = match std::str::from_utf8(&body) {
        Ok(text) if !text.trim().is_empty() => parse(text).unwrap_or(Json::Null),
        _ => Json::Null,
    };
    Ok(Some(Request {
        method,
        path,
        query,
        body: parsed,
        raw: body,
        headers,
    }))
}

fn parse_query(query: &str) -> Vec<(String, String)> {
    query
        .split('&')
        .filter(|part| !part.is_empty())
        .map(|part| match part.split_once('=') {
            Some((k, v)) => (percent_decode(k), percent_decode(v)),
            None => (percent_decode(part), String::new()),
        })
        .collect()
}

/// `%xx` and `+`, which is all a query string needs.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(byte) => {
                        out.push(byte);
                        i += 3;
                    }
                    Err(_) => {
                        out.push(bytes[i]);
                        i += 1;
                    }
                }
            }
            byte => {
                out.push(byte);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn error_doc(message: &str) -> Json {
    Json::obj([("error", Json::str(message))])
}

fn write_json(stream: &mut TcpStream, status: u16, doc: &Json) -> std::io::Result<()> {
    write_bytes(
        stream,
        status,
        "application/json; charset=utf-8",
        doc.render(0).as_bytes(),
    )
}

fn write_bytes(stream: &mut TcpStream, status: u16, content_type: &str, body: &[u8]) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        reason = reason(status),
        len = body.len(),
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        202 => "Accepted",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "OK",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_query_string_comes_apart() {
        let parsed = parse_query("limit=20&node=the%20cat&flag");
        assert_eq!(parsed[0], ("limit".to_string(), "20".to_string()));
        assert_eq!(parsed[1], ("node".to_string(), "the cat".to_string()));
        assert_eq!(parsed[2], ("flag".to_string(), String::new()));
        assert_eq!(percent_decode("a+b%2Fc"), "a b/c");
    }

    #[test]
    fn a_sink_streams_one_line_per_event_with_late_headers() {
        let mut out: Vec<u8> = Vec::new();
        {
            let mut sink = Sink::new(&mut out);
            assert!(!sink.started());
            sink.send(&Json::obj([
                ("event", Json::str("look")),
                ("from", Json::str("the cat")),
            ]));
            assert!(sink.started());
            sink.send(&Json::obj([("event", Json::str("done"))]));
            sink.finish();
            assert!(!sink.failed());
        }
        let text = String::from_utf8(out).unwrap();
        let (head, body) = text.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("HTTP/1.1 200 OK\r\n"), "{head}");
        assert!(head.contains("Content-Type: application/x-ndjson; charset=utf-8\r\n"));
        assert!(head.contains("Transfer-Encoding: chunked\r\n"));
        assert!(!head.contains("Content-Length"));
        let first = "{\"event\":\"look\",\"from\":\"the cat\"}\n";
        let second = "{\"event\":\"done\"}\n";
        assert_eq!(
            body,
            format!(
                "{:x}\r\n{first}\r\n{:x}\r\n{second}\r\n0\r\n\r\n",
                first.len(),
                second.len()
            )
        );
        // nothing sent, nothing ended: an untouched sink writes no bytes at all
        let mut out: Vec<u8> = Vec::new();
        Sink::new(&mut out).finish();
        assert!(out.is_empty());
    }

    #[test]
    fn a_static_path_cannot_climb_out() {
        let root = Path::new("/srv/dist");
        assert_eq!(safe_join(root, "index.html"), Some(root.join("index.html")));
        assert_eq!(safe_join(root, "assets/app.js"), Some(root.join("assets/app.js")));
        assert_eq!(safe_join(root, "../secrets"), None);
        assert_eq!(safe_join(root, "/etc/passwd"), None);
        assert_eq!(safe_join(root, ""), Some(root.join("index.html")));
    }

    #[test]
    fn a_content_type_follows_the_extension() {
        assert_eq!(content_type(Path::new("a/index.html")), "text/html; charset=utf-8");
        assert_eq!(content_type(Path::new("a/app.js")), "text/javascript; charset=utf-8");
        assert_eq!(content_type(Path::new("a/app.css")), "text/css; charset=utf-8");
        assert_eq!(content_type(Path::new("a/thing.bin")), "application/octet-stream");
    }
}
