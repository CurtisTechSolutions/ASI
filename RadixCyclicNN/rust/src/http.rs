//! The HTTP transport: a small HTTP/1.1 server, written out like everything
//! else here because the crate has no dependencies.
//!
//! It is the same shape as the Python and Go servers - a request comes in, a
//! route hands back JSON, and anything that is not `/api/...` is served from the
//! prebuilt frontend - and it speaks the same JSON contract, so
//! `frontend/dist` runs against this server unchanged
//! (`../../DESIGN.md` §12).
//!
//! A route reads its body one of two ways.  Most take JSON and get it parsed
//! ([`Server::route`]): the body is read whole, under [`MAX_BODY`].  A
//! *streaming* route ([`Server::stream_route`]) is handed the body as it
//! arrives instead, with no cap on its size - `POST /api/uploads` is one, so an
//! archive of any size goes straight to disk, as it does on the Go server
//! (D-035), and a client that announces `Expect: 100-continue` (curl, for a
//! large file) gets the nod before it sends.
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

/// How much of a request body a route that reads it whole will take (16 MiB):
/// the cap is what keeps a bad `Content-Length` from asking for the machine's
/// memory.  A streaming route ([`Server::stream_route`]) has no cap - an
/// upload is read as it arrives and never held.
pub const MAX_BODY: usize = 16 * 1024 * 1024;

/// One request, parsed.
pub struct Request {
    pub method: String,
    pub path: String,
    /// The query string as `[(key, value)]`, percent-decoded.
    pub query: Vec<(String, String)>,
    /// The body parsed as JSON (`Json::Null` when there is none, or when it
    /// is not JSON - an upload arrives as multipart or as raw bytes).  A
    /// streaming route's request holds no body at all: the route reads it.
    pub body: Json,
    /// The body as it arrived (empty on a streaming route).
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

/// What answers a streaming route: the state, the request without its body
/// (`raw` is empty and `body` is `Null`), and the body itself as it arrives.
///
/// The reader ends where `Content-Length` says the body ends, and fails with
/// `UnexpectedEof` when the client sent less than it declared, so nothing
/// short is ever taken for whole.  The route need not drain it.
pub type StreamHandler<S> = fn(&Arc<S>, &Request, &mut dyn Read) -> Answer;

/// A route, by how it takes its body.
enum Route<S> {
    /// The body is read whole (under [`MAX_BODY`]) and parsed as JSON first.
    Json(Handler<S>),
    /// The body is handed over as it arrives, of any size.
    Stream(StreamHandler<S>),
}

// a function pointer copies whatever `S` is, which a derive would not know
impl<S> Clone for Route<S> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<S> Copy for Route<S> {}

/// A server: the routes, and where the prebuilt frontend lives.
pub struct Server<S> {
    state: Arc<S>,
    routes: Vec<(&'static str, &'static str, Route<S>)>,
    frontend: Option<PathBuf>,
}

impl<S: Send + Sync + 'static> Server<S> {
    pub fn new(state: Arc<S>) -> Server<S> {
        Server {
            state,
            routes: Vec::new(),
            frontend: None,
        }
    }

    /// Adds one route; the first match wins.
    pub fn route(&mut self, method: &'static str, path: &'static str, handler: Handler<S>) {
        self.routes.push((method, path, Route::Json(handler)));
    }

    /// Adds a route that reads its body as it arrives, with no cap on its
    /// size: the handler gets the request without a body, and a reader of it.
    pub fn stream_route(&mut self, method: &'static str, path: &'static str, handler: StreamHandler<S>) {
        self.routes.push((method, path, Route::Stream(handler)));
    }

    /// Every route, as `"METHOD /path"`, in the order added.
    pub fn routes(&self) -> Vec<String> {
        self.routes.iter().map(|(m, p, _)| format!("{m} {p}")).collect()
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
        let head = match read_head(&mut stream) {
            Ok(Some(head)) => head,
            Ok(None) => return Ok(()),
            Err((status, message)) => {
                crate::log_warn!(LOG, "{status} (unreadable request): {message}");
                return write_json(&mut stream, status, &error_doc(&message));
            }
        };
        let Head {
            mut request,
            length,
            mut reader,
            expects_continue,
        } = head;
        let route = self.match_route(&request);
        // a streaming route takes the body as it arrives; every other request
        // reads it whole first, under the cap that keeps a bad Content-Length
        // from asking for the machine's memory
        let answer = match route {
            Some(Route::Stream(handler)) => {
                if expects_continue && length > 0 {
                    send_continue(&mut stream)?;
                }
                let mut body = Body {
                    reader: &mut reader,
                    remaining: length as u64,
                };
                let answer = handler(&self.state, &request, &mut body);
                // what the client is still sending is taken off the wire, up to
                // a point, so it reads the answer rather than a reset
                body.discard(DISCARD_BYTES);
                Some(answer)
            }
            other => {
                if length > MAX_BODY {
                    let message = format!("request body larger than {MAX_BODY} bytes");
                    crate::log_warn!(LOG, "400 (unreadable request): {message}");
                    return write_json(&mut stream, 400, &error_doc(&message));
                }
                if expects_continue && length > 0 {
                    send_continue(&mut stream)?;
                }
                match read_whole(&mut reader, length) {
                    Ok(raw) => {
                        request.body = parse_json(&raw);
                        request.raw = raw;
                    }
                    Err(message) => {
                        crate::log_warn!(LOG, "400 (unreadable request): {message}");
                        return write_json(&mut stream, 400, &error_doc(&message));
                    }
                }
                match other {
                    Some(Route::Json(handler)) => Some(handler(&self.state, &request)),
                    _ => None,
                }
            }
        };
        // an API route, the prebuilt frontend, or a 404 that says which
        if let Some(answer) = answer {
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

    fn match_route(&self, request: &Request) -> Option<Route<S>> {
        self.routes
            .iter()
            .find(|(method, path, _)| *path == request.path && *method == request.method)
            .map(|(_, _, route)| *route)
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

/// A request's head, read: the request without its body, how long the body
/// says it is, and the reader it is still in.
struct Head {
    request: Request,
    length: usize,
    reader: BufReader<TcpStream>,
    /// `Expect: 100-continue`: the client waits for a nod before it sends the body.
    expects_continue: bool,
}

/// Reads one request's line and headers; `Ok(None)` when the connection
/// closed before sending one, `Err((status, message))` for one that cannot be
/// read.  The body stays in the reader.
fn read_head(stream: &mut TcpStream) -> Result<Option<Head>, (u16, String)> {
    let unreadable = |e: std::io::Error| (400, e.to_string());
    let mut reader = BufReader::new(stream.try_clone().map_err(unreadable)?);
    let mut line = String::new();
    if reader.read_line(&mut line).map_err(unreadable)? == 0 {
        return Ok(None);
    }
    let mut parts = line.trim_end().split(' ');
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("/").to_string();
    if method.is_empty() {
        return Err((400, "empty request line".to_string()));
    }
    let mut length = 0usize;
    let mut headers: Vec<(String, String)> = Vec::new();
    loop {
        let mut header = String::new();
        if reader.read_line(&mut header).map_err(unreadable)? == 0 {
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
    // a chunked body has no length to read by: 411, as the Python server answers
    if headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked"))
    {
        return Err((
            411,
            "chunked request bodies are not supported; send a Content-Length header".to_string(),
        ));
    }
    let expects_continue = headers
        .iter()
        .any(|(name, value)| name == "expect" && value.eq_ignore_ascii_case("100-continue"));
    let (path, query) = match target.split_once('?') {
        Some((path, query)) => (path.to_string(), parse_query(query)),
        None => (target, Vec::new()),
    };
    Ok(Some(Head {
        request: Request {
            method,
            path,
            query,
            body: Json::Null,
            raw: Vec::new(),
            headers,
        },
        length,
        reader,
        expects_continue,
    }))
}

/// Reads a body whole, as a route that takes JSON needs it (`length` is
/// under [`MAX_BODY`] by the time this is called).
fn read_whole(reader: &mut BufReader<TcpStream>, length: usize) -> Result<Vec<u8>, String> {
    let mut body = vec![0u8; length];
    if length > 0 {
        reader.read_exact(&mut body).map_err(|e| e.to_string())?;
    }
    Ok(body)
}

/// The JSON a body holds; `Null` when it holds none, or something else.
fn parse_json(body: &[u8]) -> Json {
    match std::str::from_utf8(body) {
        Ok(text) if !text.trim().is_empty() => parse(text).unwrap_or(Json::Null),
        _ => Json::Null,
    }
}

/// `100 Continue`: the nod a client that sent `Expect: 100-continue` waits for
/// before it sends the body (curl does, for a large upload).
fn send_continue(stream: &mut TcpStream) -> std::io::Result<()> {
    stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n")?;
    stream.flush()
}

/// How much of a body a streaming route left unread is taken off the wire
/// before the connection closes, so the client reads the answer rather than a
/// reset (the Go server does the same).
const DISCARD_BYTES: u64 = 1 << 20;

/// The body of a request to a streaming route, as it arrives: it ends where
/// `Content-Length` says, and a client that sends less than it declared is an
/// `UnexpectedEof`, so nothing short is ever taken for whole.
struct Body<'a> {
    reader: &'a mut BufReader<TcpStream>,
    remaining: u64,
}

impl Read for Body<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if self.remaining == 0 || out.is_empty() {
            return Ok(0);
        }
        let want = out.len().min(self.remaining.min(usize::MAX as u64) as usize);
        let n = self.reader.read(&mut out[..want])?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "incomplete request body",
            ));
        }
        self.remaining -= n as u64;
        Ok(n)
    }
}

impl Body<'_> {
    /// Takes up to `limit` of what is left off the wire and drops it.
    fn discard(&mut self, limit: u64) {
        let _ = std::io::copy(&mut self.by_ref().take(limit), &mut std::io::sink());
    }
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
    let mut answer = format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {len}\r\n\
         Access-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        reason = reason(status),
        len = body.len(),
    )
    .into_bytes();
    // in one write: the head alone would go at once and the body wait for its
    // acknowledgement (Nagle), and a connection closed on a body it did not
    // read - a refused upload - discards what is still waiting to go
    answer.extend_from_slice(body);
    stream.write_all(&answer)?;
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
        411 => "Length Required",
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
