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
    /// The body parsed as JSON (`Json::Null` when there is none).
    pub body: Json,
}

impl Request {
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
}

impl From<String> for ApiError {
    /// Every model error is a bad request until something says otherwise.
    fn from(message: String) -> ApiError {
        ApiError::bad_request(message)
    }
}

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

/// A server: the routes, and where the prebuilt frontend lives.
pub struct Server<S> {
    state: Arc<S>,
    routes: Vec<(&'static str, &'static str, Handler<S>)>,
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
        self.routes.push((method, path, handler));
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
        let shared = Arc::new(self);
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let server = Arc::clone(&shared);
                    std::thread::spawn(move || {
                        let _ = server.handle(stream);
                    });
                }
                // one refused connection is not the end of the server
                Err(_) => continue,
            }
        }
        Ok(())
    }

    fn handle(&self, mut stream: TcpStream) -> std::io::Result<()> {
        let request = match read_request(&mut stream) {
            Ok(Some(request)) => request,
            Ok(None) => return Ok(()),
            Err(message) => return write_json(&mut stream, 400, &error_doc(&message)),
        };
        // an API route, the prebuilt frontend, or a 404 that says which
        if let Some(handler) = self.match_route(&request) {
            return match handler(&self.state, &request) {
                Ok(doc) => {
                    let (status, doc) = status_of(doc);
                    write_json(&mut stream, status, &doc)
                }
                Err(err) => write_json(&mut stream, err.status, &error_doc(&err.message)),
            };
        }
        if request.path.starts_with("/api/") {
            return write_json(
                &mut stream,
                404,
                &error_doc(&format!("no route {} {}", request.method, request.path)),
            );
        }
        self.serve_static(&mut stream, &request)
    }

    fn match_route(&self, request: &Request) -> Option<Handler<S>> {
        self.routes
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
    let body = match String::from_utf8(body) {
        Ok(text) if !text.trim().is_empty() => parse(&text).unwrap_or(Json::Null),
        _ => Json::Null,
    };
    Ok(Some(Request {
        method,
        path,
        query,
        body,
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
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        409 => "Conflict",
        500 => "Internal Server Error",
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
