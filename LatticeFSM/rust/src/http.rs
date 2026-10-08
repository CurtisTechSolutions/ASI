//! A small HTTP/1.1 server, written out because the crate has no dependencies.
//!
//! One request per connection, then close; every connection is a thread, and
//! the handler behind it takes a lock so the machine is never touched by two
//! requests at once.  No TLS, no HTTP/2, no chunked bodies, no keep-alive: what
//! a localhost model server needs and nothing more.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

use crate::json::{parse, Json};

/// How much of a request body is read (4 MiB): the cap keeps a bad `Content-Length` from asking for memory.
pub const MAX_BODY: usize = 4 * 1024 * 1024;

/// One request, parsed.
pub struct Request {
    pub method: String,
    pub path: String,
    /// The query string as `[(key, value)]`, percent-decoded.
    pub query: Vec<(String, String)>,
    /// The body parsed as JSON (`Json::Null` when there is none or it is not JSON).
    pub body: Json,
}

impl Request {
    pub fn param(&self, key: &str) -> Option<&str> {
        self.query.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }
}

/// One response: a status, a content type and a body.
pub struct Response {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(status: u16, value: &Json) -> Response {
        Response {
            status,
            content_type: "application/json; charset=utf-8",
            body: value.dump().into_bytes(),
        }
    }

    pub fn ok(value: &Json) -> Response {
        Response::json(200, value)
    }

    pub fn error(status: u16, message: &str) -> Response {
        Response::json(status, &Json::object().with("error", message.into()))
    }

    pub fn html(text: &str) -> Response {
        Response {
            status: 200,
            content_type: "text/html; charset=utf-8",
            body: text.as_bytes().to_vec(),
        }
    }
}

pub type Handler = dyn Fn(&Request) -> Response + Send + Sync + 'static;

/// Serve `handler` on `addr` forever (or until `max_requests`, for tests).
pub fn serve(addr: &str, handler: Arc<Handler>, max_requests: Option<usize>) -> Result<(), String> {
    let listener = TcpListener::bind(addr).map_err(|e| format!("cannot listen on {addr}: {e}"))?;
    serve_on(listener, handler, max_requests)
}

pub fn serve_on(listener: TcpListener, handler: Arc<Handler>, max_requests: Option<usize>) -> Result<(), String> {
    let mut served = 0usize;
    for stream in listener.incoming() {
        let stream = match stream {
            Ok(s) => s,
            Err(_) => continue,
        };
        let handler = Arc::clone(&handler);
        let worker = std::thread::spawn(move || handle(stream, &*handler));
        if let Some(max) = max_requests {
            let _ = worker.join();
            served += 1;
            if served >= max {
                break;
            }
        }
    }
    Ok(())
}

fn handle(mut stream: TcpStream, handler: &Handler) {
    let response = match read_request(&mut stream) {
        Ok(req) => handler(&req),
        Err(msg) => Response::error(400, &msg),
    };
    let _ = write_response(&mut stream, &response);
}

fn read_request(stream: &mut TcpStream) -> Result<Request, String> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).map_err(|e| e.to_string())?;
    let mut parts = line.split_whitespace();
    let method = parts.next().ok_or("empty request line")?.to_string();
    let target = parts.next().ok_or("no request target")?.to_string();
    let mut length = 0usize;
    loop {
        let mut header = String::new();
        reader.read_line(&mut header).map_err(|e| e.to_string())?;
        let header = header.trim_end();
        if header.is_empty() {
            break;
        }
        if let Some((name, value)) = header.split_once(':') {
            if name.trim().eq_ignore_ascii_case("content-length") {
                length = value.trim().parse().map_err(|_| "bad Content-Length")?;
            }
        }
    }
    if length > MAX_BODY {
        return Err(format!("body of {length} bytes is over the {MAX_BODY} byte cap"));
    }
    let mut raw = vec![0u8; length];
    if length > 0 {
        reader.read_exact(&mut raw).map_err(|e| e.to_string())?;
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), parse_query(q)),
        None => (target, Vec::new()),
    };
    let body = if raw.is_empty() {
        Json::Null
    } else {
        parse(&String::from_utf8_lossy(&raw)).unwrap_or(Json::Null)
    };
    Ok(Request {
        method,
        path,
        query,
        body,
    })
}

fn write_response(stream: &mut TcpStream, r: &Response) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n",
        r.status,
        reason(r.status),
        r.content_type,
        r.body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(&r.body)?;
    stream.flush()
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        500 => "Internal Server Error",
        _ => "Unknown",
    }
}

/// `a=1&b=x%20y` as pairs, percent-decoded (`+` is a space).
pub fn parse_query(q: &str) -> Vec<(String, String)> {
    q.split('&')
        .filter(|s| !s.is_empty())
        .map(|pair| match pair.split_once('=') {
            Some((k, v)) => (unquote(k), unquote(v)),
            None => (unquote(pair), String::new()),
        })
        .collect()
}

pub fn unquote(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 2 < bytes.len() => match u8::from_str_radix(&s[i + 1..i + 3], 16) {
                Ok(b) => {
                    out.push(b);
                    i += 2;
                }
                Err(_) => out.push(b'%'),
            },
            b => out.push(b),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// A client for the tests: one request, the parsed response.
pub fn request(addr: &str, method: &str, path: &str, body: Option<&Json>) -> Result<(u16, Json), String> {
    let mut stream = TcpStream::connect(addr).map_err(|e| e.to_string())?;
    let payload = body.map(Json::dump).unwrap_or_default();
    let head = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        payload.len()
    );
    stream.write_all(head.as_bytes()).map_err(|e| e.to_string())?;
    stream.write_all(payload.as_bytes()).map_err(|e| e.to_string())?;
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).map_err(|e| e.to_string())?;
    let text = String::from_utf8_lossy(&raw);
    let (head, body) = text.split_once("\r\n\r\n").ok_or("no response head")?;
    let status: u16 = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or("no status")?;
    let json = if head.contains("application/json") {
        parse(body)?
    } else {
        Json::String(body.to_string())
    };
    Ok((status, json))
}
