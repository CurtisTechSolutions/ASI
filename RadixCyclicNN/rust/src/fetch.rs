//! An HTTP client with no dependencies: plain HTTP written out over
//! `TcpStream`, HTTPS handed to the system's `curl`.
//!
//! The crate has no dependencies (D-072), and the standard library speaks no
//! TLS - which is what kept the LLM clients and browsing out of the port.  A
//! TLS stack is not a thing to write out the way the gzip inflate was, so an
//! `https://` URL goes through the one TLS client every machine this runs on
//! already has: `curl`, run as a child process, with the request body on its
//! stdin and the response on its stdout.  It honours `HTTPS_PROXY`, `NO_PROXY`
//! and `CURL_CA_BUNDLE` the way every other tool on the machine does.  Plain
//! `http://` (a local Ollama, a transcription server, a test's own listener)
//! never leaves the process.  D-076 records the trade.
//!
//! One request, one connection (`Connection: close`); a chunked or gzipped
//! answer is decoded; redirects are followed only when asked, because the
//! browsing tool checks every hop itself.
//!
//! # Headers never go on curl's command line
//!
//! A command line is public: every user on the machine can read it from
//! `/proc/<pid>/cmdline` (or `ps`) for as long as the request runs, and an
//! `Authorization: Bearer ...` header there is the API key handed out.  So
//! curl reads the request's headers from a file only this user can read
//! (`0600`, created exclusively, removed as soon as curl exits), and its
//! command line carries nothing but the method, the URL and the knobs.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::json::{parse, Json};

/// The default time one request may take.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

/// One request.
#[derive(Clone, Debug)]
pub struct Request {
    pub method: String,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    pub timeout: Duration,
    /// How many redirects to follow (0 = hand the 3xx back).
    pub redirects: usize,
    /// The most body bytes to read; `0` is no limit.
    pub max_bytes: usize,
    /// Whether the environment's proxy may carry the request (`HTTP_PROXY`,
    /// `HTTPS_PROXY`).  Off for a server the caller knows is on this machine or
    /// its network - a local Ollama - which a proxy could only fail to reach.
    pub proxy: bool,
}

impl Request {
    pub fn new(method: &str, url: &str) -> Request {
        Request {
            method: method.to_string(),
            url: url.to_string(),
            headers: Vec::new(),
            body: Vec::new(),
            timeout: DEFAULT_TIMEOUT,
            redirects: 0,
            max_bytes: 0,
            proxy: true,
        }
    }
    pub fn get(url: &str) -> Request {
        Request::new("GET", url)
    }
    /// A POST of a JSON document.
    pub fn post_json(url: &str, doc: &Json) -> Request {
        let mut r = Request::new("POST", url);
        r.body = doc.render(0).into_bytes();
        r.header("Content-Type", "application/json")
    }
    pub fn header(mut self, name: &str, value: &str) -> Request {
        self.headers.push((name.to_string(), value.to_string()));
        self
    }
    pub fn timeout(mut self, timeout: Duration) -> Request {
        if !timeout.is_zero() {
            self.timeout = timeout;
        }
        self
    }
    pub fn follow(mut self, redirects: usize) -> Request {
        self.redirects = redirects;
        self
    }
    pub fn limit(mut self, max_bytes: usize) -> Request {
        self.max_bytes = max_bytes;
        self
    }
    /// Never through a proxy, whatever the environment says (see [`Request::proxy`]).
    pub fn direct(mut self) -> Request {
        self.proxy = false;
        self
    }

    /// Sends the request.
    pub fn send(&self) -> Result<Response, FetchError> {
        let mut current = self.clone();
        let mut hops = 0;
        loop {
            let url = Url::parse(&current.url)?;
            let response = if url.scheme == "https" {
                via_curl(&current)?
            } else {
                over_tcp(&current, &url)?
            };
            let redirect = matches!(response.status, 301 | 302 | 303 | 307 | 308);
            if !redirect || hops >= current.redirects {
                return Ok(response);
            }
            let Some(location) = response.header("location") else {
                return Ok(response);
            };
            hops += 1;
            current.url = url.join(location);
            if response.status == 303
                || ((response.status == 301 || response.status == 302) && current.method == "POST")
            {
                current.method = "GET".to_string();
                current.body.clear();
            }
        }
    }
}

/// One response.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub status: u16,
    /// Header names lower-cased, in the order received.
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
    /// The URL that answered (after any redirects followed).
    pub url: String,
    /// Whether the body was cut at the request's `max_bytes`.
    pub truncated: bool,
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        let name = name.to_ascii_lowercase();
        self.headers
            .iter()
            .rev()
            .find(|(k, _)| *k == name)
            .map(|(_, v)| v.as_str())
    }
    /// The body as text (invalid UTF-8 replaced).
    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }
    /// The body as JSON.
    pub fn json(&self) -> Result<Json, FetchError> {
        parse(&self.text()).map_err(|err| FetchError::new(format!("the answer is not JSON: {err}")))
    }
    /// Whether the status is 2xx.
    pub fn ok(&self) -> bool {
        (200..300).contains(&self.status)
    }
}

/// Why a request got no answer.
#[derive(Clone, Debug, PartialEq)]
pub struct FetchError {
    pub message: String,
    /// The request ran out of time (rather than being refused or failing).
    pub timeout: bool,
}

impl FetchError {
    pub fn new<S: Into<String>>(message: S) -> FetchError {
        FetchError {
            message: message.into(),
            timeout: false,
        }
    }
    fn timed_out<S: Into<String>>(message: S) -> FetchError {
        FetchError {
            message: message.into(),
            timeout: true,
        }
    }
}

impl std::fmt::Display for FetchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<FetchError> for String {
    fn from(err: FetchError) -> String {
        err.message
    }
}

/// A URL, as far as a request needs one.
#[derive(Clone, Debug, PartialEq)]
pub struct Url {
    pub scheme: String,
    pub host: String,
    pub port: u16,
    /// The path and query, starting with `/`.
    pub target: String,
}

impl Url {
    pub fn parse(url: &str) -> Result<Url, FetchError> {
        let (scheme, rest) = url
            .split_once("://")
            .ok_or_else(|| FetchError::new(format!("{url:?} is not an absolute URL")))?;
        let scheme = scheme.to_ascii_lowercase();
        let default_port = match scheme.as_str() {
            "http" => 80,
            "https" => 443,
            other => return Err(FetchError::new(format!("unsupported URL scheme {other:?}"))),
        };
        let (authority, target) = match rest.find(['/', '?', '#']) {
            Some(at) => (&rest[..at], &rest[at..]),
            None => (rest, "/"),
        };
        let target = target.split('#').next().unwrap_or("/");
        let target = if target.starts_with('?') {
            format!("/{target}")
        } else if target.is_empty() {
            "/".to_string()
        } else {
            target.to_string()
        };
        // credentials in the URL are dropped from the host, never sent by this client
        let authority = authority.rsplit('@').next().unwrap_or(authority);
        let (host, port) = if let Some(inner) = authority.strip_prefix('[') {
            // [::1]:8080
            let (h, tail) = inner
                .split_once(']')
                .ok_or_else(|| FetchError::new(format!("bad IPv6 host in {url:?}")))?;
            let port = match tail.strip_prefix(':') {
                Some(p) => p.parse().map_err(|_| FetchError::new(format!("bad port in {url:?}")))?,
                None => default_port,
            };
            (h.to_string(), port)
        } else {
            match authority.rsplit_once(':') {
                Some((h, p)) => (
                    h.to_string(),
                    p.parse().map_err(|_| FetchError::new(format!("bad port in {url:?}")))?,
                ),
                None => (authority.to_string(), default_port),
            }
        };
        if host.is_empty() {
            return Err(FetchError::new(format!("{url:?} has no host")));
        }
        Ok(Url {
            scheme,
            host: host.to_ascii_lowercase(),
            port,
            target,
        })
    }

    /// The URL a `Location` header points at, resolved against this one.
    pub fn join(&self, location: &str) -> String {
        if location.contains("://") {
            return location.to_string();
        }
        let origin = self.origin();
        if let Some(rest) = location.strip_prefix("//") {
            return format!("{}://{rest}", self.scheme);
        }
        if location.starts_with('/') {
            return format!("{origin}{location}");
        }
        let path = self.target.split('?').next().unwrap_or("/");
        let dir = match path.rfind('/') {
            Some(at) => &path[..=at],
            None => "/",
        };
        format!("{origin}{dir}{location}")
    }

    /// `scheme://host[:port]`, the port left out when it is the scheme's own.
    pub fn origin(&self) -> String {
        let host = if self.host.contains(':') {
            format!("[{}]", self.host)
        } else {
            self.host.clone()
        };
        let default = if self.scheme == "https" { 443 } else { 80 };
        if self.port == default {
            format!("{}://{host}", self.scheme)
        } else {
            format!("{}://{host}:{}", self.scheme, self.port)
        }
    }
}

/// Whether `host` is one `NO_PROXY` exempts.
fn no_proxy(host: &str) -> bool {
    if host == "localhost" || host == "127.0.0.1" || host == "::1" {
        return true;
    }
    let list = std::env::var("NO_PROXY")
        .or_else(|_| std::env::var("no_proxy"))
        .unwrap_or_default();
    list.split(',').map(str::trim).filter(|s| !s.is_empty()).any(|entry| {
        if entry == "*" {
            return true;
        }
        let entry = entry.trim_start_matches('.');
        host == entry || host.ends_with(&format!(".{entry}"))
    })
}

/// The proxy a plain-HTTP request to `host` goes through, when one is set.
fn http_proxy(host: &str) -> Option<Url> {
    if no_proxy(host) {
        return None;
    }
    let proxy = std::env::var("HTTP_PROXY")
        .or_else(|_| std::env::var("http_proxy"))
        .ok()?;
    let proxy = if proxy.contains("://") {
        proxy
    } else {
        format!("http://{proxy}")
    };
    Url::parse(&proxy).ok().filter(|u| u.scheme == "http")
}

fn over_tcp(request: &Request, url: &Url) -> Result<Response, FetchError> {
    let proxy = if request.proxy { http_proxy(&url.host) } else { None };
    let (connect_host, connect_port, target) = match &proxy {
        Some(p) => (p.host.clone(), p.port, format!("{}{}", url.origin(), url.target)),
        None => (url.host.clone(), url.port, url.target.clone()),
    };
    let addrs: Vec<_> = (connect_host.as_str(), connect_port)
        .to_socket_addrs()
        .map_err(|err| FetchError::new(format!("cannot resolve {connect_host}: {err}")))?
        .collect();
    let mut last = FetchError::new(format!("cannot resolve {connect_host}"));
    let mut stream = None;
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, request.timeout) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(err) if err.kind() == std::io::ErrorKind::TimedOut => {
                last = FetchError::timed_out(format!("connecting to {connect_host}:{connect_port} timed out"))
            }
            Err(err) => last = FetchError::new(format!("cannot connect to {connect_host}:{connect_port}: {err}")),
        }
    }
    let mut stream = stream.ok_or(last)?;
    let _ = stream.set_read_timeout(Some(request.timeout));
    let _ = stream.set_write_timeout(Some(request.timeout));
    let host_header = if (url.scheme == "http" && url.port == 80) || (url.scheme == "https" && url.port == 443) {
        url.host.clone()
    } else {
        format!("{}:{}", url.host, url.port)
    };
    let mut head = format!(
        "{} {target} HTTP/1.1\r\nHost: {host_header}\r\nConnection: close\r\nAccept-Encoding: gzip\r\n",
        request.method
    );
    let mut has_agent = false;
    for (name, value) in &request.headers {
        has_agent |= name.eq_ignore_ascii_case("user-agent");
        head.push_str(&format!("{name}: {value}\r\n"));
    }
    if !has_agent {
        head.push_str(&format!("User-Agent: radixnet-rust/{}\r\n", env!("CARGO_PKG_VERSION")));
    }
    if !request.body.is_empty() || request.method == "POST" || request.method == "PUT" {
        head.push_str(&format!("Content-Length: {}\r\n", request.body.len()));
    }
    head.push_str("\r\n");
    let io = |err: std::io::Error| {
        if matches!(
            err.kind(),
            std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock
        ) {
            FetchError::timed_out(format!("{} timed out", request.url))
        } else {
            FetchError::new(format!("{}: {err}", request.url))
        }
    };
    // one write for the head and the body: two small writes wait on the
    // delayed ACK of the first (Nagle), which every LLM call would pay
    let mut message = head.into_bytes();
    message.extend_from_slice(&request.body);
    stream.write_all(&message).map_err(io)?;
    stream.flush().map_err(io)?;
    let mut raw = Vec::new();
    let cap = if request.max_bytes > 0 {
        // the head is on top of the body's allowance
        request.max_bytes + 64 * 1024
    } else {
        usize::MAX
    };
    let mut buf = [0u8; 16 * 1024];
    let mut truncated = false;
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&buf[..n]);
                if raw.len() >= cap {
                    truncated = true;
                    break;
                }
            }
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(io(err)),
        }
    }
    let mut response = parse_response(&raw, request.max_bytes)?;
    response.truncated |= truncated;
    response.url = request.url.clone();
    Ok(response)
}

/// Splits a raw HTTP/1.x answer into its status, headers and decoded body.
pub fn parse_response(raw: &[u8], max_bytes: usize) -> Result<Response, FetchError> {
    let end = find(raw, b"\r\n\r\n").ok_or_else(|| FetchError::new("the answer ended before its headers did"))?;
    let head = String::from_utf8_lossy(&raw[..end]).into_owned();
    let mut lines = head.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| FetchError::new(format!("not an HTTP answer: {status_line:?}")))?;
    let headers: Vec<(String, String)> = lines
        .filter_map(|l| l.split_once(':'))
        .map(|(k, v)| (k.trim().to_ascii_lowercase(), v.trim().to_string()))
        .collect();
    let mut response = Response {
        status,
        headers,
        ..Default::default()
    };
    let mut body = raw[end + 4..].to_vec();
    if response
        .header("transfer-encoding")
        .is_some_and(|v| v.to_ascii_lowercase().contains("chunked"))
    {
        body = dechunk(&body);
    }
    if response
        .header("content-encoding")
        .is_some_and(|v| v.eq_ignore_ascii_case("gzip"))
        && crate::gzip::is_gzip(&body)
    {
        body = crate::gzip::decompress(&body).map_err(FetchError::new)?;
    }
    if max_bytes > 0 && body.len() > max_bytes {
        body.truncate(max_bytes);
        response.truncated = true;
    }
    response.body = body;
    Ok(response)
}

fn find(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// A chunked body, reassembled; a body cut short keeps what arrived.
fn dechunk(mut data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    while let Some(line_end) = find(data, b"\r\n") {
        let size_text = String::from_utf8_lossy(&data[..line_end]);
        let size_text = size_text.split(';').next().unwrap_or("").trim().to_string();
        let Ok(size) = usize::from_str_radix(&size_text, 16) else {
            break;
        };
        data = &data[line_end + 2..];
        if size == 0 {
            break;
        }
        let take = size.min(data.len());
        out.extend_from_slice(&data[..take]);
        if take < size || data.len() < size + 2 {
            break;
        }
        data = &data[size + 2..];
    }
    out
}

/// Whether `curl` can be run - what an HTTPS request needs.
pub fn curl_available() -> bool {
    std::process::Command::new("curl")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

fn via_curl(request: &Request) -> Result<Response, FetchError> {
    use std::process::{Command, Stdio};
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let head_file = std::env::temp_dir().join(format!("radixnet-curl-{}-{n}.head", std::process::id()));
    let mut cmd = Command::new("curl");
    cmd.arg("--silent")
        .arg("--show-error")
        .arg("--compressed")
        .arg("--request")
        .arg(&request.method)
        .arg("--max-time")
        .arg(format!("{:.3}", request.timeout.as_secs_f64()))
        .arg("--dump-header")
        .arg(&head_file);
    if request.redirects > 0 {
        cmd.arg("--location")
            .arg("--max-redirs")
            .arg(request.redirects.to_string());
    }
    if request.max_bytes > 0 {
        cmd.arg("--max-filesize").arg((request.max_bytes * 4).to_string());
    }
    if !request.proxy {
        cmd.arg("--noproxy").arg("*");
    }
    // the headers go in a file only this user can read, never on the command
    // line (see the module docs); the guard removes it when curl is done
    let _headers = if request.headers.is_empty() {
        None
    } else {
        let path = std::env::temp_dir().join(format!("radixnet-curl-{}-{n}.headers", std::process::id()));
        let file = header_file(&request.headers, path)?;
        cmd.arg("--header").arg(format!("@{}", file.0.display()));
        Some(file)
    };
    if !request
        .headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("user-agent"))
    {
        cmd.arg("--user-agent")
            .arg(format!("radixnet-rust/{}", env!("CARGO_PKG_VERSION")));
    }
    if !request.body.is_empty() || request.method == "POST" {
        cmd.arg("--data-binary").arg("@-");
    }
    cmd.arg("--").arg(&request.url);
    cmd.stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd
        .spawn()
        .map_err(|err| FetchError::new(format!("https needs curl on the PATH: {err}")))?;
    if let Some(mut stdin) = child.stdin.take() {
        let body = request.body.clone();
        // written on its own thread, so a large body cannot deadlock against a full stdout pipe
        std::thread::spawn(move || {
            let _ = stdin.write_all(&body);
        });
    }
    let out = child
        .wait_with_output()
        .map_err(|err| FetchError::new(format!("curl: {err}")))?;
    let head = std::fs::read(&head_file).unwrap_or_default();
    let _ = std::fs::remove_file(&head_file);
    if !out.status.success() {
        let why = String::from_utf8_lossy(&out.stderr).trim().to_string();
        // curl's exit code 28 is "operation timed out"
        return Err(if out.status.code() == Some(28) {
            FetchError::timed_out(format!("{} timed out: {why}", request.url))
        } else {
            FetchError::new(format!("{}: {why}", request.url))
        });
    }
    // the last header block is the final answer's: a proxy's CONNECT and
    // every redirect wrote one before it
    let head = String::from_utf8_lossy(&head).into_owned();
    let block = head
        .split("\r\n\r\n")
        .filter(|b| b.starts_with("HTTP/"))
        .last()
        .unwrap_or("")
        .to_string();
    let mut raw = block.into_bytes();
    raw.extend_from_slice(b"\r\n\r\n");
    let mut response = parse_response(&raw, 0)?;
    // curl already decoded the transfer and content encodings
    response
        .headers
        .retain(|(k, _)| k != "transfer-encoding" && k != "content-encoding");
    response.body = out.stdout;
    if request.max_bytes > 0 && response.body.len() > request.max_bytes {
        response.body.truncate(request.max_bytes);
        response.truncated = true;
    }
    response.url = request.url.clone();
    Ok(response)
}

/// A file that is removed when it goes out of scope, however the request ends.
struct Scratch(std::path::PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// The request's headers as curl's `--header @file` reads them, one per line,
/// in a file created exclusively with `0600` permissions.
///
/// A line break inside a name or a value would start a header of its own, so
/// it is dropped rather than passed on.
fn header_file(headers: &[(String, String)], path: std::path::PathBuf) -> Result<Scratch, FetchError> {
    let open = |path: &std::path::Path| {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options.open(path)
    };
    let mut file = match open(&path) {
        Ok(file) => file,
        // a file left behind by an earlier process that had this pid
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {
            let _ = std::fs::remove_file(&path);
            open(&path).map_err(|err| FetchError::new(format!("cannot write curl's headers: {err}")))?
        }
        Err(err) => return Err(FetchError::new(format!("cannot write curl's headers: {err}"))),
    };
    let guard = Scratch(path);
    let clean = |text: &str| text.chars().filter(|c| *c != '\r' && *c != '\n').collect::<String>();
    let mut text = String::new();
    for (name, value) in headers {
        text.push_str(&format!("{}: {}\n", clean(name), clean(value)));
    }
    file.write_all(text.as_bytes())
        .map_err(|err| FetchError::new(format!("cannot write curl's headers: {err}")))?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn a_url_comes_apart() {
        let u = Url::parse("http://localhost:11434/api/tags?x=1#frag").unwrap();
        assert_eq!(
            (u.scheme.as_str(), u.host.as_str(), u.port),
            ("http", "localhost", 11434)
        );
        assert_eq!(u.target, "/api/tags?x=1");
        let u = Url::parse("https://user:pw@Example.com").unwrap();
        assert_eq!((u.host.as_str(), u.port, u.target.as_str()), ("example.com", 443, "/"));
        let u = Url::parse("http://[::1]:8080/a").unwrap();
        assert_eq!((u.host.as_str(), u.port), ("::1", 8080));
        assert!(Url::parse("ftp://x").is_err());
        assert!(Url::parse("no scheme").is_err());
        let base = Url::parse("http://h/a/b?q").unwrap();
        assert_eq!(base.join("/c"), "http://h/c");
        assert_eq!(base.join("c"), "http://h/a/c");
        assert_eq!(base.join("https://o/p"), "https://o/p");
        assert_eq!(base.join("//o/p"), "http://o/p");
    }

    #[test]
    fn a_chunked_answer_is_put_back_together() {
        let raw =
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nX-A: b\r\n\r\n5\r\nhello\r\n6\r\n world\r\n0\r\n\r\n";
        let r = parse_response(raw, 0).unwrap();
        assert_eq!(r.status, 200);
        assert_eq!(r.text(), "hello world");
        assert_eq!(r.header("x-a"), Some("b"));
        let r = parse_response(raw, 4).unwrap();
        assert_eq!((r.text().as_str(), r.truncated), ("hell", true));
    }

    #[test]
    fn it_talks_to_a_plain_http_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            // the POST, the redirect and the page it points at
            for _ in 0..3 {
                let (mut s, _) = listener.accept().unwrap();
                // the whole request - a read may return the head before the body
                let mut raw = Vec::new();
                let mut buf = vec![0u8; 4096];
                loop {
                    let n = s.read(&mut buf).unwrap();
                    raw.extend_from_slice(&buf[..n]);
                    let text = String::from_utf8_lossy(&raw);
                    let whole = text.split_once("\r\n\r\n").is_some_and(|(head, body)| {
                        let length = head
                            .lines()
                            .find_map(|l| l.strip_prefix("Content-Length: "))
                            .and_then(|v| v.trim().parse::<usize>().ok())
                            .unwrap_or(0);
                        body.len() >= length
                    });
                    if n == 0 || whole {
                        break;
                    }
                }
                let req = String::from_utf8_lossy(&raw).into_owned();
                let answer = if req.starts_with("GET /moved") {
                    "HTTP/1.1 302 Found\r\nLocation: /here\r\nContent-Length: 0\r\n\r\n".to_string()
                } else {
                    let body = format!("{{\"got\": {}}}", req.contains("{\"a\":1}"));
                    format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len())
                };
                s.write_all(answer.as_bytes()).unwrap();
            }
        });
        let url = format!("http://127.0.0.1:{port}");
        let doc = crate::json::parse("{\"a\":1}").unwrap();
        let r = Request::post_json(&format!("{url}/api"), &doc).send().unwrap();
        assert!(r.ok());
        assert_eq!(r.json().unwrap().at("got").as_bool(), Some(true));
        let r = Request::get(&format!("{url}/moved")).follow(1).send().unwrap();
        assert_eq!(r.text(), "{\"got\": false}", "the redirect was followed");
        server.join().unwrap();
    }

    #[test]
    fn a_header_file_is_private_and_goes_away() {
        let path = std::env::temp_dir().join(format!("radixnet-test-{}.headers", std::process::id()));
        let headers = vec![
            ("Authorization".to_string(), "Bearer sk-secret".to_string()),
            ("X-Evil".to_string(), "a\r\nInjected: yes".to_string()),
        ];
        {
            let file = header_file(&headers, path.clone()).unwrap();
            let text = std::fs::read_to_string(&file.0).unwrap();
            assert_eq!(text, "Authorization: Bearer sk-secret\nX-Evil: aInjected: yes\n");
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let mode = std::fs::metadata(&file.0).unwrap().permissions().mode();
                assert_eq!(mode & 0o777, 0o600, "only this user may read the key");
            }
        }
        assert!(!path.exists(), "removed when the request is done");
    }

    #[test]
    fn curl_sends_the_headers_it_reads_from_the_file() {
        if !curl_available() {
            return;
        }
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let mut raw = Vec::new();
            let mut buf = [0u8; 4096];
            // up to the end of the head and the two-byte body
            while !String::from_utf8_lossy(&raw).contains("\r\n\r\n{}") {
                let n = s.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                raw.extend_from_slice(&buf[..n]);
            }
            let body = "{\"ok\": true}";
            let answer = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}", body.len());
            s.write_all(answer.as_bytes()).unwrap();
            String::from_utf8_lossy(&raw).into_owned()
        });
        // curl speaks plain HTTP too, which is what lets a test see what it sent
        let doc = crate::json::parse("{}").unwrap();
        let request = Request::post_json(&format!("http://127.0.0.1:{port}/v1/x"), &doc)
            .header("Authorization", "Bearer sk-test")
            .direct();
        let response = via_curl(&request).unwrap();
        assert_eq!(response.json().unwrap().at("ok").as_bool(), Some(true));
        let seen = server.join().unwrap();
        assert!(seen.contains("Authorization: Bearer sk-test"), "{seen}");
        assert!(seen.contains("Content-Type: application/json"), "{seen}");
    }

    #[test]
    fn nobody_listening_is_an_error_not_a_hang() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let err = Request::get(&format!("http://127.0.0.1:{port}/"))
            .timeout(Duration::from_secs(2))
            .send()
            .unwrap_err();
        assert!(err.message.contains("cannot connect"), "{err}");
    }
}
