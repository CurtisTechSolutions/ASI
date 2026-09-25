//! Request bodies that are not (only) JSON: `multipart/form-data`, raw bytes,
//! and the JSON upload forms - the way `radixnet/api.py` reads them.
//!
//! A file reaches the server one of three ways, and the Python server takes
//! all three on the routes that carry files (`POST /api/uploads`, the image
//! encoder, speech teaching and transcription, and - here - the two recall
//! tutors, because the frontend sends them multipart too):
//!
//! * `multipart/form-data` - what a browser's `FormData` and `curl -F
//!   file=@x` send; every part with a file name is a file, plain fields are
//!   ignored, and the request's *options* come from the query string;
//! * a raw body (`curl --data-binary @photo.png`) named by `?name=`, or by the
//!   route's default name;
//! * JSON: `{name, content | content_base64}` or `{files: [...]}`, with the
//!   options beside them in the body.
//!
//! [`Form`] reads a request as the one Python would see (`_upload_body` and
//! `Fields.upload_files`), and answers an option from the JSON body when there
//! is one and from the query string otherwise (`_option`), so a route is
//! written once for all three.  The rest is what a route does with a file:
//! keep it as an upload ([`store_text`], [`store_bytes`]) and train on what it
//! became ([`save_and_train`]).
//!
//! `POST /api/uploads` is the one route that never holds its body: it streams
//! ([`store_stream`]), so an archive of any size is possible - a multipart
//! file part or a raw body goes straight to a `.part` file in the upload
//! directory as it arrives ([`stream_parts`] finds the parts on the way), and
//! is validated from there, entry by entry (D-035, as the Go server does it).
//! Only the JSON forms, which carry the file inline, are read whole.
//!
//! No dependencies, like everything else: the multipart parser (held and
//! streamed) and base64 ([`base64`]) are written out.

pub mod base64;

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::http::{accepted, Answer, ApiError, Request};
use crate::json::Json;
use crate::service::Service;

/// What this module's lines are filed under.
const LOG: &str = "media";

// -- multipart/form-data ----------------------------------------------------------------------

/// One part of a `multipart/form-data` body.
#[derive(Clone, Debug, PartialEq)]
pub struct Part {
    /// The form field it was sent as (`name=`).
    pub name: String,
    /// The file name, when the part is a file (`filename=`).
    pub filename: Option<String>,
    /// The part's own `Content-Type` (empty when it has none).
    pub content_type: String,
    /// The bytes, with any `Content-Transfer-Encoding: base64` undone.
    pub data: Vec<u8>,
}

/// A parameter of a header value (`boundary=...`, `filename="..."`), unquoted.
///
/// RFC 2231's `name*=charset''percent-encoded` form wins over the plain one,
/// as Python's `email` package lets it win.
pub fn header_param(value: &str, key: &str) -> Option<String> {
    let mut plain: Option<String> = None;
    let mut extended: Option<String> = None;
    for piece in split_params(value).into_iter().skip(1) {
        let Some((k, v)) = piece.split_once('=') else { continue };
        let k = k.trim().to_ascii_lowercase();
        let v = v.trim();
        if k == key {
            plain = Some(unquote(v));
        } else if k == format!("{key}*") {
            // charset'language'percent-encoded
            let encoded = v.splitn(3, '\'').nth(2).unwrap_or(v);
            extended = Some(percent_decode(&unquote(encoded)));
        }
    }
    extended.or(plain)
}

/// `a; b="x; y"; c` -> `["a", "b=\"x; y\"", "c"]`: semicolons inside quotes stay.
fn split_params(value: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut start, mut quoted, mut escaped) = (0, false, false);
    for (i, c) in value.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if quoted => escaped = true,
            '"' => quoted = !quoted,
            ';' if !quoted => {
                out.push(value[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(value[start..].trim());
    out
}

/// A quoted string's contents with its backslash escapes undone; a token as it is.
fn unquote(text: &str) -> String {
    let Some(inner) = text.strip_prefix('"').and_then(|t| t.strip_suffix('"')) else {
        return text.to_string();
    };
    let mut out = String::with_capacity(inner.len());
    let mut chars = inner.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            if let Some(next) = chars.next() {
                out.push(next);
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `%xx` decoding for an RFC 2231 value.
fn percent_decode(text: &str) -> String {
    let hex = |b: u8| (b as char).to_digit(16);
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(hi), Some(lo)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((hi * 16 + lo) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The position of `needle` in `hay` at or after `from`.
fn find(hay: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || hay.len() < needle.len() {
        return None;
    }
    let last = hay.len() - needle.len();
    let mut at = from;
    // the first byte is looked for on its own, which is most of the work on a
    // body of megabytes; the rest is compared where it matches
    while at <= last {
        at += hay[at..=last].iter().position(|&b| b == needle[0])?;
        if &hay[at..at + needle.len()] == needle {
            return Some(at);
        }
        at += 1;
    }
    None
}

/// The parts of a `multipart/form-data` body.
///
/// Lenient where Python's `email` parser is lenient, and read by its rules:
/// lines may end in CRLF or LF, a delimiter is a line that starts with
/// `--boundary`, a part's headers end at its first empty line, and a body
/// without the closing delimiter keeps the parts it has.
pub fn parse(body: &[u8], content_type: &str) -> Result<Vec<Part>, String> {
    let boundary = header_param(content_type, "boundary")
        .filter(|b| !b.is_empty())
        .ok_or("malformed multipart/form-data body (missing boundary?)")?;
    let delimiter = format!("--{boundary}").into_bytes();
    // the first delimiter opens the body; anything before it is a preamble
    let mut at = if body.starts_with(&delimiter) {
        0
    } else {
        match find(body, &[b"\n".as_slice(), &delimiter].concat(), 0) {
            Some(i) => i + 1,
            None => return Ok(Vec::new()),
        }
    };
    let mut parts = Vec::new();
    loop {
        let after = at + delimiter.len();
        if body[after..].starts_with(b"--") {
            break; // the closing delimiter
        }
        // the rest of the delimiter line (transport padding) is skipped
        let Some(line_end) = find(body, b"\n", after) else {
            break;
        };
        let start = line_end + 1;
        // from the delimiter line's own end: the next delimiter may follow at once
        let next = find(body, &[b"\n".as_slice(), &delimiter].concat(), line_end);
        let end = match next {
            Some(i) if i > start && body[i - 1] == b'\r' => i - 1,
            Some(i) => i,
            None => body.len(),
        };
        parts.push(parse_part(&body[start..end.max(start)])?);
        match next {
            Some(i) => at = i + 1,
            None => break,
        }
    }
    Ok(parts)
}

/// One part: its headers, a blank line, its bytes.
fn parse_part(raw: &[u8]) -> Result<Part, String> {
    // the headers end at the first empty line, whatever its ending; a part
    // without one is headers alone
    let (head, data) = match blank_line(raw) {
        Some((at, after)) => (&raw[..at], &raw[after..]),
        None => (raw, &raw[raw.len()..]),
    };
    let (mut part, encoding) = parse_head(&String::from_utf8_lossy(head));
    part.data = data.to_vec();
    if encoding == "base64" {
        part.data = decode_base64_part(&part.data)?;
    }
    Ok(part)
}

/// The first empty line of `raw` (a bare CRLF or LF): where it starts, and
/// where what follows it starts.
fn blank_line(raw: &[u8]) -> Option<(usize, usize)> {
    let mut start = 0;
    while start < raw.len() {
        let end = find(raw, b"\n", start)?;
        let line = &raw[start..end];
        if line.strip_suffix(b"\r").unwrap_or(line).is_empty() {
            return Some((start, end + 1));
        }
        start = end + 1;
    }
    None
}

/// A part's headers: the [`Part`] they describe, its data still to come, and
/// its `Content-Transfer-Encoding`, lower-cased.
fn parse_head(head: &str) -> (Part, String) {
    let mut part = Part {
        name: String::new(),
        filename: None,
        content_type: String::new(),
        data: Vec::new(),
    };
    let mut encoding = String::new();
    for line in head.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        match key.trim().to_ascii_lowercase().as_str() {
            "content-disposition" => {
                part.name = header_param(value, "name").unwrap_or_default();
                part.filename = header_param(value, "filename");
            }
            "content-type" => part.content_type = value.trim().to_string(),
            "content-transfer-encoding" => encoding = value.trim().to_ascii_lowercase(),
            _ => {}
        }
    }
    (part, encoding)
}

/// The bytes of a part sent `Content-Transfer-Encoding: base64`.
fn decode_base64_part(data: &[u8]) -> Result<Vec<u8>, String> {
    let text: String = String::from_utf8_lossy(data)
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    base64::decode(&text).map_err(|err| format!("malformed multipart/form-data body: {err}"))
}

// -- multipart/form-data, as it arrives -------------------------------------------------------

/// How much of a streamed body arrives at a time.
const CHUNK: usize = 256 << 10;

/// The longest line a part's headers may run to before the body is malformed.
const MAX_HEADER_LINE: usize = 1 << 20;

/// A window over a body being read: what has arrived and not been used yet.
struct Scanner<'a> {
    src: &'a mut dyn Read,
    buf: Vec<u8>,
    /// Where the unused bytes start in `buf`.
    at: usize,
    /// Where a chunk lands before it joins the window.
    scratch: Vec<u8>,
    eof: bool,
    /// Whether the last line read ended in `\n` (the last line of a body may not).
    terminated: bool,
}

impl<'a> Scanner<'a> {
    fn new(src: &'a mut dyn Read) -> Scanner<'a> {
        Scanner {
            src,
            buf: Vec::with_capacity(CHUNK),
            at: 0,
            scratch: vec![0u8; CHUNK],
            eof: false,
            terminated: false,
        }
    }

    /// The bytes that have arrived and not been used.
    fn window(&self) -> &[u8] {
        &self.buf[self.at..]
    }

    fn consume(&mut self, n: usize) {
        self.at += n;
    }

    /// Reads one more chunk of the body behind the window; false at its end.
    fn fill(&mut self) -> std::io::Result<bool> {
        if self.eof {
            return Ok(false);
        }
        // the used bytes go, then one chunk more arrives
        self.buf.drain(..self.at);
        self.at = 0;
        let n = loop {
            match self.src.read(&mut self.scratch) {
                Ok(n) => break n,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(err) => return Err(err),
            }
        };
        self.buf.extend_from_slice(&self.scratch[..n]);
        if n == 0 {
            self.eof = true;
        }
        Ok(n > 0)
    }

    /// One line, its `\n` (and a `\r` before it) dropped; `None` at the end
    /// of the body.
    fn line(&mut self) -> Result<Option<Vec<u8>>, ApiError> {
        loop {
            if let Some(i) = self.window().iter().position(|&b| b == b'\n') {
                let mut line = self.window()[..i].to_vec();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                self.consume(i + 1);
                self.terminated = true;
                return Ok(Some(line));
            }
            if self.window().len() > MAX_HEADER_LINE {
                return Err(ApiError::bad_request(
                    "malformed multipart/form-data body (a header line without an end)",
                ));
            }
            if !self.fill().map_err(read_failed)? {
                // the last line, without an end
                let rest = self.window().to_vec();
                self.consume(rest.len());
                self.terminated = false;
                return Ok(if rest.is_empty() { None } else { Some(rest) });
            }
        }
    }
}

/// The bytes of one part, read out of the scanner up to the delimiter that
/// ends it - which is then consumed too, so the scanner stands after
/// `--boundary` when the part is done.
struct PartBody<'s, 'a> {
    scanner: &'s mut Scanner<'a>,
    /// `\n--boundary`: what ends the part (a `\r` before it is the
    /// delimiter's too).
    needle: &'s [u8],
    /// Bytes at the front of the window that are the part's own, still to hand out.
    pending: usize,
    /// What follows them, once found: the delimiter, this many bytes to drop.
    delimiter: Option<usize>,
    /// Whether a delimiter ended the part (false: the body ended first).
    delimited: bool,
    /// Nothing handed out yet: the delimiter may stand right at the start,
    /// without a line end of its own, for a part with no data.
    at_start: bool,
    done: bool,
}

impl Read for PartBody<'_, '_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            if self.pending > 0 {
                let n = self.pending.min(out.len());
                out[..n].copy_from_slice(&self.scanner.window()[..n]);
                self.scanner.consume(n);
                self.pending -= n;
                return Ok(n);
            }
            if self.done {
                return Ok(0);
            }
            if let Some(skip) = self.delimiter.take() {
                // the part's bytes are out; the delimiter that ended it goes too
                self.scanner.consume(skip);
                self.delimited = true;
                self.done = true;
                return Ok(0);
            }
            if self.at_start {
                let delimiter = &self.needle[1..];
                if self.scanner.window().len() < delimiter.len() && !self.scanner.eof {
                    self.scanner.fill()?; // not enough to tell yet
                    continue;
                }
                self.at_start = false;
                if self.scanner.window().starts_with(delimiter) {
                    self.delimiter = Some(delimiter.len());
                    continue;
                }
            }
            let window = self.scanner.window();
            if let Some(i) = find(window, self.needle, 0) {
                let end = if i > 0 && window[i - 1] == b'\r' { i - 1 } else { i };
                self.pending = end;
                self.delimiter = Some(i - end + self.needle.len());
            } else if window.len() > self.needle.len() {
                // a delimiter cannot start earlier than the window's last
                // needle's length of bytes: everything before is the part's own
                self.pending = window.len() - self.needle.len();
            } else if !self.scanner.fill()? {
                // the body ended without a closing delimiter: the rest is the
                // part's own, as `parse` reads it
                self.pending = self.scanner.window().len();
                self.done = true;
            }
        }
    }
}

/// A read of the request body that failed, as the client is answered: a body
/// shorter than its `Content-Length` says is a 400, as it is on the Python server.
fn read_failed(err: std::io::Error) -> ApiError {
    if err.kind() == std::io::ErrorKind::UnexpectedEof {
        return ApiError::bad_request("incomplete request body");
    }
    ApiError::bad_request(format!("could not read the request body: {err}"))
}

/// Reads a `multipart/form-data` body as it arrives, one part at a time:
/// `each` gets a part's headers (a [`Part`] without its data) and a reader of
/// its bytes, which ends where the part does.  Nothing is held - what `each`
/// leaves unread is dropped - so a part may be as large as the body.
///
/// Lenient where [`parse`] is lenient: lines may end in CRLF or LF, and a body
/// without the closing delimiter keeps the parts it has.  A part sent
/// `Content-Transfer-Encoding: base64` (which nothing sends any more) is
/// decoded whole first.
pub fn stream_parts(
    body: &mut dyn Read,
    content_type: &str,
    each: &mut dyn FnMut(&Part, &mut dyn Read) -> Result<(), ApiError>,
) -> Result<(), ApiError> {
    let boundary = header_param(content_type, "boundary")
        .filter(|b| !b.is_empty())
        .ok_or_else(|| ApiError::bad_request("malformed multipart/form-data body (missing boundary?)"))?;
    let delimiter = format!("--{boundary}").into_bytes();
    let needle = [b"\n".as_slice(), &delimiter].concat();
    let mut scanner = Scanner::new(body);
    // the first delimiter opens the body; anything before it is a preamble
    let mut after_delimiter = loop {
        match scanner.line()? {
            None => return Ok(()),
            Some(line) if line.starts_with(&delimiter) => break line[delimiter.len()..].to_vec(),
            Some(_) => continue,
        }
    };
    loop {
        // "--" after a delimiter closes the body, and so does a delimiter
        // line without an end; the rest of the line is padding
        if after_delimiter.starts_with(b"--") || !scanner.terminated {
            return Ok(());
        }
        // the part's headers, up to a blank line - or to the next delimiter,
        // for a part with no data and no blank line
        let mut head = String::new();
        let mut ended_early = None;
        while let Some(line) = scanner.line()? {
            if line.is_empty() {
                break;
            }
            if line.starts_with(&delimiter) {
                ended_early = Some(line[delimiter.len()..].to_vec());
                break;
            }
            head.push_str(&String::from_utf8_lossy(&line));
            head.push('\n');
        }
        let (part, encoding) = parse_head(&head);
        if let Some(rest) = ended_early {
            each(&part, &mut std::io::empty())?;
            after_delimiter = rest;
            continue;
        }
        let mut data = PartBody {
            scanner: &mut scanner,
            needle: &needle,
            pending: 0,
            delimiter: None,
            delimited: false,
            at_start: true,
            done: false,
        };
        if encoding == "base64" {
            let mut raw = Vec::new();
            data.read_to_end(&mut raw).map_err(read_failed)?;
            let decoded = decode_base64_part(&raw).map_err(ApiError::bad_request)?;
            each(&part, &mut std::io::Cursor::new(decoded))?;
        } else {
            each(&part, &mut data)?;
            // what `each` left is dropped, so the scanner stands at the delimiter
            std::io::copy(&mut data, &mut std::io::sink()).map_err(read_failed)?;
        }
        if !data.delimited {
            return Ok(());
        }
        after_delimiter = match scanner.line()? {
            Some(rest) => rest,
            None => return Ok(()),
        };
    }
}

// -- the upload forms -------------------------------------------------------------------------

/// The content of one uploaded file: text as it was sent, or bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Text(String),
    Bytes(Vec<u8>),
}

/// One file of a request: its name and its content.
#[derive(Clone, Debug, PartialEq)]
pub struct Upload {
    pub name: String,
    pub payload: Payload,
}

/// A request that may carry files, read the way the Python server reads it.
pub struct Form<'a> {
    request: &'a Request,
    /// The JSON body (an object); empty for multipart and raw bodies, whose
    /// options arrive in the query string.
    fields: Json,
    /// The files a multipart or raw body carried.
    carried: Option<Vec<Upload>>,
}

/// Python's name for a JSON value's type, for the messages that quote one.
fn json_type(value: &Json) -> &'static str {
    match value {
        Json::Null => "null",
        Json::Bool(_) => "boolean",
        Json::Int(_) | Json::Num(_) => "number",
        Json::Str(_) => "string",
        Json::Arr(_) => "array",
        Json::Obj(_) => "object",
    }
}

/// `'name' must be ... (got <type> <value>)`, as Python's `Fields._bad` says it.
fn bad(name: &str, expected: &str, value: &Json) -> ApiError {
    ApiError::bad_request(format!(
        "'{name}' must be {expected} (got {} {})",
        json_type(value),
        value.render(0)
    ))
}

/// The JSON object a request body holds: `{}` for an empty body, 400 for
/// anything that is not an object (`api.parse_body`).
pub fn json_object(r: &Request) -> Result<Json, ApiError> {
    if r.raw.iter().all(|b| b.is_ascii_whitespace()) {
        return Ok(Json::Obj(Vec::new()));
    }
    match &r.body {
        Json::Obj(_) => Ok(r.body.clone()),
        Json::Null => {
            let why = match std::str::from_utf8(&r.raw) {
                Err(err) => format!("request body must be UTF-8 encoded JSON ({err})"),
                Ok(text) => match crate::json::parse(text) {
                    Err(err) => format!("invalid JSON body: {err}"),
                    Ok(_) => "JSON body must be an object, not null".to_string(),
                },
            };
            Err(ApiError::bad_request(why))
        }
        other => Err(ApiError::bad_request(format!(
            "JSON body must be an object, not {}",
            json_type(other)
        ))),
    }
}

impl<'a> Form<'a> {
    /// Reads the body of a route that takes files (`_upload_body`).
    ///
    /// `default_name` names a raw body sent without `?name=`; without one, a
    /// raw body needs the query parameter (the uploads route).
    pub fn read(request: &'a Request, default_name: Option<&str>) -> Result<Form<'a>, ApiError> {
        let kind = request.content_type();
        let names: Vec<&str> = request
            .query
            .iter()
            .filter(|(k, v)| k == "name" && !v.trim().is_empty())
            .map(|(_, v)| v.as_str())
            .collect();
        if kind == "multipart/form-data" {
            let content_type = request.header("content-type").unwrap_or("");
            let parts = parse(&request.raw, content_type).map_err(ApiError::bad_request)?;
            let files: Vec<Upload> = parts
                .into_iter()
                .filter_map(|p| {
                    // plain form fields are ignored
                    p.filename.filter(|f| !f.is_empty()).map(|name| Upload {
                        name,
                        payload: Payload::Bytes(p.data),
                    })
                })
                .collect();
            if files.is_empty() {
                return Err(ApiError::bad_request(
                    "multipart body contains no file parts (use -F file=@corpus.txt)",
                ));
            }
            return Ok(Form {
                request,
                fields: Json::Obj(Vec::new()),
                carried: Some(files),
            });
        }
        if kind == "application/json" || (kind.is_empty() && names.is_empty() && default_name.is_none()) {
            return Ok(Form {
                request,
                fields: json_object(request)?,
                carried: None,
            });
        }
        let name = match names.first() {
            Some(name) => name.to_string(),
            None => default_name
                .ok_or_else(|| {
                    ApiError::bad_request(
                        "raw uploads need a ?name=<file name> query parameter (or send JSON {name, content})",
                    )
                })?
                .to_string(),
        };
        Ok(Form {
            request,
            fields: Json::Obj(Vec::new()),
            carried: Some(vec![Upload {
                name,
                payload: Payload::Bytes(request.raw.clone()),
            }]),
        })
    }

    /// A JSON request, whatever its content type says: the routes that take no
    /// files read their body this way.
    pub fn json(request: &'a Request) -> Result<Form<'a>, ApiError> {
        Ok(Form {
            request,
            fields: json_object(request)?,
            carried: None,
        })
    }

    /// A body field; JSON `null` reads as absent, as it does in Python.
    pub fn field(&self, name: &str) -> Option<&Json> {
        self.fields.get(name).filter(|v| !v.is_null())
    }

    /// The files of the request (`Fields.upload_files`): what a multipart or
    /// raw body carried, else `{name, content | content_base64}`, else `{files: [...]}`.
    pub fn files(&self) -> Result<Vec<Upload>, ApiError> {
        if let Some(files) = &self.carried {
            return Ok(files.clone());
        }
        let Some(list) = self.field("files") else {
            return Ok(vec![one_upload(&self.fields)?]);
        };
        let items = match list {
            Json::Arr(items) if !items.is_empty() => items,
            other => {
                return Err(bad(
                    "files",
                    "a non-empty list of {name, content | content_base64} objects",
                    other,
                ))
            }
        };
        items
            .iter()
            .enumerate()
            .map(|(i, item)| match item {
                Json::Obj(_) => one_upload(item),
                _ => Err(ApiError::bad_request(format!(
                    "'files[{i}]' must be an object with 'name' and 'content' (or 'content_base64')"
                ))),
            })
            .collect()
    }

    /// The first file as bytes: the one image or recording a media route
    /// takes (`_audio_bytes`); `what` names it in the refusal.
    pub fn bytes(&self, what: &str) -> Result<(String, Vec<u8>), ApiError> {
        let mut files = self.files()?;
        let first = files.remove(0);
        match first.payload {
            Payload::Bytes(data) => Ok((first.name, data)),
            Payload::Text(_) => Err(ApiError::bad_request(format!(
                "send the {what} as bytes: multipart/form-data, a raw body, or JSON {{name, content_base64}}"
            ))),
        }
    }

    /// An option: the JSON body's field when there is one, else the first
    /// value of the query parameter (as a string) - Python's `_option`.
    pub fn option(&self, name: &str) -> Option<Json> {
        if let Some(value) = self.field(name) {
            return Some(value.clone());
        }
        self.request
            .query
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| Json::Str(v.clone()))
    }

    /// `str(_option(...) or default)`: a falsy value (absent, `""`, `0`,
    /// `false`) reads as the default.
    pub fn text(&self, name: &str, default: &str) -> String {
        match self.option(name) {
            Some(value) if truthy(&value) => python_str(&value),
            _ => default.to_string(),
        }
    }

    /// `_option(...) or None` as a string: `None` for a falsy value.
    pub fn maybe_text(&self, name: &str) -> Option<String> {
        self.option(name).filter(truthy).map(|v| python_str(&v))
    }

    /// `int(_option(...) or default)`; the error is Python's `int()` message.
    pub fn int(&self, name: &str, default: i64) -> Result<i64, String> {
        match self.option(name) {
            Some(value) if truthy(&value) => python_int(&value),
            _ => Ok(default),
        }
    }

    /// `int(_option(..., default))`: like [`Form::int`], but a given `0` stays
    /// `0` (Python reads `epochs` this way, without the `or`).
    pub fn exact_int(&self, name: &str, default: i64) -> Result<i64, String> {
        match self.option(name) {
            Some(Json::Null) | None => Ok(default),
            Some(value) => python_int(&value),
        }
    }

    /// `float(_option(..., default))`: like [`Form::float`], but a given `0`
    /// stays `0` (the learning rate of a training request is read this way).
    pub fn exact_float(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.option(name) {
            Some(Json::Null) | None => Ok(default),
            Some(value) => python_float(&value),
        }
    }

    /// `float(_option(...) or default)`.
    pub fn float(&self, name: &str, default: f64) -> Result<f64, String> {
        match self.option(name) {
            Some(value) if truthy(&value) => python_float(&value),
            _ => Ok(default),
        }
    }

    /// `_flag_option`: a JSON boolean as it is, anything else true when it
    /// reads as `1`, `true`, `yes` or `on`.
    pub fn flag(&self, name: &str, default: bool) -> bool {
        match self.option(name) {
            None => default,
            Some(Json::Bool(b)) => b,
            Some(value) => matches!(
                python_str(&value).trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            ),
        }
    }

    /// `Fields.texts_optional(list, one)`: a list of strings, or one string
    /// cut into its non-blank lines; absent or empty is `[]`.
    pub fn texts(&self, list: &str, one: &str) -> Result<Vec<String>, ApiError> {
        if let Some(value) = self.field(list) {
            return match value {
                Json::Arr(items) if items.iter().all(|t| matches!(t, Json::Str(_))) => {
                    Ok(items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
                }
                other => Err(bad(list, "a list of strings", other)),
            };
        }
        match self.field(one) {
            None => Ok(Vec::new()),
            Some(Json::Str(blob)) => Ok(blob
                .lines()
                .map(|l| l.trim_end_matches('\r'))
                .filter(|l| !l.trim_matches(base64::is_python_space).is_empty())
                .map(str::to_string)
                .collect()),
            Some(other) => Err(bad(one, "a string", other)),
        }
    }
}

/// `{name, content | content_base64}` (`Fields.text("name")` and `_upload_payload`).
fn one_upload(fields: &Json) -> Result<Upload, ApiError> {
    let get = |key: &str| fields.get(key).filter(|v| !v.is_null());
    let name = match get("name") {
        None => return Err(ApiError::bad_request("missing field 'name' (string)")),
        Some(Json::Str(name)) => name.clone(),
        Some(other) => return Err(bad("name", "a string", other)),
    };
    if let Some(encoded) = get("content_base64") {
        let Json::Str(text) = encoded else {
            return Err(bad("content_base64", "a base64 string", encoded));
        };
        let data = base64::decode(text)
            .map_err(|err| ApiError::bad_request(format!("'content_base64' is not valid base64: {err}")))?;
        return Ok(Upload {
            name,
            payload: Payload::Bytes(data),
        });
    }
    match get("content") {
        None => Err(ApiError::bad_request(
            "missing field 'content' (string) or 'content_base64' (base64 of a text file or ZIP archive)",
        )),
        Some(Json::Str(text)) => Ok(Upload {
            name,
            payload: Payload::Text(text.clone()),
        }),
        Some(other) => Err(bad("content", "a string", other)),
    }
}

/// Python's truthiness of a JSON value.
fn truthy(value: &Json) -> bool {
    match value {
        Json::Null => false,
        Json::Bool(b) => *b,
        Json::Int(n) => *n != 0,
        Json::Num(x) => *x != 0.0,
        Json::Str(s) => !s.is_empty(),
        Json::Arr(items) => !items.is_empty(),
        Json::Obj(pairs) => !pairs.is_empty(),
    }
}

/// `str(value)` of a JSON value, as Python prints it.
fn python_str(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        Json::Bool(true) => "True".to_string(),
        Json::Bool(false) => "False".to_string(),
        Json::Null => "None".to_string(),
        Json::Int(n) => n.to_string(),
        Json::Num(x) => crate::json::py_repr(*x),
        other => other.render(0),
    }
}

/// `int(value)`: numbers truncate, text must be a whole number.
fn python_int(value: &Json) -> Result<i64, String> {
    match value {
        Json::Int(n) => Ok(*n),
        Json::Bool(b) => Ok(*b as i64),
        Json::Num(x) if x.is_finite() => Ok(x.trunc() as i64),
        Json::Num(x) => Err(format!("cannot convert float {} to integer", crate::json::py_repr(*x))),
        Json::Str(s) => s
            .trim_matches(base64::is_python_space)
            .replace('_', "")
            .parse()
            .map_err(|_| {
                format!(
                    "invalid literal for int() with base 10: {}",
                    crate::negative::python_repr(s)
                )
            }),
        other => Err(format!(
            "int() argument must be a string or a number, not {}",
            json_type(other)
        )),
    }
}

/// `float(value)`.
fn python_float(value: &Json) -> Result<f64, String> {
    match value {
        Json::Int(n) => Ok(*n as f64),
        Json::Num(x) => Ok(*x),
        Json::Bool(b) => Ok(*b as i64 as f64),
        Json::Str(s) => {
            let t = s.trim_matches(base64::is_python_space).to_ascii_lowercase();
            match t.as_str() {
                "inf" | "+inf" | "infinity" | "+infinity" => Ok(f64::INFINITY),
                "-inf" | "-infinity" => Ok(f64::NEG_INFINITY),
                "nan" | "+nan" | "-nan" => Ok(f64::NAN),
                _ => t
                    .parse()
                    .map_err(|_| format!("could not convert string to float: {}", crate::negative::python_repr(s))),
            }
        }
        other => Err(format!(
            "float() argument must be a string or a real number, not {}",
            json_type(other)
        )),
    }
}

// -- keeping what arrived ---------------------------------------------------------------------

/// Largest upload name kept (`api.MAX_UPLOAD_NAME`).
pub const MAX_UPLOAD_NAME: usize = 128;

/// A safe file name inside the upload directory (`api.sanitize_upload_name`):
/// the base name only, odd characters replaced, never a way out.
pub fn sanitize_upload_name(name: &str) -> Result<String, ApiError> {
    let unified = name.replace('\\', "/");
    let base = unified
        .rsplit('/')
        .next()
        .unwrap_or("")
        .trim_matches(base64::is_python_space);
    let replaced: String = base
        .chars()
        .map(|c| {
            if c.is_alphanumeric() || "._- +@()".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    let clean = replaced.trim_matches([' ', '.']);
    if clean.is_empty() || clean == "." || clean == ".." {
        return Err(ApiError::bad_request(format!(
            "invalid upload name {}",
            crate::negative::python_repr(name)
        )));
    }
    Ok(clean.chars().take(MAX_UPLOAD_NAME).collect())
}

/// The upload directory, made when it is not there yet.
fn upload_dir(svc: &Service) -> Result<String, ApiError> {
    let dir = svc
        .upload_dir
        .clone()
        .ok_or_else(|| ApiError::bad_request("uploads are disabled: start the server with --upload-dir"))?;
    std::fs::create_dir_all(&dir).map_err(|err| ApiError::with_status(500, format!("cannot create {dir}: {err}")))?;
    Ok(dir)
}

/// `<path>.part`: where an upload is written before it is moved into place,
/// so a reader never sees half a file.
fn staged(path: &Path) -> PathBuf {
    path.with_extension(match path.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    })
}

/// A `.part` file of its own in the upload directory, for bytes on their way
/// in (the listing skips `.part` files, as the Python and Go listings do).
fn new_part(dir: &str) -> Result<PathBuf, ApiError> {
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = Path::new(dir).join(format!(
        ".upload-{}-{}-{stamp}.part",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .map_err(|err| ApiError::with_status(500, format!("cannot create {}: {err}", path.display())))?;
    Ok(path)
}

/// `2026-09-20T01:23:45.678+00:00`: a file's modification time as Python's
/// `isoformat(timespec="milliseconds")` writes it.
fn modified(meta: &std::fs::Metadata) -> String {
    let since = meta
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .unwrap_or_default();
    let stamp = crate::clock::iso8601(since.as_secs() as i64);
    let (date, zone) = stamp.split_at(stamp.len() - "+00:00".len());
    format!("{date}.{:03}{zone}", since.subsec_millis())
}

/// The listing record of one text upload (`api._upload_record`): its
/// characters and its non-blank lines, counted a line at a time, so a large
/// one is described without being held.
pub(crate) fn record(path: &Path) -> Result<Json, ApiError> {
    let meta = std::fs::metadata(path).map_err(|err| ApiError::with_status(500, err.to_string()))?;
    let (chars, lines) = text_counts(path).map_err(|err| ApiError::with_status(500, err.to_string()))?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Json::obj([
        ("name", Json::str(name)),
        ("bytes", Json::Int(meta.len() as i64)),
        ("chars", Json::Int(chars as i64)),
        ("lines", Json::Int(lines as i64)),
        ("modified", Json::str(modified(&meta))),
    ]))
}

/// The characters of a text file and its non-blank lines (`str.strip()`
/// blank), a line at a time.
fn text_counts(path: &Path) -> std::io::Result<(usize, usize)> {
    let mut reader = BufReader::with_capacity(CHUNK, std::fs::File::open(path)?);
    let (mut chars, mut lines) = (0usize, 0usize);
    let mut line = Vec::new();
    loop {
        line.clear();
        if reader.read_until(b'\n', &mut line)? == 0 {
            return Ok((chars, lines));
        }
        let text = String::from_utf8_lossy(&line);
        chars += text.chars().count();
        if !text.trim_matches(base64::is_python_space).is_empty() {
            lines += 1;
        }
    }
}

/// Stores `content` as the upload `name`, replacing one of that name
/// (`ModelService.upload`); the record says whether it did.
pub fn store_text(svc: &Service, name: &str, content: &str) -> Result<Json, ApiError> {
    let dir = upload_dir(svc)?;
    let path = std::path::Path::new(&dir).join(sanitize_upload_name(name)?);
    let replaced = path.is_file();
    // written beside and moved into place, so a reader never sees half a file
    let part = staged(&path);
    std::fs::write(&part, content.as_bytes())
        .and_then(|_| std::fs::rename(&part, &path))
        .map_err(|err| ApiError::with_status(500, format!("cannot store {}: {err}", path.display())))?;
    let mut doc = record(&path)?;
    if let Json::Obj(pairs) = &mut doc {
        pairs.push(("replaced".to_string(), Json::Bool(replaced)));
    }
    crate::log_info!(LOG, "upload {} ({} bytes)", path.display(), content.len());
    Ok(doc)
}

/// Stores a binary upload (`ModelService.upload_bytes`): a ZIP archive is
/// kept whole as one upload (D-034), anything else is stored as UTF-8 text
/// with the byte-order mark dropped and undecodable bytes replaced.  Answers
/// `{"uploads": [record]}`, plus `"archives": [summary]` for an archive.
pub fn store_bytes(svc: &Service, name: &str, data: &[u8]) -> Result<Json, ApiError> {
    if crate::zip::is_zip(data) {
        return store_archive(svc, name, data);
    }
    let data = data.strip_prefix(b"\xef\xbb\xbf".as_slice()).unwrap_or(data);
    let text = String::from_utf8_lossy(data);
    Ok(Json::obj([("uploads", Json::Arr(vec![store_text(svc, name, &text)?]))]))
}

/// Keeps a ZIP archive held in memory (the JSON forms carry it inline) as a
/// single upload (`ModelService.store_archive`): its bytes go to a `.part`
/// file, and [`store_part`] validates it from there and keeps it whole.
pub fn store_archive(svc: &Service, name: &str, data: &[u8]) -> Result<Json, ApiError> {
    let dir = upload_dir(svc)?;
    store_from(svc, &dir, name, &mut std::io::Cursor::new(data))
}

/// Stores every file of an upload request: `{"uploads": [records]}` - the
/// body of `POST /api/uploads` (`api._r_upload`).
pub fn store_all(svc: &Service, files: &[Upload]) -> Result<Json, ApiError> {
    let mut stored = Vec::new();
    for file in files {
        stored.push(match &file.payload {
            Payload::Text(text) => Json::obj([("uploads", Json::Arr(vec![store_text(svc, &file.name, text)?]))]),
            Payload::Bytes(data) => store_bytes(svc, &file.name, data)?,
        });
    }
    gather(&stored)
}

/// The answer of an upload request from what each of its files stored:
/// `{"uploads": [records]}`, plus `"archives": [summaries]` when any was one.
fn gather(stored: &[Json]) -> Result<Json, ApiError> {
    let mut records = Vec::new();
    let mut archives: Vec<Json> = Vec::new();
    for doc in stored {
        records.extend(doc.at("uploads").as_array().iter().cloned());
        archives.extend(doc.at("archives").as_array().iter().cloned());
    }
    if records.is_empty() {
        return Err(ApiError::bad_request("nothing was uploaded"));
    }
    let mut body = vec![("uploads".to_string(), Json::Arr(records))];
    if !archives.is_empty() {
        body.push(("archives".to_string(), Json::Arr(archives)));
    }
    Ok(Json::Obj(body))
}

// -- keeping what arrives, as it arrives -----------------------------------------------------

/// `POST /api/uploads`, read as the body arrives (the route streams:
/// [`crate::http::Server::stream_route`]), so an archive of any size goes
/// straight to disk, as it does on the Go server (D-035).  Multipart file
/// parts and raw bodies are written to a `.part` file in the upload directory
/// and kept from there ([`store_part`]); the JSON forms carry their file
/// inline and are read whole, as Python reads them.  Answers
/// `{"uploads": [records]}`, plus `"archives": [summaries]` for the archives
/// among them (`api._r_upload`).
pub fn store_stream(svc: &Service, r: &Request, body: &mut dyn Read) -> Answer {
    // refused before a byte of the body is read
    let dir = upload_dir(svc)?;
    let kind = r.content_type();
    let names: Vec<&str> = r
        .query
        .iter()
        .filter(|(k, v)| k == "name" && !v.trim().is_empty())
        .map(|(_, v)| v.as_str())
        .collect();
    let mut stored: Vec<Json> = Vec::new();
    if kind == "multipart/form-data" {
        let content_type = r.header("content-type").unwrap_or("").to_string();
        stream_parts(body, &content_type, &mut |part, data| {
            // plain form fields are ignored
            let Some(name) = part.filename.as_deref().filter(|f| !f.is_empty()) else {
                return Ok(());
            };
            stored.push(store_from(svc, &dir, name, data)?);
            Ok(())
        })?;
        if stored.is_empty() {
            return Err(ApiError::bad_request(
                "multipart body contains no file parts (use -F file=@corpus.txt)",
            ));
        }
        return gather(&stored);
    }
    if kind == "application/json" || (kind.is_empty() && names.is_empty()) {
        let mut raw = Vec::new();
        body.read_to_end(&mut raw).map_err(read_failed)?;
        let parsed = match std::str::from_utf8(&raw) {
            Ok(text) if !text.trim().is_empty() => crate::json::parse(text).unwrap_or(Json::Null),
            _ => Json::Null,
        };
        let request = Request {
            method: r.method.clone(),
            path: r.path.clone(),
            query: r.query.clone(),
            body: parsed,
            raw,
            headers: r.headers.clone(),
        };
        let files = Form::read(&request, None)?.files()?;
        return store_all(svc, &files);
    }
    let Some(name) = names.first() else {
        return Err(ApiError::bad_request(
            "raw uploads need a ?name=<file name> query parameter (or send JSON {name, content})",
        ));
    };
    stored.push(store_from(svc, &dir, name, body)?);
    gather(&stored)
}

/// Stores one file from the stream of its bytes: they go to a `.part` file
/// in the upload directory as they arrive, and [`store_part`] keeps it from
/// there.  A bad name is refused before a byte is read.
fn store_from(svc: &Service, dir: &str, name: &str, data: &mut dyn Read) -> Result<Json, ApiError> {
    let safe = sanitize_upload_name(name)?;
    let part = new_part(dir)?;
    let copied = std::fs::OpenOptions::new().write(true).open(&part).and_then(|file| {
        let mut file = BufWriter::with_capacity(CHUNK, file);
        std::io::copy(data, &mut file)?;
        file.flush()
    });
    if let Err(err) = copied {
        let _ = std::fs::remove_file(&part);
        if err.kind() == std::io::ErrorKind::UnexpectedEof {
            return Err(ApiError::bad_request("incomplete request body"));
        }
        return Err(ApiError::with_status(500, format!("cannot store {safe}: {err}")));
    }
    store_part(svc, dir, &safe, &part)
}

/// Keeps an upload whose bytes are in `part`, a file in the upload directory,
/// under the name `safe` (already made safe).  The first four bytes decide:
/// a ZIP archive is validated by reading its entries from the file - a batch
/// at a time, never whole - and kept whole under a `.zip` name (D-034);
/// anything else is a text file, stored as UTF-8 with the byte-order mark
/// dropped and undecodable bytes replaced, converted a chunk at a time.
/// `part` is gone afterwards, kept or refused.
fn store_part(svc: &Service, dir: &str, safe: &str, part: &Path) -> Result<Json, ApiError> {
    let outcome = if crate::zip::is_zip_file(part) {
        keep_archive(svc, dir, safe, part)
    } else {
        keep_text(dir, safe, part)
    };
    // nothing to do when it was moved into place; a refused one goes
    let _ = std::fs::remove_file(part);
    outcome
}

/// The archive in `part`, validated from the file and moved into place.  A
/// corrupt archive, or one without a single text entry, is refused (400); the
/// entries passed over are listed with their reasons in the summary.
fn keep_archive(svc: &Service, dir: &str, safe: &str, part: &Path) -> Result<Json, ApiError> {
    let mut archive_name = safe.to_string();
    if !archive_name.to_lowercase().ends_with(".zip") {
        archive_name.push_str(".zip");
    }
    let summary = crate::source::inspect(part, svc.workers);
    if let Some(err) = &summary.error {
        return Err(ApiError::bad_request(format!("{archive_name}: {err}")));
    }
    let extracted = summary.files();
    if extracted == 0 {
        let reasons: Vec<String> = summary
            .skipped
            .iter()
            .take(8)
            .map(|s| format!("{}: {}", s.path, s.reason))
            .collect();
        let why = if reasons.is_empty() {
            " (it is empty)".to_string()
        } else {
            format!(" ({})", reasons.join("; "))
        };
        return Err(ApiError::bad_request(format!(
            "{archive_name} holds no text files to train on{why}"
        )));
    }
    let path = Path::new(dir).join(&archive_name);
    let replaced = path.is_file();
    std::fs::rename(part, &path)
        .map_err(|err| ApiError::with_status(500, format!("cannot store {}: {err}", path.display())))?;
    let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    // the listing's record comes from the pass just made, not a second one
    crate::source::remember(&path, summary.clone());
    let mut record = crate::source::archive_record(&path, svc.workers)
        .ok_or_else(|| ApiError::with_status(500, format!("{} is not a ZIP archive", path.display())))?;
    if let Json::Obj(pairs) = &mut record {
        pairs.push(("replaced".to_string(), Json::Bool(replaced)));
    }
    let skipped = &summary.skipped;
    let doc = Json::obj([
        ("name", Json::str(archive_name.clone())),
        ("bytes", Json::Int(bytes as i64)),
        ("entries", Json::Int((extracted + skipped.len()) as i64)),
        ("extracted", Json::Int(extracted as i64)),
        ("skipped", Json::Arr(skipped.iter().map(|s| s.to_json()).collect())),
    ]);
    crate::log_info!(
        LOG,
        "upload {archive_name} ({bytes} bytes; {extracted} text file(s) inside, {} skipped)",
        skipped.len()
    );
    Ok(Json::obj([
        ("uploads", Json::Arr(vec![record])),
        ("archives", Json::Arr(vec![doc])),
    ]))
}

/// The text file in `part`, converted into place a chunk at a time.
fn keep_text(dir: &str, safe: &str, part: &Path) -> Result<Json, ApiError> {
    let path = Path::new(dir).join(safe);
    let replaced = path.is_file();
    let stage = staged(&path);
    let written = std::fs::File::open(part).and_then(|src| {
        let src = BufReader::with_capacity(CHUNK, src);
        let mut dst = BufWriter::with_capacity(CHUNK, std::fs::File::create(&stage)?);
        let written = copy_as_text(src, &mut dst)?;
        dst.flush()?;
        std::fs::rename(&stage, &path)?;
        Ok(written)
    });
    let bytes = match written {
        Ok(bytes) => bytes,
        Err(err) => {
            let _ = std::fs::remove_file(&stage);
            return Err(ApiError::with_status(
                500,
                format!("cannot store {}: {err}", path.display()),
            ));
        }
    };
    let mut doc = record(&path)?;
    if let Json::Obj(pairs) = &mut doc {
        pairs.push(("replaced".to_string(), Json::Bool(replaced)));
    }
    crate::log_info!(LOG, "upload {} ({bytes} bytes)", path.display());
    Ok(Json::obj([("uploads", Json::Arr(vec![doc]))]))
}

/// Copies bytes as text - UTF-8 with the byte-order mark dropped and anything
/// that is not UTF-8 replaced, what [`store_bytes`] makes of a small upload -
/// a chunk at a time, so a file of any size is converted without being held.
/// Returns how many bytes were written.
fn copy_as_text(mut src: impl Read, mut dst: impl Write) -> std::io::Result<u64> {
    let mut chunk = vec![0u8; CHUNK];
    // what the last chunk left: a multi-byte sequence cut short, waiting for its rest
    let mut held: Vec<u8> = Vec::new();
    let mut first = true;
    let mut written = 0u64;
    loop {
        let n = match src.read(&mut chunk) {
            Ok(n) => n,
            Err(err) if err.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(err) => return Err(err),
        };
        held.extend_from_slice(&chunk[..n]);
        let at_end = n == 0;
        if first {
            // the mark is three bytes: wait for them, or the end, before deciding
            if held.len() < 3 && !at_end {
                continue;
            }
            if held.starts_with(&[0xef, 0xbb, 0xbf]) {
                held.drain(..3);
            }
            first = false;
        }
        let keep = if at_end { 0 } else { cut_short(&held) };
        let ready = held.len() - keep;
        let text = String::from_utf8_lossy(&held[..ready]);
        dst.write_all(text.as_bytes())?;
        written += text.len() as u64;
        held.drain(..ready);
        if at_end {
            return Ok(written);
        }
    }
}

/// How many bytes at the end of `bytes` begin a multi-byte UTF-8 sequence
/// whose rest has not arrived (0 when they end cleanly, or wrongly: a
/// sequence that is wrong is replaced whatever comes next).
fn cut_short(bytes: &[u8]) -> usize {
    let from = bytes.len().saturating_sub(3);
    for i in (from..bytes.len()).rev() {
        let lead = bytes[i];
        if lead & 0xc0 == 0x80 {
            continue; // a continuation byte: the lead is before it
        }
        let need = match lead {
            0xc2..=0xdf => 2,
            0xe0..=0xef => 3,
            0xf0..=0xf4 => 4,
            _ => return 0,
        };
        let have = bytes.len() - i;
        return if have < need { have } else { 0 };
    }
    0
}

/// Starts a training job on `texts` and hands back the job, as the train route
/// does: the model is held for the run and learns the way its kind learns
/// ([`crate::kinds::train`], every epoch reported and the stop button
/// honoured), is saved when it succeeds, and the frontend follows it on
/// `/api/job`.
pub fn start_train(svc: &Arc<Service>, texts: Vec<String>, settings: crate::kinds::TrainSettings) -> Json {
    svc.start_job("train");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let outcome = crate::checkpoint::train_job(&worker, &texts, &settings, 0);
        if outcome.is_ok() {
            worker.autosave();
        }
        worker.finish_job(outcome.map(|records| records.iter().map(|r| r.to_json()).collect()));
    });
    svc.job_json()
}

/// How an encode / teach request ends (`api._r_image_encode`,
/// `api._r_speech_teach`): `save_as` keeps the texts as an upload, `train`
/// starts a training job on them and answers 202.
///
/// `out` already holds `upload: null` and `job: null`; they are filled in here.
pub fn save_and_train(svc: &Arc<Service>, form: &Form, mut out: Json, texts: &[String]) -> Answer {
    let set = |doc: &mut Json, key: &str, value: Json| {
        if let Json::Obj(pairs) = doc {
            match pairs.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => pairs.push((key.to_string(), value)),
            }
        }
    };
    if let Some(name) = form.maybe_text("save_as") {
        let record = store_text(svc, &name, &format!("{}\n", texts.join("\n")))?;
        set(&mut out, "upload", record);
    }
    if !form.flag("train", false) {
        return Ok(out);
    }
    // `TrainConfig(epochs=3, lr=0.5, act_lr=..., batch_size=8)`: one long text
    // (or a few), so the sine model's rate is high; the other kinds count
    let base = crate::radix::TrainConfig::default();
    let config = crate::radix::TrainConfig {
        epochs: form.exact_int("epochs", 3).map_err(ApiError::bad_request)?.max(0) as usize,
        lr: form.exact_float("lr", 0.5).map_err(ApiError::bad_request)?,
        act_lr: form.exact_float("act_lr", base.act_lr).map_err(ApiError::bad_request)?,
        batch_size: form.exact_int("batch_size", 8).map_err(ApiError::bad_request)?.max(0) as usize,
        ..base
    };
    config.validate().map_err(ApiError::bad_request)?;
    let settings = crate::kinds::TrainSettings {
        config,
        ..Default::default()
    };
    let job = start_train(svc, texts.to_vec(), settings);
    set(&mut out, "job", job);
    Ok(accepted(out))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn multipart_request(body: Vec<u8>, boundary: &str, query: &[(&str, &str)]) -> Request {
        Request {
            method: "POST".into(),
            path: "/api/uploads".into(),
            query: query.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            body: Json::Null,
            raw: body,
            headers: vec![(
                "content-type".into(),
                format!("multipart/form-data; boundary={boundary}"),
            )],
        }
    }

    #[test]
    fn a_browser_form_comes_apart() {
        let body = b"--XyZ\r\n\
            Content-Disposition: form-data; name=\"note\"\r\n\r\n\
            just a field\r\n\
            --XyZ\r\n\
            Content-Disposition: form-data; name=\"file\"; filename=\"a \\\"b\\\".wav\"\r\n\
            Content-Type: audio/wav\r\n\r\n\
            RIFF\r\n--not the boundary\r\n\
            --XyZ--\r\n";
        let parts = parse(body, "multipart/form-data; boundary=XyZ").unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].name, "note");
        assert_eq!(parts[0].filename, None);
        assert_eq!(parts[0].data, b"just a field");
        assert_eq!(parts[1].filename.as_deref(), Some("a \"b\".wav"));
        assert_eq!(parts[1].content_type, "audio/wav");
        assert_eq!(parts[1].data, b"RIFF\r\n--not the boundary");
        assert!(parse(body, "multipart/form-data").is_err(), "no boundary");
    }

    #[test]
    fn lf_lines_quoted_boundaries_and_rfc2231_names_are_read() {
        let body = b"preamble\n--b 1\nContent-Disposition: form-data; name=file; \
            filename*=UTF-8''caf%C3%A9.txt\n\nhello\n--b 1--\n";
        let parts = parse(body, "multipart/form-data; boundary=\"b 1\"").unwrap();
        assert_eq!(parts.len(), 1);
        assert_eq!(parts[0].filename.as_deref(), Some("café.txt"));
        assert_eq!(parts[0].data, b"hello");
    }

    #[test]
    fn the_three_forms_give_the_same_files() {
        let multipart = multipart_request(
            b"--q\r\nContent-Disposition: form-data; name=\"file\"; filename=\"x.png\"\r\n\r\n\x89PNG\r\n--q--\r\n"
                .to_vec(),
            "q",
            &[("size", "32")],
        );
        let form = Form::read(&multipart, Some("image")).unwrap();
        assert_eq!(form.bytes("image").unwrap(), ("x.png".to_string(), b"\x89PNG".to_vec()));
        assert_eq!(form.int("size", 0).unwrap(), 32, "options come from the query string");

        let mut raw = multipart_request(b"\x89PNG".to_vec(), "", &[]);
        raw.headers = vec![("content-type".into(), "image/png".into())];
        let form = Form::read(&raw, Some("image")).unwrap();
        assert_eq!(form.bytes("image").unwrap().0, "image", "the route's default name");
        assert!(Form::read(&raw, None).is_err(), "the uploads route wants ?name=");

        let doc = crate::json::parse(r#"{"name":"x.png","content_base64":"iVBORw==","size":16,"train":true}"#).unwrap();
        let json = Request::json("POST", "/api/images/encode", doc);
        let form = Form::read(&json, Some("image")).unwrap();
        assert_eq!(form.bytes("image").unwrap(), ("x.png".to_string(), b"\x89PNG".to_vec()));
        assert_eq!(form.int("size", 0).unwrap(), 16);
        assert!(form.flag("train", false));

        let text = Request::json(
            "POST",
            "/api/speech/teach",
            crate::json::parse(r#"{"name":"x.wav","content":"text, not bytes"}"#).unwrap(),
        );
        let err = Form::read(&text, Some("speech"))
            .unwrap()
            .bytes("recording")
            .unwrap_err();
        assert!(err.message.contains("bytes"), "{}", err.message);
    }

    #[test]
    fn options_read_the_way_python_reads_them() {
        let doc = crate::json::parse(
            r#"{"attempts":"many","length":0,"lead":2.9,"threshold":"7.5","blame":1,"mode":"","texts":["a"],"text":5}"#,
        )
        .unwrap();
        let r = Request::json("POST", "/api/speech/tutor", doc);
        let form = Form::read(&r, Some("speech")).unwrap();
        assert!(form.int("attempts", 1).is_err());
        assert_eq!(form.int("length", 0).unwrap(), 0);
        assert_eq!(form.int("length", 7).unwrap(), 7, "a falsy 0 reads as the default");
        assert_eq!(
            form.exact_int("length", 7).unwrap(),
            0,
            "unless the option is read exactly"
        );
        assert_eq!(form.int("lead", 0).unwrap(), 2, "int() truncates");
        assert_eq!(form.float("threshold", 6.0).unwrap(), 7.5);
        assert!(form.flag("blame", false), "1 reads as true");
        assert_eq!(form.text("mode", "beam"), "beam", "an empty string is falsy");
        assert_eq!(form.texts("texts", "text").unwrap(), vec!["a"]);
        let only_text = Request::json("POST", "/", crate::json::parse(r#"{"text":5}"#).unwrap());
        assert!(Form::read(&only_text, Some("x"))
            .unwrap()
            .texts("texts", "text")
            .is_err());
    }

    #[test]
    fn an_upload_name_is_made_safe() {
        assert_eq!(sanitize_upload_name("../../etc/passwd").unwrap(), "passwd");
        assert_eq!(
            sanitize_upload_name("C:\\x\\my photo (1).png").unwrap(),
            "my photo (1).png"
        );
        assert_eq!(sanitize_upload_name("a*b?.txt").unwrap(), "a_b_.txt");
        assert!(sanitize_upload_name("..").is_err());
        assert!(sanitize_upload_name("  ").is_err());
        assert_eq!(sanitize_upload_name(&"x".repeat(300)).unwrap().len(), MAX_UPLOAD_NAME);
    }

    #[test]
    fn a_bad_json_body_is_a_400() {
        let mut r = Request::json("POST", "/api/uploads", Json::Null);
        r.raw = b"{not json".to_vec();
        assert!(Form::read(&r, None).is_err());
        r.raw = b"   ".to_vec();
        assert!(
            Form::read(&r, None).unwrap().files().is_err(),
            "an empty body names no file"
        );
    }

    /// Reads a body one to three bytes at a time, so a delimiter is cut at
    /// every place it can be.
    struct Dribble<'a> {
        bytes: &'a [u8],
        at: usize,
        step: usize,
    }

    impl Read for Dribble<'_> {
        fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
            let n = (self.bytes.len() - self.at).min(out.len()).min(self.step);
            out[..n].copy_from_slice(&self.bytes[self.at..self.at + n]);
            self.at += n;
            self.step = self.step % 3 + 1;
            Ok(n)
        }
    }

    /// The parts of a body read as a stream, `step` bytes at a time (0: in
    /// one piece), gathered the way `parse` gathers them.
    fn streamed(body: &[u8], content_type: &str, step: usize) -> Result<Vec<Part>, ApiError> {
        let mut parts = Vec::new();
        let mut each = |head: &Part, data: &mut dyn Read| {
            let mut part = head.clone();
            data.read_to_end(&mut part.data)
                .map_err(|err| ApiError::bad_request(err.to_string()))?;
            parts.push(part);
            Ok(())
        };
        if step == 0 {
            stream_parts(&mut std::io::Cursor::new(body), content_type, &mut each)?;
        } else {
            stream_parts(
                &mut Dribble {
                    bytes: body,
                    at: 0,
                    step,
                },
                content_type,
                &mut each,
            )?;
        }
        Ok(parts)
    }

    /// Bytes with every prefix of a delimiter in them, and CRs and LFs where
    /// they hurt: what a compressed archive looks like to a boundary search.
    fn awkward_bytes(len: usize) -> Vec<u8> {
        let pieces: [&[u8]; 8] = [
            b"-",
            b"--",
            b"\n--",
            b"\r\n--",
            b"\n--X",
            b"\r\n--Xy",
            b"\n--Xy\r",
            b"\r\r\n",
        ];
        let mut out = Vec::with_capacity(len + 16);
        let mut seed = 12345u32;
        while out.len() < len {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12345);
            let pick = (seed >> 16) as usize;
            if pick % 3 == 0 {
                out.extend_from_slice(pieces[pick % pieces.len()]);
            } else {
                // never a 'Z', so the delimiter itself cannot come up by chance
                let byte = (pick % 251) as u8;
                out.push(if byte == b'Z' { b'z' } else { byte });
            }
        }
        out.truncate(len);
        out
    }

    #[test]
    fn a_streamed_body_comes_apart_as_a_held_one_does() {
        let mut two_files =
            b"--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"one.bin\"\r\n\r\n".to_vec();
        two_files.extend(awkward_bytes(700 * 1024));
        two_files.extend_from_slice(
            b"\r\n--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"two.txt\"\r\n\r\n",
        );
        two_files.extend(awkward_bytes(3000));
        two_files.extend_from_slice(b"\r\n--XyZ--\r\nepilogue\r\n");
        let bodies: [(&[u8], &str); 10] = [
            (
                b"--XyZ\r\n\
                  Content-Disposition: form-data; name=\"note\"\r\n\r\n\
                  just a field\r\n\
                  --XyZ\r\n\
                  Content-Disposition: form-data; name=\"file\"; filename=\"a \\\"b\\\".wav\"\r\n\
                  Content-Type: audio/wav\r\n\r\n\
                  RIFF\r\n--not the boundary\r\n\
                  --XyZ--\r\n",
                "multipart/form-data; boundary=XyZ",
            ),
            (
                b"preamble\n--b 1\nContent-Disposition: form-data; name=file; \
                  filename*=UTF-8''caf%C3%A9.txt\n\nhello\n--b 1--\n",
                "multipart/form-data; boundary=\"b 1\"",
            ),
            // every prefix of the delimiter inside the data, and a CR before the real one
            (
                b"--XyZ\r\nContent-Disposition: form-data; name=\"f\"; filename=\"t.bin\"\r\n\r\n\
                  -\n--\n--X\n--Xy\r\n--Xy\r\r\n--XyZ--\r\n",
                "multipart/form-data; boundary=XyZ",
            ),
            // no closing delimiter: the part keeps what came
            (
                b"--q\r\nContent-Disposition: form-data; name=\"f\"; filename=\"x.txt\"\r\n\r\nunfinished business",
                "multipart/form-data; boundary=q",
            ),
            // a part with no headers at all
            (b"--q\r\n\r\nbare\r\n--q--\r\n", "multipart/form-data; boundary=q"),
            // headers and no blank line: a part without data
            (
                b"--q\r\nContent-Disposition: form-data; name=\"x\"\r\n--q--\r\n",
                "multipart/form-data; boundary=q",
            ),
            // a base64 part, decoded whole
            (
                b"--q\r\nContent-Disposition: form-data; name=\"f\"; filename=\"b.bin\"\r\n\
                  Content-Transfer-Encoding: base64\r\n\r\naGVs\r\nbG8=\r\n--q--\r\n",
                "multipart/form-data; boundary=q",
            ),
            (b"", "multipart/form-data; boundary=q"),
            (b"nothing here\r\n", "multipart/form-data; boundary=q"),
            (&two_files, "multipart/form-data; boundary=XyZ"),
        ];
        for (i, (body, content_type)) in bodies.iter().enumerate() {
            let held = parse(body, content_type).unwrap();
            assert_eq!(streamed(body, content_type, 0).unwrap(), held, "body {i}, in one piece");
            // a large body is cut fine only where it is cheap to
            let steps: &[usize] = if body.len() > 100_000 { &[1000] } else { &[1, 2, 3] };
            for &step in steps {
                assert_eq!(
                    streamed(body, content_type, step).unwrap(),
                    held,
                    "body {i}, {step} bytes at a time"
                );
            }
        }
        let (body, content_type) = bodies[6];
        assert_eq!(streamed(body, content_type, 0).unwrap()[0].data, b"hello");
        let (body, content_type) = bodies[9];
        let parts = streamed(body, content_type, 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data.len(), 700 * 1024);
        assert_eq!(parts[1].filename.as_deref(), Some("two.txt"));
        assert!(streamed(body, "multipart/form-data", 0).is_err(), "no boundary");
    }

    #[test]
    fn a_part_left_unread_is_dropped_and_the_next_still_found() {
        let body = b"--q\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a.txt\"\r\n\r\n\
            skipped entirely\r\n--q\r\nContent-Disposition: form-data; name=\"f\"; filename=\"b.txt\"\r\n\r\n\
            read in part\r\n--q--\r\n";
        let mut names = Vec::new();
        let mut firsts = Vec::new();
        stream_parts(
            &mut Dribble {
                bytes: body,
                at: 0,
                step: 1,
            },
            "multipart/form-data; boundary=q",
            &mut |part, data| {
                names.push(part.filename.clone().unwrap_or_default());
                if names.len() == 2 {
                    let mut four = [0u8; 4];
                    data.read_exact(&mut four)
                        .map_err(|e| ApiError::bad_request(e.to_string()))?;
                    firsts.push(four.to_vec());
                }
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(names, vec!["a.txt", "b.txt"]);
        assert_eq!(firsts, vec![b"read".to_vec()]);
    }

    #[test]
    fn a_text_upload_is_converted_a_chunk_at_a_time_as_it_would_be_whole() {
        let mut bytes = "\u{feff}caf\u{e9} \u{2615} \u{1f600}\n".as_bytes().to_vec();
        bytes.extend_from_slice(b"bad \xff byte\n");
        bytes.extend_from_slice("more \u{4e16}\u{754c}".as_bytes());
        bytes.extend_from_slice(b"\xe2\x82"); // a sequence cut short at the very end
        let whole = String::from_utf8_lossy(&bytes[3..]).into_owned();
        for step in [0usize, 1, 2, 3] {
            let mut out = Vec::new();
            let written = if step == 0 {
                copy_as_text(std::io::Cursor::new(&bytes), &mut out).unwrap()
            } else {
                copy_as_text(
                    Dribble {
                        bytes: &bytes,
                        at: 0,
                        step,
                    },
                    &mut out,
                )
                .unwrap()
            };
            assert_eq!(String::from_utf8_lossy(&out), whole, "{step} bytes at a time");
            assert_eq!(written as usize, out.len());
        }
        for (input, want) in [
            (&b""[..], ""),
            (&b"\xef\xbb\xbf"[..], ""),
            (&b"ab"[..], "ab"),
            (&b"\xef\xbb\xbfab"[..], "ab"),
            (&b"\xef\xbb"[..], "\u{fffd}"),
        ] {
            let mut out = Vec::new();
            copy_as_text(
                Dribble {
                    bytes: input,
                    at: 0,
                    step: 1,
                },
                &mut out,
            )
            .unwrap();
            assert_eq!(String::from_utf8_lossy(&out), want, "{input:?}");
        }
        assert_eq!(cut_short(b"abc"), 0);
        assert_eq!(cut_short(b"ab\xc3"), 1);
        assert_eq!(cut_short(b"a\xe2\x82"), 2);
        assert_eq!(cut_short(b"\xf0\x9f\x98"), 3);
        assert_eq!(cut_short(b"\xf0\x9f\x98\x80"), 0, "complete");
        assert_eq!(cut_short(b"a\xff"), 0, "not a lead byte");
        assert_eq!(cut_short(b"\x80\x80\x80"), 0, "continuation bytes alone");
    }

    #[test]
    fn a_text_record_counts_a_line_at_a_time() {
        let dir = std::env::temp_dir().join(format!("radixnet-record-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("notes.txt");
        std::fs::write(
            &path,
            "para one line a\npara one line b\n\n  \t\npara two\r\nlast \u{e9}",
        )
        .unwrap();
        let doc = record(&path).unwrap();
        assert_eq!(doc.at("name").as_str(), Some("notes.txt"));
        assert_eq!(doc.at("lines").as_i64(), Some(4));
        assert_eq!(doc.at("chars").as_i64(), Some(53));
        assert_eq!(doc.at("bytes").as_i64(), Some(54));
        std::fs::write(&path, b"not \xff utf-8\n").unwrap();
        assert_eq!(record(&path).unwrap().at("chars").as_i64(), Some(12));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
