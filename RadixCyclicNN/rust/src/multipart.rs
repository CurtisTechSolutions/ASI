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
//! No dependencies, like everything else: the multipart parser and base64
//! ([`base64`]) are written out.

pub mod base64;

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
    (from..=hay.len() - needle.len()).find(|&i| &hay[i..i + needle.len()] == needle)
}

/// The parts of a `multipart/form-data` body.
///
/// Lenient where Python's `email` parser is lenient: lines may end in CRLF or
/// LF, and a body without the closing delimiter keeps the parts it has.
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
        let next = find(body, &[b"\n".as_slice(), &delimiter].concat(), start);
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
    let (head, data) = match (find(raw, b"\r\n\r\n", 0), find(raw, b"\n\n", 0)) {
        (Some(a), Some(b)) if b < a => (&raw[..b], &raw[b + 2..]),
        (Some(a), _) => (&raw[..a], &raw[a + 4..]),
        (None, Some(b)) => (&raw[..b], &raw[b + 2..]),
        // a part that starts with its blank line has no headers at all
        (None, None) if raw.starts_with(b"\r\n") => (&raw[..0], &raw[2..]),
        (None, None) => (raw, &raw[raw.len()..]),
    };
    let head = String::from_utf8_lossy(head);
    let mut part = Part {
        name: String::new(),
        filename: None,
        content_type: String::new(),
        data: data.to_vec(),
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
    if encoding == "base64" {
        let text: String = String::from_utf8_lossy(&part.data)
            .chars()
            .filter(|c| !c.is_whitespace())
            .collect();
        part.data = base64::decode(&text).map_err(|err| format!("malformed multipart/form-data body: {err}"))?;
    }
    Ok(part)
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

/// The listing record of one text upload (`api._upload_record`).
fn record(path: &std::path::Path) -> Result<Json, ApiError> {
    let meta = std::fs::metadata(path).map_err(|err| ApiError::with_status(500, err.to_string()))?;
    let bytes = std::fs::read(path).map_err(|err| ApiError::with_status(500, err.to_string()))?;
    let text = String::from_utf8_lossy(&bytes);
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    Ok(Json::obj([
        ("name", Json::str(name)),
        ("bytes", Json::Int(meta.len() as i64)),
        ("chars", Json::Int(text.chars().count() as i64)),
        (
            "lines",
            Json::Int(
                text.lines()
                    .filter(|l| !l.trim_matches(base64::is_python_space).is_empty())
                    .count() as i64,
            ),
        ),
        ("modified", Json::str(modified(&meta))),
    ]))
}

/// Stores `content` as the upload `name`, replacing one of that name
/// (`ModelService.upload`); the record says whether it did.
pub fn store_text(svc: &Service, name: &str, content: &str) -> Result<Json, ApiError> {
    let dir = upload_dir(svc)?;
    let path = std::path::Path::new(&dir).join(sanitize_upload_name(name)?);
    let replaced = path.is_file();
    // written beside and moved into place, so a reader never sees half a file
    let part = path.with_extension(match path.extension() {
        Some(ext) => format!("{}.part", ext.to_string_lossy()),
        None => "part".to_string(),
    });
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

/// Keeps a ZIP archive as a single upload; its text entries are unpacked in
/// memory whenever it is used (`ModelService.store_archive`).  A corrupt
/// archive, or one without a single text entry, is refused (400); the
/// entries passed over are listed with their reasons in the summary.
pub fn store_archive(svc: &Service, name: &str, data: &[u8]) -> Result<Json, ApiError> {
    let dir = upload_dir(svc)?;
    let mut archive_name = sanitize_upload_name(name)?;
    if !archive_name.to_lowercase().ends_with(".zip") {
        archive_name.push_str(".zip");
    }
    let mut archive = crate::zip::Archive::from_bytes(data.to_vec())
        .map_err(|err| ApiError::bad_request(format!("{archive_name}: {err}")))?;
    let mut extracted = 0usize;
    let skipped = crate::source::walk_texts(&mut archive, svc.workers, &mut |_, _, _| {
        extracted += 1;
        std::ops::ControlFlow::Continue(())
    });
    if extracted == 0 {
        let reasons: Vec<String> = skipped
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
    let path = std::path::Path::new(&dir).join(&archive_name);
    let replaced = path.is_file();
    let part = path.with_extension("zip.part");
    std::fs::write(&part, data)
        .and_then(|_| std::fs::rename(&part, &path))
        .map_err(|err| ApiError::with_status(500, format!("cannot store {}: {err}", path.display())))?;
    let mut record = crate::source::archive_record(&path, svc.workers)
        .ok_or_else(|| ApiError::with_status(500, format!("{} is not a ZIP archive", path.display())))?;
    if let Json::Obj(pairs) = &mut record {
        pairs.push(("replaced".to_string(), Json::Bool(replaced)));
    }
    let summary = Json::obj([
        ("name", Json::str(archive_name.clone())),
        ("bytes", Json::Int(data.len() as i64)),
        ("entries", Json::Int((extracted + skipped.len()) as i64)),
        ("extracted", Json::Int(extracted as i64)),
        ("skipped", Json::Arr(skipped.iter().map(|s| s.to_json()).collect())),
    ]);
    crate::log_info!(
        LOG,
        "upload {archive_name} ({} bytes; {extracted} text file(s) inside, {} skipped)",
        data.len(),
        skipped.len()
    );
    Ok(Json::obj([
        ("uploads", Json::Arr(vec![record])),
        ("archives", Json::Arr(vec![summary])),
    ]))
}

/// Stores every file of an upload request: `{"uploads": [records]}` - the
/// body of `POST /api/uploads` (`api._r_upload`).
pub fn store_all(svc: &Service, files: &[Upload]) -> Result<Json, ApiError> {
    let mut records = Vec::new();
    let mut archives: Vec<Json> = Vec::new();
    for file in files {
        match &file.payload {
            Payload::Text(text) => records.push(store_text(svc, &file.name, text)?),
            Payload::Bytes(data) => {
                let stored = store_bytes(svc, &file.name, data)?;
                records.extend(stored.at("uploads").as_array().iter().cloned());
                archives.extend(stored.at("archives").as_array().iter().cloned());
            }
        }
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
}
