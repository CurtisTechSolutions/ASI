//! JSON, written out: a value, a parser and a serialiser.  The crate has no
//! dependencies, and the machine file, the experiment records and the HTTP API
//! are all JSON.
//!
//! Numbers are `f64`; integers print without a fraction.  Object keys keep the
//! order they were inserted or parsed in, so a serialised machine reads in the
//! order the Python port writes it.

use std::fmt::Write as _;

/// A JSON value.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub fn object() -> Json {
        Json::Object(Vec::new())
    }

    /// Set `key` on an object (replacing an existing one); does nothing on any other value.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Object(pairs) = self {
            if let Some(pair) = pairs.iter_mut().find(|(k, _)| k == key) {
                pair.1 = value;
            } else {
                pairs.push((key.to_string(), value));
            }
        }
    }

    /// Build an object: `Json::object().with("a", 1.0.into())`.
    pub fn with(mut self, key: &str, value: Json) -> Json {
        self.set(key, value);
        self
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Object(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Number(n) => Some(*n),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        self.as_f64().map(|n| n as i64)
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&Vec<Json>> {
        match self {
            Json::Array(a) => Some(a),
            _ => None,
        }
    }

    /// `get(key)` as a number, or `default`.
    pub fn num(&self, key: &str, default: f64) -> f64 {
        self.get(key).and_then(Json::as_f64).unwrap_or(default)
    }

    pub fn str_or<'a>(&'a self, key: &str, default: &'a str) -> &'a str {
        self.get(key).and_then(Json::as_str).unwrap_or(default)
    }

    pub fn bool_or(&self, key: &str, default: bool) -> bool {
        self.get(key).and_then(Json::as_bool).unwrap_or(default)
    }

    pub fn numbers(list: &[f64]) -> Json {
        Json::Array(list.iter().map(|v| Json::Number(*v)).collect())
    }

    pub fn strings(list: &[String]) -> Json {
        Json::Array(list.iter().map(|s| Json::String(s.clone())).collect())
    }

    /// Serialise, compact.
    pub fn dump(&self) -> String {
        let mut out = String::new();
        self.write(&mut out, None, 0);
        out
    }

    /// Serialise, indented by `indent` spaces per level.
    pub fn pretty(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write(&mut out, Some(indent), 0);
        out.push('\n');
        out
    }

    fn write(&self, out: &mut String, indent: Option<usize>, level: usize) {
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Number(n) => write_number(out, *n),
            Json::String(s) => write_string(out, s),
            Json::Array(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline(out, indent, level + 1);
                    item.write(out, indent, level + 1);
                }
                newline(out, indent, level);
                out.push(']');
            }
            Json::Object(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (k, v)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    newline(out, indent, level + 1);
                    write_string(out, k);
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    v.write(out, indent, level + 1);
                }
                newline(out, indent, level);
                out.push('}');
            }
        }
    }
}

impl From<f64> for Json {
    fn from(n: f64) -> Json {
        Json::Number(n)
    }
}
impl From<i64> for Json {
    fn from(n: i64) -> Json {
        Json::Number(n as f64)
    }
}
impl From<usize> for Json {
    fn from(n: usize) -> Json {
        Json::Number(n as f64)
    }
}
impl From<bool> for Json {
    fn from(b: bool) -> Json {
        Json::Bool(b)
    }
}
impl From<&str> for Json {
    fn from(s: &str) -> Json {
        Json::String(s.to_string())
    }
}
impl From<String> for Json {
    fn from(s: String) -> Json {
        Json::String(s)
    }
}
impl From<Vec<Json>> for Json {
    fn from(items: Vec<Json>) -> Json {
        Json::Array(items)
    }
}

fn newline(out: &mut String, indent: Option<usize>, level: usize) {
    if let Some(n) = indent {
        out.push('\n');
        for _ in 0..n * level {
            out.push(' ');
        }
    }
}

fn write_number(out: &mut String, n: f64) {
    if !n.is_finite() {
        out.push_str("null");
    } else if n.fract() == 0.0 && n.abs() < 1e15 {
        let _ = write!(out, "{}", n as i64);
    } else {
        let _ = write!(out, "{}", n);
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// Parse a JSON text.
pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        chars: text.as_bytes(),
        at: 0,
        text,
    };
    let value = p.value()?;
    p.skip_ws();
    if p.at != p.chars.len() {
        return Err(format!("trailing characters at {}", p.at));
    }
    Ok(value)
}

struct Parser<'a> {
    chars: &'a [u8],
    at: usize,
    text: &'a str,
}

impl<'a> Parser<'a> {
    fn skip_ws(&mut self) {
        while self.at < self.chars.len() && matches!(self.chars[self.at], b' ' | b'\n' | b'\r' | b'\t') {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.chars.get(self.at).copied()
    }

    fn expect(&mut self, word: &str) -> Result<(), String> {
        if self.text[self.at..].starts_with(word) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.skip_ws();
        match self.peek() {
            None => Err("unexpected end of input".to_string()),
            Some(b'n') => self.expect("null").map(|_| Json::Null),
            Some(b't') => self.expect("true").map(|_| Json::Bool(true)),
            Some(b'f') => self.expect("false").map(|_| Json::Bool(false)),
            Some(b'"') => self.string().map(Json::String),
            Some(b'[') => {
                self.at += 1;
                let mut items = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek() == Some(b']') {
                        self.at += 1;
                        return Ok(Json::Array(items));
                    }
                    if !items.is_empty() {
                        self.expect(",")?;
                    }
                    items.push(self.value()?);
                }
            }
            Some(b'{') => {
                self.at += 1;
                let mut pairs = Vec::new();
                loop {
                    self.skip_ws();
                    if self.peek() == Some(b'}') {
                        self.at += 1;
                        return Ok(Json::Object(pairs));
                    }
                    if !pairs.is_empty() {
                        self.expect(",")?;
                        self.skip_ws();
                    }
                    let key = self.string()?;
                    self.skip_ws();
                    self.expect(":")?;
                    let value = self.value()?;
                    pairs.push((key, value));
                }
            }
            Some(_) => self.number(),
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        while self.at < self.chars.len()
            && matches!(self.chars[self.at], b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E')
        {
            self.at += 1;
        }
        self.text[start..self.at]
            .parse::<f64>()
            .map(Json::Number)
            .map_err(|_| format!("bad number at {start}"))
    }

    fn string(&mut self) -> Result<String, String> {
        self.expect("\"")?;
        let mut out = String::new();
        loop {
            let rest = &self.text[self.at..];
            let mut it = rest.char_indices();
            let (_, c) = it.next().ok_or("unterminated string")?;
            self.at += c.len_utf8();
            match c {
                '"' => return Ok(out),
                '\\' => {
                    let (_, e) = it.next().ok_or("unterminated escape")?;
                    self.at += e.len_utf8();
                    match e {
                        '"' => out.push('"'),
                        '\\' => out.push('\\'),
                        '/' => out.push('/'),
                        'b' => out.push('\u{8}'),
                        'f' => out.push('\u{c}'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        'u' => {
                            let code = self.hex4()?;
                            let ch = if (0xD800..0xDC00).contains(&code) {
                                self.expect("\\u")?;
                                let low = self.hex4()?;
                                0x10000 + ((code - 0xD800) << 10) + (low.wrapping_sub(0xDC00) & 0x3FF)
                            } else {
                                code
                            };
                            out.push(char::from_u32(ch).unwrap_or('\u{FFFD}'));
                        }
                        other => return Err(format!("bad escape \\{other}")),
                    }
                }
                c => out.push(c),
            }
        }
    }

    fn hex4(&mut self) -> Result<u32, String> {
        let hex = self.text.get(self.at..self.at + 4).ok_or("short \\u escape")?;
        self.at += 4;
        u32::from_str_radix(hex, 16).map_err(|_| format!("bad \\u escape {hex}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let text = r#"{"a":[1,2.5,-3e2,true,null,"x\"y\né"],"b":{}}"#;
        let v = parse(text).unwrap();
        assert_eq!(v.get("a").unwrap().as_array().unwrap().len(), 6);
        assert_eq!(parse(&v.dump()).unwrap(), v);
        assert_eq!(parse(&v.pretty(2)).unwrap(), v);
        assert_eq!(Json::Number(3.0).dump(), "3");
        assert_eq!(Json::Number(0.5).dump(), "0.5");
    }
}
