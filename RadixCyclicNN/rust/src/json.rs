//! JSON, as the Python implementation writes and reads it.
//!
//! The model file is the contract between the three implementations
//! (`../../DESIGN.md` §19), so this module is not "a JSON library": it is the
//! one Python's `json` module writes - compact separators, `ensure_ascii=False`
//! so text goes out as UTF-8, and **floats rendered exactly as `repr(float)`
//! renders them**, which is not how Rust's `{}` renders them (`0.0` prints as
//! `0`, `1e-05` as `0.00001`).  An object keeps its insertion order, because
//! Python's dicts do and the file is diffed by people.

use std::collections::VecDeque;

/// A JSON value.  Objects keep the order their keys were inserted in.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// An object from `(key, value)` pairs.
    pub fn obj<K: Into<String>, I: IntoIterator<Item = (K, Json)>>(pairs: I) -> Json {
        Json::Obj(pairs.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// A string value.
    pub fn str<S: Into<String>>(s: S) -> Json {
        Json::Str(s.into())
    }

    /// An array of integers.
    pub fn ints<I: IntoIterator<Item = i64>>(values: I) -> Json {
        Json::Arr(values.into_iter().map(Json::Int).collect())
    }

    /// An array of floats.
    pub fn nums<I: IntoIterator<Item = f64>>(values: I) -> Json {
        Json::Arr(values.into_iter().map(Json::Num).collect())
    }

    /// An array of strings.
    pub fn strs<I: IntoIterator<Item = S>, S: Into<String>>(values: I) -> Json {
        Json::Arr(values.into_iter().map(Json::str).collect())
    }

    // -- reading ------------------------------------------------------------

    /// One key of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(pairs) => pairs.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// One key, or `Json::Null` when it is absent - `dict.get(key)` in one call.
    pub fn at(&self, key: &str) -> &Json {
        self.get(key).unwrap_or(&Json::Null)
    }

    /// The value as a float (an integer converts; anything else is `None`).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(x) => Some(*x),
            Json::Int(n) => Some(*n as f64),
            Json::Bool(b) => Some(*b as i64 as f64),
            _ => None,
        }
    }

    /// The value as an integer (a whole float converts).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Json::Int(n) => Some(*n),
            Json::Num(x) if x.is_finite() => Some(*x as i64),
            Json::Bool(b) => Some(*b as i64),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// The value as an array (`&[]` when it is not one, or is null).
    pub fn as_array(&self) -> &[Json] {
        match self {
            Json::Arr(items) => items,
            _ => &[],
        }
    }

    /// An array of integers; a missing array reads as empty.
    pub fn to_i64s(&self) -> Vec<i64> {
        self.as_array().iter().filter_map(|v| v.as_i64()).collect()
    }

    /// An array of floats.
    pub fn to_f64s(&self) -> Vec<f64> {
        self.as_array().iter().filter_map(|v| v.as_f64()).collect()
    }

    /// An array of strings.
    pub fn to_strings(&self) -> Vec<String> {
        self.as_array()
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect()
    }

    pub fn is_null(&self) -> bool {
        matches!(self, Json::Null)
    }

    // -- writing ------------------------------------------------------------

    /// Renders the value the way `json.dumps(..., separators=(",", ":"),
    /// ensure_ascii=False)` renders it; `indent` > 0 pretty-prints instead.
    pub fn render(&self, indent: usize) -> String {
        let mut out = String::new();
        self.write(&mut out, indent, 0);
        out
    }

    fn write(&self, out: &mut String, indent: usize, depth: usize) {
        let pad = |out: &mut String, depth: usize| {
            if indent > 0 {
                out.push('\n');
                out.push_str(&" ".repeat(indent * depth));
            }
        };
        match self {
            Json::Null => out.push_str("null"),
            Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Json::Int(n) => out.push_str(&n.to_string()),
            Json::Num(x) => out.push_str(&py_repr(*x)),
            Json::Str(s) => write_string(out, s),
            Json::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    pad(out, depth + 1);
                    item.write(out, indent, depth + 1);
                }
                pad(out, depth);
                out.push(']');
            }
            Json::Obj(pairs) => {
                if pairs.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (i, (key, value)) in pairs.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    pad(out, depth + 1);
                    write_string(out, key);
                    out.push(':');
                    if indent > 0 {
                        out.push(' ');
                    }
                    value.write(out, indent, depth + 1);
                }
                pad(out, depth);
                out.push('}');
            }
        }
    }
}

fn write_string(out: &mut String, s: &str) {
    out.push('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A float as `repr(float)` writes it in Python: the shortest string that reads
/// back as the same double, in fixed notation while the exponent is in
/// `[-4, 16)` and in exponential notation outside it, with a two-digit exponent
/// and never without a fractional part.
///
/// Rust's `{}` is shortest-round-trip too, but formats the rest differently:
/// `0.0` prints as `0`, `1e-5` as `0.00001`, `1e300` as 301 digits.  The file
/// is read by Python and by Go, so it is written the way they write it.
pub fn py_repr(x: f64) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 {
            "-Infinity".to_string()
        } else {
            "Infinity".to_string()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0.0".to_string()
        } else {
            "0.0".to_string()
        };
    }
    let formatted = format!("{x:e}"); // shortest round-trip: "-1.2345e-7"
    let (mantissa, exponent) = formatted.split_once('e').expect("{:e} always writes an exponent");
    let exp: i32 = exponent.parse().expect("{:e} writes a decimal exponent");
    let sign = if mantissa.starts_with('-') { "-" } else { "" };
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    if !(-4..16).contains(&exp) {
        let mut m = String::from(&digits[..1]);
        if digits.len() > 1 {
            m.push('.');
            m.push_str(&digits[1..]);
        }
        let s = if exp < 0 { '-' } else { '+' };
        return format!("{sign}{m}e{s}{:02}", exp.abs());
    }
    if exp < 0 {
        return format!("{sign}0.{}{digits}", "0".repeat((-exp - 1) as usize));
    }
    let point = exp as usize + 1;
    if digits.len() > point {
        format!("{sign}{}.{}", &digits[..point], &digits[point..])
    } else {
        format!("{sign}{digits}{}.0", "0".repeat(point - digits.len()))
    }
}

// -- parsing ----------------------------------------------------------------

/// Reads a JSON document.
pub fn parse(text: &str) -> Result<Json, String> {
    let mut p = Parser {
        bytes: text.as_bytes(),
        at: 0,
    };
    p.skip_space();
    let value = p.value()?;
    p.skip_space();
    if p.at != p.bytes.len() {
        return Err(format!("trailing data at byte {}", p.at));
    }
    Ok(value)
}

struct Parser<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Parser<'_> {
    fn skip_space(&mut self) {
        while self.at < self.bytes.len() && matches!(self.bytes[self.at], b' ' | b'\t' | b'\n' | b'\r') {
            self.at += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.at).copied()
    }

    fn eat(&mut self, want: u8) -> Result<(), String> {
        if self.peek() == Some(want) {
            self.at += 1;
            Ok(())
        } else {
            Err(format!("expected {:?} at byte {}", want as char, self.at))
        }
    }

    fn literal(&mut self, word: &str) -> Result<(), String> {
        if self.bytes[self.at..].starts_with(word.as_bytes()) {
            self.at += word.len();
            Ok(())
        } else {
            Err(format!("expected {word} at byte {}", self.at))
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        match self.peek().ok_or("unexpected end of document")? {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Ok(Json::Str(self.string()?)),
            b't' => self.literal("true").map(|_| Json::Bool(true)),
            b'f' => self.literal("false").map(|_| Json::Bool(false)),
            b'n' => self.literal("null").map(|_| Json::Null),
            b'N' => self.literal("NaN").map(|_| Json::Num(f64::NAN)),
            b'I' => self.literal("Infinity").map(|_| Json::Num(f64::INFINITY)),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.eat(b'{')?;
        let mut pairs = Vec::new();
        self.skip_space();
        if self.peek() == Some(b'}') {
            self.at += 1;
            return Ok(Json::Obj(pairs));
        }
        loop {
            self.skip_space();
            let key = self.string()?;
            self.skip_space();
            self.eat(b':')?;
            self.skip_space();
            pairs.push((key, self.value()?));
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(Json::Obj(pairs));
                }
                _ => return Err(format!("expected ',' or '}}' at byte {}", self.at)),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.skip_space();
        if self.peek() == Some(b']') {
            self.at += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            self.skip_space();
            items.push(self.value()?);
            self.skip_space();
            match self.peek() {
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err(format!("expected ',' or ']' at byte {}", self.at)),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        self.eat(b'"')?;
        let mut out = String::new();
        let mut pending: Option<u16> = None; // a high surrogate waiting for its pair
        loop {
            let byte = self.peek().ok_or("unterminated string")?;
            match byte {
                b'"' => {
                    self.at += 1;
                    if pending.is_some() {
                        out.push('\u{fffd}');
                    }
                    return Ok(out);
                }
                b'\\' => {
                    self.at += 1;
                    let esc = self.peek().ok_or("unterminated escape")?;
                    self.at += 1;
                    let ch = match esc {
                        b'"' => '"',
                        b'\\' => '\\',
                        b'/' => '/',
                        b'b' => '\u{8}',
                        b'f' => '\u{c}',
                        b'n' => '\n',
                        b'r' => '\r',
                        b't' => '\t',
                        b'u' => {
                            let hex = self.bytes.get(self.at..self.at + 4).ok_or("truncated \\u escape")?;
                            let code = u16::from_str_radix(std::str::from_utf8(hex).map_err(|_| "bad \\u escape")?, 16)
                                .map_err(|_| "bad \\u escape")?;
                            self.at += 4;
                            match (pending, code) {
                                (None, 0xd800..=0xdbff) => {
                                    pending = Some(code);
                                    continue;
                                }
                                (Some(high), 0xdc00..=0xdfff) => {
                                    pending = None;
                                    let joined = 0x10000 + ((high as u32 - 0xd800) << 10) + (code as u32 - 0xdc00);
                                    char::from_u32(joined).unwrap_or('\u{fffd}')
                                }
                                _ => {
                                    pending = None;
                                    char::from_u32(code as u32).unwrap_or('\u{fffd}')
                                }
                            }
                        }
                        other => return Err(format!("unknown escape \\{}", other as char)),
                    };
                    out.push(ch);
                }
                _ => {
                    // copy the whole UTF-8 run up to the next escape or quote
                    let start = self.at;
                    while let Some(b) = self.peek() {
                        if b == b'"' || b == b'\\' {
                            break;
                        }
                        self.at += 1;
                    }
                    out.push_str(std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| "invalid UTF-8")?);
                }
            }
        }
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.at;
        if self.peek() == Some(b'-') {
            self.at += 1;
            if self.peek() == Some(b'I') {
                self.literal("Infinity")?;
                return Ok(Json::Num(f64::NEG_INFINITY));
            }
        }
        let mut float = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.at += 1,
                b'.' | b'e' | b'E' | b'+' | b'-' => {
                    float = true;
                    self.at += 1;
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.bytes[start..self.at]).map_err(|_| "invalid number")?;
        if text.is_empty() {
            return Err(format!("expected a value at byte {start}"));
        }
        if !float {
            if let Ok(n) = text.parse::<i64>() {
                return Ok(Json::Int(n));
            }
        }
        text.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad number {text:?}"))
    }
}

/// A queue of the values of an array, for readers that walk one.
pub fn queue(value: &Json) -> VecDeque<&Json> {
    value.as_array().iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_writes_what_a_parser_reads_back() {
        let doc = Json::obj([
            ("name", Json::str("a \"quoted\"\nline")),
            ("rate", Json::Num(1234.5)),
            ("count", Json::Int(7)),
            ("list", Json::Arr(vec![Json::Bool(true), Json::Null])),
        ]);
        assert_eq!(
            doc.render(0),
            r#"{"name":"a \"quoted\"\nline","rate":1234.5,"count":7,"list":[true,null]}"#
        );
        assert!(doc.render(2).contains("\n  \"name\""));
        assert_eq!(parse(&doc.render(0)).unwrap(), doc);
    }

    #[test]
    fn floats_read_as_python_writes_them() {
        // the right-hand sides are what CPython's repr() prints
        let cases: &[(f64, &str)] = &[
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            (1.0, "1.0"),
            (-1.5, "-1.5"),
            (1.0 / 3.0, "0.3333333333333333"),
            (0.5, "0.5"),
            (123.456, "123.456"),
            (1e-4, "0.0001"),
            (1e-5, "1e-05"),
            (9.999e-5, "9.999e-05"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1.5e16, "1.5e+16"),
            (1e100, "1e+100"),
            (5e-324, "5e-324"),
            (f64::MAX, "1.7976931348623157e+308"),
            (-1.3862943611198906, "-1.3862943611198906"),
            (2.220446049250313e-16, "2.220446049250313e-16"),
            (100.0, "100.0"),
            (0.1, "0.1"),
            (4.5, "4.5"),
        ];
        for (value, want) in cases {
            assert_eq!(&py_repr(*value), want, "{value}");
            assert_eq!(py_repr(*value).parse::<f64>().unwrap().to_bits(), value.to_bits());
        }
    }

    #[test]
    fn documents_survive_a_round_trip() {
        let text = r#"{"a":[1,2.5,-3e-7],"b":{"c":"é😀","d":null},"e":true,"f":[]}"#;
        let parsed = parse(text).unwrap();
        assert_eq!(parsed.at("b").at("c").as_str(), Some("é😀"));
        assert_eq!(parsed.at("a").to_f64s(), vec![1.0, 2.5, -3e-7]);
        assert_eq!(parsed.at("a").as_array()[0], Json::Int(1));
        assert_eq!(parse(&parsed.render(0)).unwrap(), parsed);
        assert!(parse("{").is_err());
        assert!(parse("[1,2] junk").is_err());
    }

    #[test]
    fn missing_keys_read_as_nothing() {
        let doc = parse(r#"{"x":1}"#).unwrap();
        assert!(doc.at("nope").is_null());
        assert_eq!(doc.at("nope").to_i64s(), Vec::<i64>::new());
        assert_eq!(doc.at("x").as_i64(), Some(1));
    }
}
