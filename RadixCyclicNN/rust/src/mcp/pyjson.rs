//! JSON read the way Python's `json.loads` reads it, refusals and all.
//!
//! [`crate::json::parse`] reads the model file, which only ever holds what a
//! writer put there.  The MCP stream and the problem and task files hold
//! whatever a person or a client typed, and what goes back when that is not
//! JSON is part of the contract: the MCP server answers a line it cannot read
//! with `invalid JSON: <Python's message>`, and a client (or a test that runs
//! both servers over one stream) cannot tell the two apart only if the message
//! is Python's to the character - `Expecting property name enclosed in double
//! quotes: line 1 column 2 (char 1)`.
//!
//! So this is CPython's scanner (`Modules/_json.c`) retold: the same grammar
//! (with `NaN`, `Infinity` and `-Infinity`, which Python accepts), the same
//! error for the same input at the same character position, and a repeated
//! key keeping its first place and its last value, as a `dict` does.

use crate::json::Json;

/// `json.loads(text)`: the value, or Python's `JSONDecodeError` message.
pub fn loads(text: &str) -> Result<Json, String> {
    let chars: Vec<char> = text.chars().collect();
    let scanner = Scanner { s: &chars };
    if chars.first() == Some(&'\u{feff}') {
        return Err(scanner.message("Unexpected UTF-8 BOM (decode using utf-8-sig)", 0));
    }
    let start = scanner.skip(0);
    let (value, end) = match scanner.value(start) {
        Ok(found) => found,
        Err(Stop::At(at)) => return Err(scanner.message("Expecting value", at)),
        Err(Stop::Error(message)) => return Err(message),
    };
    let end = scanner.skip(end);
    if end != chars.len() {
        return Err(scanner.message("Extra data", end));
    }
    Ok(value)
}

/// Why a scan stopped: nothing that is a value starts here (Python's
/// `StopIteration`, which becomes `Expecting value` at that place), or an
/// error with its message already written.
enum Stop {
    At(usize),
    Error(String),
}

struct Scanner<'a> {
    s: &'a [char],
}

impl Scanner<'_> {
    /// `msg: line L column C (char P)`, as `JSONDecodeError` words it.
    fn message(&self, msg: &str, pos: usize) -> String {
        let before = &self.s[..pos.min(self.s.len())];
        let line = before.iter().filter(|&&c| c == '\n').count() + 1;
        let column = match before.iter().rposition(|&c| c == '\n') {
            Some(newline) => pos - newline,
            None => pos + 1,
        };
        format!("{msg}: line {line} column {column} (char {pos})")
    }

    fn error(&self, msg: &str, pos: usize) -> Stop {
        Stop::Error(self.message(msg, pos))
    }

    /// Past the whitespace JSON allows (`[ \t\n\r]*`).
    fn skip(&self, mut at: usize) -> usize {
        while at < self.s.len() && matches!(self.s[at], ' ' | '\t' | '\n' | '\r') {
            at += 1;
        }
        at
    }

    fn starts(&self, at: usize, word: &str) -> bool {
        let mut i = at;
        for c in word.chars() {
            if self.s.get(i) != Some(&c) {
                return false;
            }
            i += 1;
        }
        true
    }

    /// One value from `at`: the value and where it ends.
    fn value(&self, at: usize) -> Result<(Json, usize), Stop> {
        let Some(&c) = self.s.get(at) else {
            return Err(Stop::At(at));
        };
        match c {
            '"' => {
                let (text, end) = self.string(at + 1)?;
                Ok((Json::Str(text), end))
            }
            '{' => self.object(at + 1),
            '[' => self.array(at + 1),
            'n' if self.starts(at, "null") => Ok((Json::Null, at + 4)),
            't' if self.starts(at, "true") => Ok((Json::Bool(true), at + 4)),
            'f' if self.starts(at, "false") => Ok((Json::Bool(false), at + 5)),
            'N' if self.starts(at, "NaN") => Ok((Json::Num(f64::NAN), at + 3)),
            'I' if self.starts(at, "Infinity") => Ok((Json::Num(f64::INFINITY), at + 8)),
            '-' if self.starts(at, "-Infinity") => Ok((Json::Num(f64::NEG_INFINITY), at + 9)),
            _ => self.number(at),
        }
    }

    /// `-?(0|[1-9]\d*)(\.\d+)?([eE][-+]?\d+)?`, else nothing starts here.
    fn number(&self, at: usize) -> Result<(Json, usize), Stop> {
        let digit = |i: usize| self.s.get(i).is_some_and(|c| c.is_ascii_digit());
        let mut i = at;
        if self.s.get(i) == Some(&'-') {
            i += 1;
        }
        match self.s.get(i) {
            Some('0') => i += 1,
            Some(c) if c.is_ascii_digit() => {
                while digit(i) {
                    i += 1;
                }
            }
            _ => return Err(Stop::At(at)),
        }
        let mut float = false;
        if self.s.get(i) == Some(&'.') && digit(i + 1) {
            float = true;
            i += 1;
            while digit(i) {
                i += 1;
            }
        }
        if matches!(self.s.get(i), Some('e') | Some('E')) {
            let mut j = i + 1;
            if matches!(self.s.get(j), Some('+') | Some('-')) {
                j += 1;
            }
            if digit(j) {
                while digit(j) {
                    j += 1;
                }
                float = true;
                i = j;
            }
        }
        let text: String = self.s[at..i].iter().collect();
        let value = if float {
            Json::Num(text.parse().unwrap_or(f64::NAN))
        } else {
            match text.parse::<i64>() {
                Ok(n) => Json::Int(n),
                // Python's int has no ceiling; the nearest double is what is left
                Err(_) => Json::Num(text.parse().unwrap_or(f64::NAN)),
            }
        };
        Ok((value, i))
    }

    /// The text of a string whose opening quote is just before `at`.
    fn string(&self, at: usize) -> Result<(String, usize), Stop> {
        let begin = at - 1;
        let mut out = String::new();
        let mut i = at;
        loop {
            let Some(&c) = self.s.get(i) else {
                return Err(self.error("Unterminated string starting at", begin));
            };
            match c {
                '"' => return Ok((out, i + 1)),
                '\\' => {
                    i += 1;
                    let Some(&kind) = self.s.get(i) else {
                        return Err(self.error("Unterminated string starting at", begin));
                    };
                    if kind == 'u' {
                        let unit = self.hex4(i)?;
                        i += 5;
                        if (0xd800..0xdc00).contains(&unit) && self.starts(i, "\\u") {
                            if let Ok(low) = self.hex4(i + 1) {
                                if (0xdc00..0xe000).contains(&low) {
                                    let code = 0x10000 + ((unit - 0xd800) << 10) + (low - 0xdc00);
                                    out.push(char::from_u32(code).unwrap_or('\u{fffd}'));
                                    i += 6;
                                    continue;
                                }
                            }
                        }
                        // a lone surrogate is a str in Python and cannot be one here
                        out.push(char::from_u32(unit).unwrap_or('\u{fffd}'));
                        continue;
                    }
                    let decoded = match kind {
                        '"' => '"',
                        '\\' => '\\',
                        '/' => '/',
                        'b' => '\u{8}',
                        'f' => '\u{c}',
                        'n' => '\n',
                        'r' => '\r',
                        't' => '\t',
                        _ => return Err(self.error("Invalid \\escape", i - 1)),
                    };
                    out.push(decoded);
                    i += 1;
                }
                c if (c as u32) < 0x20 => return Err(self.error("Invalid control character at", i)),
                c => {
                    out.push(c);
                    i += 1;
                }
            }
        }
    }

    /// The four hex digits after the `u` at `u`; Python also wants a character
    /// after them (the string cannot end there).
    fn hex4(&self, u: usize) -> Result<u32, Stop> {
        if u + 5 >= self.s.len() {
            return Err(self.error("Invalid \\uXXXX escape", u));
        }
        let digits: String = self.s[u + 1..u + 5].iter().collect();
        if !digits.chars().all(|c| c.is_ascii_hexdigit()) {
            return Err(self.error("Invalid \\uXXXX escape", u));
        }
        u32::from_str_radix(&digits, 16).map_err(|_| self.error("Invalid \\uXXXX escape", u))
    }

    fn object(&self, at: usize) -> Result<(Json, usize), Stop> {
        let mut pairs: Vec<(String, Json)> = Vec::new();
        let mut i = self.skip(at);
        if self.s.get(i) == Some(&'}') {
            return Ok((Json::Obj(pairs), i + 1));
        }
        loop {
            if self.s.get(i) != Some(&'"') {
                return Err(self.error("Expecting property name enclosed in double quotes", i));
            }
            let (key, end) = self.string(i + 1)?;
            i = self.skip(end);
            if self.s.get(i) != Some(&':') {
                return Err(self.error("Expecting ':' delimiter", i));
            }
            i = self.skip(i + 1);
            let (value, end) = self.value(i)?;
            // a dict: the first place, the last value
            match pairs.iter_mut().find(|(k, _)| *k == key) {
                Some(slot) => slot.1 = value,
                None => pairs.push((key, value)),
            }
            i = self.skip(end);
            match self.s.get(i) {
                Some('}') => return Ok((Json::Obj(pairs), i + 1)),
                Some(',') => i = self.skip(i + 1),
                _ => return Err(self.error("Expecting ',' delimiter", i)),
            }
        }
    }

    fn array(&self, at: usize) -> Result<(Json, usize), Stop> {
        let mut items = Vec::new();
        let mut i = self.skip(at);
        if self.s.get(i) == Some(&']') {
            return Ok((Json::Arr(items), i + 1));
        }
        loop {
            let (value, end) = self.value(i)?;
            items.push(value);
            i = self.skip(end);
            match self.s.get(i) {
                Some(']') => return Ok((Json::Arr(items), i + 1)),
                Some(',') => i = self.skip(i + 1),
                _ => return Err(self.error("Expecting ',' delimiter", i)),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_refusals_are_python_s_to_the_character() {
        // every expectation below is what CPython 3.11's json.loads raised for the input
        for (text, want) in [
            ("", "Expecting value: line 1 column 1 (char 0)"),
            ("not json at all", "Expecting value: line 1 column 1 (char 0)"),
            (
                "{not json}",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            ("{\"a\": }", "Expecting value: line 1 column 7 (char 6)"),
            ("{\"a\" 1}", "Expecting ':' delimiter: line 1 column 6 (char 5)"),
            (
                "{\"a\": 1,}",
                "Expecting property name enclosed in double quotes: line 1 column 9 (char 8)",
            ),
            (
                "{\"a\": 1 \"b\": 2}",
                "Expecting ',' delimiter: line 1 column 9 (char 8)",
            ),
            ("[1,]", "Expecting value: line 1 column 4 (char 3)"),
            ("[1 2]", "Expecting ',' delimiter: line 1 column 4 (char 3)"),
            ("\"abc", "Unterminated string starting at: line 1 column 1 (char 0)"),
            ("\"a\tb\"", "Invalid control character at: line 1 column 3 (char 2)"),
            ("\"a\\qb\"", "Invalid \\escape: line 1 column 3 (char 2)"),
            ("\"\\u12\"", "Invalid \\uXXXX escape: line 1 column 3 (char 2)"),
            ("1 2", "Extra data: line 1 column 3 (char 2)"),
            ("01", "Extra data: line 1 column 2 (char 1)"),
            ("-", "Expecting value: line 1 column 1 (char 0)"),
            ("tru", "Expecting value: line 1 column 1 (char 0)"),
            ("{\"a\":1}}", "Extra data: line 1 column 8 (char 7)"),
            ("  {\"a\"", "Expecting ':' delimiter: line 1 column 7 (char 6)"),
            (
                "{",
                "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)",
            ),
            ("[", "Expecting value: line 1 column 2 (char 1)"),
            ("{\"a\":[1,{\"b\":}]}", "Expecting value: line 1 column 14 (char 13)"),
            ("[\"x\" ,", "Expecting value: line 1 column 7 (char 6)"),
            ("[1,\n 2,\n ]", "Expecting value: line 3 column 2 (char 9)"),
        ] {
            assert_eq!(loads(text).unwrap_err(), want, "{text:?}");
        }
    }

    #[test]
    fn what_python_reads_is_read() {
        let doc = |t: &str| loads(t).unwrap().render(0);
        assert_eq!(doc("\"\\ud83d\\ude00\""), "\"😀\"");
        assert_eq!(doc("{\"a\":1,\"a\":2,\"b\":3}"), "{\"a\":2,\"b\":3}");
        assert_eq!(
            doc(" [1.0, 2, 1e5, -0.5, true, null] "),
            "[1.0,2,100000.0,-0.5,true,null]"
        );
        assert!(matches!(loads("NaN").unwrap(), Json::Num(x) if x.is_nan()));
        assert_eq!(loads("-Infinity").unwrap(), Json::Num(f64::NEG_INFINITY));
        assert_eq!(doc("\"tab\\tnew\\nline \\u00e9\""), "\"tab\\tnew\\nline é\"");
        assert_eq!(doc("12345678901234567"), "12345678901234567");
    }
}
