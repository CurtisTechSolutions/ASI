//! A small JSON writer - enough to report a benchmark run.
//!
//! The model file format is not implemented here (the Python and Go
//! implementations own it); this is the one place the port has to speak JSON,
//! so it speaks just enough of it.

/// A JSON value.
#[derive(Clone, Debug)]
pub enum Json {
    Null,
    Bool(bool),
    Int(i64),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    /// An object, in insertion order.
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

    /// Renders the value; `indent` > 0 pretty-prints it.
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
            Json::Num(x) => {
                if x.is_finite() {
                    out.push_str(&format!("{x}"));
                } else {
                    out.push_str("null");
                }
            }
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

#[cfg(test)]
mod tests {
    use super::Json;

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
    }
}
