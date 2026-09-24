//! A request body read the way the Python server reads it (`api.Fields`):
//! every field typed, every problem a 400 that names the field.
//!
//! [`crate::http::Request`]'s own readers are forgiving - a missing number is
//! its default, a string where a list was wanted is split into lines.  The LLM
//! routes answer with the Python server's refusals instead (`'count' must be
//! >= 1 (got 0)`, `missing field 'prompt' (string)`), because a teaching loop
//! that starts with a nonsense setting spends a minute of somebody's LLM before
//! it fails.  A JSON `null` counts as "not given", as it does there.

use crate::http::{ApiError, Request};
use crate::json::Json;
use crate::negative::python_repr;

/// Typed access to a request's JSON body.
pub struct Fields<'a> {
    body: &'a Json,
}

impl<'a> Fields<'a> {
    /// The fields of a request.
    pub fn of(request: &'a Request) -> Fields<'a> {
        Fields { body: &request.body }
    }

    /// The fields of a document.
    pub fn new(body: &'a Json) -> Fields<'a> {
        Fields { body }
    }

    /// A field, `None` when it is absent or `null`.
    pub fn get(&self, name: &str) -> Option<&'a Json> {
        self.body.get(name).filter(|v| !v.is_null())
    }

    /// Whether a field was given.
    pub fn present(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// A string field, `None` when it was not given.
    pub fn text(&self, name: &str) -> Result<Option<String>, ApiError> {
        match self.get(name) {
            None => Ok(None),
            Some(Json::Str(text)) => Ok(Some(text.clone())),
            Some(other) => Err(bad(name, "a string", other)),
        }
    }

    /// A string field, or `default`.
    pub fn text_or(&self, name: &str, default: &str) -> Result<String, ApiError> {
        Ok(self.text(name)?.unwrap_or_else(|| default.to_string()))
    }

    /// A string field that must be given.
    pub fn required_text(&self, name: &str) -> Result<String, ApiError> {
        self.text(name)?
            .ok_or_else(|| ApiError::bad_request(format!("missing field '{name}' (string)")))
    }

    /// A boolean field, or `default`.
    pub fn flag(&self, name: &str, default: bool) -> Result<bool, ApiError> {
        match self.get(name) {
            None => Ok(default),
            Some(Json::Bool(b)) => Ok(*b),
            Some(other) => Err(bad(name, "a boolean", other)),
        }
    }

    /// A whole-number field (`3.0` counts), `None` when it was not given.
    pub fn integer(&self, name: &str, minimum: Option<i64>) -> Result<Option<i64>, ApiError> {
        let value = match self.get(name) {
            None => return Ok(None),
            Some(Json::Int(n)) => *n,
            Some(Json::Num(x)) if x.is_finite() && x.fract() == 0.0 => *x as i64,
            Some(other) => return Err(bad(name, "an integer", other)),
        };
        if let Some(minimum) = minimum {
            if value < minimum {
                return Err(ApiError::bad_request(format!(
                    "'{name}' must be >= {minimum} (got {value})"
                )));
            }
        }
        Ok(Some(value))
    }

    /// A whole-number field, or `default`.
    pub fn integer_or(&self, name: &str, default: i64, minimum: Option<i64>) -> Result<i64, ApiError> {
        Ok(self.integer(name, minimum)?.unwrap_or(default))
    }

    /// A count: a whole number that is at least `minimum` (0 or more), or `default`.
    pub fn count_or(&self, name: &str, default: usize, minimum: usize) -> Result<usize, ApiError> {
        Ok(self.integer_or(name, default as i64, Some(minimum as i64))? as usize)
    }

    /// A finite number field, `None` when it was not given.
    pub fn number(&self, name: &str, minimum: Option<f64>) -> Result<Option<f64>, ApiError> {
        let value = match self.get(name) {
            None => return Ok(None),
            Some(Json::Int(n)) => *n as f64,
            Some(Json::Num(x)) if x.is_finite() => *x,
            Some(other) => return Err(bad(name, "a finite number", other)),
        };
        if let Some(minimum) = minimum {
            if value < minimum {
                return Err(ApiError::bad_request(format!(
                    "'{name}' must be >= {} (got {})",
                    py_number(minimum),
                    py_number_of(self.get(name))
                )));
            }
        }
        Ok(Some(value))
    }

    /// A finite number field, or `default`.
    pub fn number_or(&self, name: &str, default: f64, minimum: Option<f64>) -> Result<f64, ApiError> {
        Ok(self.number(name, minimum)?.unwrap_or(default))
    }

    /// Texts from `list` (a list of strings) or `text` (one text per line,
    /// blank lines dropped); `[]` when neither was given or they hold nothing.
    pub fn texts_optional(&self, list: &str, text: &str) -> Result<Vec<String>, ApiError> {
        if let Some(value) = self.get(list) {
            let items = match value {
                Json::Arr(items) if items.iter().all(|t| matches!(t, Json::Str(_))) => items,
                other => return Err(bad(list, "a list of strings", other)),
            };
            return Ok(items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect());
        }
        match self.get(text) {
            None => Ok(Vec::new()),
            Some(Json::Str(blob)) => Ok(crate::llm::splitlines(blob)
                .into_iter()
                .filter(|line| !line.trim().is_empty())
                .map(str::to_string)
                .collect()),
            Some(other) => Err(bad(text, "a string", other)),
        }
    }

    /// Upload names from a field that is a list of strings (or one string).
    pub fn names(&self, name: &str) -> Result<Vec<String>, ApiError> {
        match self.get(name) {
            None => Ok(Vec::new()),
            Some(Json::Str(one)) => Ok(vec![one.clone()]),
            Some(Json::Arr(items)) if items.iter().all(|t| matches!(t, Json::Str(_))) => {
                Ok(items.iter().filter_map(|t| t.as_str().map(str::to_string)).collect())
            }
            Some(other) => Err(bad(name, "a list of upload names", other)),
        }
    }
}

/// `'name' must be <expected> (got <type> <repr>)`, the Python server's words.
fn bad(name: &str, expected: &str, value: &Json) -> ApiError {
    ApiError::bad_request(format!(
        "'{name}' must be {expected} (got {} {})",
        json_type(value),
        py_repr_of(value)
    ))
}

/// The JSON type of a value, as the refusals name it.
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

/// A number as Python's `str()` writes it: an int as an int, a float with its point.
fn py_number(x: f64) -> String {
    crate::json::py_repr(x)
}

fn py_number_of(value: Option<&Json>) -> String {
    match value {
        Some(Json::Int(n)) => n.to_string(),
        Some(Json::Num(x)) => py_number(*x),
        _ => String::new(),
    }
}

/// A JSON value as Python's `repr()` writes the object it parses into.
pub fn py_repr_of(value: &Json) -> String {
    match value {
        Json::Null => "None".to_string(),
        Json::Bool(true) => "True".to_string(),
        Json::Bool(false) => "False".to_string(),
        Json::Int(n) => n.to_string(),
        Json::Num(x) if x.is_nan() => "nan".to_string(),
        Json::Num(x) if x.is_infinite() => if *x > 0.0 { "inf" } else { "-inf" }.to_string(),
        Json::Num(x) => crate::json::py_repr(*x),
        Json::Str(s) => python_repr(s),
        Json::Arr(items) => format!("[{}]", items.iter().map(py_repr_of).collect::<Vec<_>>().join(", ")),
        Json::Obj(pairs) => format!(
            "{{{}}}",
            pairs
                .iter()
                .map(|(k, v)| format!("{}: {}", python_repr(k), py_repr_of(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// A JSON value as Python's `str()` writes it: a string as itself, anything
/// else as its `repr()`.
pub fn py_str_of(value: &Json) -> String {
    match value {
        Json::Str(s) => s.clone(),
        other => py_repr_of(other),
    }
}

/// Python's truthiness of the object a JSON value parses into.
pub fn truthy(value: &Json) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;

    #[test]
    fn fields_are_typed_and_refusals_name_them() {
        let body = parse(
            "{\"prompt\": \"sea\", \"count\": 0, \"lines\": 3.0, \"t\": 1.5, \"flag\": \"yes\", \"none\": null, \
             \"texts\": [\"a\", \"b\"], \"text\": \"x\\n\\n y\", \"bad\": [1]}",
        )
        .unwrap();
        let f = Fields::new(&body);
        assert_eq!(f.required_text("prompt").unwrap(), "sea");
        assert_eq!(
            f.required_text("none").unwrap_err().message,
            "missing field 'none' (string)"
        );
        assert_eq!(
            f.count_or("count", 8, 1).unwrap_err().message,
            "'count' must be >= 1 (got 0)"
        );
        assert_eq!(f.count_or("lines", 20, 1).unwrap(), 3);
        assert_eq!(f.count_or("absent", 20, 1).unwrap(), 20);
        assert_eq!(f.number_or("t", 6.0, Some(0.0)).unwrap(), 1.5);
        assert_eq!(
            f.number("t", Some(2.0)).unwrap_err().message,
            "'t' must be >= 2.0 (got 1.5)"
        );
        assert_eq!(
            f.flag("flag", false).unwrap_err().message,
            "'flag' must be a boolean (got string 'yes')"
        );
        assert_eq!(f.texts_optional("texts", "text").unwrap(), vec!["a", "b"]);
        assert_eq!(f.texts_optional("nope", "text").unwrap(), vec!["x", " y"]);
        assert!(f.texts_optional("nope", "nothing").unwrap().is_empty());
        assert_eq!(
            f.texts_optional("bad", "text").unwrap_err().message,
            "'bad' must be a list of strings (got array [1])"
        );
        assert!(!f.present("none"));
    }

    #[test]
    fn values_print_as_python_prints_them() {
        let doc = parse("{\"a\": [1, 2.5, true, null, \"it's\"]}").unwrap();
        assert_eq!(py_repr_of(&doc), "{'a': [1, 2.5, True, None, \"it's\"]}");
        assert_eq!(py_str_of(&Json::str("plain")), "plain");
        assert!(!truthy(&Json::str("")));
        assert!(truthy(&Json::Int(3)));
    }
}
