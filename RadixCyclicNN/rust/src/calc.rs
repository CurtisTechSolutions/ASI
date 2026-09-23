//! The calculator tool's expression language (`radixnet/tools.py`
//! `safe_eval`, `go/radixnet/calc.go`).
//!
//! Python evaluates an expression in three steps: `ast.parse` it, walk the
//! tree breadth first against a whitelist of node types, and hand whatever
//! passed to `eval`.  There is no `eval` here, so the three steps are written
//! out, and each one is kept separate *because the order is the contract*:
//!
//! * a **tokenizer and a recursive-descent parser** over the part of Python's
//!   expression grammar a calculator can meet - including the parts it must
//!   refuse (`lambda`, `a if b else c`, `x.attr`, `x[0]`, `{}`, ...), since an
//!   expression Python *parses* and then refuses gets a different message from
//!   one it cannot parse at all;
//! * the **same breadth-first walk** over the same whitelist, which is what
//!   decides *which* complaint a bad expression gets: Python reports the first
//!   node `ast.walk` reaches, so `'a' * open(1)` is "only numbers" and not
//!   "unknown function".  A refused node is kept as an opaque leaf: its
//!   children are deeper than it is, so the walk can never reach them first;
//! * an **evaluator with Python's arithmetic**: `/` is true division, `//`
//!   floors, `%` takes the sign of the divisor, int op int stays an int, `and`
//!   / `or` return an operand and short-circuit (`1 or 1/0` is `1`), a tuple or
//!   a list is a value in its own right (`[1] + [2]` is `[1, 2]`), a negative
//!   number to a fractional power is complex, and every result is rendered the
//!   way `str()` renders it.  The error texts are Python's, word for word, for
//!   everything the tests and the network meet.
//!
//! The Go port parses and evaluates in one pass and flattens sequences into
//! numbers, which is simpler and disagrees with Python at the edges; this one
//! was checked against Python expression by expression.
//!
//! Two differences remain, both deliberate:
//!
//! * integers are `i128`, not Python's unbounded integers: `2 ** 126` is exact,
//!   `10 ** 40` is an error rather than forty-one digits;
//! * `math.hypot` over three or more values is a scaled square root rather
//!   than CPython's compensated one, and can differ from it in the last digit.

use std::cmp::Ordering;
use std::collections::VecDeque;

use crate::json::py_repr;
use crate::negative::python_repr;

/// The longest expression the calculator reads, in characters.
pub const MAX_EXPRESSION_CHARS: usize = 500;

/// How deeply brackets may nest - CPython's own tokenizer limit.
const MAX_NESTING: usize = 200;

/// The longest tuple or list an expression may build (`[0] * 10 ** 9` is a
/// memory bomb, not arithmetic).
const MAX_SEQUENCE: usize = 100_000;

/// What an expression's integers stop at.
const TOO_BIG: &str = "integer overflow: this calculator's integers stop at 2**127";

/// The functions an expression may call, sorted as Python lists them.
pub const FUNCTIONS: [&str; 21] = [
    "abs", "atan", "ceil", "cos", "exp", "fabs", "float", "floor", "hypot", "int", "log", "log10", "log2", "max",
    "min", "pow", "round", "sin", "sqrt", "sum", "tan",
];

/// The constants an expression may use.
pub const NAMES: [(&str, f64); 4] = [
    ("pi", std::f64::consts::PI),
    ("e", std::f64::consts::E),
    ("tau", std::f64::consts::TAU),
    ("inf", f64::INFINITY),
];

/// Python's keywords: a name that is one of these is grammar, not a value.
const KEYWORDS: [&str; 35] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
    "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
];

/// Evaluates an arithmetic expression - numbers, operators and the math
/// functions, nothing else - and renders the result as Python's `str()` would.
///
/// The error is the message the calculator tool reports, as Python words it.
pub fn safe_eval(expression: &str) -> Result<String, String> {
    evaluate(expression).map(|value| value.to_string())
}

/// Evaluates an expression to a [`Value`].
///
/// The parse and the evaluation run on a thread of their own with a generous
/// stack: 200 nested brackets are legal, and a recursive-descent parser in a
/// debug build needs more than a server thread's default stack to go that deep.
pub fn evaluate(expression: &str) -> Result<Value, String> {
    let text = expression.trim();
    if text.is_empty() {
        return Err("the expression is empty".to_string());
    }
    if text.chars().count() > MAX_EXPRESSION_CHARS {
        return Err(format!(
            "the expression is too long ({MAX_EXPRESSION_CHARS} characters max)"
        ));
    }
    let source = text.replace('^', "**");
    let run = || evaluate_source(&source);
    std::thread::scope(|scope| {
        match std::thread::Builder::new()
            .name("calculator".to_string())
            .stack_size(32 << 20)
            .spawn_scoped(scope, run)
        {
            Ok(handle) => handle
                .join()
                .unwrap_or_else(|_| Err("cannot evaluate the expression: the evaluator failed".to_string())),
            // no thread to be had: the caller's stack will have to do
            Err(_) => run(),
        }
    })
}

fn evaluate_source(source: &str) -> Result<Value, String> {
    let syntax = |message: String| format!("not an arithmetic expression: {message}");
    let tokens = tokenize(source).map_err(syntax)?;
    let tree = Parser::new(tokens).parse().map_err(syntax)?;
    validate(&tree)?;
    eval(&tree).map_err(|fail| match fail {
        Fail::Zero => "division by zero".to_string(),
        Fail::Eval(message) => format!("cannot evaluate the expression: {message}"),
    })
}

// -- values -----------------------------------------------------------------------------------------

/// A Python value an expression can produce.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Int(i128),
    Bool(bool),
    Float(f64),
    /// `(real, imag)`: what a negative number to a fractional power is.
    Complex(f64, f64),
    List(Vec<Value>),
    Tuple(Vec<Value>),
    /// A function named as a value (`sqrt`), which Python allows and prints.
    Func(&'static str),
}

impl Value {
    /// Python's name for the value's type, as its error messages use it.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Int(_) => "int",
            Value::Bool(_) => "bool",
            Value::Float(_) => "float",
            Value::Complex(..) => "complex",
            Value::List(_) => "list",
            Value::Tuple(_) => "tuple",
            Value::Func("int") | Value::Func("float") => "type",
            Value::Func(_) => "builtin_function_or_method",
        }
    }

    /// Python's truth value.
    pub fn truthy(&self) -> bool {
        match self {
            Value::Int(n) => *n != 0,
            Value::Bool(b) => *b,
            Value::Float(x) => *x != 0.0,
            Value::Complex(re, im) => *re != 0.0 || *im != 0.0,
            Value::List(items) | Value::Tuple(items) => !items.is_empty(),
            Value::Func(_) => true,
        }
    }

    fn num(&self) -> Option<Num> {
        match self {
            Value::Int(n) => Some(Num::I(*n)),
            Value::Bool(b) => Some(Num::I(*b as i128)),
            Value::Float(x) => Some(Num::F(*x)),
            Value::Complex(re, im) => Some(Num::C(*re, *im)),
            _ => None,
        }
    }
}

impl std::fmt::Display for Value {
    /// `str(value)`, as Python writes it.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{n}"),
            Value::Bool(b) => f.write_str(if *b { "True" } else { "False" }),
            Value::Float(x) => f.write_str(&float_repr(*x)),
            Value::Complex(re, im) => f.write_str(&complex_repr(*re, *im)),
            Value::List(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                write!(f, "[{}]", inner.join(", "))
            }
            Value::Tuple(items) => {
                let inner: Vec<String> = items.iter().map(|v| v.to_string()).collect();
                if items.len() == 1 {
                    write!(f, "({},)", inner[0])
                } else {
                    write!(f, "({})", inner.join(", "))
                }
            }
            Value::Func(name @ ("int" | "float")) => write!(f, "<class '{name}'>"),
            Value::Func(name) => write!(f, "<built-in function {name}>"),
        }
    }
}

/// A float as Python's `repr()` writes it (`inf`, `nan`, `1e+16`, `0.1`).
pub fn float_repr(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        py_repr(x)
    }
}

/// One part of a complex number: `repr()` without the forced `.0`.
fn complex_part(x: f64) -> String {
    let text = float_repr(x);
    match text.strip_suffix(".0") {
        Some(whole) if !text.contains('e') => whole.to_string(),
        _ => text,
    }
}

/// A complex number as Python writes it: `2j`, `(1+2j)`, `(8.6e-17+1.4j)`.
fn complex_repr(re: f64, im: f64) -> String {
    let imag = complex_part(im);
    if re == 0.0 && re.is_sign_positive() {
        return format!("{imag}j");
    }
    let sign = if imag.starts_with('-') { "" } else { "+" };
    format!("({}{sign}{imag}j)", complex_part(re))
}

/// Why an evaluation stopped: a division by zero (Python's
/// `ZeroDivisionError`, which the tool words on its own), or anything else.
#[derive(Debug)]
enum Fail {
    Zero,
    Eval(String),
}

fn fail<T, S: Into<String>>(message: S) -> Result<T, Fail> {
    Err(Fail::Eval(message.into()))
}

type Eval<T> = Result<T, Fail>;

// -- the tokenizer ----------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    /// `None` when the literal does not fit this calculator's integers.
    Int(Option<i128>),
    Float(f64),
    /// `2j`: a complex literal, which the whitelist refuses as a constant.
    Imag,
    Str(StrKind),
    Name(String),
    Op(&'static str),
    End,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum StrKind {
    Text,
    Bytes,
    Formatted,
}

/// Every operator and delimiter, longest first so `**` wins over `*`.
const OPS: [&str; 47] = [
    "**=", "//=", ">>=", "<<=", "...", "**", "//", "<<", ">>", "<=", ">=", "==", "!=", ":=", "->", "+=", "-=", "*=",
    "/=", "%=", "&=", "|=", "^=", "@=", "+", "-", "*", "/", "%", "@", "&", "|", "^", "~", "<", ">", "(", ")", "[", "]",
    "{", "}", ",", ":", ".", ";", "=",
];

fn is_name_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_name_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Splits the text into tokens, checking the brackets as CPython's tokenizer
/// does - which is why an unclosed bracket is reported before anything the
/// parser would have said.
fn tokenize(text: &str) -> Result<Vec<Tok>, String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut out = Vec::new();
    let mut stack: Vec<char> = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if c == '#' {
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if c == '\\' {
            if i + 1 < n && (chars[i + 1] == '\n' || chars[i + 1] == '\r') {
                i += 2;
                continue;
            }
            return Err("unexpected character after line continuation character".to_string());
        }
        if c == '\n' || c == '\r' {
            // a new line inside brackets is joined; outside them it ends the
            // expression, and anything after it is a second statement
            if stack.is_empty() && chars[i..].iter().any(|ch| !ch.is_whitespace()) {
                let rest: String = chars[i..].iter().collect();
                let rest = rest.trim_start();
                if !rest.starts_with('#') {
                    return Err("invalid syntax".to_string());
                }
            }
            i += 1;
            continue;
        }
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if c.is_ascii_digit() || (c == '.' && chars.get(i + 1).is_some_and(|d| d.is_ascii_digit())) {
            let (tok, next) = number(&chars, i)?;
            out.push(tok);
            i = next;
            continue;
        }
        if c == '\'' || c == '"' {
            let (tok, next) = string(&chars, i, "")?;
            out.push(tok);
            i = next;
            continue;
        }
        if is_name_start(c) {
            let start = i;
            while i < n && is_name_char(chars[i]) {
                i += 1;
            }
            let name: String = chars[start..i].iter().collect();
            let prefix = name.to_ascii_lowercase();
            if i < n
                && (chars[i] == '\'' || chars[i] == '"')
                && matches!(prefix.as_str(), "r" | "u" | "b" | "f" | "br" | "rb" | "fr" | "rf")
            {
                let (tok, next) = string(&chars, i, &prefix)?;
                out.push(tok);
                i = next;
                continue;
            }
            out.push(Tok::Name(name));
            continue;
        }
        let rest: String = chars[i..chars.len().min(i + 3)].iter().collect();
        if let Some(op) = OPS.iter().find(|op| rest.starts_with(**op)) {
            match *op {
                "(" | "[" | "{" => {
                    if stack.len() >= MAX_NESTING {
                        return Err("too many nested parentheses".to_string());
                    }
                    stack.push(c);
                }
                ")" | "]" | "}" => match stack.pop() {
                    None => return Err(format!("unmatched '{c}'")),
                    Some(open) if closer(open) != c => {
                        return Err(format!(
                            "closing parenthesis '{c}' does not match opening parenthesis '{open}'"
                        ))
                    }
                    Some(_) => {}
                },
                _ => {}
            }
            out.push(Tok::Op(op));
            i += op.chars().count();
            continue;
        }
        if c.is_ascii() {
            return Err("invalid syntax".to_string());
        }
        return Err(format!("invalid character '{c}' (U+{:04X})", c as u32));
    }
    if let Some(open) = stack.last() {
        return Err(format!("'{open}' was never closed"));
    }
    out.push(Tok::End);
    Ok(out)
}

fn closer(open: char) -> char {
    match open {
        '(' => ')',
        '[' => ']',
        _ => '}',
    }
}

/// Digits of a radix, with Python's rule for underscores: one at a time, and
/// only between digits (or right after the `0x` prefix).
fn digits(chars: &[char], mut i: usize, ok: impl Fn(char) -> bool, leading_underscore: bool) -> (usize, bool) {
    let start = i;
    let mut bad = false;
    while i < chars.len() {
        let c = chars[i];
        if ok(c) {
            i += 1;
        } else if c == '_' {
            let after = chars.get(i + 1).copied();
            if (i == start && !leading_underscore) || !after.is_some_and(&ok) {
                bad = true;
                break;
            }
            i += 1;
        } else {
            break;
        }
    }
    (i, bad)
}

/// A numeric literal starting at `i`: the token and where it ends.
fn number(chars: &[char], start: usize) -> Result<(Tok, usize), String> {
    let n = chars.len();
    let name_follows = |at: usize| at < n && is_name_char(chars[at]);
    if chars[start] == '0' && start + 1 < n && matches!(chars[start + 1], 'x' | 'X' | 'o' | 'O' | 'b' | 'B') {
        let (radix, kind) = match chars[start + 1].to_ascii_lowercase() {
            'x' => (16, "hexadecimal"),
            'o' => (8, "octal"),
            _ => (2, "binary"),
        };
        let (end, bad) = digits(chars, start + 2, |c| c.is_digit(radix), true);
        let text: String = chars[start + 2..end].iter().filter(|c| **c != '_').collect();
        if bad || text.is_empty() {
            return Err(format!("invalid {kind} literal"));
        }
        if end < n && chars[end].is_ascii_digit() {
            return Err(format!("invalid digit '{}' in {kind} literal", chars[end]));
        }
        if name_follows(end) {
            return Err(format!("invalid {kind} literal"));
        }
        return Ok((Tok::Int(i128::from_str_radix(&text, radix).ok()), end));
    }
    let decimal = |c: char| c.is_ascii_digit();
    let invalid = || "invalid decimal literal".to_string();
    let (mut i, bad) = digits(chars, start, decimal, false);
    if bad {
        return Err(invalid());
    }
    let mut float = false;
    if i < n && chars[i] == '.' {
        float = true;
        let (end, bad) = digits(chars, i + 1, decimal, false);
        if bad {
            return Err(invalid());
        }
        i = end;
    }
    if i < n && matches!(chars[i], 'e' | 'E') {
        let mut at = i + 1;
        if at < n && matches!(chars[at], '+' | '-') {
            at += 1;
        }
        if !(at < n && chars[at].is_ascii_digit()) {
            return Err(invalid());
        }
        let (end, bad) = digits(chars, at, decimal, false);
        if bad {
            return Err(invalid());
        }
        float = true;
        i = end;
    }
    if i < n && matches!(chars[i], 'j' | 'J') {
        if name_follows(i + 1) {
            return Err(invalid());
        }
        return Ok((Tok::Imag, i + 1));
    }
    if name_follows(i) {
        return Err(invalid());
    }
    let text: String = chars[start..i].iter().filter(|c| **c != '_').collect();
    if float {
        let value: f64 = text.parse().map_err(|_| invalid())?;
        return Ok((Tok::Float(value), i));
    }
    if text.len() > 1 && text.starts_with('0') && text.chars().any(|c| c != '0') {
        return Err(
            "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers"
                .to_string(),
        );
    }
    Ok((Tok::Int(text.parse::<i128>().ok()), i))
}

/// A string literal (with its prefix already read) starting at the quote.
fn string(chars: &[char], start: usize, prefix: &str) -> Result<(Tok, usize), String> {
    let quote = chars[start];
    let n = chars.len();
    let triple = start + 2 < n && chars[start + 1] == quote && chars[start + 2] == quote;
    let mut i = start + if triple { 3 } else { 1 };
    loop {
        if i >= n {
            return Err(if triple {
                "unterminated triple-quoted string literal (detected at line 1)".to_string()
            } else {
                "unterminated string literal (detected at line 1)".to_string()
            });
        }
        let c = chars[i];
        if c == '\\' {
            i += 2;
            continue;
        }
        if !triple && c == '\n' {
            return Err("unterminated string literal (detected at line 1)".to_string());
        }
        if c == quote && (!triple || (i + 2 < n && chars[i + 1] == quote && chars[i + 2] == quote)) {
            i += if triple { 3 } else { 1 };
            break;
        }
        i += 1;
    }
    let kind = if prefix.contains('f') {
        StrKind::Formatted
    } else if prefix.contains('b') {
        StrKind::Bytes
    } else {
        StrKind::Text
    };
    Ok((Tok::Str(kind), i))
}

// -- the tree ---------------------------------------------------------------------------------------

/// An operator node, named as Python's `ast` names it.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Op {
    Add,
    Sub,
    Mult,
    Div,
    FloorDiv,
    Mod,
    Pow,
    MatMult,
    BitOr,
    BitXor,
    BitAnd,
    LShift,
    RShift,
    USub,
    UAdd,
    Not,
    Invert,
    And,
    Or,
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    In,
    NotIn,
    Is,
    IsNot,
}

impl Op {
    fn name(self) -> &'static str {
        match self {
            Op::Add => "Add",
            Op::Sub => "Sub",
            Op::Mult => "Mult",
            Op::Div => "Div",
            Op::FloorDiv => "FloorDiv",
            Op::Mod => "Mod",
            Op::Pow => "Pow",
            Op::MatMult => "MatMult",
            Op::BitOr => "BitOr",
            Op::BitXor => "BitXor",
            Op::BitAnd => "BitAnd",
            Op::LShift => "LShift",
            Op::RShift => "RShift",
            Op::USub => "USub",
            Op::UAdd => "UAdd",
            Op::Not => "Not",
            Op::Invert => "Invert",
            Op::And => "And",
            Op::Or => "Or",
            Op::Eq => "Eq",
            Op::NotEq => "NotEq",
            Op::Lt => "Lt",
            Op::LtE => "LtE",
            Op::Gt => "Gt",
            Op::GtE => "GtE",
            Op::In => "In",
            Op::NotIn => "NotIn",
            Op::Is => "Is",
            Op::IsNot => "IsNot",
        }
    }

    /// Whether Python's whitelist lets the operator through.
    fn allowed(self) -> bool {
        !matches!(
            self,
            Op::MatMult
                | Op::BitOr
                | Op::BitXor
                | Op::BitAnd
                | Op::LShift
                | Op::RShift
                | Op::Invert
                | Op::In
                | Op::NotIn
                | Op::Is
                | Op::IsNot
        )
    }

    /// The symbol Python's `TypeError` messages use.
    fn symbol(self) -> &'static str {
        match self {
            Op::Add => "+",
            Op::Sub => "-",
            Op::Mult => "*",
            Op::Div => "/",
            Op::FloorDiv => "//",
            Op::Mod => "%",
            Op::Pow => "** or pow()",
            Op::Lt => "<",
            Op::LtE => "<=",
            Op::Gt => ">",
            Op::GtE => ">=",
            _ => "?",
        }
    }
}

#[derive(Debug)]
enum Const {
    Int(Option<i128>),
    Float(f64),
    Bool(bool),
    /// A string, bytes, `None`, `...` or a complex literal: parsed, then refused.
    Other,
}

#[derive(Debug)]
enum Node {
    Const(Const),
    Name(String),
    Tuple(Vec<Node>),
    List(Vec<Node>),
    Bin(Box<Node>, Op, Box<Node>),
    Unary(Op, Box<Node>),
    Bool(Op, Vec<Node>),
    Compare(Box<Node>, Vec<(Op, Node)>),
    /// The function, the positional arguments and the keyword arguments.
    Call(Box<Node>, Vec<Node>, Vec<Node>),
    /// A node type the whitelist refuses, by Python's name for it.  Kept
    /// opaque: its children are deeper than it is, so the walk never reaches
    /// them before it.
    Banned(&'static str),
}

impl Node {
    /// `type(node).__name__`, which is what "unknown function" names when the
    /// thing called is not a name.
    fn type_name(&self) -> &'static str {
        match self {
            Node::Const(_) => "Constant",
            Node::Name(_) => "Name",
            Node::Tuple(_) => "Tuple",
            Node::List(_) => "List",
            Node::Bin(..) => "BinOp",
            Node::Unary(..) => "UnaryOp",
            Node::Bool(..) => "BoolOp",
            Node::Compare(..) => "Compare",
            Node::Call(..) => "Call",
            Node::Banned(name) => name,
        }
    }
}

// -- the parser -------------------------------------------------------------------------------------

/// The binary operators by precedence, loosest first.
const LEVELS: [&[(&str, Op)]; 6] = [
    &[("|", Op::BitOr)],
    &[("^", Op::BitXor)],
    &[("&", Op::BitAnd)],
    &[("<<", Op::LShift), (">>", Op::RShift)],
    &[("+", Op::Add), ("-", Op::Sub)],
    &[
        ("*", Op::Mult),
        ("/", Op::Div),
        ("//", Op::FloorDiv),
        ("%", Op::Mod),
        ("@", Op::MatMult),
    ],
];

struct Parser {
    toks: Vec<Tok>,
    at: usize,
    /// How many brackets the parser is inside: two expressions side by side in
    /// a bracket is Python's "Perhaps you forgot a comma?".
    depth: usize,
}

type Parsed = Result<Node, String>;

fn invalid() -> String {
    "invalid syntax".to_string()
}

impl Parser {
    fn new(toks: Vec<Tok>) -> Parser {
        Parser { toks, at: 0, depth: 0 }
    }

    fn peek(&self) -> &Tok {
        &self.toks[self.at.min(self.toks.len() - 1)]
    }

    fn peek_at(&self, ahead: usize) -> &Tok {
        &self.toks[(self.at + ahead).min(self.toks.len() - 1)]
    }

    fn advance(&mut self) -> Tok {
        let tok = self.peek().clone();
        if self.at < self.toks.len() - 1 {
            self.at += 1;
        }
        tok
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Tok::Op(o) if *o == op)
    }

    fn is_kw(&self, word: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == word)
    }

    fn accept(&mut self, op: &str) -> bool {
        if self.is_op(op) {
            self.advance();
            return true;
        }
        false
    }

    fn accept_kw(&mut self, word: &str) -> bool {
        if self.is_kw(word) {
            self.advance();
            return true;
        }
        false
    }

    /// Whether the next token could start an expression.
    fn starts_expression(&self) -> bool {
        match self.peek() {
            Tok::Int(_) | Tok::Float(_) | Tok::Imag | Tok::Str(_) => true,
            Tok::Name(name) => {
                !KEYWORDS.contains(&name.as_str())
                    || matches!(name.as_str(), "True" | "False" | "None" | "not" | "lambda" | "await")
            }
            Tok::Op(op) => matches!(*op, "(" | "[" | "{" | "-" | "+" | "~" | "..."),
            Tok::End => false,
        }
    }

    /// What a token that does not fit says.
    fn unexpected(&self) -> String {
        if self.depth > 0 && self.starts_expression() {
            "invalid syntax. Perhaps you forgot a comma?".to_string()
        } else {
            invalid()
        }
    }

    fn parse(mut self) -> Parsed {
        let tree = self.expressions()?;
        if *self.peek() != Tok::End {
            return Err(self.unexpected());
        }
        Ok(tree)
    }

    /// `a, b, c` without brackets is a tuple.
    fn expressions(&mut self) -> Parsed {
        let first = self.expression()?;
        if !self.is_op(",") {
            return Ok(first);
        }
        let mut items = vec![first];
        while self.accept(",") {
            if !self.starts_expression() {
                break;
            }
            items.push(self.expression()?);
        }
        Ok(Node::Tuple(items))
    }

    /// Skips a balanced run up to the token that closes the current bracket
    /// (the tokenizer has already checked that the brackets balance).
    fn skip_to_close(&mut self) {
        let mut level = 0usize;
        loop {
            match self.peek() {
                Tok::End => return,
                Tok::Op("(" | "[" | "{") => level += 1,
                Tok::Op(")" | "]" | "}") => {
                    if level == 0 {
                        return;
                    }
                    level -= 1;
                }
                _ => {}
            }
            self.advance();
        }
    }

    fn close(&mut self, op: &str) -> Result<(), String> {
        if self.accept(op) {
            self.depth -= 1;
            return Ok(());
        }
        Err(self.unexpected())
    }

    fn expression(&mut self) -> Parsed {
        if self.accept_kw("lambda") {
            // the parameters never matter: the whitelist refuses the lambda first
            let mut level = 0usize;
            loop {
                match self.peek() {
                    Tok::End => return Err(invalid()),
                    Tok::Op("(" | "[" | "{") => level += 1,
                    Tok::Op(")" | "]" | "}") => {
                        if level == 0 {
                            return Err(invalid());
                        }
                        level -= 1;
                    }
                    Tok::Op(":") if level == 0 => break,
                    _ => {}
                }
                self.advance();
            }
            self.advance();
            self.expression()?;
            return Ok(Node::Banned("Lambda"));
        }
        let body = self.disjunction()?;
        if self.accept_kw("if") {
            self.disjunction()?;
            if !self.accept_kw("else") {
                return Err("expected 'else' after 'if' expression".to_string());
            }
            self.expression()?;
            return Ok(Node::Banned("IfExp"));
        }
        Ok(body)
    }

    fn disjunction(&mut self) -> Parsed {
        let first = self.conjunction()?;
        if !self.is_kw("or") {
            return Ok(first);
        }
        let mut values = vec![first];
        while self.accept_kw("or") {
            values.push(self.conjunction()?);
        }
        Ok(Node::Bool(Op::Or, values))
    }

    fn conjunction(&mut self) -> Parsed {
        let first = self.inversion()?;
        if !self.is_kw("and") {
            return Ok(first);
        }
        let mut values = vec![first];
        while self.accept_kw("and") {
            values.push(self.inversion()?);
        }
        Ok(Node::Bool(Op::And, values))
    }

    fn inversion(&mut self) -> Parsed {
        if self.accept_kw("not") {
            return Ok(Node::Unary(Op::Not, Box::new(self.inversion()?)));
        }
        self.comparison()
    }

    fn comparison(&mut self) -> Parsed {
        let left = self.binary(0)?;
        let mut pairs = Vec::new();
        loop {
            let op = match self.peek() {
                Tok::Op("==") => Op::Eq,
                Tok::Op("!=") => Op::NotEq,
                Tok::Op("<") => Op::Lt,
                Tok::Op("<=") => Op::LtE,
                Tok::Op(">") => Op::Gt,
                Tok::Op(">=") => Op::GtE,
                Tok::Name(n) if n == "in" => Op::In,
                Tok::Name(n) if n == "not" && matches!(self.peek_at(1), Tok::Name(m) if m == "in") => {
                    self.advance();
                    Op::NotIn
                }
                Tok::Name(n) if n == "is" => {
                    if matches!(self.peek_at(1), Tok::Name(m) if m == "not") {
                        self.advance();
                        Op::IsNot
                    } else {
                        Op::Is
                    }
                }
                _ => break,
            };
            self.advance();
            pairs.push((op, self.binary(0)?));
        }
        if pairs.is_empty() {
            return Ok(left);
        }
        Ok(Node::Compare(Box::new(left), pairs))
    }

    /// The binary operators, by precedence climbing (one frame per level
    /// rather than one function per level, which keeps deep brackets cheap).
    fn binary(&mut self, level: usize) -> Parsed {
        if level == LEVELS.len() {
            return self.factor();
        }
        let mut left = self.binary(level + 1)?;
        loop {
            let found = LEVELS[level].iter().find(|(sym, _)| self.is_op(sym)).map(|(_, op)| *op);
            let Some(op) = found else { return Ok(left) };
            self.advance();
            let right = self.binary(level + 1)?;
            left = Node::Bin(Box::new(left), op, Box::new(right));
        }
    }

    fn factor(&mut self) -> Parsed {
        let op = match self.peek() {
            Tok::Op("-") => Op::USub,
            Tok::Op("+") => Op::UAdd,
            Tok::Op("~") => Op::Invert,
            _ => return self.power(),
        };
        self.advance();
        Ok(Node::Unary(op, Box::new(self.factor()?)))
    }

    fn power(&mut self) -> Parsed {
        let base = if self.accept_kw("await") {
            self.primary()?;
            Node::Banned("Await")
        } else {
            self.primary()?
        };
        if self.accept("**") {
            let exponent = self.factor()?;
            return Ok(Node::Bin(Box::new(base), Op::Pow, Box::new(exponent)));
        }
        Ok(base)
    }

    fn primary(&mut self) -> Parsed {
        let mut node = self.atom()?;
        loop {
            if self.accept("(") {
                self.depth += 1;
                let (args, keywords) = self.arguments()?;
                self.close(")")?;
                node = Node::Call(Box::new(node), args, keywords);
            } else if self.accept("[") {
                self.depth += 1;
                if self.is_op("]") {
                    return Err(invalid());
                }
                self.skip_to_close();
                self.close("]")?;
                node = Node::Banned("Subscript");
            } else if self.accept(".") {
                match self.advance() {
                    Tok::Name(name) if !KEYWORDS.contains(&name.as_str()) => {}
                    _ => return Err(invalid()),
                }
                node = Node::Banned("Attribute");
            } else {
                return Ok(node);
            }
        }
    }

    /// A call's arguments, up to (not including) its `)`.
    fn arguments(&mut self) -> Result<(Vec<Node>, Vec<Node>), String> {
        let mut args = Vec::new();
        let mut keywords = Vec::new();
        while !self.is_op(")") {
            if self.accept("*") {
                self.expression()?;
                args.push(Node::Banned("Starred"));
            } else if self.accept("**") {
                self.expression()?;
                keywords.push(Node::Banned("keyword"));
            } else if matches!(self.peek(), Tok::Name(_)) && matches!(self.peek_at(1), Tok::Op("=")) {
                self.advance();
                self.advance();
                self.expression()?;
                keywords.push(Node::Banned("keyword"));
            } else {
                let arg = self.named_expression()?;
                if self.is_kw("for") || self.is_kw("async") {
                    self.skip_to_close();
                    args.push(Node::Banned("GeneratorExp"));
                    break;
                }
                args.push(arg);
            }
            if !self.accept(",") {
                break;
            }
        }
        Ok((args, keywords))
    }

    /// An expression that may be `name := value` (only inside brackets).
    fn named_expression(&mut self) -> Parsed {
        if matches!(self.peek(), Tok::Name(_)) && matches!(self.peek_at(1), Tok::Op(":=")) {
            self.advance();
            self.advance();
            self.expression()?;
            return Ok(Node::Banned("NamedExpr"));
        }
        self.expression()
    }

    /// One element of a tuple or list display: `*x` is allowed there.
    fn element(&mut self) -> Parsed {
        if self.accept("*") {
            self.binary(0)?;
            return Ok(Node::Banned("Starred"));
        }
        self.named_expression()
    }

    fn atom(&mut self) -> Parsed {
        match self.peek().clone() {
            Tok::Int(value) => {
                self.advance();
                Ok(Node::Const(Const::Int(value)))
            }
            Tok::Float(value) => {
                self.advance();
                Ok(Node::Const(Const::Float(value)))
            }
            Tok::Imag => {
                self.advance();
                Ok(Node::Const(Const::Other))
            }
            Tok::Str(_) => {
                let mut formatted = false;
                let mut bytes = None;
                while let Tok::Str(kind) = self.peek().clone() {
                    self.advance();
                    formatted |= kind == StrKind::Formatted;
                    let is_bytes = kind == StrKind::Bytes;
                    if bytes.is_some_and(|b| b != is_bytes) {
                        return Err("cannot mix bytes and nonbytes literals".to_string());
                    }
                    bytes = Some(is_bytes);
                }
                Ok(if formatted {
                    Node::Banned("JoinedStr")
                } else {
                    Node::Const(Const::Other)
                })
            }
            Tok::Name(name) => {
                self.advance();
                match name.as_str() {
                    "True" => Ok(Node::Const(Const::Bool(true))),
                    "False" => Ok(Node::Const(Const::Bool(false))),
                    "None" => Ok(Node::Const(Const::Other)),
                    word if KEYWORDS.contains(&word) => Err(invalid()),
                    _ => Ok(Node::Name(name)),
                }
            }
            Tok::Op("...") => {
                self.advance();
                Ok(Node::Const(Const::Other))
            }
            Tok::Op("(") => {
                self.advance();
                self.depth += 1;
                if self.accept(")") {
                    self.depth -= 1;
                    return Ok(Node::Tuple(Vec::new()));
                }
                if self.is_kw("yield") {
                    self.skip_to_close();
                    self.close(")")?;
                    return Ok(Node::Banned("Yield"));
                }
                let starred = self.is_op("*");
                let first = self.element()?;
                if self.is_kw("for") || self.is_kw("async") {
                    self.skip_to_close();
                    self.close(")")?;
                    return Ok(Node::Banned("GeneratorExp"));
                }
                if self.is_op(")") {
                    if starred {
                        return Err("cannot use starred expression here".to_string());
                    }
                    self.close(")")?;
                    return Ok(first);
                }
                if !self.is_op(",") {
                    return Err(self.unexpected());
                }
                let mut items = vec![first];
                while self.accept(",") {
                    if self.is_op(")") {
                        break;
                    }
                    items.push(self.element()?);
                }
                self.close(")")?;
                Ok(Node::Tuple(items))
            }
            Tok::Op("[") => {
                self.advance();
                self.depth += 1;
                let mut items = Vec::new();
                while !self.is_op("]") {
                    items.push(self.element()?);
                    if items.len() == 1 && (self.is_kw("for") || self.is_kw("async")) {
                        self.skip_to_close();
                        self.close("]")?;
                        return Ok(Node::Banned("ListComp"));
                    }
                    if !self.accept(",") {
                        break;
                    }
                }
                self.close("]")?;
                Ok(Node::List(items))
            }
            Tok::Op("{") => {
                self.advance();
                self.depth += 1;
                if self.accept("}") {
                    self.depth -= 1;
                    return Ok(Node::Banned("Dict"));
                }
                let dict = if self.accept("**") {
                    self.binary(0)?;
                    true
                } else {
                    self.element()?;
                    if self.accept(":") {
                        self.expression()?;
                        true
                    } else {
                        false
                    }
                };
                let comprehension = self.is_kw("for") || self.is_kw("async");
                self.skip_to_close();
                self.close("}")?;
                Ok(Node::Banned(match (dict, comprehension) {
                    (true, false) => "Dict",
                    (true, true) => "DictComp",
                    (false, false) => "Set",
                    (false, true) => "SetComp",
                }))
            }
            _ => Err(self.unexpected()),
        }
    }
}

// -- the whitelist ----------------------------------------------------------------------------------

/// One thing `ast.walk` yields: a node, an operator node, or a `Load` context.
enum Item<'a> {
    Node(&'a Node),
    Op(Op),
    Load,
}

/// Walks the tree breadth first, as `ast.walk` does, and refuses the first
/// node the whitelist does not allow - so a bad expression gets the same
/// complaint it gets from Python.
fn validate(tree: &Node) -> Result<(), String> {
    let mut queue: VecDeque<Item> = VecDeque::new();
    queue.push_back(Item::Node(tree));
    while let Some(item) = queue.pop_front() {
        let node = match item {
            Item::Load => continue,
            Item::Op(op) => {
                if !op.allowed() {
                    return Err(format!("{} is not allowed in an expression", op.name()));
                }
                continue;
            }
            Item::Node(node) => node,
        };
        match node {
            Node::Banned(name) => return Err(format!("{name} is not allowed in an expression")),
            Node::Const(Const::Other) => return Err("only numbers are allowed in an expression".to_string()),
            Node::Const(_) => {}
            Node::Name(name) => {
                if !FUNCTIONS.contains(&name.as_str()) && !NAMES.iter().any(|(n, _)| *n == name.as_str()) {
                    return Err(format!("unknown name {}", python_repr(name)));
                }
                queue.push_back(Item::Load);
            }
            Node::Tuple(items) | Node::List(items) => {
                queue.extend(items.iter().map(Item::Node));
                queue.push_back(Item::Load);
            }
            Node::Bin(left, op, right) => {
                queue.push_back(Item::Node(left));
                queue.push_back(Item::Op(*op));
                queue.push_back(Item::Node(right));
            }
            Node::Unary(op, operand) => {
                queue.push_back(Item::Op(*op));
                queue.push_back(Item::Node(operand));
            }
            Node::Bool(op, values) => {
                queue.push_back(Item::Op(*op));
                queue.extend(values.iter().map(Item::Node));
            }
            Node::Compare(left, pairs) => {
                queue.push_back(Item::Node(left));
                queue.extend(pairs.iter().map(|(op, _)| Item::Op(*op)));
                queue.extend(pairs.iter().map(|(_, right)| Item::Node(right)));
            }
            Node::Call(func, args, keywords) => {
                let known = matches!(func.as_ref(), Node::Name(n) if FUNCTIONS.contains(&n.as_str()));
                if !known {
                    let name = match func.as_ref() {
                        Node::Name(n) => n.clone(),
                        other => other.type_name().to_string(),
                    };
                    return Err(format!(
                        "unknown function {} (have: {})",
                        python_repr(&name),
                        FUNCTIONS.join(", ")
                    ));
                }
                queue.push_back(Item::Node(func));
                queue.extend(args.iter().map(Item::Node));
                queue.extend(keywords.iter().map(Item::Node));
            }
        }
    }
    Ok(())
}

// -- the evaluator ----------------------------------------------------------------------------------

/// A number: an integer (bools count), a float or a complex.
#[derive(Clone, Copy, Debug)]
enum Num {
    I(i128),
    F(f64),
    C(f64, f64),
}

impl Num {
    fn float(self) -> f64 {
        match self {
            Num::I(n) => n as f64,
            Num::F(x) => x,
            Num::C(re, _) => re,
        }
    }
    fn complex(self) -> (f64, f64) {
        match self {
            Num::C(re, im) => (re, im),
            other => (other.float(), 0.0),
        }
    }
    fn value(self) -> Value {
        match self {
            Num::I(n) => Value::Int(n),
            Num::F(x) => Value::Float(x),
            Num::C(re, im) => Value::Complex(re, im),
        }
    }
}

fn eval(node: &Node) -> Eval<Value> {
    match node {
        Node::Const(Const::Int(Some(n))) => Ok(Value::Int(*n)),
        Node::Const(Const::Int(None)) => fail(TOO_BIG),
        Node::Const(Const::Float(x)) => Ok(Value::Float(*x)),
        Node::Const(Const::Bool(b)) => Ok(Value::Bool(*b)),
        Node::Const(Const::Other) | Node::Banned(_) => fail("the expression was not checked"),
        Node::Name(name) => {
            if let Some((_, value)) = NAMES.iter().find(|(n, _)| *n == name.as_str()) {
                return Ok(Value::Float(*value));
            }
            match FUNCTIONS.iter().find(|f| **f == name.as_str()) {
                Some(f) => Ok(Value::Func(f)),
                None => fail(format!("name {} is not defined", python_repr(name))),
            }
        }
        Node::Tuple(items) => Ok(Value::Tuple(items.iter().map(eval).collect::<Eval<_>>()?)),
        Node::List(items) => Ok(Value::List(items.iter().map(eval).collect::<Eval<_>>()?)),
        Node::Bin(left, op, right) => {
            let a = eval(left)?;
            let b = eval(right)?;
            binary(*op, &a, &b)
        }
        Node::Unary(op, operand) => unary(*op, eval(operand)?),
        Node::Bool(op, values) => {
            // Python returns the operand that decided, and never evaluates the rest
            let mut result = eval(&values[0])?;
            for value in &values[1..] {
                if result.truthy() == (*op == Op::Or) {
                    return Ok(result);
                }
                result = eval(value)?;
            }
            Ok(result)
        }
        Node::Compare(left, pairs) => {
            let mut a = eval(left)?;
            for (op, right) in pairs {
                let b = eval(right)?;
                if !compare(*op, &a, &b)? {
                    return Ok(Value::Bool(false));
                }
                a = b;
            }
            Ok(Value::Bool(true))
        }
        Node::Call(func, args, _) => {
            let Node::Name(name) = func.as_ref() else {
                return fail("the expression was not checked");
            };
            let args = args.iter().map(eval).collect::<Eval<Vec<_>>>()?;
            call(name, args)
        }
    }
}

fn unsupported<T>(op: Op, a: &Value, b: &Value) -> Eval<T> {
    fail(format!(
        "unsupported operand type(s) for {}: '{}' and '{}'",
        op.symbol(),
        a.type_name(),
        b.type_name()
    ))
}

/// A sequence repeated `n` times, within [`MAX_SEQUENCE`].
fn repeat(items: &[Value], n: i128) -> Eval<Vec<Value>> {
    if n <= 0 || items.is_empty() {
        return Ok(Vec::new());
    }
    let total = (items.len() as i128).saturating_mul(n);
    if total > MAX_SEQUENCE as i128 {
        return fail(format!("the sequence would hold {total} items ({MAX_SEQUENCE} max)"));
    }
    Ok(items.iter().cloned().cycle().take(total as usize).collect())
}

fn int_of(v: &Value) -> Option<i128> {
    match v {
        Value::Int(n) => Some(*n),
        Value::Bool(b) => Some(*b as i128),
        _ => None,
    }
}

fn binary(op: Op, a: &Value, b: &Value) -> Eval<Value> {
    if let (Some(x), Some(y)) = (a.num(), b.num()) {
        return arithmetic(op, x, y).map(Num::value);
    }
    match op {
        Op::Add => match (a, b) {
            (Value::List(x), Value::List(y)) | (Value::Tuple(x), Value::Tuple(y)) => {
                if x.len() + y.len() > MAX_SEQUENCE {
                    return fail(format!(
                        "the sequence would hold {} items ({MAX_SEQUENCE} max)",
                        x.len() + y.len()
                    ));
                }
                let joined: Vec<Value> = x.iter().chain(y.iter()).cloned().collect();
                Ok(if matches!(a, Value::List(_)) {
                    Value::List(joined)
                } else {
                    Value::Tuple(joined)
                })
            }
            (Value::List(_) | Value::Tuple(_), other) => fail(format!(
                "can only concatenate {} (not \"{}\") to {}",
                a.type_name(),
                other.type_name(),
                a.type_name()
            )),
            _ => unsupported(op, a, b),
        },
        Op::Mult => match (a, b) {
            (Value::List(items), n) | (n, Value::List(items)) if int_of(n).is_some() => {
                Ok(Value::List(repeat(items, int_of(n).unwrap_or(0))?))
            }
            (Value::Tuple(items), n) | (n, Value::Tuple(items)) if int_of(n).is_some() => {
                Ok(Value::Tuple(repeat(items, int_of(n).unwrap_or(0))?))
            }
            (Value::List(_) | Value::Tuple(_), other) => fail(format!(
                "can't multiply sequence by non-int of type '{}'",
                other.type_name()
            )),
            (other, Value::List(_) | Value::Tuple(_)) => fail(format!(
                "can't multiply sequence by non-int of type '{}'",
                other.type_name()
            )),
            _ => unsupported(op, a, b),
        },
        _ => unsupported(op, a, b),
    }
}

fn arithmetic(op: Op, a: Num, b: Num) -> Eval<Num> {
    if let (Num::C(..), _) | (_, Num::C(..)) = (a, b) {
        return complex_arithmetic(op, a.complex(), b.complex(), a, b);
    }
    if let (Num::I(x), Num::I(y)) = (a, b) {
        let too_big = || Fail::Eval(TOO_BIG.to_string());
        return match op {
            Op::Add => x.checked_add(y).map(Num::I).ok_or_else(too_big),
            Op::Sub => x.checked_sub(y).map(Num::I).ok_or_else(too_big),
            Op::Mult => x.checked_mul(y).map(Num::I).ok_or_else(too_big),
            Op::Div => {
                if y == 0 {
                    return Err(Fail::Zero);
                }
                Ok(Num::F(int_true_div(x, y)))
            }
            Op::FloorDiv => {
                if y == 0 {
                    return Err(Fail::Zero);
                }
                let q = x.checked_div(y).ok_or_else(too_big)?;
                Ok(Num::I(if x % y != 0 && ((x < 0) != (y < 0)) { q - 1 } else { q }))
            }
            Op::Mod => {
                if y == 0 {
                    return Err(Fail::Zero);
                }
                let r = x.checked_rem(y).unwrap_or(0);
                Ok(Num::I(if r != 0 && ((r < 0) != (y < 0)) { r + y } else { r }))
            }
            Op::Pow => {
                if y >= 0 {
                    // 0, 1 and -1 stay small whatever the exponent
                    match x {
                        0 => return Ok(Num::I(i128::from(y == 0))),
                        1 => return Ok(Num::I(1)),
                        -1 => return Ok(Num::I(if y % 2 == 0 { 1 } else { -1 })),
                        _ => {}
                    }
                    let exponent = u32::try_from(y).map_err(|_| too_big())?;
                    return x.checked_pow(exponent).map(Num::I).ok_or_else(too_big);
                }
                // a negative exponent makes it float arithmetic, in Python too
                float_pow(x as f64, y as f64)
            }
            _ => fail("unknown operator"),
        };
    }
    let (x, y) = (a.float(), b.float());
    match op {
        Op::Add => Ok(Num::F(x + y)),
        Op::Sub => Ok(Num::F(x - y)),
        Op::Mult => Ok(Num::F(x * y)),
        Op::Div => {
            if y == 0.0 {
                return Err(Fail::Zero);
            }
            Ok(Num::F(x / y))
        }
        Op::FloorDiv => float_divmod(x, y).map(|(d, _)| Num::F(d)),
        Op::Mod => float_divmod(x, y).map(|(_, m)| Num::F(m)),
        Op::Pow => float_pow(x, y),
        _ => fail("unknown operator"),
    }
}

/// `x / y` for two integers, correctly rounded as Python's `long_true_divide`
/// rounds it.  Converting both to floats first rounds twice once either is
/// past 2**53, so the quotient is built bit by bit instead: its first 55
/// significant bits and whether anything was left over, then one rounding to
/// even.
fn int_true_div(x: i128, y: i128) -> f64 {
    const EXACT: u128 = 1 << 53;
    let (a, b) = (x.unsigned_abs(), y.unsigned_abs());
    if a <= EXACT && b <= EXACT {
        return x as f64 / y as f64;
    }
    let negative = (x < 0) != (y < 0);
    if a == 0 {
        // nothing to build bits from: a zero, signed as Python signs it
        return if negative { -0.0 } else { 0.0 };
    }
    let bits = |n: u128| 128 - n.leading_zeros();
    let mut m = a / b;
    let mut r = a % b;
    let mut e: i32 = 0;
    let mut sticky = false;
    while bits(m) < 55 {
        // r < b <= 2**127, so doubling it cannot overflow
        r <<= 1;
        m <<= 1;
        if r >= b {
            r -= b;
            m |= 1;
        }
        e -= 1;
    }
    if bits(m) > 55 {
        let extra = bits(m) - 55;
        sticky |= m & ((1u128 << extra) - 1) != 0;
        m >>= extra;
        e += extra as i32;
    }
    sticky |= r != 0;
    // 55 bits to 53: the two dropped bits and the sticky one decide
    let dropped = m & 3;
    m >>= 2;
    e += 2;
    if dropped & 2 != 0 && (dropped & 1 != 0 || sticky || m & 1 != 0) {
        m += 1;
        if m == EXACT {
            m >>= 1;
            e += 1;
        }
    }
    let value = m as f64 * 2f64.powi(e);
    if negative {
        -value
    } else {
        value
    }
}

/// CPython's `float_divmod`: the floor quotient and a remainder with the sign
/// of the divisor, including its care for `-0.0` and for a quotient one ulp
/// off an integer.
fn float_divmod(vx: f64, wx: f64) -> Eval<(f64, f64)> {
    if wx == 0.0 {
        return Err(Fail::Zero);
    }
    let mut m = vx % wx;
    let mut div = (vx - m) / wx;
    if m != 0.0 {
        if (wx < 0.0) != (m < 0.0) {
            m += wx;
            div -= 1.0;
        }
    } else {
        m = 0.0f64.copysign(wx);
    }
    let floordiv = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 {
            f += 1.0;
        }
        f
    } else {
        0.0f64.copysign(vx / wx)
    };
    Ok((floordiv, m))
}

fn odd_integer(x: f64) -> bool {
    x.is_finite() && (x % 2.0).abs() == 1.0
}

/// CPython's `float_pow`: its special cases, the complex answer for a negative
/// base and a fractional exponent, and its overflow error.
fn float_pow(iv: f64, iw: f64) -> Eval<Num> {
    if iw == 0.0 {
        return Ok(Num::F(1.0));
    }
    if iv.is_nan() {
        return Ok(Num::F(iv));
    }
    if iw.is_nan() {
        return Ok(Num::F(if iv == 1.0 { 1.0 } else { iw }));
    }
    if iw.is_infinite() {
        let a = iv.abs();
        if a == 1.0 {
            return Ok(Num::F(1.0));
        }
        return Ok(Num::F(if (iw > 0.0) == (a > 1.0) { iw.abs() } else { 0.0 }));
    }
    if iv.is_infinite() {
        let odd = odd_integer(iw);
        return Ok(Num::F(if iw > 0.0 {
            if odd {
                iv
            } else {
                iv.abs()
            }
        } else if odd {
            0.0f64.copysign(iv)
        } else {
            0.0
        }));
    }
    if iv == 0.0 {
        if iw < 0.0 {
            return Err(Fail::Zero);
        }
        return Ok(Num::F(if odd_integer(iw) { iv } else { 0.0 }));
    }
    let mut base = iv;
    let mut negate = false;
    if iv < 0.0 {
        if iw != iw.floor() {
            // a negative number to a fractional power is complex in Python
            let (re, im) = complex_pow((iv, 0.0), (iw, 0.0))?;
            return Ok(Num::C(re, im));
        }
        base = -iv;
        negate = odd_integer(iw);
    }
    if base == 1.0 {
        return Ok(Num::F(if negate { -1.0 } else { 1.0 }));
    }
    let x = base.powf(iw);
    if x.is_infinite() {
        return fail("(34, 'Numerical result out of range')");
    }
    Ok(Num::F(if negate { -x } else { x }))
}

fn c_mul(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// CPython's `_Py_c_quot`: division scaled by the larger part of the divisor.
fn c_quot(a: (f64, f64), b: (f64, f64)) -> Eval<(f64, f64)> {
    let (abs_re, abs_im) = (b.0.abs(), b.1.abs());
    if abs_re >= abs_im {
        if abs_re == 0.0 {
            return Err(Fail::Zero);
        }
        let ratio = b.1 / b.0;
        let denom = b.0 + b.1 * ratio;
        Ok(((a.0 + a.1 * ratio) / denom, (a.1 - a.0 * ratio) / denom))
    } else if abs_im >= abs_re {
        let ratio = b.0 / b.1;
        let denom = b.0 * ratio + b.1;
        Ok(((a.0 * ratio + a.1) / denom, (a.1 * ratio - a.0) / denom))
    } else {
        Ok((f64::NAN, f64::NAN))
    }
}

/// CPython's `complex_pow`: repeated squaring for a small integral exponent,
/// the polar form otherwise.
fn complex_pow(a: (f64, f64), b: (f64, f64)) -> Eval<(f64, f64)> {
    if b.1 == 0.0 && b.0 == b.0.trunc() && b.0.abs() <= 100.0 {
        let n = b.0 as i64;
        let powu = |mut p: (f64, f64), n: u64| {
            let mut r = (1.0, 0.0);
            let mut mask = 1u64;
            while mask > 0 && n >= mask {
                if n & mask != 0 {
                    r = c_mul(r, p);
                }
                mask <<= 1;
                p = c_mul(p, p);
            }
            r
        };
        return if n > 0 {
            Ok(powu(a, n as u64))
        } else {
            c_quot((1.0, 0.0), powu(a, n.unsigned_abs()))
        };
    }
    if b.0 == 0.0 && b.1 == 0.0 {
        return Ok((1.0, 0.0));
    }
    if a.0 == 0.0 && a.1 == 0.0 {
        if b.1 != 0.0 || b.0 < 0.0 {
            return Err(Fail::Zero);
        }
        return Ok((0.0, 0.0));
    }
    let vabs = a.0.hypot(a.1);
    let mut len = vabs.powf(b.0);
    let at = a.1.atan2(a.0);
    let mut phase = at * b.0;
    if b.1 != 0.0 {
        len /= (at * b.1).exp();
        phase += b.1 * vabs.ln();
    }
    let out = (len * phase.cos(), len * phase.sin());
    if out.0.is_infinite() || out.1.is_infinite() {
        return fail("complex exponentiation");
    }
    Ok(out)
}

fn complex_arithmetic(op: Op, x: (f64, f64), y: (f64, f64), a: Num, b: Num) -> Eval<Num> {
    let wrap = |(re, im): (f64, f64)| Num::C(re, im);
    match op {
        Op::Add => Ok(wrap((x.0 + y.0, x.1 + y.1))),
        Op::Sub => Ok(wrap((x.0 - y.0, x.1 - y.1))),
        Op::Mult => Ok(wrap(c_mul(x, y))),
        Op::Div => c_quot(x, y).map(wrap),
        Op::Pow => complex_pow(x, y).map(wrap),
        _ => unsupported(op, &a.value(), &b.value()),
    }
}

fn unary(op: Op, v: Value) -> Eval<Value> {
    match op {
        Op::Not => Ok(Value::Bool(!v.truthy())),
        Op::USub => match v.num() {
            Some(Num::I(n)) => n
                .checked_neg()
                .map(Value::Int)
                .ok_or_else(|| Fail::Eval(TOO_BIG.to_string())),
            Some(Num::F(x)) => Ok(Value::Float(-x)),
            Some(Num::C(re, im)) => Ok(Value::Complex(-re, -im)),
            None => fail(format!("bad operand type for unary -: '{}'", v.type_name())),
        },
        Op::UAdd => match v.num() {
            Some(n) => Ok(n.value()),
            None => fail(format!("bad operand type for unary +: '{}'", v.type_name())),
        },
        _ => fail("unknown operator"),
    }
}

/// An integer against a float, exactly (Python never rounds the integer).
fn cmp_int_float(i: i128, f: f64) -> Option<Ordering> {
    if f.is_nan() {
        return None;
    }
    if f.is_infinite() {
        return Some(if f > 0.0 { Ordering::Less } else { Ordering::Greater });
    }
    let whole = f.floor();
    if whole >= 1.7e38 {
        return Some(Ordering::Less);
    }
    if whole < -1.7e38 {
        return Some(Ordering::Greater);
    }
    let w = whole as i128;
    Some(match i.cmp(&w) {
        Ordering::Equal if f > whole => Ordering::Less,
        other => other,
    })
}

fn num_cmp(a: Num, b: Num) -> Option<Ordering> {
    match (a, b) {
        (Num::I(x), Num::I(y)) => Some(x.cmp(&y)),
        (Num::I(x), Num::F(y)) => cmp_int_float(x, y),
        (Num::F(x), Num::I(y)) => cmp_int_float(y, x).map(Ordering::reverse),
        (x, y) => x.float().partial_cmp(&y.float()),
    }
}

fn equal(a: &Value, b: &Value) -> bool {
    match (a.num(), b.num()) {
        (Some(x @ Num::C(..)), Some(y)) | (Some(x), Some(y @ Num::C(..))) => x.complex() == y.complex(),
        (Some(x), Some(y)) => num_cmp(x, y) == Some(Ordering::Equal),
        _ => match (a, b) {
            (Value::List(x), Value::List(y)) | (Value::Tuple(x), Value::Tuple(y)) => {
                x.len() == y.len() && x.iter().zip(y).all(|(p, q)| equal(p, q))
            }
            (Value::Func(x), Value::Func(y)) => x == y,
            _ => false,
        },
    }
}

/// One comparison, as Python makes it.
fn compare(op: Op, a: &Value, b: &Value) -> Eval<bool> {
    match op {
        Op::Eq => return Ok(equal(a, b)),
        Op::NotEq => return Ok(!equal(a, b)),
        _ => {}
    }
    let refuse = || {
        fail(format!(
            "'{}' not supported between instances of '{}' and '{}'",
            op.symbol(),
            a.type_name(),
            b.type_name()
        ))
    };
    let ordering = match (a.num(), b.num()) {
        (Some(Num::C(..)), Some(_)) | (Some(_), Some(Num::C(..))) => return refuse(),
        (Some(x), Some(y)) => num_cmp(x, y),
        _ => match (a, b) {
            (Value::List(x), Value::List(y)) | (Value::Tuple(x), Value::Tuple(y)) => {
                // the first pair that differs decides, with the operator itself
                match x.iter().zip(y).find(|(p, q)| !equal(p, q)) {
                    Some((p, q)) => return compare(op, p, q),
                    None => Some(x.len().cmp(&y.len())),
                }
            }
            _ => return refuse(),
        },
    };
    let Some(ordering) = ordering else { return Ok(false) };
    Ok(match op {
        Op::Lt => ordering == Ordering::Less,
        Op::LtE => ordering != Ordering::Greater,
        Op::Gt => ordering == Ordering::Greater,
        Op::GtE => ordering != Ordering::Less,
        _ => false,
    })
}

// -- the functions ----------------------------------------------------------------------------------

/// A real number out of a value, or the `TypeError` a math function raises.
fn real(v: &Value) -> Eval<f64> {
    match v.num() {
        Some(Num::I(n)) => Ok(n as f64),
        Some(Num::F(x)) => Ok(x),
        _ => fail(format!("must be real number, not {}", v.type_name())),
    }
}

/// A float that has to become an integer.
fn to_int(x: f64) -> Eval<i128> {
    if x.is_infinite() {
        return fail("cannot convert float infinity to integer");
    }
    if x.is_nan() {
        return fail("cannot convert float NaN to integer");
    }
    if x.abs() >= 1.7e38 {
        return fail(TOO_BIG);
    }
    Ok(x as i128)
}

fn one_arg<'a>(name: &str, args: &'a [Value]) -> Eval<&'a Value> {
    if args.len() != 1 {
        return fail(format!(
            "math.{name}() takes exactly one argument ({} given)",
            args.len()
        ));
    }
    Ok(&args[0])
}

fn domain<T>() -> Eval<T> {
    fail("math domain error")
}

fn call(name: &str, args: Vec<Value>) -> Eval<Value> {
    let n = args.len();
    match name {
        "sqrt" => {
            let x = real(one_arg(name, &args)?)?;
            if x < 0.0 {
                return domain();
            }
            Ok(Value::Float(x.sqrt()))
        }
        "floor" | "ceil" => {
            let v = one_arg(name, &args)?;
            if let Some(i) = int_of(v) {
                return Ok(Value::Int(i));
            }
            let x = real(v)?;
            Ok(Value::Int(to_int(if name == "floor" { x.floor() } else { x.ceil() })?))
        }
        "log" => {
            if n == 0 || n > 2 {
                return fail("math.log requires 1 to 2 arguments");
            }
            let ln = |v: &Value| -> Eval<f64> {
                let x = real(v)?;
                if x <= 0.0 {
                    return domain();
                }
                Ok(x.ln())
            };
            let num = ln(&args[0])?;
            if n == 1 {
                return Ok(Value::Float(num));
            }
            let den = ln(&args[1])?;
            if den == 0.0 {
                return Err(Fail::Zero);
            }
            Ok(Value::Float(num / den))
        }
        "log2" | "log10" => {
            let x = real(one_arg(name, &args)?)?;
            if x <= 0.0 {
                return domain();
            }
            Ok(Value::Float(if name == "log2" { x.log2() } else { x.log10() }))
        }
        "exp" => {
            let x = real(one_arg(name, &args)?)?;
            let y = x.exp();
            if y.is_infinite() && x.is_finite() {
                return fail("math range error");
            }
            Ok(Value::Float(y))
        }
        "sin" | "cos" | "tan" => {
            let x = real(one_arg(name, &args)?)?;
            if x.is_infinite() {
                return domain();
            }
            Ok(Value::Float(match name {
                "sin" => x.sin(),
                "cos" => x.cos(),
                _ => x.tan(),
            }))
        }
        "atan" => Ok(Value::Float(real(one_arg(name, &args)?)?.atan())),
        "fabs" => Ok(Value::Float(real(one_arg(name, &args)?)?.abs())),
        "hypot" => {
            let values = args.iter().map(real).collect::<Eval<Vec<f64>>>()?;
            Ok(Value::Float(hypot(&values)))
        }
        "pow" => {
            if n != 2 {
                return fail(format!("pow expected 2 arguments, got {n}"));
            }
            math_pow(real(&args[0])?, real(&args[1])?).map(Value::Float)
        }
        "abs" => {
            if n != 1 {
                return fail(format!("abs() takes exactly one argument ({n} given)"));
            }
            match args[0].num() {
                Some(Num::I(i)) => i
                    .checked_abs()
                    .map(Value::Int)
                    .ok_or_else(|| Fail::Eval(TOO_BIG.to_string())),
                Some(Num::F(x)) => Ok(Value::Float(x.abs())),
                Some(Num::C(re, im)) => {
                    let r = re.hypot(im);
                    if r.is_infinite() && re.is_finite() && im.is_finite() {
                        return fail("absolute value too large");
                    }
                    Ok(Value::Float(r))
                }
                None => fail(format!("bad operand type for abs(): '{}'", args[0].type_name())),
            }
        }
        "round" => round(args),
        "min" | "max" => {
            if n == 0 {
                return fail(format!("{name} expected at least 1 argument, got 0"));
            }
            let items = if n == 1 {
                match &args[0] {
                    Value::List(items) | Value::Tuple(items) => items.clone(),
                    other => return fail(format!("'{}' object is not iterable", other.type_name())),
                }
            } else {
                args
            };
            let Some(mut best) = items.first().cloned() else {
                return fail(format!("{name}() arg is an empty sequence"));
            };
            let op = if name == "min" { Op::Lt } else { Op::Gt };
            for item in &items[1..] {
                if compare(op, item, &best)? {
                    best = item.clone();
                }
            }
            Ok(best)
        }
        "sum" => {
            if n == 0 {
                return fail("sum() takes at least 1 positional argument (0 given)");
            }
            if n > 2 {
                return fail(format!("sum() takes at most 2 arguments ({n} given)"));
            }
            let items = match &args[0] {
                Value::List(items) | Value::Tuple(items) => items,
                other => return fail(format!("'{}' object is not iterable", other.type_name())),
            };
            let mut total = args.get(1).cloned().unwrap_or(Value::Int(0));
            for item in items {
                total = binary(Op::Add, &total, item)?;
            }
            Ok(total)
        }
        "int" => match n {
            0 => Ok(Value::Int(0)),
            1 => match args[0].num() {
                Some(Num::I(i)) => Ok(Value::Int(i)),
                Some(Num::F(x)) => Ok(Value::Int(to_int(x.trunc())?)),
                _ => fail(format!(
                    "int() argument must be a string, a bytes-like object or a real number, not '{}'",
                    args[0].type_name()
                )),
            },
            // the base is checked before the number is
            2 => match int_of(&args[1]) {
                None => fail(format!(
                    "'{}' object cannot be interpreted as an integer",
                    args[1].type_name()
                )),
                Some(base) if (base != 0 && base < 2) || base > 36 => fail("int() base must be >= 2 and <= 36, or 0"),
                Some(_) => fail("int() can't convert non-string with explicit base"),
            },
            _ => fail(format!("int() takes at most 2 arguments ({n} given)")),
        },
        "float" => match n {
            0 => Ok(Value::Float(0.0)),
            1 => match args[0].num() {
                Some(Num::I(i)) => Ok(Value::Float(i as f64)),
                Some(Num::F(x)) => Ok(Value::Float(x)),
                _ => fail(format!(
                    "float() argument must be a string or a real number, not '{}'",
                    args[0].type_name()
                )),
            },
            _ => fail(format!("float expected at most 1 argument, got {n}")),
        },
        other => fail(format!("name {} is not defined", python_repr(other))),
    }
}

/// `math.hypot` over any number of coordinates: infinity wins over NaN, as in
/// Python, and the sum of squares is scaled by the largest so it neither
/// overflows nor underflows.
fn hypot(values: &[f64]) -> f64 {
    if values.iter().any(|x| x.is_infinite()) {
        return f64::INFINITY;
    }
    if values.iter().any(|x| x.is_nan()) {
        return f64::NAN;
    }
    match values {
        [] => 0.0,
        [x] => x.abs(),
        [x, y] => x.hypot(*y),
        _ => {
            let max = values.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            if max == 0.0 {
                return 0.0;
            }
            if (1e-150..1e150).contains(&max) {
                // no square can overflow or vanish: the plain sum is the exact one
                return values.iter().map(|x| x * x).sum::<f64>().sqrt();
            }
            let total: f64 = values.iter().map(|x| (x / max) * (x / max)).sum();
            max * total.sqrt()
        }
    }
}

/// `math.pow`: the C `pow` with Python's error checking, which is not `**`'s
/// (a negative base and a fractional exponent is a domain error here).
fn math_pow(x: f64, y: f64) -> Eval<f64> {
    if !x.is_finite() || !y.is_finite() {
        if x.is_nan() {
            return Ok(if y == 0.0 { 1.0 } else { x });
        }
        if y.is_nan() {
            return Ok(if x == 1.0 { 1.0 } else { y });
        }
        if x.is_infinite() {
            let odd = y.is_finite() && (y.abs() % 2.0) == 1.0;
            return Ok(if y > 0.0 {
                if odd {
                    x
                } else {
                    x.abs()
                }
            } else if y == 0.0 {
                1.0
            } else if odd {
                0.0f64.copysign(x)
            } else {
                0.0
            });
        }
        // y is infinite
        return Ok(if x.abs() == 1.0 {
            1.0
        } else if (y > 0.0) == (x.abs() > 1.0) {
            f64::INFINITY
        } else {
            0.0
        });
    }
    let r = x.powf(y);
    if r.is_nan() {
        return domain();
    }
    if r.is_infinite() {
        return if x == 0.0 { domain() } else { fail("math range error") };
    }
    Ok(r)
}

/// `round(number[, ndigits])`, including Python's correctly rounded decimal
/// rounding of a float (`round(2.675, 2)` is 2.67: the double is below it).
fn round(args: Vec<Value>) -> Eval<Value> {
    let n = args.len();
    if n == 0 {
        return fail("round() missing required argument 'number' (pos 1)");
    }
    if n > 2 {
        return fail(format!("round() takes at most 2 arguments ({n} given)"));
    }
    // the number is asked for its __round__ before the digits are read
    if !matches!(args[0].num(), Some(Num::I(_) | Num::F(_))) {
        return fail(format!("type {} doesn't define __round__ method", args[0].type_name()));
    }
    let digits = match args.get(1) {
        None => None,
        Some(v) => match int_of(v) {
            Some(d) => Some(d),
            None => {
                return fail(format!(
                    "'{}' object cannot be interpreted as an integer",
                    v.type_name()
                ))
            }
        },
    };
    match args[0].num() {
        Some(Num::I(i)) => match digits {
            Some(d) if d < 0 => Ok(Value::Int(round_int(i, d.unsigned_abs())?)),
            _ => Ok(Value::Int(i)),
        },
        Some(Num::F(x)) => match digits {
            None => {
                let r = x.round_ties_even();
                Ok(Value::Int(to_int(r)?))
            }
            Some(d) => Ok(Value::Float(round_decimal(x, d)?)),
        },
        _ => fail(format!("type {} doesn't define __round__ method", args[0].type_name())),
    }
}

/// An integer rounded half to even at `10 ** k`.
fn round_int(i: i128, k: u128) -> Eval<i128> {
    if k >= 39 {
        return Ok(0);
    }
    let pow = 10i128.pow(k as u32);
    let q = i.div_euclid(pow);
    let r = i.rem_euclid(pow);
    let up = 2 * r > pow || (2 * r == pow && q % 2 != 0);
    let q = if up { q + 1 } else { q };
    q.checked_mul(pow).ok_or_else(|| Fail::Eval(TOO_BIG.to_string()))
}

/// A float rounded to `digits` decimals the way CPython's `double_round` does:
/// through the exact decimal expansion, half to even, and back.
pub fn round_float(x: f64, digits: i128) -> Result<f64, String> {
    round_decimal(x, digits).map_err(|fail| match fail {
        Fail::Zero => "division by zero".to_string(),
        Fail::Eval(message) => message,
    })
}

fn round_decimal(x: f64, digits: i128) -> Eval<f64> {
    if !x.is_finite() || x == 0.0 {
        return Ok(x);
    }
    if digits > 323 {
        return Ok(x);
    }
    if digits < -308 {
        return Ok(0.0 * x);
    }
    if digits >= 0 {
        let text = format!("{:.*}", digits as usize, x);
        let rounded: f64 = text.parse().unwrap_or(x);
        if rounded.is_infinite() {
            return fail("rounded value too large to represent");
        }
        return Ok(rounded);
    }
    // a negative count rounds inside the integer part: the digits are exact
    let k = digits.unsigned_abs() as usize;
    let whole = x.abs().trunc();
    let fraction = x.abs() != whole;
    let text = format!("{whole:.0}");
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    if text.len() < k {
        return Ok(0.0 * sign);
    }
    let (head, tail) = text.split_at(text.len() - k);
    let first = tail.as_bytes()[0];
    let rest_nonzero = tail[1..].bytes().any(|b| b != b'0') || fraction;
    let head_odd = head.bytes().last().is_some_and(|b| (b - b'0') % 2 == 1);
    let up = first > b'5' || (first == b'5' && (rest_nonzero || head_odd));
    let mut digits_out: Vec<u8> = if head.is_empty() {
        vec![b'0']
    } else {
        head.bytes().collect()
    };
    if up {
        let mut at = digits_out.len();
        loop {
            if at == 0 {
                digits_out.insert(0, b'1');
                break;
            }
            at -= 1;
            if digits_out[at] == b'9' {
                digits_out[at] = b'0';
            } else {
                digits_out[at] += 1;
                break;
            }
        }
    }
    let mut out = String::from_utf8(digits_out).unwrap_or_default();
    out.push_str(&"0".repeat(k));
    let rounded: f64 = out.parse().unwrap_or(0.0);
    if rounded.is_infinite() {
        return fail("rounded value too large to represent");
    }
    Ok(sign * rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(expression: &str) -> String {
        safe_eval(expression).unwrap_or_else(|err| panic!("{expression}: {err}"))
    }

    fn err(expression: &str) -> String {
        match safe_eval(expression) {
            Ok(value) => panic!("{expression} was evaluated to {value}"),
            Err(message) => message,
        }
    }

    #[test]
    fn it_answers_what_python_answers() {
        for (expression, want) in [
            ("2 * (3 + 4)", "14"),
            ("sqrt(841)", "29.0"),
            ("2 ** 10", "1024"),
            ("7 / 2", "3.5"),
            ("7 // 2", "3"),
            ("-7 % 3", "2"),
            ("round(2.5)", "2"),
            ("round(3.14159, 2)", "3.14"),
            ("min(3, 1, 2)", "1"),
            ("max([3, 1, 2])", "3"),
            ("sum([1, 2, 3])", "6"),
            ("2 ^ 8", "256"),
            ("1 < 2", "True"),
            ("not 0", "True"),
            ("1 and 2", "2"),
            ("abs(-3)", "3"),
            ("int(3.9)", "3"),
            ("float(3)", "3.0"),
            ("log(100, 10)", "2.0"),
            ("hypot(3, 4)", "5.0"),
            ("1e3", "1000.0"),
            ("0.1 + 0.2", "0.30000000000000004"),
            ("pi", "3.141592653589793"),
            ("2 ** 100", "1267650600228229401496703205376"),
            ("-2 ** 2", "-4"),
            ("2 ** -1", "0.5"),
            ("1e-7", "1e-07"),
            ("1e16", "1e+16"),
            ("0x10 + 0b11 + 0o7 + 1_000", "1026"),
            (".5 + 5.", "5.5"),
            ("-7 // 2.0", "-4.0"),
            ("5 % -3.0", "-1.0"),
            ("-1 % inf", "inf"),
            ("inf - inf", "nan"),
            ("1 < 2 < 3", "True"),
            ("1 < 2 > 1.5", "True"),
            ("1 or 1/0", "1"),
            ("0 and 1/0", "0"),
            ("1, 2", "(1, 2)"),
            ("(1,)", "(1,)"),
            ("()", "()"),
            ("[1] + [2]", "[1, 2]"),
            ("[1, 2] * 2", "[1, 2, 1, 2]"),
            ("[1, 2] == [1, 2.0]", "True"),
            ("(1,) == [1]", "False"),
            ("(1, 2) < (1, 3)", "True"),
            ("min((1, 2), (0, 5))", "(0, 5)"),
            ("max(True, 1)", "True"),
            ("max(1, 2.0)", "2.0"),
            ("min(1, 2, 1.0)", "1"),
            ("sum([1.5, 2])", "3.5"),
            ("sum([[1]], [])", "[1]"),
            ("True + True", "2"),
            ("abs(True)", "1"),
            ("round(2.675, 2)", "2.67"),
            ("round(0.125, 2)", "0.12"),
            ("round(1234.5, -2)", "1200.0"),
            ("round(1234, -2)", "1200"),
            ("round(15, -1)", "20"),
            ("round(25, -1)", "20"),
            ("round(2.5, 0)", "2.0"),
            ("round(-0.5)", "0"),
            ("round(1e300, -300)", "1e+300"),
            ("ceil(-0.5)", "0"),
            ("fabs(-0.0)", "0.0"),
            ("-0.0", "-0.0"),
            ("hypot(3, 4, 12)", "13.0"),
            ("hypot()", "0.0"),
            ("int()", "0"),
            ("float()", "0.0"),
            ("sqrt", "<built-in function sqrt>"),
            ("[sqrt, int]", "[<built-in function sqrt>, <class 'int'>]"),
            ("(-8) ** (1/3)", "(1.0000000000000002+1.7320508075688772j)"),
            ("(-8) ** (1/3) + 1", "(2+1.7320508075688772j)"),
            ("abs((-8) ** (1/3))", "2.0"),
            ("(-2) ** 0.5", "(8.659560562354934e-17+1.4142135623730951j)"),
            ("2 ** 1000.0", "1.0715086071862673e+301"),
            ("(-2.0) ** 3", "-8.0"),
            ("0 ** 0", "1"),
            ("123456789 * 987654321", "121932631112635269"),
            ("tan(1e308)", "-0.5086861259107568"),
            ("1 # a comment", "1"),
        ] {
            assert_eq!(ok(expression), want, "{expression}");
        }
    }

    #[test]
    fn it_refuses_with_python_s_words() {
        for (expression, want) in [
            ("", "the expression is empty"),
            ("1 +", "not an arithmetic expression: invalid syntax"),
            ("(1", "not an arithmetic expression: '(' was never closed"),
            ("1)", "not an arithmetic expression: unmatched ')'"),
            (
                "(1]",
                "not an arithmetic expression: closing parenthesis ']' does not match opening parenthesis '('",
            ),
            ("[1 2]", "not an arithmetic expression: invalid syntax. Perhaps you forgot a comma?"),
            ("1 2", "not an arithmetic expression: invalid syntax"),
            ("x = 1", "not an arithmetic expression: invalid syntax"),
            ("1 $ (", "not an arithmetic expression: invalid syntax"),
            ("1__0", "not an arithmetic expression: invalid decimal literal"),
            ("1e", "not an arithmetic expression: invalid decimal literal"),
            ("1 if 2", "not an arithmetic expression: expected 'else' after 'if' expression"),
            ("10 / 0", "division by zero"),
            ("3 % 0", "division by zero"),
            ("0 ** -1", "division by zero"),
            ("log(8, 1)", "division by zero"),
            ("sqrt(-1)", "cannot evaluate the expression: math domain error"),
            ("exp(1000)", "cannot evaluate the expression: math range error"),
            ("2.0 ** 10000", "cannot evaluate the expression: (34, 'Numerical result out of range')"),
            ("pow(-8, 1/3)", "cannot evaluate the expression: math domain error"),
            ("floor(1e400)", "cannot evaluate the expression: cannot convert float infinity to integer"),
            ("min(5)", "cannot evaluate the expression: 'int' object is not iterable"),
            ("min([])", "cannot evaluate the expression: min() arg is an empty sequence"),
            ("sqrt()", "cannot evaluate the expression: math.sqrt() takes exactly one argument (0 given)"),
            (
                "sqrt + 1",
                "cannot evaluate the expression: unsupported operand type(s) for +: 'builtin_function_or_method' and 'int'",
            ),
            (
                "[1] + 1",
                "cannot evaluate the expression: can only concatenate list (not \"int\") to list",
            ),
            (
                "[1] * 2.0",
                "cannot evaluate the expression: can't multiply sequence by non-int of type 'float'",
            ),
            (
                "min(1, [1])",
                "cannot evaluate the expression: '<' not supported between instances of 'list' and 'int'",
            ),
            (
                "max([1], 1)",
                "cannot evaluate the expression: '>' not supported between instances of 'int' and 'list'",
            ),
            ("round(5, 1.0)", "cannot evaluate the expression: 'float' object cannot be interpreted as an integer"),
            ("'a' * 3", "only numbers are allowed in an expression"),
            ("None + 1", "only numbers are allowed in an expression"),
            ("1j * 1", "only numbers are allowed in an expression"),
            ("[].__class__", "Attribute is not allowed in an expression"),
            ("lambda: 1", "Lambda is not allowed in an expression"),
            ("1 if 1 else 2", "IfExp is not allowed in an expression"),
            ("~1", "Invert is not allowed in an expression"),
            ("1 << 2", "LShift is not allowed in an expression"),
            ("1 in [1]", "In is not allowed in an expression"),
            ("a[0]", "Subscript is not allowed in an expression"),
            ("{}", "Dict is not allowed in an expression"),
            ("{1}", "Set is not allowed in an expression"),
            ("[i for i in [1]]", "ListComp is not allowed in an expression"),
            ("sum(i for i in [1])", "GeneratorExp is not allowed in an expression"),
            ("sqrt(x=1)", "keyword is not allowed in an expression"),
            ("sqrt(*[1])", "Starred is not allowed in an expression"),
            ("f'{1}'", "JoinedStr is not allowed in an expression"),
            ("nosuchname", "unknown name 'nosuchname'"),
        ] {
            assert_eq!(err(expression), want, "{expression}");
        }
        let unknown = err("open('x')");
        assert!(
            unknown.starts_with("unknown function 'open' (have: abs, atan, ceil"),
            "{unknown}"
        );
        assert!(err("(1)(2)").starts_with("unknown function 'Constant'"));
        assert!(err("pi(2)").starts_with("unknown function 'pi'"));
        assert_eq!(err(&"9".repeat(600)), "the expression is too long (500 characters max)");
        assert!(
            err("10 ** 40").contains("2**127"),
            "an honest limit, not a wrong answer"
        );
    }

    #[test]
    fn integer_division_is_rounded_once() {
        assert_eq!(ok("(2**100 + 1) / (2**53 + 1)"), "140737488355327.98");
        assert_eq!(ok("(2**60 + 1) / 3"), "3.843071682022823e+17");
        assert_eq!(ok("-(2**70 + 1) / 2"), "-5.902958103587057e+20");
        assert_eq!(ok("1 / 2**126"), "1.1754943508222875e-38");
        assert_eq!(ok("0 / 1000000000000000007"), "0.0");
        assert_eq!(ok("0 / -1000000000000000007"), "-0.0");
        assert_eq!(ok("(2**126 - 1) / 3"), "2.8356863910078204e+37");
        assert_eq!(ok("(2**126 + 2**73 + 1) / 2**73"), "9007199254740994.0");
        assert_eq!(ok("(2**126 + 2**73) / 2**73"), "9007199254740992.0");
    }

    #[test]
    fn deep_brackets_are_fine_and_deeper_ones_are_python_s_error() {
        let deep = format!("{}1{}", "(".repeat(200), ")".repeat(200));
        assert_eq!(ok(&deep), "1");
        let deeper = format!("{}1{}", "(".repeat(201), ")".repeat(201));
        assert_eq!(
            err(&deeper),
            "not an arithmetic expression: too many nested parentheses"
        );
        assert_eq!(ok(&format!("{}1", "-".repeat(499))), "-1");
    }

    #[test]
    fn float_rounding_is_decimal_and_half_even() {
        assert_eq!(round_float(0.375, 2).unwrap(), 0.38);
        assert_eq!(round_float(-1.0, -2).unwrap().to_bits(), (-0.0f64).to_bits());
        assert_eq!(round_float(1250.0, -2).unwrap(), 1200.0);
        assert_eq!(round_float(1350.0, -2).unwrap(), 1400.0);
        assert_eq!(round_float(1250.5, -2).unwrap(), 1300.0);
        assert_eq!(round_float(999.0, -1).unwrap(), 1000.0);
        assert_eq!(float_repr(f64::NEG_INFINITY), "-inf");
        assert_eq!(complex_repr(0.0, 2.0), "2j");
        assert_eq!(complex_repr(1.0, -2.0), "(1-2j)");
    }
}
