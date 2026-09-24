//! Learning-rate schedules: a rate as a *graph function* of the epoch.
//!
//! A schedule is a small arithmetic expression evaluated once per epoch:
//!
//! ```text
//! linear(lr0, 4 * lr0)          ramp from the base rate to four times it
//! lr0 * 1.25 ** i               grow 25 % every epoch
//! 0.05 + 0.45 * t               from 0.05 at the first epoch to 0.5 at the last
//! warmup(lr0 / 10, lr0, 3)      warm up over three epochs, then hold
//! lr / 10                       (activation schedule) a tenth of this epoch's rate
//! ```
//!
//! This is a port of `radixnet/schedule.py`, and the reason it is a parser
//! rather than a table of presets is that Python's is Python: it parses the
//! text with `ast`, walks the tree against a whitelist of node types, names and
//! functions, and evaluates it with `eval`.  An expression that one side
//! accepts and the other refuses, or that the two evaluate to different
//! doubles, would make a schedule a thing that changes meaning with the
//! implementation that trained the model - so everything a schedule can see is
//! reproduced here, down to the words of the errors:
//!
//! * **the grammar** is Python's expression grammar, parsed by recursive
//!   descent into a tree shaped like `ast`'s (operators and the `Load` context
//!   are nodes of their own), because the whitelist is checked breadth first,
//!   as `ast.walk` visits, and the first node it refuses names the error;
//! * **the values** follow Python's numeric tower - `bool`, `int`, `float` and
//!   `complex` - because `floor` and `round` hand back integers, `True + 1` is
//!   `2`, and `(-8.0) ** 0.5` is a complex number whose `abs` is a fine rate;
//! * **the arithmetic** is CPython's: floor division and modulo take the sign
//!   of the divisor, `**` has its special cases before the platform `pow` is
//!   asked, and the `math` functions raise where CPython raises;
//! * **the errors** are CPython 3.11's texts, wrapped the way `schedule.py`
//!   wraps them.
//!
//! What is not reproduced: integers past `i128` (Python's are unbounded; here
//! they continue as floats), and the parser's error recovery in its rarest
//! corners, where CPython may name a different syntax error for input that is
//! wrong either way.

use std::collections::VecDeque;
use std::f64::consts::{E, PI};

use crate::json::{py_repr, Json};
use crate::negative::python_repr;

/// The variables an expression may read.
pub const VARIABLES: [&str; 7] = ["epoch", "i", "epochs", "t", "lr0", "act_lr0", "lr"];
/// The named constants, sorted as `describe()` lists them.
pub const CONSTANTS: [&str; 2] = ["e", "pi"];
/// The functions, sorted as `describe()` lists them.
pub const FUNCTIONS: [&str; 17] = [
    "abs", "ceil", "clamp", "cos", "exp", "floor", "log", "log10", "log2", "max", "min", "pow", "round", "sin", "sqrt",
    "tan", "tanh",
];
/// The schedule helpers.
pub const HELPERS: [&str; 5] = ["linear", "geometric", "cosine", "step", "warmup"];

/// One ready-made pair of schedules, as the frontend's preset menu lists them.
pub struct Preset {
    pub name: &'static str,
    pub lr: &'static str,
    pub act_lr: &'static str,
    pub description: &'static str,
}

/// The presets, in Python's order.
pub const PRESETS: [Preset; 8] = [
    Preset {
        name: "constant",
        lr: "lr0",
        act_lr: "act_lr0",
        description: "the base rates every epoch",
    },
    Preset {
        name: "linear ramp x4",
        lr: "linear(lr0, 4 * lr0)",
        act_lr: "linear(act_lr0, 4 * act_lr0)",
        description: "grow linearly from the base rate to four times it",
    },
    Preset {
        name: "exponential ramp x4",
        lr: "geometric(lr0, 4 * lr0)",
        act_lr: "geometric(act_lr0, 4 * act_lr0)",
        description: "grow by the same factor every epoch, reaching four times the base rate",
    },
    Preset {
        name: "cosine ramp x4",
        lr: "cosine(lr0, 4 * lr0)",
        act_lr: "cosine(act_lr0, 4 * act_lr0)",
        description: "a smooth S-shaped rise from the base rate to four times it",
    },
    Preset {
        name: "step x1.5 every 2 epochs",
        lr: "step(lr0, 1.5, 2)",
        act_lr: "step(act_lr0, 1.5, 2)",
        description: "multiply by 1.5 every two epochs",
    },
    Preset {
        name: "grow 25% per epoch",
        lr: "lr0 * 1.25 ** i",
        act_lr: "act_lr0 * 1.25 ** i",
        description: "compound growth of a quarter per epoch",
    },
    Preset {
        name: "warm-up over 3 epochs",
        lr: "warmup(lr0 / 10, lr0, 3)",
        act_lr: "warmup(act_lr0 / 10, act_lr0, 3)",
        description: "start at a tenth of the rate, reach it after three epochs, then hold",
    },
    Preset {
        name: "activation follows lr / 10",
        lr: "lr0",
        act_lr: "lr / 10",
        description: "keep the activation rate at a tenth of whatever the learning rate is",
    },
];

/// What a schedule expression may use - for help texts and `GET /api/schedule`.
pub fn describe() -> Json {
    Json::obj([
        ("variables", Json::strs(VARIABLES)),
        ("constants", Json::strs(CONSTANTS)),
        ("functions", Json::strs(FUNCTIONS)),
        (
            "helpers",
            Json::strs([
                "linear(a, b): a at the first epoch, b at the last",
                "geometric(a, b): a to b by a constant factor per epoch (a > 0)",
                "cosine(a, b): a to b along a half cosine",
                "step(a, factor, every): a times factor every `every` epochs",
                "warmup(a, b, n): a to b over the first n epochs, then b",
            ]),
        ),
        ("presets", presets_json()),
    ])
}

/// The presets as `describe()` lists them.
pub fn presets_json() -> Json {
    Json::Arr(
        PRESETS
            .iter()
            .map(|p| {
                Json::obj([
                    ("name", Json::str(p.name)),
                    ("lr", Json::str(p.lr)),
                    ("act_lr", Json::str(p.act_lr)),
                    ("description", Json::str(p.description)),
                ])
            })
            .collect(),
    )
}

// -- the rates ------------------------------------------------------------------------------------

/// The rates of one epoch.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Point {
    pub epoch: i64,
    pub lr: f64,
    pub act_lr: f64,
}

impl Point {
    /// `{"epoch", "lr", "act_lr"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("epoch", Json::Int(self.epoch)),
            ("lr", Json::Num(self.lr)),
            ("act_lr", Json::Num(self.act_lr)),
        ])
    }
}

/// `[{"epoch", "lr", "act_lr"}]` for every epoch; the activation schedule sees
/// the epoch's `lr`.
///
/// `reverse` plays the schedule backwards: the rates of the last epoch come
/// first (a ramp up becomes a ramp down), and the pairing of `lr` and `act_lr`
/// within an epoch is kept.  Both expressions are parsed before either is
/// evaluated, as Python parses them.
pub fn preview_points(
    lr_schedule: Option<&str>,
    act_lr_schedule: Option<&str>,
    epochs: i64,
    lr: f64,
    act_lr: f64,
    reverse: bool,
) -> Result<Vec<Point>, String> {
    if epochs < 0 {
        return Err("epochs must be >= 0".to_string());
    }
    let lr_fn = parse_schedule(lr_schedule)?;
    let act_fn = parse_schedule(act_lr_schedule)?;
    let mut points = Vec::with_capacity(epochs as usize);
    for epoch in 1..=epochs {
        let rate = match &lr_fn {
            Some(f) => f.rate(epoch, epochs, lr, act_lr, lr)?,
            None => lr,
        };
        let act = match &act_fn {
            Some(f) => f.rate(epoch, epochs, act_lr, act_lr, rate)?,
            None => act_lr,
        };
        points.push(Point {
            epoch,
            lr: rate,
            act_lr: act,
        });
    }
    if reverse {
        points.reverse();
        for (i, p) in points.iter_mut().enumerate() {
            p.epoch = i as i64 + 1;
        }
    }
    Ok(points)
}

/// `None` or blank is a constant rate; anything else must be a valid schedule.
pub fn parse_schedule(expression: Option<&str>) -> Result<Option<Schedule>, String> {
    match expression {
        None => Ok(None),
        Some(text) if text.trim().is_empty() => Ok(None),
        Some(text) => Schedule::new(text).map(Some),
    }
}

/// A validated schedule expression; [`Schedule::rate`] evaluates it.
#[derive(Clone, Debug)]
pub struct Schedule {
    pub expression: String,
    tree: Node,
}

impl Schedule {
    /// Parses and validates an expression (`ScheduleError` texts on failure).
    pub fn new(expression: &str) -> Result<Schedule, String> {
        let text = expression.trim();
        if text.is_empty() {
            return Err("the schedule expression is empty".to_string());
        }
        let tree = parse(text).map_err(|msg| format!("invalid schedule expression {}: {msg}", python_repr(text)))?;
        validate(&tree)?;
        Ok(Schedule {
            expression: text.to_string(),
            tree,
        })
    }

    /// The rate for `epoch` (1-based) of `epochs`; `base` is `lr0`, and
    /// `act_lr0` and `lr` are the other two names a schedule may read.
    pub fn rate(&self, epoch: i64, epochs: i64, base: f64, act_lr0: f64, lr: f64) -> Result<f64, String> {
        let i = (epoch - 1) as i128;
        let t = if epochs > 1 {
            true_div_int(i, (epochs - 1) as i128)
        } else {
            0.0
        };
        let env = Env {
            epoch: epoch as f64,
            i,
            epochs: epochs as f64,
            t,
            lr0: base,
            act_lr0,
            lr,
        };
        let expr = python_repr(&self.expression);
        let value = eval(&self.tree, &env).map_err(|msg| format!("schedule {expr} failed at epoch {epoch}: {msg}"))?;
        let rate = match value {
            Val::Float(x) => x,
            Val::Int(n) => n as f64,
            other => {
                return Err(format!("schedule {expr} must give a number, got {}", other.type_name()));
            }
        };
        if !rate.is_finite() || rate < 0.0 {
            return Err(format!(
                "schedule {expr} gave {} at epoch {epoch}; rates must be finite and >= 0",
                py_str(rate)
            ));
        }
        Ok(rate)
    }

    /// The rates of epochs `1..=epochs`.
    pub fn preview(&self, epochs: i64, base: f64, act_lr0: f64, lr: f64) -> Result<Vec<f64>, String> {
        (1..=epochs).map(|e| self.rate(e, epochs, base, act_lr0, lr)).collect()
    }
}

/// `str(float)` as Python writes it in a message: `inf`, `nan`, else `repr`.
fn py_str(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        py_repr(x)
    }
}

// -- the tree -------------------------------------------------------------------------------------

/// A constant as the tokenizer read it.
#[derive(Clone, Debug, PartialEq)]
enum Const {
    /// Every number: `_Floats` turns an integer literal into a float before
    /// anything is evaluated, so there is only one kind to keep.
    Num(f64),
    Imag(f64),
    Str(String),
    Bytes(String),
    True,
    False,
    None,
    Ellipsis,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Bin {
    Add,
    Sub,
    Mult,
    Div,
    Pow,
    Mod,
    FloorDiv,
    BitOr,
    BitXor,
    BitAnd,
    LShift,
    RShift,
    MatMult,
}

impl Bin {
    fn name(self) -> &'static str {
        match self {
            Bin::Add => "Add",
            Bin::Sub => "Sub",
            Bin::Mult => "Mult",
            Bin::Div => "Div",
            Bin::Pow => "Pow",
            Bin::Mod => "Mod",
            Bin::FloorDiv => "FloorDiv",
            Bin::BitOr => "BitOr",
            Bin::BitXor => "BitXor",
            Bin::BitAnd => "BitAnd",
            Bin::LShift => "LShift",
            Bin::RShift => "RShift",
            Bin::MatMult => "MatMult",
        }
    }

    fn allowed(self) -> bool {
        matches!(
            self,
            Bin::Add | Bin::Sub | Bin::Mult | Bin::Div | Bin::Pow | Bin::Mod | Bin::FloorDiv
        )
    }

    /// The symbol a `TypeError` names.
    fn symbol(self) -> &'static str {
        match self {
            Bin::Add => "+",
            Bin::Sub => "-",
            Bin::Mult => "*",
            Bin::Div => "/",
            Bin::Pow => "** or pow()",
            Bin::Mod => "%",
            Bin::FloorDiv => "//",
            Bin::BitOr => "|",
            Bin::BitXor => "^",
            Bin::BitAnd => "&",
            Bin::LShift => "<<",
            Bin::RShift => ">>",
            Bin::MatMult => "@",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Un {
    Plus,
    Minus,
    Not,
    Invert,
}

impl Un {
    fn name(self) -> &'static str {
        match self {
            Un::Plus => "UAdd",
            Un::Minus => "USub",
            Un::Not => "Not",
            Un::Invert => "Invert",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Cmp {
    Lt,
    LtE,
    Gt,
    GtE,
    Eq,
    NotEq,
    In,
    NotIn,
    Is,
    IsNot,
}

impl Cmp {
    fn name(self) -> &'static str {
        match self {
            Cmp::Lt => "Lt",
            Cmp::LtE => "LtE",
            Cmp::Gt => "Gt",
            Cmp::GtE => "GtE",
            Cmp::Eq => "Eq",
            Cmp::NotEq => "NotEq",
            Cmp::In => "In",
            Cmp::NotIn => "NotIn",
            Cmp::Is => "Is",
            Cmp::IsNot => "IsNot",
        }
    }

    fn allowed(self) -> bool {
        matches!(self, Cmp::Lt | Cmp::LtE | Cmp::Gt | Cmp::GtE | Cmp::Eq | Cmp::NotEq)
    }

    fn symbol(self) -> &'static str {
        match self {
            Cmp::Lt => "<",
            Cmp::LtE => "<=",
            Cmp::Gt => ">",
            Cmp::GtE => ">=",
            Cmp::Eq => "==",
            Cmp::NotEq => "!=",
            Cmp::In => "in",
            Cmp::NotIn => "not in",
            Cmp::Is => "is",
            Cmp::IsNot => "is not",
        }
    }
}

/// A node of the tree, shaped like `ast`'s where the whitelist can see it.
///
/// `Other` is every node type the whitelist refuses outright (`Attribute`,
/// `Lambda`, `Tuple`, ...): it is refused on its own visit, before any of its
/// children would be, so only its type and its place in the walk matter.
#[derive(Clone, Debug)]
enum Node {
    Const(Const),
    Name(String),
    BinOp(Box<Node>, Bin, Box<Node>),
    Unary(Un, Box<Node>),
    Compare(Box<Node>, Vec<Cmp>, Vec<Node>),
    BoolOp(bool, Vec<Node>),
    IfExp(Box<Node>, Box<Node>, Box<Node>),
    /// The function, the positional arguments, and whether any argument was
    /// a keyword or a `*` / `**` unpacking.
    Call(Box<Node>, Vec<Node>, bool),
    Other(&'static str),
    /// An operator or context node (`Add`, `Lt`, `And`, `Load`): visited by
    /// the walk like any other.
    Op(&'static str),
}

/// The children of a node in `ast`'s field order - what `ast.walk` queues.
fn children(node: &Node) -> Vec<Node> {
    match node {
        Node::Const(_) | Node::Other(_) | Node::Op(_) => Vec::new(),
        Node::Name(_) => vec![Node::Op("Load")],
        Node::BinOp(left, op, right) => vec![(**left).clone(), Node::Op(op.name()), (**right).clone()],
        Node::Unary(op, operand) => vec![Node::Op(op.name()), (**operand).clone()],
        Node::Compare(left, ops, rest) => {
            let mut out = vec![(**left).clone()];
            out.extend(ops.iter().map(|op| Node::Op(op.name())));
            out.extend(rest.iter().cloned());
            out
        }
        Node::BoolOp(and, values) => {
            let mut out = vec![Node::Op(if *and { "And" } else { "Or" })];
            out.extend(values.iter().cloned());
            out
        }
        Node::IfExp(test, body, orelse) => vec![(**test).clone(), (**body).clone(), (**orelse).clone()],
        Node::Call(func, args, _) => {
            let mut out = vec![(**func).clone()];
            out.extend(args.iter().cloned());
            out
        }
    }
}

// -- validation: the whitelist, breadth first ---------------------------------------------------------

fn allowed_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = VARIABLES.to_vec();
    names.extend(["pi", "e"]);
    names.extend(FUNCTIONS);
    names.extend(HELPERS);
    names.sort_unstable();
    names.dedup();
    names
}

fn callable_names() -> Vec<&'static str> {
    let mut names: Vec<&'static str> = FUNCTIONS.to_vec();
    names.extend(HELPERS);
    names.sort_unstable();
    names
}

/// Python's `_validate`: every node, in the order `ast.walk` visits them,
/// against the whitelist; the first one refused names the error.
fn validate(tree: &Node) -> Result<(), String> {
    let mut queue: VecDeque<Node> = VecDeque::new();
    queue.push_back(tree.clone());
    while let Some(node) = queue.pop_front() {
        queue.extend(children(&node));
        check(&node)?;
    }
    Ok(())
}

fn check(node: &Node) -> Result<(), String> {
    match node {
        Node::Const(c) => match c {
            Const::Num(_) => Ok(()),
            other => Err(format!(
                "only numbers are allowed as constants, not {}",
                const_repr(other)
            )),
        },
        Node::BinOp(_, op, _) if !op.allowed() => Err(format!("operator {} is not allowed", op.name())),
        Node::Unary(Un::Invert, _) => Err("operator Invert is not allowed".to_string()),
        Node::Compare(_, ops, _) if !ops.iter().all(|op| op.allowed()) => {
            Err("only <, <=, >, >=, == and != comparisons are allowed".to_string())
        }
        Node::Name(id) => {
            let names = allowed_names();
            if names.contains(&id.as_str()) {
                Ok(())
            } else {
                Err(format!(
                    "unknown name {}; allowed: {}",
                    python_repr(id),
                    names.join(", ")
                ))
            }
        }
        Node::Call(func, _, extra) => {
            let callables = callable_names();
            let name = match &**func {
                Node::Name(id) if callables.contains(&id.as_str()) => id.clone(),
                _ => return Err(format!("only these functions may be called: {}", callables.join(", "))),
            };
            if *extra {
                return Err(format!("{name}() takes positional arguments only"));
            }
            Ok(())
        }
        Node::Other(kind) => Err(format!("{kind} is not allowed in a schedule expression")),
        Node::Op(name) => match *name {
            "Load" | "And" | "Or" | "Add" | "Sub" | "Mult" | "Div" | "Pow" | "Mod" | "FloorDiv" | "UAdd" | "USub"
            | "Not" | "Lt" | "LtE" | "Gt" | "GtE" | "Eq" | "NotEq" => Ok(()),
            other => Err(format!("{other} is not allowed in a schedule expression")),
        },
        _ => Ok(()),
    }
}

/// `repr()` of a refused constant.
fn const_repr(c: &Const) -> String {
    match c {
        Const::Num(x) => py_repr(*x),
        Const::Imag(x) => {
            let text = py_repr(*x);
            format!("{}j", text.strip_suffix(".0").unwrap_or(&text))
        }
        Const::Str(s) => python_repr(s),
        Const::Bytes(s) => format!("b{}", python_repr(s)),
        Const::True => "True".to_string(),
        Const::False => "False".to_string(),
        Const::None => "None".to_string(),
        Const::Ellipsis => "Ellipsis".to_string(),
    }
}

// -- the tokenizer --------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Tok {
    Num(f64),
    Imag(f64),
    Name(String),
    Str {
        value: String,
        bytes: bool,
        fstring: bool,
    },
    Op(&'static str),
    Newline,
    End,
    /// A tokenizer error, where the parser meets it.
    Error(String),
}

#[derive(Clone, Debug)]
struct Token {
    tok: Tok,
    /// The bracket depth the token sits at (after an opener, before a closer).
    level: usize,
}

const KEYWORDS: [&str; 35] = [
    "False", "None", "True", "and", "as", "assert", "async", "await", "break", "class", "continue", "def", "del",
    "elif", "else", "except", "finally", "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
];

/// Longest first, so `**=` is not read as `**` and `=`.
const OPERATORS: &[&str] = &[
    "**=", "//=", ">>=", "<<=", "...", "!=", "%=", "&=", "**", "*=", "+=", "-=", "->", "//", "/=", ":=", "<<", "<=",
    "==", ">=", ">>", "@=", "^=", "|=", "<>", "%", "&", "(", ")", "*", "+", ",", "-", ".", "/", ":", ";", "<", "=",
    ">", "@", "[", "]", "^", "{", "|", "}", "~",
];

fn is_id_start(c: char) -> bool {
    c == '_' || c.is_alphabetic()
}

fn is_id_char(c: char) -> bool {
    c == '_' || c.is_alphanumeric()
}

/// Cuts the text into tokens the way CPython's tokenizer does for one logical
/// line; a tokenizer error becomes the last token, where the parser meets it.
fn tokenize(text: &str) -> Vec<Token> {
    let chars: Vec<char> = text.chars().collect();
    let mut out: Vec<Token> = Vec::new();
    let mut brackets: Vec<char> = Vec::new();
    let mut pos = 0usize;
    let push = |out: &mut Vec<Token>, tok: Tok, level: usize| out.push(Token { tok, level });
    while pos < chars.len() {
        let c = chars[pos];
        if c == ' ' || c == '\t' || c == '\x0c' {
            pos += 1;
            continue;
        }
        if c == '#' {
            while pos < chars.len() && chars[pos] != '\n' {
                pos += 1;
            }
            continue;
        }
        if c == '\\' && chars.get(pos + 1) == Some(&'\n') {
            pos += 2;
            continue;
        }
        if c == '\n' || c == '\r' {
            pos += 1;
            if brackets.is_empty() {
                push(&mut out, Tok::Newline, 0);
            }
            continue;
        }
        // numbers, including those that start with a point
        if c.is_ascii_digit() || (c == '.' && chars.get(pos + 1).is_some_and(|d| d.is_ascii_digit())) {
            match number(&chars, &mut pos) {
                Ok(tok) => push(&mut out, tok, brackets.len()),
                Err(msg) => {
                    push(&mut out, Tok::Error(msg), brackets.len());
                    return out;
                }
            }
            continue;
        }
        // strings, with their prefixes
        if let Some((prefix_len, bytes, fstring)) = string_prefix(&chars, pos) {
            pos += prefix_len;
            match string(&chars, &mut pos) {
                Ok(value) => push(&mut out, Tok::Str { value, bytes, fstring }, brackets.len()),
                Err(msg) => {
                    push(&mut out, Tok::Error(msg), brackets.len());
                    return out;
                }
            }
            continue;
        }
        if is_id_start(c) {
            let start = pos;
            while pos < chars.len() && is_id_char(chars[pos]) {
                pos += 1;
            }
            let name: String = chars[start..pos].iter().collect();
            push(&mut out, Tok::Name(name), brackets.len());
            continue;
        }
        let rest: String = chars[pos..chars.len().min(pos + 3)].iter().collect();
        if let Some(op) = OPERATORS.iter().find(|op| rest.starts_with(**op)) {
            pos += op.chars().count();
            match *op {
                "(" | "[" | "{" => {
                    push(&mut out, Tok::Op(op), brackets.len());
                    brackets.push(op.chars().next().expect("an opener"));
                }
                ")" | "]" | "}" => {
                    let close = op.chars().next().expect("a closer");
                    let want = match close {
                        ')' => '(',
                        ']' => '[',
                        _ => '{',
                    };
                    match brackets.last() {
                        None => {
                            push(&mut out, Tok::Error(format!("unmatched '{close}'")), 0);
                            return out;
                        }
                        Some(&open) if open != want => {
                            push(
                                &mut out,
                                Tok::Error(format!(
                                    "closing parenthesis '{close}' does not match opening parenthesis '{open}'"
                                )),
                                brackets.len(),
                            );
                            return out;
                        }
                        Some(_) => {
                            brackets.pop();
                            push(&mut out, Tok::Op(op), brackets.len());
                        }
                    }
                }
                _ => push(&mut out, Tok::Op(op), brackets.len()),
            }
            continue;
        }
        if c.is_ascii() {
            // `$`, `?`, `!` on its own: the parser has nothing to do with it
            push(&mut out, Tok::Op("?"), brackets.len());
            pos += 1;
            continue;
        }
        push(
            &mut out,
            Tok::Error(format!("invalid character '{c}' (U+{:04X})", c as u32)),
            brackets.len(),
        );
        return out;
    }
    // the end of the input with a bracket still open is where CPython says so
    match brackets.last() {
        Some(open) => push(
            &mut out,
            Tok::Error(format!("'{open}' was never closed")),
            brackets.len(),
        ),
        None => push(&mut out, Tok::End, 0),
    }
    out
}

/// A string prefix (`r`, `b`, `f`, `u` and their combinations) followed by a
/// quote: `(length, bytes, fstring)`.
fn string_prefix(chars: &[char], pos: usize) -> Option<(usize, bool, bool)> {
    let mut i = pos;
    while i < chars.len() && i - pos < 2 && matches!(chars[i].to_ascii_lowercase(), 'r' | 'b' | 'f' | 'u') {
        i += 1;
    }
    if i < chars.len() && (chars[i] == '\'' || chars[i] == '"') {
        let prefix: String = chars[pos..i].iter().map(|c| c.to_ascii_lowercase()).collect();
        let valid = matches!(prefix.as_str(), "" | "r" | "b" | "f" | "u" | "rb" | "br" | "fr" | "rf");
        if valid {
            return Some((i - pos, prefix.contains('b'), prefix.contains('f')));
        }
    }
    None
}

/// A quoted string, its common escapes decoded; `pos` is at the opening quote.
fn string(chars: &[char], pos: &mut usize) -> Result<String, String> {
    let quote = chars[*pos];
    let triple = chars.get(*pos + 1) == Some(&quote) && chars.get(*pos + 2) == Some(&quote);
    *pos += if triple { 3 } else { 1 };
    let mut value = String::new();
    loop {
        let Some(&c) = chars.get(*pos) else {
            return Err(if triple {
                "unterminated triple-quoted string literal (detected at line 1)".to_string()
            } else {
                "unterminated string literal (detected at line 1)".to_string()
            });
        };
        if c == quote {
            if !triple {
                *pos += 1;
                return Ok(value);
            }
            if chars.get(*pos + 1) == Some(&quote) && chars.get(*pos + 2) == Some(&quote) {
                *pos += 3;
                return Ok(value);
            }
        }
        if c == '\n' && !triple {
            return Err("unterminated string literal (detected at line 1)".to_string());
        }
        if c == '\\' {
            if let Some(&next) = chars.get(*pos + 1) {
                value.push(match next {
                    'n' => '\n',
                    't' => '\t',
                    'r' => '\r',
                    '0' => '\0',
                    other => other,
                });
                *pos += 2;
                continue;
            }
        }
        value.push(c);
        *pos += 1;
    }
}

/// A number literal, CPython's rules for underscores, prefixes and leading
/// zeros included; `pos` is at its first character.
fn number(chars: &[char], pos: &mut usize) -> Result<Tok, String> {
    let start = *pos;
    let at = |i: usize| chars.get(i).copied().unwrap_or('\0');
    // digits with single underscores between them
    fn decimal_tail(chars: &[char], pos: &mut usize) -> Result<(), String> {
        loop {
            while chars.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
                *pos += 1;
            }
            if chars.get(*pos) != Some(&'_') {
                return Ok(());
            }
            *pos += 1;
            if !chars.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
                return Err("invalid decimal literal".to_string());
            }
        }
    }
    let text_of = |from: usize, to: usize| -> String { chars[from..to].iter().filter(|c| **c != '_').collect() };
    let mut is_float = false;
    if at(*pos) == '0' && matches!(at(*pos + 1), 'x' | 'X' | 'o' | 'O' | 'b' | 'B') {
        let kind = at(*pos + 1).to_ascii_lowercase();
        *pos += 2;
        let (radix, name) = match kind {
            'x' => (16, "hexadecimal"),
            'o' => (8, "octal"),
            _ => (2, "binary"),
        };
        let digits_start = *pos;
        loop {
            if at(*pos) == '_' {
                *pos += 1;
            }
            let c = at(*pos);
            if !c.is_digit(radix) {
                if radix != 16 && c.is_ascii_digit() {
                    return Err(format!("invalid digit '{c}' in {name} literal"));
                }
                return Err(format!("invalid {name} literal"));
            }
            while at(*pos).is_digit(radix) {
                *pos += 1;
            }
            if at(*pos) != '_' {
                break;
            }
        }
        if radix != 16 && at(*pos).is_ascii_digit() {
            return Err(format!("invalid digit '{}' in {name} literal", at(*pos)));
        }
        verify_end(chars, *pos, name)?;
        let digits = text_of(digits_start, *pos);
        let value = u128::from_str_radix(&digits, radix)
            .map(|v| v as f64)
            .unwrap_or(f64::INFINITY);
        return Ok(Tok::Num(value));
    }
    if at(*pos) == '.' {
        // `.5`: straight to the fraction
        *pos += 1;
        decimal_tail(chars, pos)?;
        is_float = true;
    } else if at(*pos) == '0' {
        // zeros, then maybe the rest of a float; `09` is an old octal
        let mut nonzero = false;
        loop {
            if at(*pos) == '_' {
                *pos += 1;
                if !at(*pos).is_ascii_digit() {
                    return Err("invalid decimal literal".to_string());
                }
            }
            if at(*pos) != '0' {
                break;
            }
            *pos += 1;
        }
        if at(*pos).is_ascii_digit() {
            nonzero = true;
            decimal_tail(chars, pos)?;
        }
        if at(*pos) == '.' {
            *pos += 1;
            if at(*pos).is_ascii_digit() {
                decimal_tail(chars, pos)?;
            }
            is_float = true;
        } else if !matches!(at(*pos), 'e' | 'E' | 'j' | 'J') && nonzero {
            return Err(
                "leading zeros in decimal integer literals are not permitted; use an 0o prefix for octal integers"
                    .to_string(),
            );
        }
    } else {
        decimal_tail(chars, pos)?;
        if at(*pos) == '.' {
            *pos += 1;
            if at(*pos).is_ascii_digit() {
                decimal_tail(chars, pos)?;
            }
            is_float = true;
        }
    }
    if matches!(at(*pos), 'e' | 'E') {
        let e_at = *pos;
        *pos += 1;
        if matches!(at(*pos), '+' | '-') {
            *pos += 1;
            if !at(*pos).is_ascii_digit() {
                return Err("invalid decimal literal".to_string());
            }
        } else if !at(*pos).is_ascii_digit() {
            // `1e` followed by something that is not an exponent
            *pos = e_at;
            verify_end(chars, e_at, "decimal")?;
            return Ok(Tok::Num(literal_value(&text_of(start, e_at))));
        }
        decimal_tail(chars, pos)?;
        is_float = true;
    }
    let _ = is_float;
    let value = literal_value(&text_of(start, *pos));
    if matches!(at(*pos), 'j' | 'J') {
        *pos += 1;
        verify_end(chars, *pos, "imaginary")?;
        return Ok(Tok::Imag(value));
    }
    verify_end(chars, *pos, "decimal")?;
    Ok(Tok::Num(value))
}

/// The value of a decimal literal (underscores already gone), correctly
/// rounded as `float()` rounds it: `.5`, `5.` and `1.e5` are spelt the way
/// Rust's parser wants them first.
fn literal_value(text: &str) -> f64 {
    let mut spelt = String::with_capacity(text.len() + 2);
    let mut prev = '\0';
    for c in text.chars() {
        if prev == '.' && !c.is_ascii_digit() {
            spelt.push('0');
        }
        if c == '.' && !prev.is_ascii_digit() {
            spelt.push('0');
        }
        spelt.push(c);
        prev = c;
    }
    if prev == '.' {
        spelt.push('0');
    }
    spelt.parse().unwrap_or(0.0)
}

/// CPython's `verify_end_of_number`: a number glued to a name is an error,
/// unless the name is one of the keywords that can follow a number.
fn verify_end(chars: &[char], pos: usize, kind: &str) -> Result<(), String> {
    let Some(&c) = chars.get(pos) else { return Ok(()) };
    let rest: String = chars[pos..chars.len().min(pos + 5)].iter().collect();
    let keyword = ["and", "else", "for", "if", "in", "is", "or", "not"]
        .iter()
        .any(|k| rest.starts_with(k));
    if keyword {
        return Ok(());
    }
    if is_id_char(c) {
        return Err(format!("invalid {kind} literal"));
    }
    Ok(())
}

// -- the parser -----------------------------------------------------------------------------------

/// How a parse failed.
#[derive(Debug)]
enum Fail {
    /// A generic failure at a token: "invalid syntax", unless the tokens
    /// around it say something more specific.
    At(usize),
    /// A specific error, reported as it is.
    Say(String),
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
}

type Parsed = Result<Node, Fail>;

/// Parses an expression the way `ast.parse(text, mode="eval")` does.
fn parse(text: &str) -> Result<Node, String> {
    let toks = tokenize(text);
    let mut p = Parser { toks, pos: 0 };
    match p.top() {
        Ok(node) => Ok(node),
        Err(Fail::Say(msg)) => Err(msg),
        Err(Fail::At(at)) => Err(p.explain(at)),
    }
}

impl Parser {
    fn peek(&self) -> &Tok {
        &self.toks[self.pos.min(self.toks.len() - 1)].tok
    }

    fn peek_at(&self, k: usize) -> &Tok {
        &self.toks[(self.pos + k).min(self.toks.len() - 1)].tok
    }

    fn is_op(&self, op: &str) -> bool {
        matches!(self.peek(), Tok::Op(o) if *o == op)
    }

    fn is_kw(&self, kw: &str) -> bool {
        matches!(self.peek(), Tok::Name(n) if n == kw)
    }

    fn advance(&mut self) {
        if self.pos < self.toks.len() - 1 {
            self.pos += 1;
        }
    }

    /// Fails here: a tokenizer error the parser has reached is that error,
    /// anything else is a generic failure at this token.
    fn fail(&self) -> Fail {
        match self.peek() {
            Tok::Error(msg) => Fail::Say(msg.clone()),
            _ => Fail::At(self.pos),
        }
    }

    fn expect_op(&mut self, op: &str) -> Result<(), Fail> {
        if self.is_op(op) {
            self.advance();
            Ok(())
        } else {
            Err(self.fail())
        }
    }

    /// What a generic failure at token `at` is called.
    ///
    /// CPython's second pass looks for a better name than "invalid syntax":
    /// an expression that follows another inside brackets is a missing comma,
    /// and one that runs into the end of the input with a bracket open is the
    /// bracket's fault.  Past that, a tokenizer error anywhere in the input
    /// replaces the generic error (`_PyPegen_tokenize_full_source_to_check_for_errors`).
    fn explain(&mut self, at: usize) -> String {
        let starts_expression = match &self.toks[at].tok {
            Tok::Num(_) | Tok::Imag(_) | Tok::Str { .. } => true,
            Tok::Name(n) => {
                !KEYWORDS.contains(&n.as_str())
                    || matches!(n.as_str(), "True" | "False" | "None" | "not" | "lambda" | "await")
            }
            Tok::Op(o) => matches!(*o, "(" | "[" | "{" | "-" | "+" | "~" | "..."),
            _ => false,
        };
        let after_expression = at > 0
            && matches!(
                &self.toks[at - 1].tok,
                Tok::Num(_)
                    | Tok::Imag(_)
                    | Tok::Str { .. }
                    | Tok::Name(_)
                    | Tok::Op(")")
                    | Tok::Op("]")
                    | Tok::Op("}")
            );
        if starts_expression && after_expression {
            self.pos = at;
            match self.expression() {
                Err(Fail::Say(msg)) if msg.ends_with("was never closed") => return msg,
                Ok(_) => {
                    let last = self.pos.saturating_sub(1);
                    if self.toks[last].level > 0 {
                        return "invalid syntax. Perhaps you forgot a comma?".to_string();
                    }
                }
                _ => {}
            }
        }
        for t in &self.toks {
            if let Tok::Error(msg) = &t.tok {
                if !msg.ends_with("was never closed") {
                    return msg.clone();
                }
            }
        }
        "invalid syntax".to_string()
    }

    /// `eval: expressions NEWLINE* ENDMARKER`.
    fn top(&mut self) -> Parsed {
        let first = self.expression()?;
        let mut node = first;
        if self.is_op(",") {
            // `a, b` and `a,` are tuples
            while self.is_op(",") {
                self.advance();
                if self.at_end() {
                    break;
                }
                self.expression()?;
            }
            node = Node::Other("Tuple");
        }
        while matches!(self.peek(), Tok::Newline) {
            self.advance();
        }
        match self.peek() {
            Tok::End => Ok(node),
            _ => Err(self.fail()),
        }
    }

    fn at_end(&self) -> bool {
        matches!(self.peek(), Tok::End | Tok::Newline)
    }

    /// `expression: disjunction ['if' disjunction 'else' expression] | lambdef`.
    fn expression(&mut self) -> Parsed {
        if self.is_kw("lambda") {
            return self.lambda();
        }
        let body = self.disjunction()?;
        if self.is_kw("if") {
            self.advance();
            let test = self.disjunction()?;
            if !self.is_kw("else") {
                if let Tok::Error(msg) = self.peek() {
                    return Err(Fail::Say(msg.clone()));
                }
                return Err(Fail::Say("expected 'else' after 'if' expression".to_string()));
            }
            self.advance();
            let orelse = self.expression()?;
            return Ok(Node::IfExp(Box::new(test), Box::new(body), Box::new(orelse)));
        }
        Ok(body)
    }

    fn lambda(&mut self) -> Parsed {
        self.advance();
        // parameters: names, defaults, * and **, up to the colon
        loop {
            match self.peek() {
                Tok::Op(":") => break,
                Tok::Op(",") | Tok::Op("*") | Tok::Op("**") | Tok::Op("/") => self.advance(),
                Tok::Name(n) if !KEYWORDS.contains(&n.as_str()) => {
                    self.advance();
                    if self.is_op("=") {
                        self.advance();
                        self.expression()?;
                    }
                }
                _ => return Err(self.fail()),
            }
        }
        self.advance();
        self.expression()?;
        Ok(Node::Other("Lambda"))
    }

    fn disjunction(&mut self) -> Parsed {
        let first = self.conjunction()?;
        if !self.is_kw("or") {
            return Ok(first);
        }
        let mut values = vec![first];
        while self.is_kw("or") {
            self.advance();
            values.push(self.conjunction()?);
        }
        Ok(Node::BoolOp(false, values))
    }

    fn conjunction(&mut self) -> Parsed {
        let first = self.inversion()?;
        if !self.is_kw("and") {
            return Ok(first);
        }
        let mut values = vec![first];
        while self.is_kw("and") {
            self.advance();
            values.push(self.inversion()?);
        }
        Ok(Node::BoolOp(true, values))
    }

    fn inversion(&mut self) -> Parsed {
        if self.is_kw("not") {
            self.advance();
            let operand = self.inversion()?;
            return Ok(Node::Unary(Un::Not, Box::new(operand)));
        }
        self.comparison()
    }

    fn compare_op(&mut self) -> Option<Cmp> {
        let op = match self.peek() {
            Tok::Op("<") => Cmp::Lt,
            Tok::Op("<=") => Cmp::LtE,
            Tok::Op(">") => Cmp::Gt,
            Tok::Op(">=") => Cmp::GtE,
            Tok::Op("==") => Cmp::Eq,
            Tok::Op("!=") => Cmp::NotEq,
            Tok::Name(n) if n == "in" => Cmp::In,
            Tok::Name(n) if n == "is" => {
                if matches!(self.peek_at(1), Tok::Name(m) if m == "not") {
                    self.advance();
                    Cmp::IsNot
                } else {
                    Cmp::Is
                }
            }
            Tok::Name(n) if n == "not" && matches!(self.peek_at(1), Tok::Name(m) if m == "in") => {
                self.advance();
                Cmp::NotIn
            }
            _ => return None,
        };
        self.advance();
        Some(op)
    }

    fn comparison(&mut self) -> Parsed {
        let left = self.bitwise(0)?;
        let mut ops = Vec::new();
        let mut rest = Vec::new();
        while let Some(op) = self.compare_op() {
            ops.push(op);
            rest.push(self.bitwise(0)?);
        }
        if ops.is_empty() {
            return Ok(left);
        }
        Ok(Node::Compare(Box::new(left), ops, rest))
    }

    /// The binary operators below the comparisons, loosest first:
    /// `|`, `^`, `&`, shifts, `+ -`, `* / // % @`.
    fn bitwise(&mut self, level: usize) -> Parsed {
        const LEVELS: [&[(&str, Bin)]; 6] = [
            &[("|", Bin::BitOr)],
            &[("^", Bin::BitXor)],
            &[("&", Bin::BitAnd)],
            &[("<<", Bin::LShift), (">>", Bin::RShift)],
            &[("+", Bin::Add), ("-", Bin::Sub)],
            &[
                ("*", Bin::Mult),
                ("/", Bin::Div),
                ("//", Bin::FloorDiv),
                ("%", Bin::Mod),
                ("@", Bin::MatMult),
            ],
        ];
        if level == LEVELS.len() {
            return self.factor();
        }
        let mut left = self.bitwise(level + 1)?;
        loop {
            let found = LEVELS[level]
                .iter()
                .find(|(sym, _)| matches!(self.peek(), Tok::Op(o) if o == sym))
                .map(|(_, op)| *op);
            let Some(op) = found else { return Ok(left) };
            self.advance();
            let right = self.bitwise(level + 1)?;
            left = Node::BinOp(Box::new(left), op, Box::new(right));
        }
    }

    /// `factor: ('+' | '-' | '~') factor | power`.
    fn factor(&mut self) -> Parsed {
        let op = match self.peek() {
            Tok::Op("+") => Some(Un::Plus),
            Tok::Op("-") => Some(Un::Minus),
            Tok::Op("~") => Some(Un::Invert),
            _ => None,
        };
        if let Some(op) = op {
            self.advance();
            let operand = self.factor()?;
            return Ok(Node::Unary(op, Box::new(operand)));
        }
        self.power()
    }

    /// `power: await_primary ['**' factor]`.
    fn power(&mut self) -> Parsed {
        let base = if self.is_kw("await") {
            self.advance();
            self.primary()?;
            Node::Other("Await")
        } else {
            self.primary()?
        };
        if self.is_op("**") {
            self.advance();
            let exponent = self.factor()?;
            return Ok(Node::BinOp(Box::new(base), Bin::Pow, Box::new(exponent)));
        }
        Ok(base)
    }

    /// An atom and its trailers: calls, subscripts, attributes.
    fn primary(&mut self) -> Parsed {
        let mut node = self.atom()?;
        loop {
            if self.is_op("(") {
                self.advance();
                let (args, extra) = self.arguments()?;
                node = Node::Call(Box::new(node), args, extra);
            } else if self.is_op("[") {
                self.advance();
                self.skip_to_close("]")?;
                node = Node::Other("Subscript");
            } else if self.is_op(".") {
                self.advance();
                match self.peek() {
                    Tok::Name(n) if !KEYWORDS.contains(&n.as_str()) => self.advance(),
                    _ => return Err(self.fail()),
                }
                node = Node::Other("Attribute");
            } else {
                return Ok(node);
            }
        }
    }

    /// Call arguments after `(`: the positional ones, and whether any was a
    /// keyword or an unpacking.
    fn arguments(&mut self) -> Result<(Vec<Node>, bool), Fail> {
        let mut args = Vec::new();
        let mut extra = false;
        loop {
            if self.is_op(")") {
                self.advance();
                return Ok((args, extra));
            }
            if self.is_op("*") || self.is_op("**") {
                self.advance();
                self.expression()?;
                extra = true;
            } else if matches!(self.peek(), Tok::Name(n) if !KEYWORDS.contains(&n.as_str()))
                && matches!(self.peek_at(1), Tok::Op("="))
            {
                self.advance();
                self.advance();
                self.expression()?;
                extra = true;
            } else {
                let arg = self.expression()?;
                if matches!(self.peek(), Tok::Name(n) if n == "for") {
                    // a generator argument
                    self.skip_to_close(")")?;
                    args.push(Node::Other("GeneratorExp"));
                    return Ok((args, extra));
                }
                args.push(arg);
            }
            if self.is_op(",") {
                self.advance();
                continue;
            }
            if self.is_op(")") {
                continue;
            }
            return Err(self.fail());
        }
    }

    /// Skips balanced tokens up to (and past) the closer of the bracket just
    /// opened - the inside of a subscript or a comprehension, which is refused
    /// whatever it holds.
    fn skip_to_close(&mut self, close: &str) -> Result<(), Fail> {
        let mut depth = 0usize;
        loop {
            match self.peek() {
                Tok::Error(msg) => return Err(Fail::Say(msg.clone())),
                Tok::End => return Err(self.fail()),
                Tok::Op(o) if matches!(*o, "(" | "[" | "{") => depth += 1,
                Tok::Op(o) if *o == close && depth == 0 => {
                    self.advance();
                    return Ok(());
                }
                Tok::Op(o) if matches!(*o, ")" | "]" | "}") => depth = depth.saturating_sub(1),
                _ => {}
            }
            self.advance();
        }
    }

    fn atom(&mut self) -> Parsed {
        let tok = self.peek().clone();
        match tok {
            Tok::Num(x) => {
                self.advance();
                Ok(Node::Const(Const::Num(x)))
            }
            Tok::Imag(x) => {
                self.advance();
                Ok(Node::Const(Const::Imag(x)))
            }
            Tok::Str { .. } => {
                // adjacent strings concatenate; an f-string makes the whole a JoinedStr
                let mut value = String::new();
                let (mut bytes, mut fstring) = (false, false);
                while let Tok::Str {
                    value: v,
                    bytes: b,
                    fstring: f,
                } = self.peek().clone()
                {
                    value.push_str(&v);
                    bytes |= b;
                    fstring |= f;
                    self.advance();
                }
                if fstring {
                    return Ok(Node::Other("JoinedStr"));
                }
                Ok(Node::Const(if bytes { Const::Bytes(value) } else { Const::Str(value) }))
            }
            Tok::Name(name) => match name.as_str() {
                "True" => {
                    self.advance();
                    Ok(Node::Const(Const::True))
                }
                "False" => {
                    self.advance();
                    Ok(Node::Const(Const::False))
                }
                "None" => {
                    self.advance();
                    Ok(Node::Const(Const::None))
                }
                n if KEYWORDS.contains(&n) => Err(self.fail()),
                _ => {
                    self.advance();
                    Ok(Node::Name(name))
                }
            },
            Tok::Op("...") => {
                self.advance();
                Ok(Node::Const(Const::Ellipsis))
            }
            Tok::Op("(") => {
                self.advance();
                self.group()
            }
            Tok::Op("[") => {
                self.advance();
                self.display("]", "List", "ListComp")
            }
            Tok::Op("{") => {
                self.advance();
                self.braces()
            }
            _ => Err(self.fail()),
        }
    }

    /// After `(`: a parenthesised expression, a tuple, `yield`, a walrus or a
    /// generator.
    fn group(&mut self) -> Parsed {
        if self.is_op(")") {
            self.advance();
            return Ok(Node::Other("Tuple"));
        }
        if self.is_kw("yield") {
            self.skip_to_close(")")?;
            return Ok(Node::Other("Yield"));
        }
        if matches!(self.peek(), Tok::Name(n) if !KEYWORDS.contains(&n.as_str()))
            && matches!(self.peek_at(1), Tok::Op(":="))
        {
            self.advance();
            self.advance();
            self.expression()?;
            self.expect_op(")")?;
            return Ok(Node::Other("NamedExpr"));
        }
        let first = if self.is_op("*") {
            self.advance();
            self.expression()?;
            Node::Other("Starred")
        } else {
            self.expression()?
        };
        if self.is_kw("for") {
            self.skip_to_close(")")?;
            return Ok(Node::Other("GeneratorExp"));
        }
        if self.is_op(")") {
            self.advance();
            return Ok(first);
        }
        if !self.is_op(",") {
            return Err(self.fail());
        }
        while self.is_op(",") {
            self.advance();
            if self.is_op(")") {
                break;
            }
            self.expression()?;
        }
        self.expect_op(")")?;
        Ok(Node::Other("Tuple"))
    }

    /// A bracketed display (`[...]`): its elements or a comprehension.
    fn display(&mut self, close: &str, kind: &'static str, comp: &'static str) -> Parsed {
        if self.is_op(close) {
            self.advance();
            return Ok(Node::Other(kind));
        }
        self.expression()?;
        if self.is_kw("for") {
            self.skip_to_close(close)?;
            return Ok(Node::Other(comp));
        }
        loop {
            if self.is_op(close) {
                self.advance();
                return Ok(Node::Other(kind));
            }
            self.expect_op(",")?;
            if self.is_op(close) {
                continue;
            }
            self.expression()?;
        }
    }

    /// After `{`: a dict, a set, or a comprehension of either.
    fn braces(&mut self) -> Parsed {
        if self.is_op("}") {
            self.advance();
            return Ok(Node::Other("Dict"));
        }
        self.expression()?;
        if self.is_op(":") {
            self.advance();
            self.expression()?;
            if self.is_kw("for") {
                self.skip_to_close("}")?;
                return Ok(Node::Other("DictComp"));
            }
            loop {
                if self.is_op("}") {
                    self.advance();
                    return Ok(Node::Other("Dict"));
                }
                self.expect_op(",")?;
                if self.is_op("}") {
                    continue;
                }
                self.expression()?;
                self.expect_op(":")?;
                self.expression()?;
            }
        }
        if self.is_kw("for") {
            self.skip_to_close("}")?;
            return Ok(Node::Other("SetComp"));
        }
        loop {
            if self.is_op("}") {
                self.advance();
                return Ok(Node::Other("Set"));
            }
            self.expect_op(",")?;
            if self.is_op("}") {
                continue;
            }
            self.expression()?;
        }
    }
}

// -- values: Python's numeric tower ---------------------------------------------------------------

/// The functions a schedule can call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fun {
    Sin,
    Cos,
    Tan,
    Exp,
    Log,
    Log2,
    Log10,
    Sqrt,
    Pow,
    Abs,
    Floor,
    Ceil,
    Round,
    Min,
    Max,
    Tanh,
    Clamp,
    Linear,
    Geometric,
    Cosine,
    Step,
    Warmup,
}

impl Fun {
    fn named(name: &str) -> Option<Fun> {
        Some(match name {
            "sin" => Fun::Sin,
            "cos" => Fun::Cos,
            "tan" => Fun::Tan,
            "exp" => Fun::Exp,
            "log" => Fun::Log,
            "log2" => Fun::Log2,
            "log10" => Fun::Log10,
            "sqrt" => Fun::Sqrt,
            "pow" => Fun::Pow,
            "abs" => Fun::Abs,
            "floor" => Fun::Floor,
            "ceil" => Fun::Ceil,
            "round" => Fun::Round,
            "min" => Fun::Min,
            "max" => Fun::Max,
            "tanh" => Fun::Tanh,
            "clamp" => Fun::Clamp,
            "linear" => Fun::Linear,
            "geometric" => Fun::Geometric,
            "cosine" => Fun::Cosine,
            "step" => Fun::Step,
            "warmup" => Fun::Warmup,
            _ => return None,
        })
    }

    /// Python's type name for it: the `math` functions and the builtins are
    /// C functions, `clamp` and the helpers are lambdas.
    fn type_name(self) -> &'static str {
        match self {
            Fun::Clamp | Fun::Linear | Fun::Geometric | Fun::Cosine | Fun::Step | Fun::Warmup => "function",
            _ => "builtin_function_or_method",
        }
    }
}

/// A Python value an expression can produce.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Val {
    Bool(bool),
    Int(i128),
    Float(f64),
    Complex(f64, f64),
    Func(Fun),
}

impl Val {
    fn type_name(self) -> &'static str {
        match self {
            Val::Bool(_) => "bool",
            Val::Int(_) => "int",
            Val::Float(_) => "float",
            Val::Complex(..) => "complex",
            Val::Func(f) => f.type_name(),
        }
    }

    fn truthy(self) -> bool {
        match self {
            Val::Bool(b) => b,
            Val::Int(n) => n != 0,
            Val::Float(x) => x != 0.0,
            Val::Complex(re, im) => re != 0.0 || im != 0.0,
            Val::Func(_) => true,
        }
    }

    /// `float(x)` for a real number (`PyFloat_AsDouble`), else the `TypeError`.
    fn real(self) -> Result<f64, String> {
        match self {
            Val::Bool(b) => Ok(b as i64 as f64),
            Val::Int(n) => Ok(n as f64),
            Val::Float(x) => Ok(x),
            other => Err(format!("must be real number, not {}", other.type_name())),
        }
    }

    /// Booleans take part in arithmetic as the integers they are.
    fn numeric(self) -> Val {
        match self {
            Val::Bool(b) => Val::Int(b as i128),
            other => other,
        }
    }

    fn rank(self) -> u8 {
        match self {
            Val::Bool(_) | Val::Int(_) => 1,
            Val::Float(_) => 2,
            Val::Complex(..) => 3,
            Val::Func(_) => 4,
        }
    }

    fn as_complex(self) -> (f64, f64) {
        match self {
            Val::Complex(re, im) => (re, im),
            Val::Bool(b) => (b as i64 as f64, 0.0),
            Val::Int(n) => (n as f64, 0.0),
            Val::Float(x) => (x, 0.0),
            Val::Func(_) => (f64::NAN, f64::NAN),
        }
    }
}

/// `int / int` as Python divides: the correctly rounded quotient.
fn true_div_int(a: i128, b: i128) -> f64 {
    a as f64 / b as f64
}

/// What an expression is evaluated against.
struct Env {
    epoch: f64,
    /// `i` is an `int` inside the helpers (`i = epoch - 1`) and a float as a
    /// variable (`float(i)`), exactly as `_environment` builds it.
    i: i128,
    epochs: f64,
    t: f64,
    lr0: f64,
    act_lr0: f64,
    lr: f64,
}

fn eval(node: &Node, env: &Env) -> Result<Val, String> {
    match node {
        Node::Const(Const::Num(x)) => Ok(Val::Float(*x)),
        Node::Const(_) | Node::Other(_) | Node::Op(_) => Err("a refused node survived the whitelist".to_string()),
        Node::Name(name) => Ok(match name.as_str() {
            "epoch" => Val::Float(env.epoch),
            "i" => Val::Float(env.i as f64),
            "epochs" => Val::Float(env.epochs),
            "t" => Val::Float(env.t),
            "lr0" => Val::Float(env.lr0),
            "act_lr0" => Val::Float(env.act_lr0),
            "lr" => Val::Float(env.lr),
            "pi" => Val::Float(PI),
            "e" => Val::Float(E),
            other => Val::Func(Fun::named(other).ok_or_else(|| format!("name {other:?} is not defined"))?),
        }),
        Node::BinOp(left, op, right) => {
            let a = eval(left, env)?;
            let b = eval(right, env)?;
            binop(*op, a, b)
        }
        Node::Unary(op, operand) => {
            let v = eval(operand, env)?;
            match op {
                Un::Not => Ok(Val::Bool(!v.truthy())),
                Un::Plus => match v.numeric() {
                    Val::Func(f) => Err(format!("bad operand type for unary +: '{}'", f.type_name())),
                    other => Ok(other),
                },
                Un::Minus => match v.numeric() {
                    Val::Int(n) => Ok(n.checked_neg().map(Val::Int).unwrap_or(Val::Float(-(n as f64)))),
                    Val::Float(x) => Ok(Val::Float(-x)),
                    Val::Complex(re, im) => Ok(Val::Complex(-re, -im)),
                    other => Err(format!("bad operand type for unary -: '{}'", other.type_name())),
                },
                Un::Invert => Err("operator Invert is not allowed".to_string()),
            }
        }
        Node::Compare(left, ops, rest) => {
            let mut a = eval(left, env)?;
            let mut result = Val::Bool(true);
            for (op, right) in ops.iter().zip(rest) {
                let b = eval(right, env)?;
                let holds = compare(*op, a, b)?;
                result = Val::Bool(holds);
                if !holds {
                    return Ok(result);
                }
                a = b;
            }
            Ok(result)
        }
        Node::BoolOp(and, values) => {
            let mut last = Val::Bool(false);
            for (k, value) in values.iter().enumerate() {
                last = eval(value, env)?;
                let done = if *and { !last.truthy() } else { last.truthy() };
                if done || k + 1 == values.len() {
                    return Ok(last);
                }
            }
            Ok(last)
        }
        Node::IfExp(test, body, orelse) => {
            if eval(test, env)?.truthy() {
                eval(body, env)
            } else {
                eval(orelse, env)
            }
        }
        Node::Call(func, args, _) => {
            let f = match eval(func, env)? {
                Val::Func(f) => f,
                other => return Err(format!("'{}' object is not callable", other.type_name())),
            };
            let values: Vec<Val> = args.iter().map(|a| eval(a, env)).collect::<Result<_, _>>()?;
            call(f, &values, env)
        }
    }
}

// -- arithmetic -----------------------------------------------------------------------------------

fn binop(op: Bin, a: Val, b: Val) -> Result<Val, String> {
    let unsupported = || {
        format!(
            "unsupported operand type(s) for {}: '{}' and '{}'",
            op.symbol(),
            a.type_name(),
            b.type_name()
        )
    };
    if matches!(a, Val::Func(_)) || matches!(b, Val::Func(_)) {
        return Err(unsupported());
    }
    let (x, y) = (a.numeric(), b.numeric());
    match x.rank().max(y.rank()) {
        1 => {
            let (Val::Int(p), Val::Int(q)) = (x, y) else {
                unreachable!("both are integers")
            };
            int_op(op, p, q)
        }
        2 => float_op(op, x.real()?, y.real()?),
        _ => {
            if matches!(op, Bin::FloorDiv | Bin::Mod) {
                return Err(unsupported());
            }
            complex_op(op, x.as_complex(), y.as_complex())
        }
    }
}

fn int_op(op: Bin, a: i128, b: i128) -> Result<Val, String> {
    let fallback = |f: fn(f64, f64) -> f64| Val::Float(f(a as f64, b as f64));
    Ok(match op {
        Bin::Add => a.checked_add(b).map(Val::Int).unwrap_or_else(|| fallback(|x, y| x + y)),
        Bin::Sub => a.checked_sub(b).map(Val::Int).unwrap_or_else(|| fallback(|x, y| x - y)),
        Bin::Mult => a.checked_mul(b).map(Val::Int).unwrap_or_else(|| fallback(|x, y| x * y)),
        Bin::Div => {
            if b == 0 {
                return Err("division by zero".to_string());
            }
            Val::Float(true_div_int(a, b))
        }
        Bin::FloorDiv => {
            if b == 0 {
                return Err("integer division or modulo by zero".to_string());
            }
            let mut q = a / b;
            if a % b != 0 && ((a < 0) != (b < 0)) {
                q -= 1;
            }
            Val::Int(q)
        }
        Bin::Mod => {
            if b == 0 {
                return Err("integer modulo by zero".to_string());
            }
            let mut r = a % b;
            if r != 0 && ((r < 0) != (b < 0)) {
                r += b;
            }
            Val::Int(r)
        }
        Bin::Pow => {
            if b < 0 {
                // a negative exponent is a float power (`long_pow`)
                return float_pow(a as f64, b as f64);
            }
            match u32::try_from(b).ok().and_then(|e| a.checked_pow(e)) {
                Some(v) => Val::Int(v),
                None => float_pow(a as f64, b as f64)?,
            }
        }
        _ => return Err(format!("operator {} is not allowed", op.name())),
    })
}

fn float_op(op: Bin, a: f64, b: f64) -> Result<Val, String> {
    Ok(Val::Float(match op {
        Bin::Add => a + b,
        Bin::Sub => a - b,
        Bin::Mult => a * b,
        Bin::Div => {
            if b == 0.0 {
                return Err("float division by zero".to_string());
            }
            a / b
        }
        Bin::FloorDiv => {
            if b == 0.0 {
                return Err("float floor division by zero".to_string());
            }
            float_divmod(a, b).0
        }
        Bin::Mod => {
            if b == 0.0 {
                return Err("float modulo".to_string());
            }
            float_divmod(a, b).1
        }
        Bin::Pow => return float_pow(a, b),
        _ => return Err(format!("operator {} is not allowed", op.name())),
    }))
}

/// CPython's `_float_div_mod`: floor division and modulo with the sign of the
/// divisor, exactly as `float_floor_div` and `float_rem` compute them.
fn float_divmod(vx: f64, wx: f64) -> (f64, f64) {
    let mut modulo = vx % wx; // C fmod
    let mut div = (vx - modulo) / wx;
    if modulo != 0.0 {
        if (wx < 0.0) != (modulo < 0.0) {
            modulo += wx;
            div -= 1.0;
        }
    } else {
        modulo = 0.0f64.copysign(wx);
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
    (floordiv, modulo)
}

fn is_odd_integer(x: f64) -> bool {
    x.is_finite() && (x % 2.0).abs() == 1.0
}

/// `float.__pow__`: CPython's special cases, then the platform `pow`; a
/// negative base to a fractional power is a complex number, as in Python.
fn float_pow(iv: f64, iw: f64) -> Result<Val, String> {
    if iw == 0.0 {
        return Ok(Val::Float(1.0));
    }
    if iv.is_nan() {
        return Ok(Val::Float(iv));
    }
    if iw.is_nan() {
        return Ok(Val::Float(if iv == 1.0 { 1.0 } else { iw }));
    }
    if iw.is_infinite() {
        let v = iv.abs();
        if v == 1.0 {
            return Ok(Val::Float(1.0));
        }
        if (iw > 0.0) == (v > 1.0) {
            return Ok(Val::Float(iw.abs()));
        }
        return Ok(Val::Float(0.0));
    }
    if iv.is_infinite() {
        let odd = is_odd_integer(iw);
        if iw > 0.0 {
            return Ok(Val::Float(if odd { iv } else { iv.abs() }));
        }
        return Ok(Val::Float(if odd { 0.0f64.copysign(iv) } else { 0.0 }));
    }
    if iv == 0.0 {
        let odd = is_odd_integer(iw);
        if iw < 0.0 {
            return Err("0.0 cannot be raised to a negative power".to_string());
        }
        return Ok(Val::Float(if odd { iv } else { 0.0 }));
    }
    let mut base = iv;
    let mut negate = false;
    if iv < 0.0 {
        if iw != iw.floor() {
            return complex_pow((iv, 0.0), (iw, 0.0));
        }
        base = -iv;
        negate = is_odd_integer(iw);
    }
    if base == 1.0 {
        return Ok(Val::Float(if negate { -1.0 } else { 1.0 }));
    }
    let mut ix = base.powf(iw);
    if ix.is_infinite() {
        return Err("(34, 'Numerical result out of range')".to_string());
    }
    if negate {
        ix = -ix;
    }
    Ok(Val::Float(ix))
}

fn c_prod(a: (f64, f64), b: (f64, f64)) -> (f64, f64) {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

/// `_Py_c_quot`: Smith's division, as CPython divides complex numbers.
fn c_quot(a: (f64, f64), b: (f64, f64)) -> Result<(f64, f64), String> {
    let (abs_re, abs_im) = (b.0.abs(), b.1.abs());
    if abs_re >= abs_im {
        if abs_re == 0.0 {
            return Err("complex division by zero".to_string());
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

fn complex_op(op: Bin, a: (f64, f64), b: (f64, f64)) -> Result<Val, String> {
    let (re, im) = match op {
        Bin::Add => (a.0 + b.0, a.1 + b.1),
        Bin::Sub => (a.0 - b.0, a.1 - b.1),
        Bin::Mult => c_prod(a, b),
        Bin::Div => c_quot(a, b)?,
        Bin::Pow => return complex_pow(a, b),
        _ => return Err(format!("operator {} is not allowed", op.name())),
    };
    Ok(Val::Complex(re, im))
}

/// `complex.__pow__`: repeated squaring for a small integer exponent, the
/// polar form otherwise.
fn complex_pow(a: (f64, f64), b: (f64, f64)) -> Result<Val, String> {
    let (re, im) = if b.1 == 0.0 && b.0 == b.0.floor() && b.0.abs() <= 100.0 {
        let n = b.0 as i64;
        let powu = |x: (f64, f64), n: i64| {
            let mut r = (1.0, 0.0);
            let mut p = x;
            let mut mask: i64 = 1;
            while mask > 0 && n >= mask {
                if n & mask != 0 {
                    r = c_prod(r, p);
                }
                mask <<= 1;
                p = c_prod(p, p);
            }
            r
        };
        if n > 0 {
            powu(a, n)
        } else {
            c_quot((1.0, 0.0), powu(a, -n)).map_err(|_| "0.0 to a negative or complex power".to_string())?
        }
    } else if b.0 == 0.0 && b.1 == 0.0 {
        (1.0, 0.0)
    } else if a.0 == 0.0 && a.1 == 0.0 {
        if b.1 != 0.0 || b.0 < 0.0 {
            return Err("0.0 to a negative or complex power".to_string());
        }
        (0.0, 0.0)
    } else {
        let vabs = a.0.hypot(a.1);
        let mut len = vabs.powf(b.0);
        let at = a.1.atan2(a.0);
        let mut phase = at * b.0;
        if b.1 != 0.0 {
            len /= (at * b.1).exp();
            phase += b.1 * vabs.ln();
        }
        (len * phase.cos(), len * phase.sin())
    };
    if re.is_infinite() || im.is_infinite() {
        return Err("complex exponentiation".to_string());
    }
    Ok(Val::Complex(re, im))
}

/// An int and a float compared exactly, as Python compares them.
fn cmp_int_float(i: i128, f: f64) -> Option<std::cmp::Ordering> {
    use std::cmp::Ordering;
    if f.is_nan() {
        return None;
    }
    if f == f64::INFINITY || f >= 1.7014118346046923e38 {
        return Some(Ordering::Less);
    }
    if f == f64::NEG_INFINITY || f < -1.7014118346046923e38 {
        return Some(Ordering::Greater);
    }
    let fl = f.floor();
    let fi = fl as i128;
    Some(match i.cmp(&fi) {
        Ordering::Equal if f > fl => Ordering::Less,
        other => other,
    })
}

fn compare(op: Cmp, a: Val, b: Val) -> Result<bool, String> {
    use std::cmp::Ordering;
    let (x, y) = (a.numeric(), b.numeric());
    let order: Option<Ordering> = match (x, y) {
        (Val::Func(f), Val::Func(g)) if matches!(op, Cmp::Eq | Cmp::NotEq) => {
            return Ok((f == g) == (op == Cmp::Eq));
        }
        (Val::Func(_), _) | (_, Val::Func(_)) | (Val::Complex(..), _) | (_, Val::Complex(..)) => {
            if matches!(op, Cmp::Eq | Cmp::NotEq) {
                let equal = match (x, y) {
                    (Val::Func(_), _) | (_, Val::Func(_)) => false,
                    _ => x.as_complex() == y.as_complex(),
                };
                return Ok(equal == (op == Cmp::Eq));
            }
            return Err(format!(
                "'{}' not supported between instances of '{}' and '{}'",
                op.symbol(),
                a.type_name(),
                b.type_name()
            ));
        }
        (Val::Int(p), Val::Int(q)) => Some(p.cmp(&q)),
        (Val::Int(p), Val::Float(q)) => cmp_int_float(p, q),
        (Val::Float(p), Val::Int(q)) => cmp_int_float(q, p).map(Ordering::reverse),
        (Val::Float(p), Val::Float(q)) => p.partial_cmp(&q),
        _ => None,
    };
    Ok(match op {
        Cmp::Lt => order == Some(Ordering::Less),
        Cmp::LtE => matches!(order, Some(Ordering::Less | Ordering::Equal)),
        Cmp::Gt => order == Some(Ordering::Greater),
        Cmp::GtE => matches!(order, Some(Ordering::Greater | Ordering::Equal)),
        Cmp::Eq => order == Some(Ordering::Equal),
        Cmp::NotEq => order != Some(Ordering::Equal),
        _ => return Err("only <, <=, >, >=, == and != comparisons are allowed".to_string()),
    })
}

// -- the functions --------------------------------------------------------------------------------

/// `math_1`: a one-argument `math` function with CPython's error rules.
fn math_1(values: &[Val], name: &str, f: fn(f64) -> f64, can_overflow: bool) -> Result<Val, String> {
    if values.len() != 1 {
        return Err(format!(
            "math.{name}() takes exactly one argument ({} given)",
            values.len()
        ));
    }
    let x = values[0].real()?;
    let r = f(x);
    if r.is_nan() && !x.is_nan() {
        return Err("math domain error".to_string());
    }
    if r.is_infinite() && x.is_finite() {
        return Err(if can_overflow {
            "math range error"
        } else {
            "math domain error"
        }
        .to_string());
    }
    Ok(Val::Float(r))
}

/// `loghelper`: the logarithm of a real number, a domain error for anything
/// that is not positive.
fn log_of(v: Val, f: fn(f64) -> f64) -> Result<f64, String> {
    let x = v.real()?;
    if x.is_nan() {
        return Ok(x);
    }
    if x <= 0.0 {
        return Err("math domain error".to_string());
    }
    Ok(f(x))
}

/// `PyLong_FromDouble`: a float as the integer `floor` / `ceil` / `round` hand back.
fn to_int(x: f64) -> Result<Val, String> {
    if x.is_nan() {
        return Err("cannot convert float NaN to integer".to_string());
    }
    if x.is_infinite() {
        return Err("cannot convert float infinity to integer".to_string());
    }
    if x.abs() < 1.7014118346046923e38 {
        Ok(Val::Int(x as i128))
    } else {
        Ok(Val::Float(x))
    }
}

fn count_error(name: &str, params: &[&str], got: usize, qualname: &str) -> String {
    if got < params.len() {
        let missing: Vec<String> = params[got..].iter().map(|p| format!("'{p}'")).collect();
        let n = missing.len();
        let list = match n {
            1 => missing[0].clone(),
            2 => format!("{} and {}", missing[0], missing[1]),
            _ => format!("{}, and {}", missing[..n - 1].join(", "), missing[n - 1]),
        };
        return format!(
            "{qualname}() missing {n} required positional argument{}: {list}",
            if n == 1 { "" } else { "s" }
        );
    }
    let _ = name;
    format!(
        "{qualname}() takes {} positional argument{} but {got} {} given",
        params.len(),
        if params.len() == 1 { "" } else { "s" },
        if got == 1 { "was" } else { "were" }
    )
}

/// The builtin `min` / `max` over two or more values: the first extreme one,
/// compared with `<` (min) or `>` (max) as CPython's `min_max` does.
fn min_max(values: &[Val], name: &str, is_max: bool) -> Result<Val, String> {
    match values.len() {
        0 => return Err(format!("{name} expected at least 1 argument, got 0")),
        1 => return Err(format!("'{}' object is not iterable", values[0].type_name())),
        _ => {}
    }
    let mut best = values[0];
    for &item in &values[1..] {
        let better = if is_max {
            compare(Cmp::Gt, item, best)?
        } else {
            compare(Cmp::Lt, item, best)?
        };
        if better {
            best = item;
        }
    }
    Ok(best)
}

/// Python's `round(x, ndigits)` for a float: the double nearest the decimal
/// string correctly rounded to `ndigits` places (ties to even), as
/// `double_round` computes it through `dtoa`.
fn round_float(x: f64, ndigits: i128) -> Result<f64, String> {
    if !x.is_finite() || ndigits > 323 {
        return Ok(x);
    }
    if ndigits < -308 {
        return Ok(0.0 * x);
    }
    if ndigits >= 0 {
        let text = format!("{:.*}", ndigits as usize, x);
        return Ok(text.parse().unwrap_or(x));
    }
    // to a power of ten left of the point: round the significant digits
    let places = (-ndigits) as i32;
    let exp10: i32 = format!("{:e}", x.abs())
        .split_once('e')
        .and_then(|(_, e)| e.parse().ok())
        .unwrap_or(0);
    let significant = exp10 + 1 - places;
    if significant > 0 {
        let text = format!("{:.*e}", (significant - 1) as usize, x);
        let rounded: f64 = text.parse().unwrap_or(x);
        if rounded.is_infinite() {
            return Err("rounded value too large to represent".to_string());
        }
        return Ok(rounded);
    }
    if significant == 0 {
        let half = 0.5 * 10f64.powi(exp10 + 1);
        if x.abs() > half {
            return Ok(10f64.powi(exp10 + 1).copysign(x));
        }
    }
    Ok(0.0 * x)
}

fn call(f: Fun, values: &[Val], env: &Env) -> Result<Val, String> {
    match f {
        Fun::Sin => math_1(values, "sin", f64::sin, false),
        Fun::Cos => math_1(values, "cos", f64::cos, false),
        Fun::Tan => math_1(values, "tan", f64::tan, false),
        Fun::Exp => math_1(values, "exp", f64::exp, true),
        Fun::Sqrt => math_1(values, "sqrt", f64::sqrt, false),
        Fun::Tanh => math_1(values, "tanh", f64::tanh, false),
        Fun::Log => {
            if values.is_empty() || values.len() > 2 {
                return Err("math.log requires 1 to 2 arguments".to_string());
            }
            let num = log_of(values[0], f64::ln)?;
            if values.len() == 1 {
                return Ok(Val::Float(num));
            }
            let den = log_of(values[1], f64::ln)?;
            if den == 0.0 {
                return Err("float division by zero".to_string());
            }
            Ok(Val::Float(num / den))
        }
        Fun::Log2 | Fun::Log10 => {
            let name = if f == Fun::Log2 { "log2" } else { "log10" };
            if values.len() != 1 {
                return Err(format!(
                    "math.{name}() takes exactly one argument ({} given)",
                    values.len()
                ));
            }
            let func: fn(f64) -> f64 = if f == Fun::Log2 { f64::log2 } else { f64::log10 };
            Ok(Val::Float(log_of(values[0], func)?))
        }
        Fun::Pow => {
            if values.len() != 2 {
                return Err(format!("pow expected 2 arguments, got {}", values.len()));
            }
            let (x, y) = (values[0].real()?, values[1].real()?);
            math_pow(x, y).map(Val::Float)
        }
        Fun::Abs => {
            if values.len() != 1 {
                return Err(format!("abs() takes exactly one argument ({} given)", values.len()));
            }
            match values[0].numeric() {
                Val::Int(n) => Ok(n.checked_abs().map(Val::Int).unwrap_or(Val::Float((n as f64).abs()))),
                Val::Float(x) => Ok(Val::Float(x.abs())),
                Val::Complex(re, im) => Ok(Val::Float(re.hypot(im))),
                other => Err(format!("bad operand type for abs(): '{}'", other.type_name())),
            }
        }
        Fun::Floor | Fun::Ceil => {
            let name = if f == Fun::Floor { "floor" } else { "ceil" };
            if values.len() != 1 {
                return Err(format!(
                    "math.{name}() takes exactly one argument ({} given)",
                    values.len()
                ));
            }
            match values[0].numeric() {
                Val::Int(n) => Ok(Val::Int(n)),
                other => {
                    let x = other.real()?;
                    to_int(if f == Fun::Floor { x.floor() } else { x.ceil() })
                }
            }
        }
        Fun::Round => {
            match values.len() {
                0 => return Err("round() missing required argument 'number' (pos 1)".to_string()),
                1 | 2 => {}
                n => return Err(format!("round() takes at most 2 arguments ({n} given)")),
            }
            let x = values[0].numeric();
            if let Val::Complex(..) | Val::Func(_) = x {
                return Err(format!("type {} doesn't define __round__ method", x.type_name()));
            }
            if values.len() == 1 {
                return match x {
                    Val::Int(n) => Ok(Val::Int(n)),
                    Val::Float(v) => to_int(v.round_ties_even()),
                    _ => unreachable!("a real number"),
                };
            }
            let nd = match values[1].numeric() {
                Val::Int(n) => n,
                other => {
                    return Err(format!(
                        "'{}' object cannot be interpreted as an integer",
                        other.type_name()
                    ))
                }
            };
            match x {
                Val::Int(n) => {
                    if nd >= 0 {
                        return Ok(Val::Int(n));
                    }
                    let Some(pow) = 10i128.checked_pow((-nd) as u32) else {
                        return Ok(Val::Int(0));
                    };
                    let (mut q, r) = (n.div_euclid(pow), n.rem_euclid(pow));
                    if 2 * r > pow || (2 * r == pow && q % 2 != 0) {
                        q += 1;
                    }
                    Ok(Val::Int(q * pow))
                }
                Val::Float(v) => round_float(v, nd).map(Val::Float),
                _ => unreachable!("a real number"),
            }
        }
        Fun::Min => min_max(values, "min", false),
        Fun::Max => min_max(values, "max", true),
        Fun::Clamp => {
            if values.len() != 3 {
                return Err(count_error("clamp", &["x", "lo", "hi"], values.len(), "<lambda>"));
            }
            let inner = min_max(&[values[2], values[0]], "min", false)?;
            min_max(&[values[1], inner], "max", true)
        }
        Fun::Linear | Fun::Geometric | Fun::Cosine | Fun::Step | Fun::Warmup => helper(f, values, env),
    }
}

/// `math.pow`: always a float, and CPython's errors.
fn math_pow(x: f64, y: f64) -> Result<f64, String> {
    if !x.is_finite() || !y.is_finite() {
        let r = if x.is_nan() {
            if y == 0.0 {
                1.0
            } else {
                x
            }
        } else if y.is_nan() {
            if x == 1.0 {
                1.0
            } else {
                y
            }
        } else if x.is_infinite() {
            let odd = y.is_finite() && (y.abs() % 2.0) == 1.0;
            if y > 0.0 {
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
            }
        } else if x.abs() == 1.0 {
            1.0
        } else if (y > 0.0 && x.abs() > 1.0) || (y < 0.0 && x.abs() < 1.0) {
            f64::INFINITY
        } else {
            0.0
        };
        return Ok(r);
    }
    let r = x.powf(y);
    if r.is_nan() {
        return Err("math domain error".to_string());
    }
    if r.is_infinite() {
        return Err(if x == 0.0 {
            "math domain error"
        } else {
            "math range error"
        }
        .to_string());
    }
    Ok(r)
}

/// The five helpers, as the lambdas of `_environment` compute them - over
/// Python values, so an `int` stays an `int` where Python's does.
fn helper(f: Fun, v: &[Val], env: &Env) -> Result<Val, String> {
    const QUALNAME: &str = "_environment.<locals>.<lambda>";
    let params: &[&str] = match f {
        Fun::Linear | Fun::Geometric | Fun::Cosine => &["a", "b"],
        Fun::Step => &["a", "factor", "every"],
        _ => &["a", "b", "n"],
    };
    if v.len() != params.len() {
        return Err(count_error("helper", params, v.len(), QUALNAME));
    }
    let t = Val::Float(env.t);
    let i = Val::Int(env.i);
    match f {
        // a + (b - a) * t
        Fun::Linear => binop(Bin::Add, v[0], binop(Bin::Mult, binop(Bin::Sub, v[1], v[0])?, t)?),
        // a * (b / a) ** t
        Fun::Geometric => binop(Bin::Mult, v[0], binop(Bin::Pow, binop(Bin::Div, v[1], v[0])?, t)?),
        // b + (a - b) * (1.0 + math.cos(math.pi * t)) / 2.0
        Fun::Cosine => {
            let wave = Val::Float(1.0 + (PI * env.t).cos());
            let scaled = binop(Bin::Mult, binop(Bin::Sub, v[0], v[1])?, wave)?;
            binop(Bin::Add, v[1], binop(Bin::Div, scaled, Val::Float(2.0))?)
        }
        // a * factor ** math.floor(i / max(1.0, every))
        Fun::Step => {
            let every = min_max(&[Val::Float(1.0), v[2]], "max", true)?;
            let turns = call(Fun::Floor, &[binop(Bin::Div, i, every)?], env)?;
            binop(Bin::Mult, v[0], binop(Bin::Pow, v[1], turns)?)
        }
        // b if i >= n else a + (b - a) * i / max(1.0, n)
        _ => {
            if compare(Cmp::GtE, i, v[2])? {
                return Ok(v[1]);
            }
            let n = min_max(&[Val::Float(1.0), v[2]], "max", true)?;
            let rise = binop(Bin::Div, binop(Bin::Mult, binop(Bin::Sub, v[1], v[0])?, i)?, n)?;
            binop(Bin::Add, v[0], rise)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rates(expr: &str) -> Result<Vec<f64>, String> {
        Schedule::new(expr)?.preview(4, 0.05, 0.005, 0.05)
    }

    #[test]
    fn the_presets_give_pythons_rates() {
        // python3 -c "from radixnet.schedule import Schedule; ..."
        assert_eq!(
            rates("linear(lr0, 4 * lr0)").unwrap(),
            vec![0.05, 0.1, 0.15000000000000002, 0.2]
        );
        assert_eq!(
            rates("cosine(lr0, 4 * lr0)").unwrap(),
            vec![0.04999999999999999, 0.0875, 0.16249999999999998, 0.2]
        );
        assert_eq!(
            rates("geometric(lr0, 4 * lr0)").unwrap(),
            vec![0.05, 0.07937005259840997, 0.12599210498948732, 0.2]
        );
        assert_eq!(
            rates("step(lr0, 1.5, 2)").unwrap(),
            vec![0.05, 0.05, 0.07500000000000001, 0.07500000000000001]
        );
        assert_eq!(
            rates("warmup(lr0 / 10, lr0, 3)").unwrap(),
            vec![0.005, 0.02, 0.035, 0.05]
        );
        assert_eq!(
            rates("lr0 * 1.25 ** i").unwrap(),
            vec![0.05, 0.0625, 0.078125, 0.09765625]
        );
        assert_eq!(
            rates("t or 0.5").unwrap(),
            vec![0.5, 0.3333333333333333, 0.6666666666666666, 1.0]
        );
        assert_eq!(rates("abs((-8.0) ** 0.5)").unwrap()[0], 2.8284271247461903);
        assert_eq!(rates("round(0.125, floor(2))").unwrap()[0], 0.12);
        assert_eq!(rates("floor(-7) % floor(2)").unwrap()[0], 1.0);
        assert_eq!(rates("(-7.5) % 2").unwrap()[0], 0.5);
    }

    #[test]
    fn the_whitelist_refuses_in_pythons_words() {
        let refuse = |expr: &str| Schedule::new(expr).err().unwrap_or_default();
        assert_eq!(refuse("lr0.real"), "Attribute is not allowed in a schedule expression");
        assert_eq!(refuse("'x'"), "only numbers are allowed as constants, not 'x'");
        assert_eq!(refuse("True + 1"), "only numbers are allowed as constants, not True");
        assert_eq!(refuse("lr0 | 1"), "operator BitOr is not allowed");
        assert_eq!(refuse("~lr0"), "operator Invert is not allowed");
        assert_eq!(
            refuse("lr0 in 1"),
            "only <, <=, >, >=, == and != comparisons are allowed"
        );
        assert_eq!(refuse("min(a=1)"), "min() takes positional arguments only");
        assert!(refuse("foo").starts_with("unknown name 'foo'; allowed: abs, act_lr0, ceil"));
        assert!(refuse("epoch(1)").starts_with("only these functions may be called: abs, ceil, clamp"));
        assert_eq!(refuse("(1, 2)"), "Tuple is not allowed in a schedule expression");
        assert_eq!(refuse(""), "the schedule expression is empty");
    }

    #[test]
    fn a_syntax_error_says_what_cpython_says() {
        let refuse = |expr: &str| Schedule::new(expr).err().unwrap_or_default();
        assert_eq!(refuse("1 +"), "invalid schedule expression '1 +': invalid syntax");
        assert_eq!(refuse("(1"), "invalid schedule expression '(1': '(' was never closed");
        assert_eq!(refuse("1)"), "invalid schedule expression '1)': unmatched ')'");
        assert_eq!(
            refuse("lr0 if t"),
            "invalid schedule expression 'lr0 if t': expected 'else' after 'if' expression"
        );
        assert_eq!(
            refuse("min(1 2)"),
            "invalid schedule expression 'min(1 2)': invalid syntax. Perhaps you forgot a comma?"
        );
        assert_eq!(
            refuse("1 2 ("),
            "invalid schedule expression '1 2 (': '(' was never closed"
        );
        assert_eq!(refuse("1 +* ("), "invalid schedule expression '1 +* (': invalid syntax");
        assert_eq!(
            refuse("0x"),
            "invalid schedule expression '0x': invalid hexadecimal literal"
        );
        assert_eq!(
            refuse("012"),
            "invalid schedule expression '012': leading zeros in decimal integer literals are not permitted; \
             use an 0o prefix for octal integers"
        );
    }

    #[test]
    fn evaluation_errors_name_the_epoch() {
        let fail = |expr: &str| rates(expr).err().unwrap_or_default();
        assert_eq!(
            fail("1 / 0"),
            "schedule '1 / 0' failed at epoch 1: float division by zero"
        );
        assert_eq!(
            fail("10.0 ** 400"),
            "schedule '10.0 ** 400' failed at epoch 1: (34, 'Numerical result out of range')"
        );
        assert_eq!(fail("epoch < 3"), "schedule 'epoch < 3' must give a number, got bool");
        assert_eq!(
            fail("(-8.0) ** (1/3)"),
            "schedule '(-8.0) ** (1/3)' must give a number, got complex"
        );
        assert_eq!(
            fail("-lr0"),
            "schedule '-lr0' gave -0.05 at epoch 1; rates must be finite and >= 0"
        );
        assert_eq!(
            fail("lr0 * 1e400"),
            "schedule 'lr0 * 1e400' gave inf at epoch 1; rates must be finite and >= 0"
        );
        assert_eq!(
            fail("warmup()"),
            "schedule 'warmup()' failed at epoch 1: _environment.<locals>.<lambda>() missing 3 required positional \
             arguments: 'a', 'b', and 'n'"
        );
        assert_eq!(
            fail("sin"),
            "schedule 'sin' must give a number, got builtin_function_or_method"
        );
    }

    #[test]
    fn a_preview_can_run_backwards() {
        let forward = preview_points(Some("linear(lr0, 4 * lr0)"), Some("lr / 10"), 3, 0.05, 0.005, false).unwrap();
        let back = preview_points(Some("linear(lr0, 4 * lr0)"), Some("lr / 10"), 3, 0.05, 0.005, true).unwrap();
        assert_eq!(back[0].lr, forward[2].lr);
        assert_eq!(back[0].act_lr, forward[2].act_lr);
        assert_eq!(back.iter().map(|p| p.epoch).collect::<Vec<_>>(), vec![1, 2, 3]);
        assert!(preview_points(None, None, -1, 0.05, 0.005, false).is_err());
        let constant = preview_points(None, None, 2, 0.05, 0.005, false).unwrap();
        assert_eq!(
            constant[1],
            Point {
                epoch: 2,
                lr: 0.05,
                act_lr: 0.005
            }
        );
    }
}
