//! Programs written by the network, run in a sandbox and judged
//! (`radixnet/codegen.py`, `go/radixnet/codegen.go`, `go/server/codegen.go`).
//!
//! ```text
//! problem -> Python program -> sandbox run -> style check -> judge -> 2NRL (punish / reward)
//! ```
//!
//! Two semi-supervised phases over the same list of problems:
//!
//! * `teacher` - the tutor (a local Ollama model, or ChatGPT) writes a
//!   solution, the sandbox runs it, the tutor fixes what failed, and the judge
//!   (the same LLM plus an objective PEP 8 / naming check) confirms it.  The
//!   network then learns question + answer: every wrong attempt is 2NRL
//!   garbage, the correct one the fine-tune - wrong answers *before* the right
//!   one.
//! * `model` - the network itself continues each problem into code.  Every
//!   attempt is run and judged the same way; a program that errors or is judged
//!   wrong is punished and the loop tries again, a correct one is rewarded.
//!   When the network never succeeds, the teacher supplies the answer.
//!
//! # Why the style check needs Python
//!
//! The formatting rules are text and are checked here.  The naming rules and
//! the blank-line rule need to know where the definitions are (Python's `ast`),
//! so they run in the same interpreter the sandbox runs programs with
//! ([`STYLE_HELPER`], the Go port's split).  A Python parser written out in
//! Rust would disagree with `ast.walk` on real programs, which is worse than
//! asking the real one; and codegen needs an interpreter anyway, because the
//! programs it writes are Python.
//!
//! # The same conversation, byte for byte
//!
//! The teacher and the judge are sent Python's prompts character for
//! character - `tests/test_rust_parity_agent.py` compares what one fake Ollama
//! receives from both implementations - and the verdict is combined the way
//! Python combines it, so the same programs are punished and rewarded and the
//! model comes out the same - whatever its kind: the 2NRL goes through
//! [`crate::kinds`], so a sine model descends at the configured rates where a
//! count model counts, as Python's two do.
//!
//! # Where the networks live
//!
//! [`Nets`] is the one thing the command line and the server do differently:
//! the command owns its models for the length of the run, the server takes its
//! lock for one step at a time - a prediction, a round of learning - so that
//! nothing is locked while a program runs or the tutor thinks, which is most
//! of a run.  The agent ([`crate::agent`]) and the MCP server share it.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::blame::{classify, severity_of, teach, Fault, TeachOptions, TeachReport, CODE_SEVERITY};
use crate::checkpoint::{Checkpoints, Schedule};
use crate::cli::{negative_stats, Ctx};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::kinds;
use crate::llm::fields::{py_str_of, truthy, Fields};
use crate::llm::{
    clip_chars, format_g, loads_lenient, new_client, normalise_provider, LlmClient, LlmError, LlmOptions, CHATGPT,
    DEFAULT_PROVIDER, OLLAMA, PROVIDERS,
};
use crate::model::{EpochRecord, Model, PredictOptions};
use crate::negative::python_repr;
use crate::radix::Feedback;
use crate::report::stats;
use crate::service::Service;
use crate::toolbox::sandbox::{python_binary, RunResult, Sandbox, SandboxOptions};

/// What this module's own lines are filed under.
const LOG: &str = "codegen";

/// The two ways a program is produced.
pub const PHASES: &[&str] = &["teacher", "model"];

/// How strictly a verdict is reached: `strict` also wants PEP 8 formatting
/// and naming, `lenient` only a program that runs and does the task.
pub const STRICTNESS: &[&str] = &["strict", "lenient"];

/// The longest line PEP 8 allows.
pub const MAX_LINE_LENGTH: usize = 79;

/// The model that writes, fixes and judges when none is named:
/// `$RADIXNET_CODEGEN_MODEL` (else `gemma4`) for Ollama, the provider's own
/// default for ChatGPT.
pub fn default_teacher_model(provider: &str) -> String {
    if normalise_provider(provider) == Ok(CHATGPT) {
        return crate::chatgpt::default_model();
    }
    let named = crate::llm::env("RADIXNET_CODEGEN_MODEL");
    if named.is_empty() {
        "gemma4".to_string()
    } else {
        named
    }
}

// -- problems ------------------------------------------------------------------------------------

/// A task to solve: a prompt, optional test code (appended to the program) and
/// optional exact stdout.
#[derive(Clone, Debug, PartialEq)]
pub struct Problem {
    pub id: String,
    pub prompt: String,
    pub tests: Option<String>,
    pub expected_output: Option<String>,
}

/// Python's name for the type of a decoded JSON value, as a refusal names it.
pub(crate) fn py_type_name(value: &Json) -> &'static str {
    match value {
        Json::Null => "NoneType",
        Json::Bool(_) => "bool",
        Json::Int(_) => "int",
        Json::Num(_) => "float",
        Json::Str(_) => "str",
        Json::Arr(_) => "list",
        Json::Obj(_) => "dict",
    }
}

/// `dict.get(first, dict.get(second))`: the first key when it is there at all
/// (a `null` counts as there), else the second.
pub(crate) fn get_either<'a>(item: &'a Json, first: &str, second: &str) -> &'a Json {
    match item.get(first) {
        Some(value) => value,
        None => item.at(second),
    }
}

impl Problem {
    /// A problem with neither tests nor an expected output.
    pub fn new(id: &str, prompt: &str) -> Problem {
        Problem {
            id: id.to_string(),
            prompt: prompt.to_string(),
            tests: None,
            expected_output: None,
        }
    }

    /// `{"id", "prompt", "tests", "expected_output"}` - what [`parse_problems`]
    /// reads back.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("id", Json::str(self.id.clone())),
            ("prompt", Json::str(self.prompt.clone())),
            ("tests", self.tests.clone().map_or(Json::Null, Json::Str)),
            (
                "expected_output",
                self.expected_output.clone().map_or(Json::Null, Json::Str),
            ),
        ])
    }

    /// One problem out of a string or an object (`prompt` / `problem` / `task`
    /// / `question`, `tests`, `expected_output` / `expected`, `id`); `index`
    /// is its place in the list, from 1.
    pub fn from_json(item: &Json, index: usize) -> Result<Problem, String> {
        match item {
            Json::Str(text) => {
                if text.trim().is_empty() {
                    return Err(format!("problem {index}: empty prompt"));
                }
                Ok(Problem::new(&format!("p{index}"), text.trim()))
            }
            Json::Obj(_) => {
                let prompt = ["prompt", "problem", "task", "question"]
                    .iter()
                    .find_map(|key| item.at(key).as_str().filter(|t| !t.trim().is_empty()))
                    .map(|t| t.trim().to_string())
                    .ok_or_else(|| format!("problem {index}: missing 'prompt'"))?;
                let tests = match item.at("tests") {
                    Json::Null => None,
                    Json::Str(code) => Some(code.clone()),
                    _ => return Err(format!("problem {index}: 'tests' must be a string of Python code")),
                };
                let expected_output = match get_either(item, "expected_output", "expected") {
                    Json::Null => None,
                    Json::Str(text) => Some(text.clone()),
                    _ => return Err(format!("problem {index}: 'expected_output' must be a string")),
                };
                let id = match item.at("id") {
                    Json::Null => String::new(),
                    raw => py_str_of(raw).trim().to_string(),
                };
                Ok(Problem {
                    id: if id.is_empty() { format!("p{index}") } else { id },
                    prompt,
                    tests,
                    expected_output,
                })
            }
            other => Err(format!(
                "problem {index}: expected a string or an object, got {}",
                py_type_name(other)
            )),
        }
    }
}

/// Problems from strings and objects; a repeated id gets a numeric suffix.
pub fn parse_problems(items: &[Json]) -> Result<Vec<Problem>, String> {
    let mut problems: Vec<Problem> = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let mut problem = Problem::from_json(item, i + 1)?;
        let (base, mut n) = (problem.id.clone(), 2);
        while problems.iter().any(|p| p.id == problem.id) {
            problem.id = format!("{base}-{n}");
            n += 1;
        }
        problems.push(problem);
    }
    if problems.is_empty() {
        return Err("no problems given".to_string());
    }
    Ok(problems)
}

/// A problem file's content: `.jsonl` one object per line, `.json` a list or
/// `{"problems": [...]}`, anything else one prompt per non-blank line (`#`
/// lines are comments).
pub fn parse_problem_file(text: &str, ext: &str) -> Result<Vec<Problem>, String> {
    let items: Vec<Json> = match ext.to_lowercase().as_str() {
        ".jsonl" => crate::llm::splitlines(text)
            .into_iter()
            .filter(|line| !line.trim().is_empty())
            .map(crate::mcp::pyjson::loads)
            .collect::<Result<_, _>>()?,
        ".json" => match crate::mcp::pyjson::loads(text)? {
            Json::Arr(items) => items,
            doc @ Json::Obj(_) => match doc.get("problems") {
                None => Vec::new(),
                Some(Json::Arr(items)) => items.clone(),
                Some(_) => return Err("a JSON problem file must hold a list or {\"problems\": [...]}".to_string()),
            },
            _ => return Err("a JSON problem file must hold a list or {\"problems\": [...]}".to_string()),
        },
        _ => crate::llm::splitlines(text)
            .into_iter()
            .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
            .map(|line| Json::str(line.trim()))
            .collect(),
    };
    parse_problems(&items)
}

/// The extension Python's `os.path.splitext` finds: the last `.suffix` of
/// the file name, unless the name only starts with the dot.
pub(crate) fn extension(path: &str) -> String {
    let name = path.rsplit('/').next().unwrap_or(path);
    let stem_start = name.len() - name.trim_start_matches('.').len();
    match name[stem_start..].rfind('.') {
        Some(at) => name[stem_start + at..].to_string(),
        None => String::new(),
    }
}

/// A problem file from disk.
pub fn load_problems(path: &str) -> Result<Vec<Problem>, String> {
    let text = std::fs::read_to_string(path).map_err(|err| err.to_string())?;
    parse_problem_file(&text, &extension(path))
}

// -- the objective style check -------------------------------------------------------------------

/// What the style checker found.
#[derive(Clone, Debug, PartialEq)]
pub struct StyleReport {
    pub ok: bool,
    pub syntax_ok: bool,
    pub pep8_ok: bool,
    pub naming_ok: bool,
    pub issues: Vec<String>,
}

impl StyleReport {
    /// `{"ok", "syntax_ok", "pep8_ok", "naming_ok", "issues"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("ok", Json::Bool(self.ok)),
            ("syntax_ok", Json::Bool(self.syntax_ok)),
            ("pep8_ok", Json::Bool(self.pep8_ok)),
            ("naming_ok", Json::Bool(self.naming_ok)),
            ("issues", Json::strs(self.issues.clone())),
        ])
    }
}

/// The Python half of the style check: the rules that need to know where the
/// definitions are.  It reports a syntax error, the E302 blank-line rule and
/// the naming rules as JSON, with `radixnet/codegen.py`'s own regexes and
/// messages.  The source arrives as UTF-8 bytes on stdin, so no newline is
/// translated on the way in.
pub const STYLE_HELPER: &str = r##"import ast, json, re, sys

SNAKE = re.compile(r"^_*[a-z][a-z0-9_]*$|^_+$")
CAPWORDS = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
CONSTANT = re.compile(r"^_*[A-Z][A-Z0-9_]*$")
DUNDER = re.compile(r"^__[a-z0-9_]+__$")

code = sys.stdin.buffer.read().decode("utf-8", "replace")
try:
    tree = ast.parse(code)
except SyntaxError as exc:
    print(json.dumps({"syntax_error": f"syntax error: {exc.msg} (line {exc.lineno})"}))
    raise SystemExit(0)
lines = code.split("\n")
formatting = []
body = tree.body
for previous, node in zip(body, body[1:]):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        first = min([node.lineno] + [d.lineno for d in node.decorator_list])
        end = getattr(previous, "end_lineno", previous.lineno)
        blank = [not lines[i - 1].strip() for i in range(end + 1, first) if 1 <= i - 1 < len(lines)]
        comment_only = all(lines[i - 1].lstrip().startswith("#") or not lines[i - 1].strip()
                           for i in range(end + 1, first))
        if comment_only and sum(blank) < 2:
            formatting.append(f"E302 line {first}: expected 2 blank lines before a top-level definition")
naming = []
for node in ast.walk(tree):
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if not (SNAKE.match(node.name) or DUNDER.match(node.name)):
            naming.append(f"N802 line {node.lineno}: function name '{node.name}' should be snake_case")
        args = node.args
        for arg in args.posonlyargs + args.args + args.kwonlyargs + [a for a in (args.vararg, args.kwarg) if a]:
            if not SNAKE.match(arg.arg):
                naming.append(f"N803 line {node.lineno}: argument '{arg.arg}' should be snake_case")
    elif isinstance(node, ast.ClassDef):
        if not CAPWORDS.match(node.name):
            naming.append(f"N801 line {node.lineno}: class name '{node.name}' should use CapWords")
    elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
        name = node.id
        if not (SNAKE.match(name) or CONSTANT.match(name) or CAPWORDS.match(name)):
            naming.append(f"N806 line {node.lineno}: variable '{name}' should be snake_case "
                          "(or UPPER_CASE for a constant)")
seen = set()
naming = [n for n in naming if not (n in seen or seen.add(n))]
print(json.dumps({"formatting": formatting, "naming": naming}))
"##;

/// The part of PEP 8 that is text: the final newline, whitespace on blank
/// lines, tabs and indentation, line length and trailing whitespace - in
/// Python's order and words, lengths counted in characters.
pub fn check_formatting(code: &str) -> Vec<String> {
    let mut issues = Vec::new();
    if !code.is_empty() && !code.ends_with('\n') {
        issues.push("W292 no newline at end of file".to_string());
    }
    for (i, line) in code.split('\n').enumerate() {
        let number = i + 1;
        let stripped = line.trim_start_matches([' ', '\t']);
        if stripped.is_empty() {
            if !line.is_empty() {
                issues.push(format!("W293 line {number}: whitespace on a blank line"));
            }
            continue;
        }
        let indent = &line[..line.len() - stripped.len()];
        if indent.contains('\t') {
            issues.push(format!("W191 line {number}: indentation contains tabs"));
        } else if indent.len() % 4 != 0 {
            issues.push(format!("E111 line {number}: indentation is not a multiple of four"));
        }
        let length = line.chars().count();
        if length > MAX_LINE_LENGTH {
            issues.push(format!(
                "E501 line {number}: line too long ({length} > {MAX_LINE_LENGTH})"
            ));
        }
        if line.trim_end() != line {
            issues.push(format!("W291 line {number}: trailing whitespace"));
        }
    }
    issues
}

/// PEP 8 formatting and naming, without external tools.
///
/// `python` is the interpreter the AST half runs in (the sandbox's; `None`
/// for [`python_binary`]).  When it cannot be run the formatting rules still
/// apply and the report says the naming pass was skipped, rather than claiming
/// the names are fine.
pub fn check_style(code: &str, python: Option<&str>) -> StyleReport {
    let python = python.map(str::to_string).unwrap_or_else(python_binary);
    let parsed = run_style_helper(&python, code);
    if let Some(Json::Str(error)) = parsed.as_ref().and_then(|doc| doc.get("syntax_error")) {
        return StyleReport {
            ok: false,
            syntax_ok: false,
            pep8_ok: false,
            naming_ok: false,
            issues: vec![error.clone()],
        };
    }
    let mut formatting = check_formatting(code);
    let naming = match &parsed {
        Some(doc) => {
            formatting.extend(doc.at("formatting").to_strings());
            doc.at("naming").to_strings()
        }
        None => {
            // say so rather than pass silently: an unchecked name is not a good name
            formatting
                .push("W000 the naming check was skipped: no usable Python interpreter for the AST pass".to_string());
            Vec::new()
        }
    };
    StyleReport {
        ok: formatting.is_empty() && naming.is_empty(),
        syntax_ok: true,
        pep8_ok: formatting.is_empty(),
        naming_ok: naming.is_empty(),
        issues: formatting.into_iter().chain(naming).collect(),
    }
}

/// Runs [`STYLE_HELPER`] over `code`; `None` when the interpreter cannot be
/// run or answers nothing readable.
fn run_style_helper(python: &str, code: &str) -> Option<Json> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new(python)
        .args(["-I", "-B", "-c", STYLE_HELPER])
        .env("PYTHONIOENCODING", "utf-8")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    let input = code.as_bytes().to_vec();
    let mut stdin = child.stdin.take()?;
    let writer = std::thread::spawn(move || {
        let _ = stdin.write_all(&input);
    });
    let mut stdout = child.stdout.take()?;
    let reader = std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = std::io::Read::read_to_end(&mut stdout, &mut out);
        out
    });
    let deadline = Instant::now() + Duration::from_secs(20);
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(2)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    };
    let _ = writer.join();
    let out = reader.join().ok()?;
    if !status.success() {
        return None;
    }
    crate::json::parse(String::from_utf8_lossy(&out).trim()).ok()
}

// -- texts: code blocks, training texts, model prefixes --------------------------------------------

/// Where a fenced block's header ends, when a fence starts at `at`:
/// `` ```(?:python3?|py)?[ \t]*\r?\n`` read case-insensitively.
fn fence_header(text: &str, at: usize) -> Option<usize> {
    let rest = &text[at..];
    if !rest.starts_with("```") {
        return None;
    }
    let after = at + 3;
    for tag in ["python3", "python", "py", ""] {
        let Some(candidate) = text.get(after..after + tag.len()) else {
            continue;
        };
        if !candidate.eq_ignore_ascii_case(tag) {
            continue;
        }
        let mut i = after + tag.len();
        let bytes = text.as_bytes();
        while i < bytes.len() && matches!(bytes[i], b' ' | b'\t') {
            i += 1;
        }
        if bytes.get(i) == Some(&b'\r') {
            i += 1;
        }
        if bytes.get(i) == Some(&b'\n') {
            return Some(i + 1);
        }
    }
    None
}

/// The Python program inside an LLM answer: the longest fenced block, else
/// the whole text (a lone fence stripped).
pub fn extract_code(text: &str) -> String {
    let mut blocks: Vec<&str> = Vec::new();
    let mut at = 0;
    while at < text.len() {
        let Some(body) = fence_header(text, at) else {
            at += text[at..].chars().next().map_or(1, char::len_utf8);
            continue;
        };
        match text[body..].find("```") {
            Some(close) => {
                blocks.push(&text[body..body + close]);
                at = body + close + 3;
            }
            None => break,
        }
    }
    let mut longest: Option<&str> = None;
    for block in blocks.into_iter().filter(|b| !b.trim().is_empty()) {
        if longest.is_none_or(|best| block.chars().count() > best.chars().count()) {
            longest = Some(block);
        }
    }
    if let Some(block) = longest {
        return format!("{}\n", block.trim_matches('\n'));
    }
    let mut stripped = text.trim().to_string();
    if stripped.starts_with("```") {
        stripped = stripped.trim_matches('`').to_string();
        if stripped.to_lowercase().starts_with("python") {
            stripped = stripped.chars().skip(6).collect();
        }
        stripped = stripped.trim().to_string();
    }
    if stripped.is_empty() {
        String::new()
    } else {
        stripped + "\n"
    }
}

/// What the network trains on: the question, a newline, the answer.
pub fn solution_text(problem: &Problem, code: &str) -> String {
    format!("{}\n{}\n", problem.prompt.trim(), code.trim())
}

/// The prefix the network continues into code (`{problem}` is the prompt).
pub fn model_prefix(template: &str, problem: &Problem) -> Result<String, String> {
    if !template.contains("{problem}") {
        return Err("model_prompt must contain {problem}".to_string());
    }
    Ok(template.replace("{problem}", problem.prompt.trim()))
}

// -- the teacher and the judge -------------------------------------------------------------------

/// What the teacher is told it is (Python's `TEACHER_SYSTEM`).
pub const TEACHER_SYSTEM: &str = "You are an expert Python programmer writing small, self-contained programs. Answer \
with exactly one ```python code block and nothing else: no explanations before or after it. The program must run with \
`python3 solution.py` on a plain Python 3 installation (standard library only), must not read input unless the task \
says so, must finish on its own, and must print its result when the task asks for output. Follow PEP 8: four-space \
indentation, lines of at most 79 characters, snake_case function and variable names, CapWords class names, UPPER_CASE \
constants, two blank lines before top-level definitions, a newline at the end of the file.";

/// What the judge is told it is (Python's `JUDGE_SYSTEM`).
pub const JUDGE_SYSTEM: &str = "You are a strict reviewer of small Python programs written for a stated task. Decide \
whether the program genuinely accomplishes the task (not merely runs), whether its formatting follows PEP 8, and whether \
its naming follows PEP 8 (snake_case functions and variables, CapWords classes, UPPER_CASE constants, descriptive \
names). Be adversarial: look for wrong results, missing requirements, unhandled cases and sloppy names. Reply with JSON \
only, exactly of the form {\"task_accomplished\": true or false, \"pep8\": true or false, \"naming\": true or false, \
\"score\": <0-10>, \"issues\": [\"...\"], \"critique\": \"one sentence\"}.";

/// The task as the teacher and the judge are shown it.
fn problem_block(problem: &Problem) -> String {
    let mut text = format!("Task:\n{}\n", problem.prompt.trim());
    if let Some(expected) = &problem.expected_output {
        text.push_str(&format!(
            "\nThe program's standard output must be exactly:\n{}\n",
            expected.trim()
        ));
    }
    if let Some(tests) = problem.tests.as_deref().filter(|t| !t.is_empty()) {
        text.push_str(&format!(
            "\nThis test code is appended to the program and must pass:\n{}\n",
            tests.trim()
        ));
    }
    text
}

/// The options of a teacher's call: its instruction, its model and the
/// temperature Python writes with.
fn teacher_options(model: &str) -> LlmOptions {
    LlmOptions::default()
        .system(TEACHER_SYSTEM)
        .model(model)
        .temperature(0.3)
}

/// Asks the tutor for a first program.
pub fn teacher_generate(
    client: &dyn LlmClient,
    problem: &Problem,
    extra: Option<&str>,
    model: &str,
) -> Result<String, LlmError> {
    let mut user = problem_block(problem);
    if let Some(extra) = extra.filter(|e| !e.trim().is_empty()) {
        user.push_str(&format!("\nAdditional instructions:\n{}\n", extra.trim()));
    }
    user.push_str("\nWrite the program now.");
    Ok(extract_code(&client.generate(&user, &teacher_options(model))?))
}

/// Asks the tutor to correct a program that failed or was judged wrong.
pub fn teacher_fix(
    client: &dyn LlmClient,
    problem: &Problem,
    attempt: &Attempt,
    extra: Option<&str>,
    model: &str,
) -> Result<String, LlmError> {
    let mut user = problem_block(problem);
    user.push_str(&format!(
        "\nThis program is not acceptable yet:\n```python\n{}\n```\n\nWhat went wrong:\n{}\n",
        attempt.code.trim_end(),
        attempt.feedback()
    ));
    if let Some(extra) = extra.filter(|e| !e.trim().is_empty()) {
        user.push_str(&format!("\nAdditional instructions:\n{}\n", extra.trim()));
    }
    user.push_str("\nReturn the complete corrected program as one ```python block and nothing else.");
    Ok(extract_code(&client.generate(&user, &teacher_options(model))?))
}

/// The many ways an LLM writes a boolean (Python's `_as_bool`).
pub(crate) fn as_bool(value: &Json, default: bool) -> bool {
    match value {
        Json::Bool(b) => *b,
        Json::Int(n) => *n != 0,
        Json::Num(x) => *x != 0.0,
        Json::Str(text) => match text.trim().to_lowercase().as_str() {
            "true" | "yes" | "y" | "1" | "pass" | "passed" | "ok" => true,
            "false" | "no" | "n" | "0" | "fail" | "failed" => false,
            _ => default,
        },
        _ => default,
    }
}

/// `float(value)`: a number, a boolean, or text Python reads as a float.
pub(crate) fn py_float(value: &Json) -> Option<f64> {
    match value {
        Json::Int(n) => Some(*n as f64),
        Json::Num(x) => Some(*x),
        Json::Bool(b) => Some(f64::from(u8::from(*b))),
        Json::Str(text) => {
            let text = text.trim();
            let lower = text.to_ascii_lowercase();
            let bare = lower.trim_start_matches(['+', '-']);
            if matches!(bare, "inf" | "infinity" | "nan") {
                return text.parse().ok().or(Some(if bare == "nan" {
                    f64::NAN
                } else if lower.starts_with('-') {
                    f64::NEG_INFINITY
                } else {
                    f64::INFINITY
                }));
            }
            if text.contains('_') {
                return None;
            }
            text.parse().ok()
        }
        _ => None,
    }
}

/// `max(0.0, min(10.0, x))` with Python's comparisons, so a NaN lands where
/// Python puts it.
pub(crate) fn clamp_score(x: f64) -> f64 {
    let low = if x < 10.0 { x } else { 10.0 };
    if low > 0.0 {
        low
    } else {
        0.0
    }
}

/// A Python object as a `dict` would hold it: a repeated key keeps its first
/// place and its last value.
pub(crate) fn as_dict(value: Json) -> Json {
    match value {
        Json::Obj(pairs) => {
            let mut out: Vec<(String, Json)> = Vec::with_capacity(pairs.len());
            for (key, item) in pairs {
                match out.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = item,
                    None => out.push((key, item)),
                }
            }
            Json::Obj(out)
        }
        other => other,
    }
}

/// What the LLM judge said.
#[derive(Clone, Debug, PartialEq)]
pub struct JudgeOpinion {
    pub task: bool,
    pub pep8: bool,
    pub naming: bool,
    pub score: Option<f64>,
    pub issues: Vec<String>,
    pub critique: String,
}

/// The first `n` characters (Python's `text[:n]`).
pub(crate) fn head(text: &str, n: usize) -> &str {
    clip_chars(text, n)
}

/// The last `n` characters (Python's `text[-n:]`).
pub(crate) fn tail(text: &str, n: usize) -> &str {
    let count = text.chars().count();
    if count <= n {
        return text;
    }
    let skip = count - n;
    match text.char_indices().nth(skip) {
        Some((at, _)) => &text[at..],
        None => "",
    }
}

/// Asks the judge whether the program accomplishes the task.
pub fn judge_with_llm(
    client: &dyn LlmClient,
    problem: &Problem,
    code: &str,
    run: &RunResult,
    style: &StyleReport,
    model: &str,
) -> Result<JudgeOpinion, LlmError> {
    let mut user = problem_block(problem);
    let empty = |text: &str| -> String {
        if text.is_empty() {
            "(empty)".to_string()
        } else {
            text.to_string()
        }
    };
    user.push_str(&format!(
        "\nProgram:\n```python\n{}\n```\n\nExecution: exit code {}, {:.2}s\nstdout:\n{}\nstderr:\n{}\n",
        code.trim_end(),
        run.exit_code.map_or("None".to_string(), |c| c.to_string()),
        run.seconds,
        empty(head(&run.stdout, 1500)),
        empty(head(&run.stderr, 800)),
    ));
    if let Some(matched) = run.expected_ok {
        user.push_str(&format!(
            "\nThe stdout {} the expected output.\n",
            if matched { "matches" } else { "does NOT match" }
        ));
    }
    if problem.tests.as_deref().is_some_and(|t| !t.is_empty()) {
        user.push_str(&format!(
            "\nThe appended tests {}.\n",
            if run.ok { "passed" } else { "FAILED" }
        ));
    }
    let summary = if style.ok {
        "no issues".to_string()
    } else {
        style.issues.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
    };
    user.push_str(&format!("\nAutomated style check: {summary}\n\nReturn the JSON now."));
    let o = LlmOptions::default()
        .system(JUDGE_SYSTEM)
        .model(model)
        .json()
        .temperature(0.1);
    let raw = client.generate(&user, &o)?;
    let data = match loads_lenient(&raw).map(as_dict) {
        Some(data @ Json::Obj(_)) => data,
        _ => {
            return Err(LlmError::new(
                client.provider(),
                "the judge did not answer with a JSON object",
            ))
        }
    };
    let task_value = match data.get("task_accomplished") {
        Some(value) => value,
        None => get_either(&data, "correct", "task"),
    };
    let issues: Vec<String> = match data.at("issues") {
        Json::Str(one) => vec![one.trim().to_string()]
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect(),
        Json::Arr(items) => items
            .iter()
            .map(|i| py_str_of(i).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        _ => Vec::new(),
    };
    let critique = if truthy(data.at("critique")) {
        py_str_of(data.at("critique"))
    } else if truthy(data.at("reason")) {
        py_str_of(data.at("reason"))
    } else {
        String::new()
    };
    Ok(JudgeOpinion {
        task: as_bool(task_value, false),
        pep8: as_bool(get_either(&data, "pep8", "formatting"), true),
        naming: as_bool(data.at("naming"), true),
        score: py_float(data.at("score")).map(clamp_score),
        issues: issues.into_iter().take(10).collect(),
        critique: critique.trim().to_string(),
    })
}

// -- the verdict ---------------------------------------------------------------------------------

/// The one answer the sandbox, the style report and the judge come to.
#[derive(Clone, Debug, PartialEq)]
pub struct CodeVerdict {
    pub correct: bool,
    pub runs: bool,
    /// `None` when nothing could say (no judge, no expected output, no tests).
    pub task: Option<bool>,
    pub pep8: bool,
    pub naming: bool,
    pub score: Option<f64>,
    pub issues: Vec<String>,
    pub critique: String,
    /// `sandbox`, `tests`, `ollama`, `chatgpt` or `none`.
    pub judged_by: String,
}

impl CodeVerdict {
    /// The verdict in Python's key order.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("correct", Json::Bool(self.correct)),
            ("runs", Json::Bool(self.runs)),
            ("task", self.task.map_or(Json::Null, Json::Bool)),
            ("pep8", Json::Bool(self.pep8)),
            ("naming", Json::Bool(self.naming)),
            ("score", self.score.map_or(Json::Null, Json::Num)),
            ("issues", Json::strs(self.issues.clone())),
            ("critique", Json::str(self.critique.clone())),
            ("judged_by", Json::str(self.judged_by.clone())),
        ])
    }
}

/// Combines the sandbox result, the style report and the LLM's opinion into
/// one verdict; `judged_by` names the provider behind `llm`.
pub fn decide(
    run: &RunResult,
    style: &StyleReport,
    llm: Option<&JudgeOpinion>,
    strictness: &str,
    judged_by: &str,
) -> Result<CodeVerdict, String> {
    if !STRICTNESS.contains(&strictness) {
        return Err(format!("strictness must be one of {}", STRICTNESS.join(", ")));
    }
    let runs = run.ok;
    let mut issues: Vec<String> = Vec::new();
    if !runs {
        issues.push(
            run.error
                .clone()
                .filter(|e| !e.is_empty())
                .unwrap_or_else(|| "the program did not run".to_string()),
        );
    }
    if run.expected_ok == Some(false) {
        issues.push("stdout differs from the expected output".to_string());
    }
    let mut judged_by = judged_by.to_string();
    let task = if !runs || run.expected_ok == Some(false) {
        judged_by = "sandbox".to_string();
        Some(false)
    } else if let Some(opinion) = llm {
        Some(opinion.task)
    } else if run.expected_ok == Some(true) {
        judged_by = "tests".to_string();
        Some(true)
    } else {
        // nothing to judge with: running counts
        judged_by = "none".to_string();
        None
    };
    let pep8 = style.pep8_ok && llm.is_none_or(|o| o.pep8);
    let naming = style.naming_ok && llm.is_none_or(|o| o.naming);
    issues.extend(style.issues.iter().take(6).cloned());
    if let Some(opinion) = llm {
        issues.extend(opinion.issues.iter().take(6).cloned());
    }
    let strict = strictness == "strict";
    let correct =
        runs && run.expected_ok != Some(false) && task != Some(false) && (!strict || (pep8 && naming && style.ok));
    Ok(CodeVerdict {
        correct,
        runs,
        task,
        pep8,
        naming,
        score: llm.and_then(|o| o.score),
        issues,
        critique: llm.map(|o| o.critique.clone()).unwrap_or_default(),
        judged_by,
    })
}

/// One program, run, marked and judged.
#[derive(Clone, Debug)]
pub struct Attempt {
    pub index: usize,
    /// The tutor's provider (`ollama` / `chatgpt`) or `model`.
    pub source: String,
    pub code: String,
    /// What the network learns from it ([`solution_text`]).
    pub text: String,
    pub run: RunResult,
    pub style: StyleReport,
    pub verdict: CodeVerdict,
    pub seconds: f64,
}

impl Attempt {
    /// Why the program was rejected, in the words the teacher's fix prompt
    /// is given (Python's `Attempt.feedback`).
    pub fn feedback(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        if !self.run.ok {
            parts.push(format!(
                "It did not run: {}",
                self.run.error.as_deref().unwrap_or("None")
            ));
            if !self.run.stderr.is_empty() {
                parts.push(format!("stderr:\n{}", tail(&self.run.stderr, 1200)));
            }
        }
        if self.run.expected_ok == Some(false) {
            let out = tail(&self.run.stdout, 800);
            parts.push(format!(
                "Its output differed from the expected output. Actual stdout:\n{}",
                if out.is_empty() { "(empty)" } else { out }
            ));
        }
        if !self.style.issues.is_empty() {
            parts.push(format!(
                "Style checker: {}",
                self.style.issues.iter().take(8).cloned().collect::<Vec<_>>().join("; ")
            ));
        }
        if PROVIDERS.contains(&self.verdict.judged_by.as_str()) && self.verdict.task == Some(false) {
            parts.push(format!(
                "Reviewer: the task is not accomplished. {}",
                self.verdict.critique
            ));
        } else if !self.verdict.critique.is_empty() {
            parts.push(format!("Reviewer: {}", self.verdict.critique));
        }
        if !self.verdict.issues.is_empty() {
            let mut unique: Vec<&str> = Vec::new();
            for issue in &self.verdict.issues {
                if !unique.contains(&issue.as_str()) {
                    unique.push(issue);
                }
            }
            // Python clips the list, not the prefix: "Issues: " + "; ".join(...)[:1500]
            parts.push(format!("Issues: {}", head(&unique.join("; "), 1500)));
        }
        if parts.is_empty() {
            "It was judged incorrect.".to_string()
        } else {
            parts.join("\n")
        }
    }

    /// The attempt as the records and the API carry it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("index", Json::Int(self.index as i64)),
            ("source", Json::str(self.source.clone())),
            ("code", Json::str(self.code.clone())),
            ("text_chars", Json::Int(self.text.chars().count() as i64)),
            ("run", self.run.to_json()),
            ("style", self.style.to_json()),
            ("verdict", self.verdict.to_json()),
            ("correct", Json::Bool(self.verdict.correct)),
            ("seconds", Json::Num(self.seconds)),
        ])
    }
}

// -- teaching the negative network ----------------------------------------------------------------

/// Why a program was rejected, from the sandbox, the style report and the
/// judge, in the order those matter (Python's `blame.code_reason`).
pub fn code_reason(attempt: &Attempt) -> String {
    let (run, style, verdict) = (&attempt.run, &attempt.style, &attempt.verdict);
    if run.timed_out {
        return "timeout".to_string();
    }
    if !run.ok {
        return "crash".to_string();
    }
    if run.expected_ok == Some(false) {
        return "wrong-output".to_string();
    }
    if verdict.task == Some(false) {
        return "task-not-done".to_string();
    }
    if !verdict.naming || !style.naming_ok {
        return "naming".to_string();
    }
    if !verdict.pep8 || !style.pep8_ok {
        return "style".to_string();
    }
    let words = if verdict.critique.is_empty() {
        verdict.issues.iter().take(3).cloned().collect::<Vec<_>>().join("; ")
    } else {
        verdict.critique.clone()
    };
    classify(&words, "", None, "")
}

/// `(faults, correct_texts)` of a problem's attempts: a rejected program is
/// blamed for what the sandbox, the style checker or the judge found, with the
/// tutor's feedback as the note; the accepted ones come back to clear blame
/// (Python's `blame.faults_from_attempts`).
pub fn faults_from_attempts(attempts: &[Attempt], source: &str) -> (Vec<Fault>, Vec<String>) {
    let mut faults = Vec::new();
    let mut correct = Vec::new();
    for attempt in attempts {
        let text = if attempt.text.is_empty() {
            attempt.code.clone()
        } else {
            attempt.text.clone()
        };
        if text.is_empty() {
            continue;
        }
        if attempt.verdict.correct {
            correct.push(text);
            continue;
        }
        let reason = code_reason(attempt);
        let severity = severity_of(CODE_SEVERITY, &reason);
        faults.push(Fault::new(&text, &reason, severity, &attempt.feedback(), source));
    }
    (faults, correct)
}

/// Feeds a problem's rejected attempts into the negative network, the
/// accepted ones clearing blame (Python's `blame.teach_attempts`).
pub fn teach_attempts(
    negative: &mut Model,
    attempts: &[Attempt],
    clear_passes: bool,
    source: &str,
    o: &TeachOptions,
) -> Result<TeachReport, String> {
    let source = if source.is_empty() { "codegen" } else { source };
    let (faults, correct) = faults_from_attempts(attempts, source);
    let passed = if clear_passes { correct.clone() } else { Vec::new() };
    let mut report = teach(negative, &faults, &passed, o)?;
    report.source = source.to_string();
    report.faults = faults;
    report.passed = correct.len();
    Ok(report)
}

// -- where the networks live ----------------------------------------------------------------------

/// Where a loop's networks live: owned by a command for the length of the
/// run, behind the server's locks, or behind mutexes of their own (the MCP
/// server's tools) - the last two taken for one step at a time.
pub enum Nets<'a> {
    /// A command's own models.
    Owned {
        model: &'a mut Model,
        negative: Option<&'a mut Model>,
    },
    /// The server's running model and - when `negative` - its negative
    /// network, which the caller has loaded ([`Service::ensure_negative`]).
    Server { svc: &'a Service, negative: bool },
    /// Models behind mutexes of their own.
    Locked {
        model: &'a Mutex<Model>,
        negative: Option<&'a Mutex<Model>>,
    },
}

impl Nets<'_> {
    /// Runs `f` on the positive model, holding whatever lock guards it for as
    /// long as `f` runs and no longer.
    pub fn model<T>(&mut self, f: impl FnOnce(&mut Model) -> T) -> T {
        match self {
            Nets::Owned { model, .. } => f(model),
            Nets::Server { svc, .. } => svc.with_model(f),
            Nets::Locked { model, .. } => f(&mut model.lock().unwrap_or_else(|e| e.into_inner())),
        }
    }

    /// Whether a negative network takes part.
    pub fn has_negative(&self) -> bool {
        match self {
            Nets::Owned { negative, .. } => negative.is_some(),
            Nets::Server { negative, .. } => *negative,
            Nets::Locked { negative, .. } => negative.is_some(),
        }
    }

    /// Runs `f` on the negative network; `None` when there is none.  Never
    /// called with the positive model held, so the two locks are never taken
    /// in the wrong order.
    pub fn negative<T>(&mut self, f: impl FnOnce(&mut Model) -> T) -> Result<Option<T>, String> {
        match self {
            Nets::Owned { negative, .. } => Ok(negative.as_deref_mut().map(f)),
            Nets::Server { svc, negative: true } => svc.with_negative(f).map(Some).map_err(|e| e.message),
            Nets::Server { .. } => Ok(None),
            Nets::Locked { negative, .. } => Ok(negative.map(|n| f(&mut n.lock().unwrap_or_else(|e| e.into_inner())))),
        }
    }
}

/// Why a loop stopped short: an LLM that did not answer (a 502 to a client),
/// or anything else (the sandbox, the model, the settings).
#[derive(Debug)]
pub enum LoopError {
    Llm(LlmError),
    Other(String),
}

impl std::fmt::Display for LoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopError::Llm(err) => f.write_str(&err.message),
            LoopError::Other(message) => f.write_str(message),
        }
    }
}

impl From<LlmError> for LoopError {
    fn from(err: LlmError) -> LoopError {
        LoopError::Llm(err)
    }
}

impl From<String> for LoopError {
    fn from(message: String) -> LoopError {
        LoopError::Other(message)
    }
}

impl From<LoopError> for String {
    fn from(err: LoopError) -> String {
        err.to_string()
    }
}

impl From<LoopError> for ApiError {
    /// The LLM's failure is the upstream's (502); anything else is the request's.
    fn from(err: LoopError) -> ApiError {
        match err {
            LoopError::Llm(err) => err.into(),
            LoopError::Other(message) => ApiError::bad_request(message),
        }
    }
}

/// Sets `key` in a record: replaced where it already is, else added at the
/// end - `dict.update`, which keeps an existing key's place.
pub(crate) fn set(record: &mut Vec<(String, Json)>, key: &str, value: Json) {
    match record.iter_mut().find(|(k, _)| k == key) {
        Some(slot) => slot.1 = value,
        None => record.push((key.to_string(), value)),
    }
}

/// `list(dict.fromkeys(texts))`: the texts without repeats, in first-seen order.
pub(crate) fn unique(texts: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(texts.len());
    for text in texts {
        if !out.contains(text) {
            out.push(text.clone());
        }
    }
    out
}

/// The loss of the last record of a pass, or `null`.
pub(crate) fn last_loss(records: &[crate::model::EpochRecord]) -> Json {
    records.last().map_or(Json::Null, |r| Json::Num(r.loss))
}

/// The settings a trainer's 2NRL, reward and punish passes are given - what
/// Python passes `model.two_nrl` / `reward` / `punish`: the epochs, the
/// learning rates and the batch size at strength 1.  Each kind reads what
/// applies to it ([`crate::kinds`]).
pub(crate) fn feedback(neg_epochs: usize, pos_epochs: usize, neg_lr: f64, pos_lr: f64, batch_size: usize) -> Feedback {
    Feedback {
        neg_epochs,
        pos_epochs,
        neg_lr,
        pos_lr,
        batch_size: Some(batch_size),
        ..Feedback::two_nrl()
    }
}

// -- the trainer ---------------------------------------------------------------------------------

/// The loop's settings, Python's `CodeGenConfig` field for field.
///
/// The learning rates and the batch size are the sine model's; the count and
/// phase models accept and ignore them (a pass rewards or penalises by one
/// unit), as Python's do.
#[derive(Clone, Debug, PartialEq)]
pub struct CodeGenConfig {
    /// `ollama` or `chatgpt`: who tutors.
    pub teacher_provider: String,
    /// `""` resolves to [`default_teacher_model`].
    pub teacher_model: String,
    /// `""` resolves to the tutor's provider.
    pub judge_provider: String,
    /// `None`: the tutor's model on the tutor's provider, else that provider's default.
    pub judge_model: Option<String>,
    pub phases: Vec<String>,
    pub rounds: usize,
    pub teacher_attempts: usize,
    pub model_attempts: usize,
    /// The network's first attempt is its cheapest path, not a sample.
    pub first_attempt_dijkstra: bool,
    pub temperature: f64,
    /// Characters the network may generate per attempt.
    pub max_length: usize,
    pub strictness: String,
    pub use_judge: bool,
    /// When the network never succeeds, the teacher supplies the answer.
    pub fallback_teacher: bool,
    /// `problem` or `round`: when 2NRL runs.
    pub twonrl_per: String,
    pub replay: bool,
    pub replay_limit: usize,
    pub teacher_prompt: Option<String>,
    /// The prefix the network continues; `{problem}` is the prompt.
    pub model_prompt: String,
    pub neg_epochs: usize,
    pub pos_epochs: usize,
    pub neg_lr: f64,
    pub pos_lr: f64,
    pub batch_size: usize,
    /// Checkpoint every N problems (0 = off).
    pub checkpoint_every: usize,
}

impl Default for CodeGenConfig {
    /// Python's defaults, the providers resolved.
    fn default() -> CodeGenConfig {
        let mut config = CodeGenConfig {
            teacher_provider: DEFAULT_PROVIDER.to_string(),
            teacher_model: String::new(),
            judge_provider: String::new(),
            judge_model: None,
            phases: PHASES.iter().map(|p| p.to_string()).collect(),
            rounds: 1,
            teacher_attempts: 3,
            model_attempts: 4,
            first_attempt_dijkstra: true,
            temperature: 1.0,
            max_length: 800,
            strictness: "strict".to_string(),
            use_judge: true,
            fallback_teacher: true,
            twonrl_per: "problem".to_string(),
            replay: true,
            replay_limit: 64,
            teacher_prompt: None,
            model_prompt: "{problem}\n".to_string(),
            neg_epochs: 2,
            pos_epochs: 3,
            neg_lr: 0.5,
            pos_lr: 0.1,
            batch_size: 4,
            checkpoint_every: 0,
        };
        let _ = config.resolve();
        config
    }
}

impl CodeGenConfig {
    /// Resolves the providers and fills in the models they imply, so reports
    /// name the real model (Python's `__post_init__`).
    pub fn resolve(&mut self) -> Result<(), String> {
        let provider_error = || {
            format!(
                "teacher_provider and judge_provider must be one of {}",
                PROVIDERS.join(", ")
            )
        };
        self.teacher_provider = normalise_provider(&self.teacher_provider)
            .map_err(|_| provider_error())?
            .to_string();
        self.judge_provider = if self.judge_provider.trim().is_empty() {
            self.teacher_provider.clone()
        } else {
            normalise_provider(&self.judge_provider)
                .map_err(|_| provider_error())?
                .to_string()
        };
        if self.teacher_model.trim().is_empty() {
            self.teacher_model = default_teacher_model(&self.teacher_provider);
        }
        if self.judge_model.as_deref().is_some_and(|m| m.trim().is_empty()) {
            self.judge_model = None;
        }
        Ok(())
    }

    /// The model the judge answers with.
    pub fn resolved_judge_model(&self) -> String {
        if let Some(model) = self.judge_model.as_deref().filter(|m| !m.is_empty()) {
            return model.to_string();
        }
        if self.judge_provider == self.teacher_provider {
            return self.teacher_model.clone();
        }
        default_teacher_model(&self.judge_provider)
    }

    /// Refuses settings the loop cannot run with, in Python's words.
    pub fn validate(&self) -> Result<(), String> {
        if !PROVIDERS.contains(&self.teacher_provider.as_str()) || !PROVIDERS.contains(&self.judge_provider.as_str()) {
            return Err(format!(
                "teacher_provider and judge_provider must be one of {}",
                PROVIDERS.join(", ")
            ));
        }
        if self.phases.is_empty() || self.phases.iter().any(|p| !PHASES.contains(&p.as_str())) {
            return Err(format!("phases must be a non-empty subset of {}", PHASES.join(", ")));
        }
        if self.rounds < 1 {
            return Err("rounds must be >= 1".to_string());
        }
        if self.teacher_attempts < 1 || self.model_attempts < 1 {
            return Err("teacher_attempts and model_attempts must be >= 1".to_string());
        }
        if self.temperature.is_nan() || self.temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        if self.max_length < 1 {
            return Err("max_length must be >= 1".to_string());
        }
        if !STRICTNESS.contains(&self.strictness.as_str()) {
            return Err(format!("strictness must be one of {}", STRICTNESS.join(", ")));
        }
        if self.twonrl_per != "problem" && self.twonrl_per != "round" {
            return Err("twonrl_per must be 'problem' or 'round'".to_string());
        }
        if self.neg_lr < 0.0 || self.pos_lr < 0.0 {
            return Err("learning rates must be >= 0".to_string());
        }
        if self.batch_size < 1 {
            return Err("batch_size must be >= 1".to_string());
        }
        model_prefix(&self.model_prompt, &Problem::new("check", "x"))?;
        Ok(())
    }

    /// The settings as a document (`dataclasses.asdict` of Python's config).
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("teacher_provider", Json::str(self.teacher_provider.clone())),
            ("teacher_model", Json::str(self.teacher_model.clone())),
            ("judge_provider", Json::str(self.judge_provider.clone())),
            ("judge_model", self.judge_model.clone().map_or(Json::Null, Json::Str)),
            ("phases", Json::strs(self.phases.clone())),
            ("rounds", Json::Int(self.rounds as i64)),
            ("teacher_attempts", Json::Int(self.teacher_attempts as i64)),
            ("model_attempts", Json::Int(self.model_attempts as i64)),
            ("first_attempt_dijkstra", Json::Bool(self.first_attempt_dijkstra)),
            ("temperature", Json::Num(self.temperature)),
            ("max_length", Json::Int(self.max_length as i64)),
            ("strictness", Json::str(self.strictness.clone())),
            ("use_judge", Json::Bool(self.use_judge)),
            ("fallback_teacher", Json::Bool(self.fallback_teacher)),
            ("twonrl_per", Json::str(self.twonrl_per.clone())),
            ("replay", Json::Bool(self.replay)),
            ("replay_limit", Json::Int(self.replay_limit as i64)),
            (
                "teacher_prompt",
                self.teacher_prompt.clone().map_or(Json::Null, Json::Str),
            ),
            ("model_prompt", Json::str(self.model_prompt.clone())),
            ("neg_epochs", Json::Int(self.neg_epochs as i64)),
            ("pos_epochs", Json::Int(self.pos_epochs as i64)),
            ("neg_lr", Json::Num(self.neg_lr)),
            ("pos_lr", Json::Num(self.pos_lr)),
            ("batch_size", Json::Int(self.batch_size as i64)),
            ("checkpoint_every", Json::Int(self.checkpoint_every as i64)),
        ])
    }
}

/// What a run is handed besides the problems: where each record goes, when
/// to stop, and where checkpoints are written.
pub struct Hooks<'h> {
    /// Sees every record as it is written (attempts, problems, rounds).
    pub progress: &'h mut dyn FnMut(&Json),
    /// Asked between steps; `true` finishes the step in hand and stops.
    pub stop: &'h (dyn Fn() -> bool + Sync),
    pub checkpoints: Option<&'h Checkpoints>,
}

/// What one problem comes to: its record, and the wrong and correct solution
/// texts 2NRL learns from.
pub type ProblemOutcome = (Vec<(String, Json)>, Vec<String>, Vec<String>);

/// Runs the teacher / model phases over problems and applies 2NRL to the network.
pub struct CodeGenTrainer<'c> {
    /// The tutor.
    pub client: &'c dyn LlmClient,
    /// The judge: the tutor unless the settings name another provider or URL.
    pub judge: &'c dyn LlmClient,
    pub sandbox: Sandbox,
    pub config: CodeGenConfig,
    /// Every record written, in order.
    pub history: Vec<Json>,
    pub replay_buffer: Vec<String>,
    /// The correct solution text of each problem solved, by id, first solved first.
    pub solved: Vec<(String, String)>,
}

impl<'c> CodeGenTrainer<'c> {
    /// The loop, once its settings are known to be sound.
    pub fn new(
        client: &'c dyn LlmClient,
        judge: &'c dyn LlmClient,
        sandbox: Sandbox,
        mut config: CodeGenConfig,
    ) -> Result<CodeGenTrainer<'c>, String> {
        config.resolve()?;
        config.validate()?;
        Ok(CodeGenTrainer {
            client,
            judge,
            sandbox,
            config,
            history: Vec::new(),
            replay_buffer: Vec::new(),
            solved: Vec::new(),
        })
    }

    /// Runs a program, checks its style and - when it runs and the judge is
    /// on - asks the judge.
    pub fn evaluate(&self, problem: &Problem, code: &str, source: &str, index: usize) -> Result<Attempt, LoopError> {
        let started = Instant::now();
        let mut code = code.to_string();
        if !code.ends_with('\n') {
            code.push('\n');
        }
        let run = if code.trim().is_empty() {
            RunResult {
                ok: false,
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                error: Some("empty program".to_string()),
                timed_out: false,
                seconds: 0.0,
                expected_ok: None,
                network_isolated: self.sandbox.network_isolated(),
            }
        } else {
            self.sandbox
                .run(&code, problem.tests.as_deref(), problem.expected_output.as_deref(), "")?
        };
        let style = check_style(&code, Some(&self.sandbox.python));
        let opinion = if run.ok && self.config.use_judge {
            Some(judge_with_llm(
                self.judge,
                problem,
                &code,
                &run,
                &style,
                &self.config.resolved_judge_model(),
            )?)
        } else {
            None
        };
        let verdict = decide(
            &run,
            &style,
            opinion.as_ref(),
            &self.config.strictness,
            self.judge.provider(),
        )?;
        Ok(Attempt {
            index,
            source: source.to_string(),
            text: solution_text(problem, &code),
            code,
            run,
            style,
            verdict,
            seconds: started.elapsed().as_secs_f64(),
        })
    }

    fn emit(&mut self, hooks: &mut Hooks, record: Json) {
        (hooks.progress)(&record);
        self.history.push(record);
    }

    fn emit_attempt(&mut self, hooks: &mut Hooks, phase: &str, round: usize, problem: &Problem, attempt: &Attempt) {
        let v = &attempt.verdict;
        let record = Json::obj([
            ("kind", Json::str("attempt")),
            ("phase", Json::str(phase)),
            ("round", Json::Int(round as i64)),
            ("problem", Json::str(problem.id.clone())),
            ("attempt", Json::Int(attempt.index as i64 + 1)),
            ("source", Json::str(attempt.source.clone())),
            ("runs", Json::Bool(v.runs)),
            ("correct", Json::Bool(v.correct)),
            ("score", v.score.map_or(Json::Null, Json::Num)),
            ("judged_by", Json::str(v.judged_by.clone())),
            ("error", attempt.run.error.clone().map_or(Json::Null, Json::Str)),
            ("issues", Json::strs(v.issues.iter().take(4).cloned())),
            ("code", Json::str(head(&attempt.code, 2000))),
            ("stdout", Json::str(head(&attempt.run.stdout, 500))),
            ("seconds", Json::Num(attempt.seconds)),
        ]);
        self.emit(hooks, record);
    }

    /// The tutor writes a program and fixes its own until one is accepted or
    /// its attempts run out.
    pub fn solve_with_teacher(
        &mut self,
        problem: &Problem,
        phase: &str,
        round: usize,
        start_index: usize,
        hooks: &mut Hooks,
    ) -> Result<Vec<Attempt>, LoopError> {
        let cfg = self.config.clone();
        let source = self.client.provider();
        let mut code = teacher_generate(self.client, problem, cfg.teacher_prompt.as_deref(), &cfg.teacher_model)?;
        let mut attempts = Vec::new();
        for i in 0..cfg.teacher_attempts {
            let attempt = self.evaluate(problem, &code, source, start_index + i)?;
            self.emit_attempt(hooks, phase, round, problem, &attempt);
            let done = attempt.verdict.correct || i == cfg.teacher_attempts - 1 || (hooks.stop)();
            attempts.push(attempt);
            if done {
                break;
            }
            code = teacher_fix(
                self.client,
                problem,
                attempts.last().expect("just pushed"),
                cfg.teacher_prompt.as_deref(),
                &cfg.teacher_model,
            )?;
        }
        Ok(attempts)
    }

    /// The network's `index`-th program: its cheapest path first (unless
    /// `first_attempt_dijkstra` is off), samples after.
    pub fn generate_with_model(&self, model: &mut Model, problem: &Problem, index: usize) -> Result<String, String> {
        let cfg = &self.config;
        let prefix = model_prefix(&cfg.model_prompt, problem)?;
        let dijkstra = index == 0 && cfg.first_attempt_dijkstra;
        let found = model.predict(
            &prefix,
            &PredictOptions {
                length: 1,
                mode: if dijkstra { "dijkstra" } else { "sample" }.to_string(),
                temperature: cfg.temperature,
                max_length: Some(cfg.max_length),
                to_end: dijkstra,
                ..Default::default()
            },
        )?;
        Ok(found.best.text)
    }

    /// The network tries; when it never succeeds (and the settings allow)
    /// the teacher supplies the answer.
    pub fn solve_with_model(
        &mut self,
        nets: &mut Nets,
        problem: &Problem,
        phase: &str,
        round: usize,
        hooks: &mut Hooks,
    ) -> Result<Vec<Attempt>, LoopError> {
        let mut attempts = Vec::new();
        for i in 0..self.config.model_attempts {
            let code = nets.model(|m| self.generate_with_model(m, problem, i))?;
            let attempt = self.evaluate(problem, &code, "model", i)?;
            self.emit_attempt(hooks, phase, round, problem, &attempt);
            let done = attempt.verdict.correct || (hooks.stop)();
            attempts.push(attempt);
            if done {
                break;
            }
        }
        if !attempts.iter().any(|a| a.verdict.correct) && self.config.fallback_teacher && !(hooks.stop)() {
            let start = attempts.len();
            attempts.extend(self.solve_with_teacher(problem, phase, round, start, hooks)?);
        }
        Ok(attempts)
    }

    /// 2NRL on the wrong (`bad`) and correct (`good`) solution texts; replay
    /// keeps earlier successes in every positive phase.  The keys it adds to a
    /// record: `bad`, `good`, `action`, `neg_loss`, `pos_loss`.
    ///
    /// Every kind learns its own way ([`crate::kinds`]): the sine model by
    /// gradient descent at the configured learning rates and batch size, the
    /// count and phase models by counting at strength 1 - as Python hands the
    /// same call to whichever model is loaded.  `stop` ends a pass between
    /// epochs, as Python's `stop_event` does.
    pub fn learn(
        &mut self,
        nets: &mut Nets,
        bad: &[String],
        good: &[String],
        stop: &(dyn Fn() -> bool + Sync),
    ) -> Result<Vec<(String, Json)>, String> {
        let cfg = self.config.clone();
        let mut good_all = unique(good);
        if cfg.replay {
            let extra: Vec<String> = self
                .replay_buffer
                .iter()
                .filter(|t| !good_all.contains(t))
                .cloned()
                .collect();
            if cfg.replay_limit > 0 {
                good_all.extend(extra[extra.len().saturating_sub(cfg.replay_limit)..].iter().cloned());
            }
        }
        let mut result = vec![
            ("bad".to_string(), Json::Int(bad.len() as i64)),
            ("good".to_string(), Json::Int(good_all.len() as i64)),
            ("action".to_string(), Json::Null),
            ("neg_loss".to_string(), Json::Null),
            ("pos_loss".to_string(), Json::Null),
        ];
        if bad.is_empty() && good_all.is_empty() {
            return Ok(result);
        }
        let o = feedback(cfg.neg_epochs, cfg.pos_epochs, cfg.neg_lr, cfg.pos_lr, cfg.batch_size);
        let mut going = |_: &EpochRecord| !stop();
        if !bad.is_empty() && !good_all.is_empty() {
            let (negative, positive) = nets.model(|m| kinds::two_nrl(m, bad, &good_all, &o, &mut going))?;
            set(&mut result, "action", Json::str("2nrl"));
            set(&mut result, "neg_loss", last_loss(&negative));
            set(&mut result, "pos_loss", last_loss(&positive));
        } else if !good_all.is_empty() {
            let records = nets.model(|m| kinds::reward(m, &good_all, &o, &mut going))?;
            set(&mut result, "action", Json::str("reward"));
            set(&mut result, "pos_loss", last_loss(&records));
        } else {
            // punish only: the negative phase, then (the sine model) the
            // inversion that makes the wrong programs unlikely
            let records = nets.model(|m| kinds::punish(m, bad, &o, &mut going))?;
            set(&mut result, "action", Json::str("punish"));
            set(&mut result, "neg_loss", last_loss(&records));
        }
        for text in good {
            if !self.replay_buffer.contains(text) {
                self.replay_buffer.push(text.clone());
            }
        }
        if cfg.replay_limit > 0 && self.replay_buffer.len() > cfg.replay_limit {
            let excess = self.replay_buffer.len() - cfg.replay_limit;
            self.replay_buffer.drain(..excess);
        }
        Ok(result)
    }

    /// Hands a problem's attempts to the negative network: the sandbox, the
    /// style checker and the judge are its tutor.
    fn teach_negative(&self, nets: &mut Nets, attempts: &[Attempt]) -> Result<Option<TeachReport>, String> {
        if !nets.has_negative() || attempts.is_empty() {
            return Ok(None);
        }
        let o = TeachOptions::default();
        nets.negative(|negative| teach_attempts(negative, attempts, true, "codegen", &o))?
            .transpose()
    }

    /// One problem: its attempts, what they came to, and the texts 2NRL
    /// learns from - `(record, bad, good)`.
    pub fn run_problem(
        &mut self,
        nets: &mut Nets,
        problem: &Problem,
        phase: &str,
        round: usize,
        hooks: &mut Hooks,
    ) -> Result<ProblemOutcome, LoopError> {
        let started = Instant::now();
        let attempts = if phase == "teacher" {
            self.solve_with_teacher(problem, phase, round, 0, hooks)?
        } else {
            self.solve_with_model(nets, problem, phase, round, hooks)?
        };
        let Some(last) = attempts.last() else {
            return Err(LoopError::Other(format!(
                "no attempt was produced for {}",
                python_repr(&problem.id)
            )));
        };
        let correct = attempts.iter().find(|a| a.verdict.correct);
        let good: Vec<String> = attempts
            .iter()
            .filter(|a| a.verdict.correct)
            .map(|a| a.text.clone())
            .collect();
        let bad: Vec<String> = attempts
            .iter()
            .filter(|a| !a.verdict.correct)
            .map(|a| a.text.clone())
            .collect();
        if let Some(correct) = correct {
            set_solved(&mut self.solved, &problem.id, &correct.text);
        }
        let shown = correct.unwrap_or(last);
        let mut record = vec![
            ("kind".to_string(), Json::str("problem")),
            ("phase".to_string(), Json::str(phase)),
            ("round".to_string(), Json::Int(round as i64)),
            ("problem".to_string(), Json::str(problem.id.clone())),
            ("prompt".to_string(), Json::str(head(&problem.prompt, 200))),
            ("attempts".to_string(), Json::Int(attempts.len() as i64)),
            ("correct".to_string(), Json::Bool(correct.is_some())),
            (
                "solved_by".to_string(),
                correct.map_or(Json::Null, |a| Json::str(a.source.clone())),
            ),
            (
                "model_solved".to_string(),
                Json::Bool(attempts.iter().any(|a| a.source == "model" && a.verdict.correct)),
            ),
            ("bad".to_string(), Json::Int(bad.len() as i64)),
            ("good".to_string(), Json::Int(good.len() as i64)),
            ("score".to_string(), shown.verdict.score.map_or(Json::Null, Json::Num)),
            ("code".to_string(), Json::str(head(&shown.code, 2000))),
            (
                "issues".to_string(),
                Json::strs(shown.verdict.issues.iter().take(4).cloned()),
            ),
            ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
        ];
        if let Some(taught) = self.teach_negative(nets, &attempts)? {
            record.push(("negative_blamed".to_string(), Json::Int(taught.blamed as i64)));
            record.push((
                "negative_reasons".to_string(),
                Json::Obj(
                    taught
                        .reasons
                        .iter()
                        .map(|(r, n)| (r.clone(), Json::Int(*n as i64)))
                        .collect(),
                ),
            ));
        }
        Ok((record, bad, good))
    }

    /// Every round and phase over `problems`; the problem and round records
    /// (the attempts go to the progress hook too).
    pub fn run(&mut self, nets: &mut Nets, problems: &[Problem], hooks: &mut Hooks) -> Result<Vec<Json>, LoopError> {
        if problems.is_empty() {
            return Err(LoopError::Other("no problems to solve".to_string()));
        }
        let cfg = self.config.clone();
        let mut records = Vec::new();
        let mut step = 0usize;
        for round in 1..=cfg.rounds {
            for phase in &cfg.phases {
                let started = Instant::now();
                let (mut round_bad, mut round_good) = (Vec::new(), Vec::new());
                let (mut solved, mut model_solved, mut done) = (0, 0, 0);
                for problem in problems {
                    if (hooks.stop)() {
                        break;
                    }
                    let begun = Instant::now();
                    let (mut record, bad, good) = self.run_problem(nets, problem, phase, round, hooks)?;
                    if cfg.twonrl_per == "problem" {
                        for (key, value) in self.learn(nets, &bad, &good, hooks.stop)? {
                            set(&mut record, &key, value);
                        }
                    } else {
                        round_bad.extend(bad);
                        round_good.extend(good);
                    }
                    done += 1;
                    let record = {
                        set(&mut record, "seconds", Json::Num(begun.elapsed().as_secs_f64()));
                        Json::Obj(record)
                    };
                    if record.at("correct") == &Json::Bool(true) {
                        solved += 1;
                    }
                    if record.at("model_solved") == &Json::Bool(true) {
                        model_solved += 1;
                    }
                    crate::log_info!(
                        LOG,
                        "{phase} {}: {} after {} attempt(s)",
                        problem.id,
                        if record.at("correct") == &Json::Bool(true) {
                            "solved"
                        } else {
                            "unsolved"
                        },
                        record.at("attempts").as_i64().unwrap_or(0)
                    );
                    self.emit(hooks, record.clone());
                    records.push(record.clone());
                    step += 1;
                    if let Some(manager) = hooks.checkpoints.filter(|_| cfg.checkpoint_every > 0) {
                        if step % cfg.checkpoint_every == 0 {
                            let metrics = Json::obj([
                                ("phase", record.at("phase").clone()),
                                ("round", record.at("round").clone()),
                                ("problem", record.at("problem").clone()),
                                ("correct", record.at("correct").clone()),
                            ]);
                            nets.model(|m| manager.save(m, step as i64, "codegen", Some(metrics)))?;
                        }
                    }
                }
                let learned = if cfg.twonrl_per == "round"
                    && (!round_bad.is_empty() || !round_good.is_empty())
                    && !(hooks.stop)()
                {
                    self.learn(nets, &round_bad, &round_good, hooks.stop)?
                } else {
                    Vec::new()
                };
                let mut summary = vec![
                    ("kind".to_string(), Json::str("round")),
                    ("phase".to_string(), Json::str(phase.clone())),
                    ("round".to_string(), Json::Int(round as i64)),
                    ("problems".to_string(), Json::Int(done)),
                    ("solved".to_string(), Json::Int(solved)),
                    ("model_solved".to_string(), Json::Int(model_solved)),
                    ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
                ];
                summary.extend(learned);
                let summary = Json::Obj(summary);
                self.emit(hooks, summary.clone());
                records.push(summary);
                if (hooks.stop)() {
                    return Ok(records);
                }
            }
        }
        Ok(records)
    }
}

/// `solved[id] = text`: a problem solved again keeps its first place.
pub(crate) fn set_solved(solved: &mut Vec<(String, String)>, id: &str, text: &str) {
    match solved.iter_mut().find(|(k, _)| k == id) {
        Some(slot) => slot.1 = text.to_string(),
        None => solved.push((id.to_string(), text.to_string())),
    }
}

/// `{id: text, ...}` of the solutions found so far.
pub(crate) fn solutions_json(solved: &[(String, String)]) -> Json {
    Json::Obj(solved.iter().map(|(k, v)| (k.clone(), Json::str(v.clone()))).collect())
}

// -- the command line ------------------------------------------------------------------------------

/// A count flag that must be at least `minimum`, in the words of the other
/// commands' flags.
fn whole(ctx: &Ctx, name: &str, default: usize, minimum: usize) -> Result<usize, String> {
    crate::llm::count_flag(ctx, name, default as i64, minimum as i64)
}

/// A number flag that must be finite and at least `minimum`.
pub(crate) fn number_at_least(ctx: &Ctx, name: &str, default: f64, minimum: f64) -> Result<f64, String> {
    let value = ctx.args.float(name, default)?;
    if !value.is_finite() || value < minimum {
        return Err(format!(
            "--{name} must be a number >= {}, got {value}",
            format_g(minimum)
        ));
    }
    Ok(value)
}

/// One of a few words, lower-cased.
pub(crate) fn choice(ctx: &Ctx, name: &str, default: &str, allowed: &[&str]) -> Result<String, String> {
    let value = ctx.args.str(name, default).trim().to_lowercase();
    if !allowed.contains(&value.as_str()) {
        return Err(format!(
            "argument --{name}: invalid choice: {} (choose from {})",
            python_repr(&value),
            allowed.iter().map(|a| python_repr(a)).collect::<Vec<_>>().join(", ")
        ));
    }
    Ok(value)
}

/// Where a command's model came from: `{"kind": "model", "path"}` for a file,
/// `{"kind": "new", "path": null}` for a fresh one (Python's `Origin`).
pub(crate) fn origin_json(ctx: &Ctx) -> Json {
    if std::path::Path::new(&ctx.model_path).is_file() {
        Json::obj([
            ("kind", Json::str("model")),
            ("path", Json::str(ctx.model_path.clone())),
        ])
    } else {
        Json::obj([("kind", Json::str("new")), ("path", Json::Null)])
    }
}

/// Saves the negative network and reports what the tutor taught it
/// (Python's `_save_negative`).
pub(crate) fn save_negative(negative: &mut Model, path: &str) -> Result<Json, String> {
    negative.save(path)?;
    Ok(Json::obj([
        ("path", Json::str(path)),
        ("saved", crate::llm::saved(path)),
        ("reasons", crate::review::reason_rows(negative)),
        ("stats", negative_stats(negative)),
    ]))
}

/// The settings of `radixnet codegen`, from its flags.
fn config_from_args(ctx: &Ctx, every: usize) -> Result<CodeGenConfig, String> {
    let args = &ctx.args;
    let d = CodeGenConfig::default();
    let phase = choice(ctx, "phase", "both", &["both", "teacher", "model"])?;
    let teacher_provider =
        normalise_provider(&args.str("teacher-provider", &args.str("provider", DEFAULT_PROVIDER)))?.to_string();
    let judge_provider = match args.get("judge-provider").filter(|p| !p.is_empty()) {
        Some(p) => normalise_provider(p)?.to_string(),
        None => teacher_provider.clone(),
    };
    let teacher_model = match args.get("teacher-model").filter(|m| !m.is_empty()) {
        Some(model) => model.to_string(),
        None => default_teacher_model(&teacher_provider),
    };
    let mut config = CodeGenConfig {
        teacher_provider,
        teacher_model,
        judge_provider,
        judge_model: args.get("judge-model").map(str::to_string),
        phases: if phase == "both" { d.phases.clone() } else { vec![phase] },
        rounds: whole(ctx, "rounds", d.rounds, 1)?,
        teacher_attempts: whole(ctx, "teacher-attempts", d.teacher_attempts, 1)?,
        model_attempts: whole(ctx, "model-attempts", d.model_attempts, 1)?,
        first_attempt_dijkstra: !args.on("sample-first"),
        temperature: crate::llm::nonneg_flag(ctx, "temperature", d.temperature)?,
        max_length: whole(ctx, "max-length", d.max_length, 1)?,
        strictness: choice(ctx, "strictness", "strict", STRICTNESS)?,
        use_judge: !args.on("no-judge"),
        fallback_teacher: !args.on("no-fallback-teacher"),
        twonrl_per: choice(ctx, "twonrl-per", "problem", &["problem", "round"])?,
        replay: !args.on("no-replay"),
        replay_limit: d.replay_limit,
        teacher_prompt: args.get("teacher-prompt").map(str::to_string),
        model_prompt: args.str("model-prompt", &d.model_prompt),
        neg_epochs: whole(ctx, "neg-epochs", d.neg_epochs, 0)?,
        pos_epochs: whole(ctx, "pos-epochs", d.pos_epochs, 0)?,
        neg_lr: crate::llm::nonneg_flag(ctx, "neg-lr", d.neg_lr)?,
        pos_lr: crate::llm::nonneg_flag(ctx, "pos-lr", d.pos_lr)?,
        batch_size: whole(ctx, "batch-size", d.batch_size, 1)?,
        checkpoint_every: every,
    };
    config.resolve()?;
    Ok(config)
}

/// `radixnet codegen --problems FILE`: the teacher writes, the sandbox runs,
/// the judge marks and 2NRL teaches the network - then the model is saved to
/// `--out` (default `--model`), and with `--blame` the negative network
/// beside it.  Flags as Python's: `--phase both|teacher|model`, `--rounds`,
/// `--teacher-provider` (`--provider`), `--teacher-model`, `--judge-provider`,
/// `--judge-model`, `--url`, `--judge-url`, `--timeout`, `--teacher-attempts`,
/// `--model-attempts`, `--sample-first`, `--temperature`, `--max-length`,
/// `--strictness`, `--no-judge`, `--no-fallback-teacher`, `--twonrl-per`,
/// `--no-replay`, `--teacher-prompt`, `--model-prompt`, `--sandbox-timeout`,
/// `--memory-mb`, `--no-network-isolation`, the 2NRL epochs, the checkpoint
/// options, `--report FILE` and `--negative PATH`.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    let path = args
        .get("problems")
        .filter(|p| !p.is_empty())
        .ok_or("codegen needs --problems FILE (one prompt per line, or .json / .jsonl objects)")?
        .to_string();
    let problems = load_problems(&path).map_err(|err| format!("cannot load problems from {path}: {err}"))?;
    let schedule = Schedule::from_args(args)?;
    let config = config_from_args(
        ctx,
        if schedule.manager().is_some() {
            schedule.every
        } else {
            0
        },
    )?;
    let judged = config.use_judge && config.judge_provider == CHATGPT;
    if (config.teacher_provider == CHATGPT || judged) && !crate::chatgpt::key_configured() {
        return Err(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT tutor, or use \
             --teacher-provider ollama"
                .to_string(),
        );
    }
    config.validate()?;
    let timeout = crate::llm::timeout_flag(ctx)?;
    let client = new_client(
        &config.teacher_provider,
        &args.str("url", ""),
        &config.teacher_model,
        timeout,
    )?;
    let judge_url = args.str("judge-url", "");
    let separate = if config.judge_provider != config.teacher_provider || !judge_url.is_empty() {
        Some(new_client(
            &config.judge_provider,
            &judge_url,
            &config.resolved_judge_model(),
            timeout,
        )?)
    } else {
        None
    };
    let judge: &dyn LlmClient = separate.as_deref().unwrap_or(client.as_ref());
    let sandbox = Sandbox::new(SandboxOptions {
        timeout: Duration::from_secs_f64(number_at_least(ctx, "sandbox-timeout", 10.0, 0.1)?),
        memory_mb: ctx.args.usize("memory-mb", 256)? as u64,
        isolate_network: !args.on("no-network-isolation"),
        python: args.get("python").map(str::to_string),
        ..Default::default()
    })?;
    let origin = origin_json(ctx);
    let mut model = ctx.open(false)?;
    let mut negative = if args.on("blame") {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let out = ctx.out.clone().unwrap_or_else(|| ctx.model_path.clone());
    crate::log_info!(
        LOG,
        "{} problem(s) from {path}; {}, {} round(s); teacher {}: {} at {}; judge {}; sandbox timeout={}s memory={}MB \
         network={}",
        problems.len(),
        config.phases.join(" -> "),
        config.rounds,
        config.teacher_provider,
        config.teacher_model,
        client.url(),
        if config.use_judge {
            format!(
                "{}: {} at {}",
                config.judge_provider,
                config.resolved_judge_model(),
                judge.url()
            )
        } else {
            "off (sandbox, expected output and tests only)".to_string()
        },
        format_g(sandbox.timeout.as_secs_f64()),
        sandbox.memory_mb,
        if sandbox.network_isolated() {
            "isolated"
        } else {
            "NOT isolated"
        }
    );
    let mut trainer = CodeGenTrainer::new(client.as_ref(), judge, sandbox, config.clone())?;
    let mut attempts_seen = 0usize;
    let records = {
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: negative.as_mut(),
        };
        let mut progress = |record: &Json| {
            if record.at("kind").as_str() == Some("attempt") {
                attempts_seen += 1;
                crate::log_info!(
                    LOG,
                    "  {} attempt {} [{}] {}",
                    record.at("problem").as_str().unwrap_or(""),
                    record.at("attempt").as_i64().unwrap_or(0),
                    record.at("source").as_str().unwrap_or(""),
                    if record.at("correct") == &Json::Bool(true) {
                        "correct"
                    } else if record.at("runs") == &Json::Bool(true) {
                        "runs but rejected"
                    } else {
                        "error"
                    }
                );
            }
        };
        let mut hooks = Hooks {
            progress: &mut progress,
            stop: &|| false,
            checkpoints: schedule.manager(),
        };
        trainer.run(&mut nets, &problems, &mut hooks)?
    };
    model.save(&out)?;
    let problem_records: Vec<&Json> = records
        .iter()
        .filter(|r| r.at("kind").as_str() == Some("problem"))
        .collect();
    let solved = problem_records
        .iter()
        .filter(|r| r.at("correct") == &Json::Bool(true))
        .count();
    let by_model = problem_records
        .iter()
        .filter(|r| r.at("model_solved") == &Json::Bool(true))
        .count();
    crate::log_info!(
        LOG,
        "{solved}/{} problem runs solved ({by_model} by the model); {} of {} problems have a correct solution",
        problem_records.len(),
        trainer.solved.len(),
        problems.len()
    );
    let mut doc = vec![
        ("model".to_string(), origin),
        ("out".to_string(), Json::str(out.clone())),
        (
            "problems".to_string(),
            Json::Arr(problems.iter().map(Problem::to_json).collect()),
        ),
        ("config".to_string(), config.to_json()),
        ("records".to_string(), Json::Arr(records.clone())),
        ("attempts".to_string(), Json::Int(attempts_seen as i64)),
        ("solved".to_string(), Json::Int(solved as i64)),
        ("model_solved".to_string(), Json::Int(by_model as i64)),
        ("solutions".to_string(), solutions_json(&trainer.solved)),
        ("interrupted".to_string(), Json::Bool(false)),
        ("saved".to_string(), crate::llm::saved(&out)),
        ("stats".to_string(), stats(&model)),
    ];
    if let Some(negative) = negative.as_mut() {
        doc.push(("negative".to_string(), save_negative(negative, &ctx.negative_path())?));
    }
    let doc = Json::Obj(doc);
    if let Some(report) = args.get("report").filter(|r| !r.is_empty()) {
        let kept = Json::obj(
            ["problems", "config", "records", "solutions", "solved", "model_solved"].map(|k| (k, doc.at(k).clone())),
        );
        std::fs::write(report, kept.render(2)).map_err(|err| format!("cannot write {report}: {err}"))?;
        crate::log_info!(LOG, "wrote report to {report}");
    }
    ctx.emit(doc);
    Ok(())
}

// -- the server ------------------------------------------------------------------------------------

/// What the server keeps for this area: the attempt, problem and round
/// records of every codegen run, oldest first.
#[derive(Default)]
pub struct State {
    history: Mutex<Vec<Json>>,
}

impl State {
    /// Reads the `serve` command's flags for this area (it has none).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    fn push(&self, record: Json) {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).push(record);
    }

    /// Every record so far.
    pub fn history(&self) -> Vec<Json> {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }
}

/// This area's routes:
/// `POST /api/codegen/start`
/// `POST /api/codegen/solve`
/// `POST /api/codegen/run`
/// `GET /api/codegen/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/codegen/start", start_route);
    server.route("GET", "/api/codegen/history", history_route);
    server.route("POST", "/api/codegen/solve", solve_route);
    server.route("POST", "/api/codegen/run", run_route);
}

/// `[(file name, text)]` of an upload: one pair for a text file, one per text
/// entry of a ZIP archive (Python's `upload_entries`); 404 when it is not there.
pub(crate) fn upload_entries(svc: &Service, name: &str) -> Result<Vec<(String, String)>, ApiError> {
    let dir = svc
        .upload_dir
        .as_ref()
        .ok_or_else(|| ApiError::bad_request("no upload directory is configured"))?;
    let path = crate::service::safe_upload(dir, name)?;
    let base = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if !path.is_file() {
        return Err(ApiError::not_found(format!("no upload named {}", python_repr(&base))));
    }
    if let Some(entries) = crate::source::archive_entries(&path, svc.workers)
        .map_err(|err| ApiError::bad_request(format!("{base}: {err}")))?
    {
        return Ok(entries);
    }
    let bytes = std::fs::read(&path).map_err(|err| ApiError::bad_request(format!("cannot read {base}: {err}")))?;
    Ok(vec![(base, crate::source::decode_text(&bytes))])
}

/// The problems of a request: `problems` (strings / objects), `problems_text`
/// (one per line) and `problem_files` (uploads).
fn problems_from(svc: &Service, r: &Request) -> Result<Vec<Problem>, ApiError> {
    let f = Fields::of(r);
    let mut items: Vec<Json> = Vec::new();
    match f.get("problems") {
        None => {}
        Some(Json::Arr(list)) => items.extend(list.iter().cloned()),
        Some(_) => {
            return Err(ApiError::bad_request(
                "'problems' must be a list of strings or {prompt, tests, expected_output} objects",
            ))
        }
    }
    if let Some(text) = f.text("problems_text")? {
        items.extend(
            crate::llm::splitlines(&text)
                .into_iter()
                .filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#'))
                .map(|line| Json::str(line.trim())),
        );
    }
    for name in f.names("problem_files")? {
        for (entry, content) in upload_entries(svc, &name)? {
            let parsed = parse_problem_file(&content, &extension(&entry)).map_err(|err| {
                let base = std::path::Path::new(&name)
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| name.clone());
                let place = if entry == base {
                    format!("upload {}", python_repr(&name))
                } else {
                    format!("upload {} ({entry})", python_repr(&name))
                };
                ApiError::bad_request(format!("{place}: {err}"))
            })?;
            items.extend(parsed.iter().map(Problem::to_json));
        }
    }
    if items.is_empty() {
        return Err(ApiError::bad_request(
            "missing 'problems' (list of strings / objects), 'problems_text' (one per line) or 'problem_files' \
             (upload names)",
        ));
    }
    parse_problems(&items).map_err(ApiError::bad_request)
}

/// A provider named by a request field (or its alias), `default` when neither
/// is given; a name that is not one is refused with the field's name.
pub(crate) fn provider_field(f: &Fields, name: &str, alias: Option<&str>, default: &str) -> Result<String, ApiError> {
    let mut raw = f.text(name)?.filter(|t| !t.is_empty());
    if raw.is_none() {
        if let Some(alias) = alias {
            raw = f.text(alias)?.filter(|t| !t.is_empty());
        }
    }
    let Some(raw) = raw else {
        return Ok(default.to_string());
    };
    normalise_provider(&raw).map(str::to_string).map_err(|_| {
        ApiError::bad_request(format!(
            "'{name}' must be one of {} (got {})",
            PROVIDERS.join(", "),
            python_repr(&raw)
        ))
    })
}

/// The model this server tutors and judges with when a request names none:
/// the ChatGPT default of the server, else codegen's own Ollama default.
fn default_llm_model(svc: &Service, provider: &str) -> String {
    if provider == CHATGPT {
        svc.llm.chatgpt_model.clone()
    } else {
        default_teacher_model(OLLAMA)
    }
}

/// The codegen settings of a request body, defaults filled in; every bad one
/// is a 400 before anything starts.
fn config_from_fields(svc: &Service, f: &Fields) -> Result<CodeGenConfig, ApiError> {
    let d = CodeGenConfig::default();
    let raw_phases = match f.get("phases") {
        Some(value) => Some(value),
        None => f.get("phase"),
    };
    let phases_error = || ApiError::bad_request("'phases' must be 'both', 'teacher', 'model' or a list of those");
    let phases = match raw_phases {
        None => d.phases.clone(),
        Some(Json::Str(text)) => {
            let name = text.trim().to_lowercase();
            if name == "both" {
                PHASES.iter().map(|p| p.to_string()).collect()
            } else {
                vec![name]
            }
        }
        Some(Json::Arr(items)) if items.iter().all(|i| matches!(i, Json::Str(_))) => items
            .iter()
            .map(|i| i.as_str().unwrap_or("").trim().to_lowercase())
            .collect(),
        Some(_) => return Err(phases_error()),
    };
    let teacher_provider = provider_field(f, "teacher_provider", Some("provider"), &d.teacher_provider)?;
    let judge_provider = provider_field(f, "judge_provider", None, &teacher_provider)?;
    let teacher_model = match f.text("teacher_model")?.filter(|m| !m.is_empty()) {
        Some(model) => model,
        None => match f.text("model")?.filter(|m| !m.is_empty()) {
            Some(model) => model,
            None => default_llm_model(svc, &teacher_provider),
        },
    };
    let mut judge_model = f.text("judge_model")?;
    if judge_model.is_none() && judge_provider != teacher_provider {
        judge_model = Some(default_llm_model(svc, &judge_provider));
    }
    let mut config = CodeGenConfig {
        teacher_provider,
        teacher_model,
        judge_provider,
        judge_model,
        phases,
        rounds: f.count_or("rounds", d.rounds, 1)?,
        teacher_attempts: f.count_or("teacher_attempts", d.teacher_attempts, 1)?,
        model_attempts: f.count_or("model_attempts", d.model_attempts, 1)?,
        first_attempt_dijkstra: f.flag("first_attempt_dijkstra", d.first_attempt_dijkstra)?,
        temperature: f.number_or("temperature", d.temperature, Some(0.0))?,
        max_length: f.count_or("max_length", d.max_length, 1)?,
        strictness: f.text_or("strictness", &d.strictness)?.trim().to_lowercase(),
        use_judge: f.flag("judge", d.use_judge)?,
        fallback_teacher: f.flag("fallback_teacher", d.fallback_teacher)?,
        twonrl_per: f.text_or("twonrl_per", &d.twonrl_per)?.trim().to_lowercase(),
        replay: f.flag("replay", d.replay)?,
        replay_limit: f.count_or("replay_limit", d.replay_limit, 0)?,
        teacher_prompt: f.text("teacher_prompt")?,
        model_prompt: f.text_or("model_prompt", &d.model_prompt)?,
        neg_epochs: f.count_or("neg_epochs", d.neg_epochs, 0)?,
        pos_epochs: f.count_or("pos_epochs", d.pos_epochs, 0)?,
        neg_lr: f.number_or("neg_lr", d.neg_lr, Some(0.0))?,
        pos_lr: f.number_or("pos_lr", d.pos_lr, Some(0.0))?,
        batch_size: f.count_or("batch_size", d.batch_size, 1)?,
        checkpoint_every: f.count_or("checkpoint_every", d.checkpoint_every, 0)?,
    };
    config.resolve().map_err(ApiError::bad_request)?;
    config.validate().map_err(ApiError::bad_request)?;
    Ok(config)
}

/// A request's tutor, and its judge when that is another client.
type Clients = (Box<dyn LlmClient>, Option<Box<dyn LlmClient>>);

/// The tutor and the judge of a request: `url` / `judge_url` override the
/// server's defaults, and the judge is the tutor unless a provider or URL of
/// its own is named.
fn clients_from(svc: &Service, f: &Fields, config: &CodeGenConfig) -> Result<Clients, ApiError> {
    let timeout = f.number("timeout", Some(1.0))?.and_then(crate::llm::seconds);
    let url = f.text("url")?.filter(|u| !u.is_empty());
    let client = svc.llm.client(
        &config.teacher_provider,
        url.as_deref(),
        Some(&config.teacher_model),
        timeout,
    )?;
    let judge_url = f.text("judge_url")?.filter(|u| !u.is_empty());
    if config.judge_provider == config.teacher_provider && judge_url.is_none() {
        return Ok((client, None));
    }
    let judge = svc.llm.client(
        &config.judge_provider,
        judge_url.as_deref(),
        Some(&config.resolved_judge_model()),
        timeout,
    )?;
    Ok((client, Some(judge)))
}

/// The sandbox of a request: `sandbox_timeout`, `memory_mb`, `network_isolation`.
fn sandbox_from(f: &Fields) -> Result<Sandbox, ApiError> {
    Sandbox::new(SandboxOptions {
        timeout: Duration::from_secs_f64(f.number_or("sandbox_timeout", 10.0, Some(0.1))?),
        memory_mb: f.count_or("memory_mb", 256, 0)? as u64,
        isolate_network: f.flag("network_isolation", true)?,
        ..Default::default()
    })
    .map_err(ApiError::bad_request)
}

/// Starts a `codegen` job: rounds of the teacher and the network solving the
/// problems, every attempt run and judged, and 2NRL on what came out right and
/// wrong.  With `blame` the rejected programs also teach the negative network
/// why they were rejected.  Answers 202 with the job, the problem ids, the
/// settings and the sandbox; the records follow on `/api/job` and
/// `/api/codegen/history`.
fn start_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let problems = problems_from(svc, r)?;
    let config = config_from_fields(svc, &f)?;
    let (client, judge) = clients_from(svc, &f, &config)?;
    let sandbox = sandbox_from(&f)?;
    let blame = f.flag("blame", false)?;
    if config.checkpoint_every > 0 {
        crate::checkpoint::require(svc, "checkpoint_every")?;
    }
    if blame {
        // loaded now, so a file that is not a negative network is this request's 400
        svc.ensure_negative()?;
    }
    svc.ensure_idle()?;
    let sandbox_doc = Json::obj([
        ("timeout", Json::Num(sandbox.timeout.as_secs_f64())),
        ("memory_mb", Json::Int(sandbox.memory_mb as i64)),
        ("network_isolated", Json::Bool(sandbox.network_isolated())),
    ]);
    let ids = Json::strs(problems.iter().map(|p| p.id.clone()));
    let config_doc = config.to_json();
    svc.start_job("codegen");
    let worker = Arc::clone(svc);
    std::thread::spawn(move || {
        let mut emitted: Vec<Json> = Vec::new();
        let outcome = (|| -> Result<(), String> {
            let judge_ref: &dyn LlmClient = judge.as_deref().unwrap_or(client.as_ref());
            let mut trainer = CodeGenTrainer::new(client.as_ref(), judge_ref, sandbox, config)?;
            let mut nets = Nets::Server {
                svc: &worker,
                negative: blame,
            };
            let manager = crate::checkpoint::manager(&worker);
            let mut progress = |record: &Json| {
                worker.codegen.push(record.clone());
                worker.job_progress(record.clone());
                emitted.push(record.clone());
            };
            let mut hooks = Hooks {
                progress: &mut progress,
                stop: &|| worker.stopping(),
                checkpoints: manager,
            };
            trainer.run(&mut nets, &problems, &mut hooks)?;
            Ok(())
        })();
        worker.finish_job(outcome.map(|()| emitted));
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("problems", ids),
        ("config", config_doc),
        ("sandbox", sandbox_doc),
    ])))
}

/// The attempt, problem and round records of every codegen run.
fn history_route(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(Json::obj([("history", Json::Arr(svc.codegen.history()))]))
}

/// One problem now, without training: `{problem, source: model|teacher,
/// attempts, ...the codegen settings}`.  The network writes under the model's
/// lock; the sandbox and the judge run without it.
fn solve_route(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let raw = f.get("problem").ok_or_else(|| {
        ApiError::bad_request("missing field 'problem' (a string or {prompt, tests, expected_output})")
    })?;
    let problem = Problem::from_json(raw, 1).map_err(ApiError::bad_request)?;
    let source = f.text_or("source", "model")?.trim().to_lowercase();
    if source != "model" && source != "teacher" {
        return Err(ApiError::bad_request(format!(
            "'source' must be 'model' or 'teacher' (got {})",
            python_repr(&source)
        )));
    }
    let count = f.count_or("attempts", 1, 1)?;
    let mut config = config_from_fields(svc, &f)?;
    config.teacher_attempts = count;
    config.model_attempts = count;
    config.fallback_teacher = false;
    let (client, judge) = clients_from(svc, &f, &config)?;
    let judge_ref: &dyn LlmClient = judge.as_deref().unwrap_or(client.as_ref());
    let sandbox = sandbox_from(&f)?;
    let isolated = sandbox.network_isolated();
    let mut trainer =
        CodeGenTrainer::new(client.as_ref(), judge_ref, sandbox, config).map_err(ApiError::bad_request)?;
    let mut attempts: Vec<Attempt> = Vec::new();
    if source == "model" {
        let codes = svc.with_model(|m| {
            (0..count)
                .map(|i| trainer.generate_with_model(m, &problem, i))
                .collect::<Result<Vec<String>, String>>()
        })?;
        // the sandbox and the judge run without the model lock
        for (i, code) in codes.iter().enumerate() {
            let attempt = trainer.evaluate(&problem, code, "model", i)?;
            let done = attempt.verdict.correct;
            attempts.push(attempt);
            if done {
                break;
            }
        }
    } else {
        let mut hooks = Hooks {
            progress: &mut |_| {},
            stop: &|| false,
            checkpoints: None,
        };
        attempts = trainer.solve_with_teacher(&problem, "teacher", 0, 0, &mut hooks)?;
    }
    Ok(Json::obj([
        ("problem", problem.to_json()),
        ("source", Json::str(source)),
        ("attempts", Json::Arr(attempts.iter().map(Attempt::to_json).collect())),
        ("correct", Json::Bool(attempts.iter().any(|a| a.verdict.correct))),
        ("sandbox", Json::obj([("network_isolated", Json::Bool(isolated))])),
    ]))
}

/// Runs a program in the sandbox and checks it, no LLM involved: `{code,
/// tests, expected_output, stdin, strictness, sandbox_timeout, memory_mb,
/// network_isolation}` -> `{run, style, verdict}`.
fn run_route(_svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let mut code = f.required_text("code")?;
    if code.trim().is_empty() {
        return Err(ApiError::bad_request("'code' is empty"));
    }
    if !code.ends_with('\n') {
        code.push('\n');
    }
    let strictness = f.text_or("strictness", "strict")?.trim().to_lowercase();
    let sandbox = sandbox_from(&f)?;
    let tests = f.text("tests")?;
    let expected = f.text("expected_output")?;
    let stdin = f.text_or("stdin", "")?;
    let run = sandbox
        .run(&code, tests.as_deref(), expected.as_deref(), &stdin)
        .map_err(ApiError::bad_request)?;
    let style = check_style(&code, Some(&sandbox.python));
    let verdict = decide(&run, &style, None, &strictness, DEFAULT_PROVIDER).map_err(ApiError::bad_request)?;
    Ok(Json::obj([
        ("run", run.to_json()),
        ("style", style.to_json()),
        ("verdict", verdict.to_json()),
    ]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::negative::NegativeOptions;
    use crate::toolbox::sandbox::python_available;

    /// A tutor that hands out scripted programs and judges by content: a
    /// program that prints `BAD_ANSWER` does not do the task.  It keeps every
    /// prompt it was sent.
    struct Scripted {
        programs: Mutex<Vec<String>>,
        prompts: Mutex<Vec<(String, LlmOptions)>>,
    }

    impl Scripted {
        fn new(programs: &[&str]) -> Scripted {
            Scripted {
                programs: Mutex::new(programs.iter().map(|p| p.to_string()).collect()),
                prompts: Mutex::new(Vec::new()),
            }
        }
    }

    impl LlmClient for Scripted {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://scripted"
        }
        fn model(&self) -> &str {
            "scripted"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.prompts.lock().unwrap().push((prompt.to_string(), o.clone()));
            if o.json {
                let task = !prompt.contains("BAD_ANSWER");
                return Ok(format!(
                    "{{\"task_accomplished\": {task}, \"pep8\": true, \"naming\": true, \"score\": {}, \
                     \"issues\": {}, \"critique\": \"{}\"}}",
                    if task { 9 } else { 2 },
                    if task { "[]" } else { "[\"prints the wrong text\"]" },
                    if task { "fine" } else { "does not solve the task" }
                ));
            }
            let mut programs = self.programs.lock().unwrap();
            Ok(if programs.is_empty() {
                "```python\nprint(\"hello\")\n```".to_string()
            } else {
                programs.remove(0)
            })
        }
        fn chat(&self, _m: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn hello() -> Problem {
        Problem {
            expected_output: Some("hello".to_string()),
            ..Problem::new("hello", "Print the word hello.")
        }
    }

    fn sandbox() -> Sandbox {
        Sandbox::new(SandboxOptions {
            timeout: Duration::from_secs(20),
            ..Default::default()
        })
        .unwrap()
    }

    fn run(ok: bool, expected: Option<bool>) -> RunResult {
        RunResult {
            ok,
            exit_code: Some(if ok { 0 } else { 1 }),
            stdout: String::new(),
            stderr: String::new(),
            error: (!ok).then(|| "NameError: name 'hello' is not defined".to_string()),
            timed_out: false,
            seconds: 0.1,
            expected_ok: expected,
            network_isolated: false,
        }
    }

    fn clean() -> StyleReport {
        StyleReport {
            ok: true,
            syntax_ok: true,
            pep8_ok: true,
            naming_ok: true,
            issues: Vec::new(),
        }
    }

    #[test]
    fn problems_are_read_the_way_python_reads_them() {
        let items = parse(
            "[\"first\", {\"task\": \" second \", \"expected\": \"2\", \"id\": 7}, {\"prompt\": \"third\", \"id\": \
             \"p1\"}, {\"prompt\": \"fourth\", \"id\": \"p1\"}]",
        )
        .unwrap();
        let problems = parse_problems(items.as_array()).unwrap();
        let ids: Vec<&str> = problems.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, ["p1", "7", "p1-2", "p1-3"]);
        assert_eq!(problems[1].prompt, "second");
        assert_eq!(problems[1].expected_output.as_deref(), Some("2"));
        assert_eq!(
            problems[0].to_json().render(0),
            "{\"id\":\"p1\",\"prompt\":\"first\",\"tests\":null,\"expected_output\":null}"
        );
        for (bad, why) in [
            ("[\"  \"]", "problem 1: empty prompt"),
            ("[{\"text\": \"x\"}]", "problem 1: missing 'prompt'"),
            (
                "[{\"prompt\": \"x\", \"tests\": 3}]",
                "problem 1: 'tests' must be a string of Python code",
            ),
            (
                "[{\"prompt\": \"x\", \"expected_output\": []}]",
                "problem 1: 'expected_output' must be a string",
            ),
            ("[3]", "problem 1: expected a string or an object, got int"),
            ("[]", "no problems given"),
        ] {
            assert_eq!(
                parse_problems(parse(bad).unwrap().as_array()).unwrap_err(),
                why,
                "{bad}"
            );
        }
        let lines = parse_problem_file("# comment\n\nOne.\n  Two.  \n", ".txt").unwrap();
        assert_eq!(
            lines.iter().map(|p| p.prompt.as_str()).collect::<Vec<_>>(),
            ["One.", "Two."]
        );
        let json = parse_problem_file("{\"problems\": [\"a\"]}", ".JSON").unwrap();
        assert_eq!(json[0].prompt, "a");
        assert_eq!(
            parse_problem_file("{\"other\": 1}", ".json").unwrap_err(),
            "no problems given"
        );
        let jsonl = parse_problem_file("{\"prompt\": \"a\"}\n\n\"b\"\n", ".jsonl").unwrap();
        assert_eq!(jsonl.len(), 2);
        assert_eq!(
            parse_problem_file("{nope}", ".jsonl").unwrap_err(),
            "Expecting property name enclosed in double quotes: line 1 column 2 (char 1)"
        );
        assert_eq!(extension("dir.v1/problems.jsonl"), ".jsonl");
        assert_eq!(extension(".hidden"), "");
        assert_eq!(extension("plain"), "");
    }

    #[test]
    fn code_comes_out_of_an_answer() {
        assert_eq!(extract_code("Sure:\n```python\nprint(1)\n```\nthanks"), "print(1)\n");
        assert_eq!(
            extract_code("```py\nx = 1\n```\n```Python3 \nprint('a much longer one')\n```"),
            "print('a much longer one')\n"
        );
        assert_eq!(extract_code("```\n\n\nprint(2)\n\n```"), "print(2)\n");
        assert_eq!(extract_code("print(3)"), "print(3)\n");
        assert_eq!(extract_code("```python print(4)```"), "print(4)\n");
        assert_eq!(
            extract_code("```pyx\nprint(5)\n```"),
            "pyx\nprint(5)\n",
            "no header: the fence is stripped"
        );
        assert_eq!(extract_code("   "), "");
        assert_eq!(extract_code("```python\r\nprint(6)\r\n```"), "print(6)\r\n");
        let problem = hello();
        assert_eq!(
            solution_text(&problem, "  print('hello')\n\n"),
            "Print the word hello.\nprint('hello')\n"
        );
        assert_eq!(
            model_prefix("# {problem}\n", &problem).unwrap(),
            "# Print the word hello.\n"
        );
        assert_eq!(
            model_prefix("no placeholder", &problem).unwrap_err(),
            "model_prompt must contain {problem}"
        );
    }

    #[test]
    fn the_text_rules_of_pep8_are_python_s() {
        let code = "def f():\n  x = 1\t\n\treturn x\n   \n".to_string() + &"y".repeat(80) + "\nz = 1";
        assert_eq!(
            check_formatting(&code),
            [
                "W292 no newline at end of file",
                "E111 line 2: indentation is not a multiple of four",
                "W291 line 2: trailing whitespace",
                "W191 line 3: indentation contains tabs",
                "W293 line 4: whitespace on a blank line",
                "E501 line 5: line too long (80 > 79)",
            ]
        );
        assert!(check_formatting("print('é')\n").is_empty());
    }

    #[test]
    fn the_style_check_asks_python_for_names_and_blank_lines() {
        if !python_available() {
            return;
        }
        let good = check_style("def add(a, b):\n    return a + b\n", None);
        assert_eq!(good, clean());
        let bad = check_style("import os\ndef sloppyName(X):\n    myVar = 1\n    return myVar\n", None);
        assert!(!bad.ok && bad.syntax_ok && !bad.pep8_ok && !bad.naming_ok);
        assert_eq!(
            bad.issues,
            [
                "E302 line 2: expected 2 blank lines before a top-level definition",
                "N802 line 2: function name 'sloppyName' should be snake_case",
                "N803 line 2: argument 'X' should be snake_case",
                "N806 line 3: variable 'myVar' should be snake_case (or UPPER_CASE for a constant)",
            ]
        );
        let broken = check_style("def f(:\n", None);
        assert!(!broken.syntax_ok && !broken.ok);
        assert!(broken.issues[0].starts_with("syntax error: "), "{:?}", broken.issues);
        let skipped = check_style("x = 1\n", Some("/no/such/python"));
        assert!(skipped.issues[0].starts_with("W000"));
        assert!(!skipped.pep8_ok);
    }

    #[test]
    fn the_verdict_combines_the_three_signals() {
        let opinion = JudgeOpinion {
            task: true,
            pep8: true,
            naming: false,
            score: Some(8.0),
            issues: vec!["sloppy".to_string()],
            critique: "names".to_string(),
        };
        let v = decide(&run(true, None), &clean(), Some(&opinion), "strict", "ollama").unwrap();
        assert!(!v.correct && v.runs && v.task == Some(true) && !v.naming);
        assert_eq!(v.judged_by, "ollama");
        let lenient = decide(&run(true, None), &clean(), Some(&opinion), "lenient", "ollama").unwrap();
        assert!(lenient.correct);
        let crashed = decide(&run(false, None), &clean(), None, "strict", "ollama").unwrap();
        assert_eq!(
            (crashed.task, crashed.judged_by.as_str(), crashed.issues[0].as_str()),
            (Some(false), "sandbox", "NameError: name 'hello' is not defined")
        );
        let tested = decide(&run(true, Some(true)), &clean(), None, "strict", "ollama").unwrap();
        assert_eq!((tested.correct, tested.judged_by.as_str()), (true, "tests"));
        let unjudged = decide(&run(true, None), &clean(), None, "strict", "ollama").unwrap();
        assert_eq!(
            (unjudged.correct, unjudged.task, unjudged.judged_by.as_str()),
            (true, None, "none")
        );
        let wrong = decide(&run(true, Some(false)), &clean(), None, "strict", "ollama").unwrap();
        assert_eq!(wrong.issues, ["stdout differs from the expected output"]);
        assert!(decide(&run(true, None), &clean(), None, "sloppy", "ollama").is_err());
        assert_eq!(
            v.to_json().render(0),
            "{\"correct\":false,\"runs\":true,\"task\":true,\"pep8\":true,\"naming\":false,\"score\":8.0,\
             \"issues\":[\"sloppy\"],\"critique\":\"names\",\"judged_by\":\"ollama\"}"
        );
    }

    #[test]
    fn feedback_and_the_reason_say_what_went_wrong() {
        let crashed = run(false, None);
        let mut attempt = Attempt {
            index: 0,
            source: "ollama".to_string(),
            code: "print(hello)\n".to_string(),
            text: "Print hello.\nprint(hello)\n".to_string(),
            run: RunResult {
                stderr: "Traceback ...\nNameError: name 'hello' is not defined".to_string(),
                ..crashed.clone()
            },
            style: clean(),
            verdict: decide(&crashed, &clean(), None, "strict", "ollama").unwrap(),
            seconds: 0.2,
        };
        assert_eq!(
            attempt.feedback(),
            "It did not run: NameError: name 'hello' is not defined\nstderr:\nTraceback ...\nNameError: name 'hello' \
             is not defined\nIssues: NameError: name 'hello' is not defined"
        );
        assert_eq!(code_reason(&attempt), "crash");
        attempt.run = run(true, Some(false));
        attempt.verdict = decide(&attempt.run, &clean(), None, "strict", "ollama").unwrap();
        assert_eq!(code_reason(&attempt), "wrong-output");
        assert!(attempt
            .feedback()
            .contains("Its output differed from the expected output. Actual stdout:\n(empty)"));
        attempt.run.timed_out = true;
        assert_eq!(code_reason(&attempt), "timeout");
        let (faults, correct) = faults_from_attempts(std::slice::from_ref(&attempt), "codegen");
        assert_eq!((faults.len(), correct.len()), (1, 0));
        assert_eq!((faults[0].reason.as_str(), faults[0].severity), ("timeout", 1.5));
        assert_eq!(faults[0].note, attempt.feedback());
    }

    #[test]
    fn a_program_blamed_and_cleared_in_one_run_weighs_what_a_reloaded_one_does() {
        // a regression: the rows a clear touched were written with the count
        // weights, so a run that blamed and cleared in one process disagreed
        // with Python (and with the same network saved and loaded in between)
        let bad = "Print the word hello.\nprint(hello)\n".to_string();
        let good = "Print the word hello.\nprint(\"hello\")\n".to_string();
        let fault = Fault::new(&bad, "crash", 1.5, "", "codegen");
        let o = TeachOptions::default();
        let mut once = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        teach(&mut once, std::slice::from_ref(&fault), std::slice::from_ref(&good), &o).unwrap();
        let mut blamed = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        teach(&mut blamed, &[fault], &[], &o).unwrap();
        let path = std::env::temp_dir().join(format!("radixnet-codegen-neg-{}.json", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        blamed.save(&path).unwrap();
        let mut reloaded = Model::load(&path).unwrap();
        reloaded.clear(&[good], 1.0, 1).unwrap();
        let _ = std::fs::remove_file(&path);
        let weights = |m: &mut Model| m.g.to_doc().at("edges").at("w").render(0);
        assert_eq!(weights(&mut once), weights(&mut reloaded));
        assert_eq!(once.history.last().unwrap().loss, reloaded.history.last().unwrap().loss);
    }

    #[test]
    fn the_judge_is_read_leniently() {
        struct Answers(&'static str);
        impl LlmClient for Answers {
            fn provider(&self) -> &'static str {
                "ollama"
            }
            fn url(&self) -> &str {
                ""
            }
            fn model(&self) -> &str {
                ""
            }
            fn models(&self) -> Result<Vec<Json>, LlmError> {
                Ok(Vec::new())
            }
            fn generate(&self, _p: &str, _o: &LlmOptions) -> Result<String, LlmError> {
                Ok(self.0.to_string())
            }
            fn chat(&self, _m: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
                Ok(String::new())
            }
        }
        let judged = |answer: &'static str| {
            judge_with_llm(
                &Answers(answer),
                &hello(),
                "print(1)\n",
                &run(true, None),
                &clean(),
                "m",
            )
        };
        let opinion = judged(
            "```json\n{\"correct\": \"yes\", \"formatting\": \"fail\", \"score\": \"12\", \"issues\": \"one\", \
             \"reason\": \" why \"}\n```",
        )
        .unwrap();
        assert_eq!(
            opinion,
            JudgeOpinion {
                task: true,
                pep8: false,
                naming: true,
                score: Some(10.0),
                issues: vec!["one".to_string()],
                critique: "why".to_string()
            }
        );
        let odd =
            judged("{\"task_accomplished\": null, \"task\": true, \"score\": null, \"issues\": [1, \" \"]}").unwrap();
        assert!(!odd.task, "a present null is Python's None, not a fall-through");
        assert_eq!((odd.score, odd.issues.clone()), (None, vec!["1".to_string()]));
        assert_eq!(
            judged("not json").unwrap_err().message,
            "the judge did not answer with a JSON object"
        );
        assert_eq!(clamp_score(f64::NAN), 10.0);
        assert_eq!(clamp_score(-3.0), 0.0);
    }

    #[test]
    fn the_teacher_is_sent_python_s_prompts() {
        let client = Scripted::new(&["```python\nprint(hello)\n```"]);
        let problem = Problem {
            tests: Some("assert True\n".to_string()),
            ..hello()
        };
        let code = teacher_generate(&client, &problem, Some(" be brief "), "tutor").unwrap();
        assert_eq!(code, "print(hello)\n");
        let (prompt, options) = client.prompts.lock().unwrap()[0].clone();
        assert_eq!(
            prompt,
            "Task:\nPrint the word hello.\n\nThe program's standard output must be exactly:\nhello\n\nThis test code \
             is appended to the program and must pass:\nassert True\n\nAdditional instructions:\nbe brief\n\nWrite the \
             program now."
        );
        assert_eq!(options.system, TEACHER_SYSTEM);
        assert_eq!(options.model, "tutor");
        assert_eq!(options.options, vec![("temperature".to_string(), Json::Num(0.3))]);
    }

    #[test]
    fn the_teacher_fixes_what_failed_and_the_network_learns_it() {
        if !python_available() {
            return;
        }
        let client = Scripted::new(&["```python\nprint(hello)\n```", "```python\nprint(\"hello\")\n```"]);
        let config = CodeGenConfig {
            phases: vec!["teacher".to_string()],
            teacher_attempts: 2,
            ..Default::default()
        };
        let mut trainer = CodeGenTrainer::new(&client, &client, sandbox(), config).unwrap();
        let mut model = Model::new(0, Default::default()).unwrap();
        let mut negative = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let mut seen: Vec<String> = Vec::new();
        let records = {
            let mut nets = Nets::Owned {
                model: &mut model,
                negative: Some(&mut negative),
            };
            let mut progress = |r: &Json| seen.push(r.at("kind").as_str().unwrap_or("").to_string());
            let mut hooks = Hooks {
                progress: &mut progress,
                stop: &|| false,
                checkpoints: None,
            };
            trainer.run(&mut nets, &[hello()], &mut hooks).unwrap()
        };
        assert_eq!(seen, ["attempt", "attempt", "problem", "round"]);
        let problem = &records[0];
        assert_eq!(problem.at("correct"), &Json::Bool(true));
        assert_eq!(problem.at("solved_by").as_str(), Some("ollama"));
        assert_eq!(problem.at("action").as_str(), Some("2nrl"));
        assert_eq!(
            (problem.at("bad").as_i64(), problem.at("good").as_i64()),
            (Some(1), Some(1))
        );
        assert_eq!(problem.at("negative_reasons").render(0), "{\"crash\":1}");
        let keys: Vec<&str> = match problem {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            keys,
            [
                "kind",
                "phase",
                "round",
                "problem",
                "prompt",
                "attempts",
                "correct",
                "solved_by",
                "model_solved",
                "bad",
                "good",
                "score",
                "code",
                "issues",
                "seconds",
                "negative_blamed",
                "negative_reasons",
                "action",
                "neg_loss",
                "pos_loss"
            ]
        );
        assert_eq!(
            trainer.solved,
            [(
                "hello".to_string(),
                "Print the word hello.\nprint(\"hello\")\n".to_string()
            )]
        );
        assert!(
            model.g.num_nodes() > 0,
            "the network learned the question and the answer"
        );
        let prompts = client.prompts.lock().unwrap();
        assert_eq!(prompts.len(), 3, "write, fix, judge the fix");
        assert!(prompts[1]
            .0
            .contains("This program is not acceptable yet:\n```python\nprint(hello)\n```"));
        assert!(prompts[1]
            .0
            .contains("What went wrong:\nIt did not run: NameError: name 'hello' is not defined"));
    }

    #[test]
    fn learning_replays_earlier_successes() {
        let client = Scripted::new(&[]);
        let mut trainer = CodeGenTrainer::new(&client, &client, sandbox(), CodeGenConfig::default()).unwrap();
        let mut model = Model::new(0, Default::default()).unwrap();
        let mut nets = Nets::Owned {
            model: &mut model,
            negative: None,
        };
        let good = vec!["a solved problem\nprint(1)\n".to_string()];
        let first = trainer.learn(&mut nets, &[], &good, &|| false).unwrap();
        assert_eq!(Json::Obj(first.clone()).at("action").as_str(), Some("reward"));
        let second = trainer
            .learn(&mut nets, &["broken\nprint(x)\n".to_string()], &[], &|| false)
            .unwrap();
        let second = Json::Obj(second);
        assert_eq!(
            second.at("action").as_str(),
            Some("2nrl"),
            "the replayed success is the positive phase"
        );
        assert_eq!(second.at("good").as_i64(), Some(1));
        let nothing = Json::Obj(
            CodeGenTrainer::new(
                &client,
                &client,
                sandbox(),
                CodeGenConfig {
                    replay: false,
                    ..Default::default()
                },
            )
            .unwrap()
            .learn(&mut nets, &[], &[], &|| false)
            .unwrap(),
        );
        assert_eq!(nothing.at("action"), &Json::Null);
    }

    #[test]
    fn the_settings_are_checked_as_python_checks_them() {
        let d = CodeGenConfig::default();
        assert_eq!(
            (d.teacher_provider.as_str(), d.judge_provider.as_str()),
            ("ollama", "ollama")
        );
        assert_eq!(d.resolved_judge_model(), d.teacher_model);
        for (config, why) in [
            (
                CodeGenConfig {
                    phases: vec![],
                    ..Default::default()
                },
                "phases must be a non-empty subset of teacher, model",
            ),
            (
                CodeGenConfig {
                    rounds: 0,
                    ..Default::default()
                },
                "rounds must be >= 1",
            ),
            (
                CodeGenConfig {
                    strictness: "loose".to_string(),
                    ..Default::default()
                },
                "strictness must be one of strict, lenient",
            ),
            (
                CodeGenConfig {
                    twonrl_per: "epoch".to_string(),
                    ..Default::default()
                },
                "twonrl_per must be 'problem' or 'round'",
            ),
            (
                CodeGenConfig {
                    model_prompt: "x".to_string(),
                    ..Default::default()
                },
                "model_prompt must contain {problem}",
            ),
        ] {
            assert_eq!(config.validate().unwrap_err(), why);
        }
        let keys: Vec<String> = match d.to_json() {
            Json::Obj(pairs) => pairs.into_iter().map(|(k, _)| k).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys.len(), 25);
        assert_eq!(keys[0], "teacher_provider");
        assert_eq!(keys[24], "checkpoint_every");
        let mut other = CodeGenConfig {
            judge_provider: "openai".to_string(),
            ..Default::default()
        };
        other.resolve().unwrap();
        assert_eq!(other.judge_provider, "chatgpt");
        assert_eq!(other.resolved_judge_model(), crate::chatgpt::default_model());
    }

    #[test]
    fn the_routes_answer_the_frontend_s_json() {
        if !python_available() {
            return;
        }
        let model = Model::new(0, Default::default()).unwrap();
        let svc = Arc::new(Service::new(model, String::new(), 0, 1));
        let call = |body: &str| run_route(&svc, &Request::json("POST", "/api/codegen/run", parse(body).unwrap()));
        let doc =
            call("{\"code\": \"print('hello')\", \"expected_output\": \"hello\", \"sandbox_timeout\": 20}").unwrap();
        assert_eq!(doc.at("run").at("expected_ok"), &Json::Bool(true));
        assert_eq!(doc.at("style").at("ok"), &Json::Bool(true));
        assert_eq!(doc.at("verdict").at("judged_by").as_str(), Some("tests"));
        assert_eq!(call("{\"code\": \"  \"}").unwrap_err().message, "'code' is empty");
        assert_eq!(call("{}").unwrap_err().message, "missing field 'code' (string)");
        assert_eq!(
            call("{\"code\": \"x = 1\", \"strictness\": \"odd\"}")
                .unwrap_err()
                .message,
            "strictness must be one of strict, lenient"
        );
        let history = history_route(&svc, &Request::json("GET", "/api/codegen/history", Json::Null)).unwrap();
        assert_eq!(history.render(0), "{\"history\":[]}");
        let start = |body: &str| start_route(&svc, &Request::json("POST", "/api/codegen/start", parse(body).unwrap()));
        assert!(start("{}").unwrap_err().message.starts_with("missing 'problems'"));
        assert_eq!(
            start("{\"problems\": [\"x\"], \"phases\": 3}").unwrap_err().message,
            "'phases' must be 'both', 'teacher', 'model' or a list of those"
        );
        assert_eq!(
            start("{\"problems\": [\"x\"], \"teacher_provider\": \"bard\"}")
                .unwrap_err()
                .message,
            "'teacher_provider' must be one of ollama, chatgpt (got 'bard')"
        );
        assert_eq!(
            start("{\"problems\": [\"x\"], \"rounds\": 0}").unwrap_err().message,
            "'rounds' must be >= 1 (got 0)"
        );
        let solve = solve_route(
            &svc,
            &Request::json(
                "POST",
                "/api/codegen/solve",
                parse("{\"source\": \"x\", \"problem\": \"p\"}").unwrap(),
            ),
        );
        assert_eq!(
            solve.unwrap_err().message,
            "'source' must be 'model' or 'teacher' (got 'x')"
        );
    }
}
