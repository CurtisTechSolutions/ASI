//! The built-in tools: browsing, the calculator, the sandbox and the uploads
//! (`radixnet/tools.py` `default_toolbox`, `go/radixnet/toolbox.go`).
//!
//! Same names, same parameters, same descriptions and the same output text
//! as Python's - a transcript written by one implementation is a transcript
//! the others read, and a network trained on one set of transcripts learns
//! the tool calls the others make.  The outputs are the network's training
//! data too, so their wording (`1. Title — URL`, `(the program printed
//! nothing)`, the ` ...` of a clipped page) is part of the contract rather
//! than decoration.
//!
//! The sandbox the `python` tool runs in is [`sandbox`]; code generation runs
//! its programs in the same one.

pub mod sandbox;

use std::sync::Arc;

use crate::calc::safe_eval;
use crate::json::Json;
use crate::negative::python_repr;
use crate::tools::{Param, Tool, ToolBox};
use crate::web::{WebClient, WebOptions};
use sandbox::Sandbox;

/// A text argument a handler was given (they arrive coerced).
fn text_arg(args: &Json, name: &str) -> String {
    args.at(name).as_str().unwrap_or("").to_string()
}

/// An integer argument a handler was given, clamped to `[low, high]`.
fn int_arg(args: &Json, name: &str, fallback: i64, low: i64, high: i64) -> usize {
    let value = match args.at(name) {
        Json::Int(n) => *n,
        Json::Num(x) => *x as i64,
        _ => fallback,
    };
    value.clamp(low, high) as usize
}

/// The first `n` characters of a text.
fn clip(text: &str, n: usize) -> String {
    text.chars().take(n).collect()
}

/// `web_search`: a numbered list of titles with their URLs.
pub fn web_search_tool(web: Arc<WebClient>) -> Tool {
    let mut tool = Tool::new(
        "web_search",
        "Search the web and get back a numbered list of titles with their URLs.",
        vec![
            Param::new("query", "string", "what to search for"),
            Param::optional("limit", "integer", "how many results (1-10)", Json::Int(5)),
        ],
        move |args| {
            let query = text_arg(args, "query");
            let limit = int_arg(args, "limit", 5, 1, 10);
            let results = web.search(&query, limit)?;
            if results.is_empty() {
                return Ok((
                    format!("no results for {}", python_repr(&query)),
                    Json::obj([("results", Json::Arr(Vec::new()))]),
                ));
            }
            let lines: Vec<String> = results
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    let mut line = format!("{}. {} — {}", i + 1, r.title, r.url);
                    if !r.snippet.is_empty() {
                        line.push_str(&format!(" — {}", r.snippet));
                    }
                    line
                })
                .collect();
            Ok((
                lines.join("\n"),
                Json::obj([("results", Json::Arr(results.iter().map(|r| r.to_json()).collect()))]),
            ))
        },
    );
    tool.network = true;
    tool
}

/// `web_fetch`: a page as plain text, the title first; its links go in the
/// metadata, because they are what an exploration follows next.
pub fn web_fetch_tool(web: Arc<WebClient>) -> Tool {
    let mut tool = Tool::new(
        "web_fetch",
        "Open a web page and read it as plain text (the title, then the body).",
        vec![
            Param::new("url", "string", "the address of the page"),
            Param::optional("max_chars", "integer", "how much of the page to read", Json::Int(2000)),
        ],
        move |args| {
            let page = web.page(&text_arg(args, "url"))?;
            let limit = int_arg(args, "max_chars", 2000, 100, 20000);
            let chars = page.text.chars().count();
            let mut body = clip(&page.text, limit);
            if chars > limit {
                body.push_str(" ...");
            }
            let header = if page.title.is_empty() {
                format!("{}\n", page.url)
            } else {
                format!("{} — {}\n", page.title, page.url)
            };
            let meta = Json::obj([
                ("url", Json::str(page.url.clone())),
                ("title", Json::str(page.title.clone())),
                ("status", Json::Int(i64::from(page.status))),
                ("chars", Json::Int(chars as i64)),
                ("truncated", Json::Bool(page.truncated || chars > limit)),
                ("link_count", Json::Int(page.links.len() as i64)),
                (
                    "links",
                    Json::Arr(page.links.iter().take(40).map(|l| l.to_json()).collect()),
                ),
            ]);
            Ok((header + &body, meta))
        },
    );
    tool.network = true;
    tool
}

/// `web_links`: the links on a page, so another page can be opened from it.
pub fn web_links_tool(web: Arc<WebClient>) -> Tool {
    let mut tool = Tool::new(
        "web_links",
        "List the links on a web page, so another page can be opened from it.",
        vec![
            Param::new("url", "string", "the address of the page"),
            Param::optional("limit", "integer", "how many links (1-100)", Json::Int(20)),
        ],
        move |args| {
            let page = web.page(&text_arg(args, "url"))?;
            let limit = int_arg(args, "limit", 20, 1, 100);
            let links: Vec<_> = page.links.iter().take(limit).collect();
            if links.is_empty() {
                return Ok((
                    format!("no links on {}", page.url),
                    Json::obj([("links", Json::Arr(Vec::new()))]),
                ));
            }
            let lines: Vec<String> = links
                .iter()
                .enumerate()
                .map(|(i, link)| format!("{}. {} — {}", i + 1, link.text, link.url))
                .collect();
            Ok((
                lines.join("\n"),
                Json::obj([
                    ("links", Json::Arr(links.iter().map(|l| l.to_json()).collect())),
                    ("url", Json::str(page.url.clone())),
                ]),
            ))
        },
    );
    tool.network = true;
    tool
}

/// `calculator`: an arithmetic expression ([`crate::calc`]).
pub fn calculator_tool() -> Tool {
    Tool::new(
        "calculator",
        "Work out an arithmetic expression, for example 2 * (3 + 4) or sqrt(841).",
        vec![Param::new("expression", "string", "the expression to evaluate")],
        |args| Ok((safe_eval(&text_arg(args, "expression"))?, Json::Null)),
    )
}

/// `python`: a short program run in the sandbox; what it printed, or why it
/// failed with the tail of its traceback.
pub fn python_tool(sandbox: Arc<Sandbox>) -> Tool {
    Tool::new(
        "python",
        "Run a short Python program in a sandbox and get back what it printed.",
        vec![Param::new("code", "string", "the program to run")],
        move |args| {
            let run = sandbox.run(&text_arg(args, "code"), None, None, "")?;
            if run.ok {
                let out = run.stdout.trim();
                let out = if out.is_empty() {
                    "(the program printed nothing)"
                } else {
                    out
                };
                return Ok((
                    out.to_string(),
                    Json::obj([
                        ("seconds", Json::Num(run.seconds)),
                        (
                            "exit_code",
                            run.exit_code.map_or(Json::Null, |c| Json::Int(i64::from(c))),
                        ),
                    ]),
                ));
            }
            let tail: Vec<char> = run.stderr.chars().collect();
            let tail: String = tail[tail.len().saturating_sub(400)..].iter().collect();
            Err(format!("the program failed: {}\n{tail}", run.error.unwrap_or_default())
                .trim()
                .to_string())
        },
    )
}

/// `read_file`: one of the files uploaded to the server, by its bare name.
pub fn read_file_tool(upload_dir: String) -> Tool {
    Tool::new(
        "read_file",
        "Read one of the files uploaded to the server.",
        vec![
            Param::new("name", "string", "the file name"),
            Param::optional("max_chars", "integer", "how much to read", Json::Int(2000)),
        ],
        move |args| {
            let raw = text_arg(args, "name");
            // the bare name only: whatever directories it names are dropped
            let cleaned = raw.trim().replace('\\', "/");
            let safe = cleaned.rsplit('/').next().unwrap_or("").to_string();
            if safe.is_empty() || safe == "." || safe == ".." {
                return Err(format!("{} is not a file name", python_repr(&raw)));
            }
            let path = std::path::Path::new(&upload_dir).join(&safe);
            if !path.is_file() {
                let mut have: Vec<String> = std::fs::read_dir(&upload_dir)
                    .map(|entries| {
                        entries
                            .flatten()
                            .filter(|e| e.path().is_file())
                            .map(|e| e.file_name().to_string_lossy().into_owned())
                            .collect()
                    })
                    .unwrap_or_default();
                have.sort();
                have.truncate(20);
                let listed = if have.is_empty() {
                    "none".to_string()
                } else {
                    have.join(", ")
                };
                return Err(format!(
                    "no uploaded file named {} (have: {listed})",
                    python_repr(&safe)
                ));
            }
            let limit = int_arg(args, "max_chars", 2000, 100, 100_000);
            let mut blob = Vec::new();
            {
                use std::io::Read;
                let file = std::fs::File::open(&path).map_err(|err| err.to_string())?;
                file.take(limit as u64 + 1)
                    .read_to_end(&mut blob)
                    .map_err(|err| err.to_string())?;
            }
            let decoded = String::from_utf8_lossy(&blob);
            let text = decoded.trim_start_matches('\u{feff}');
            let chars = text.chars().count();
            let mut out = clip(text, limit);
            if chars > limit {
                out.push_str(" ...");
            }
            Ok((
                out,
                Json::obj([("name", Json::str(safe)), ("chars", Json::Int(chars as i64))]),
            ))
        },
    )
}

/// What a caller may change about the built-in tools.
#[derive(Default)]
pub struct ToolOptions {
    /// No web tools at all.
    pub offline: bool,
    /// How the web tools browse.
    pub web: WebOptions,
    /// When set, the `python` tool is offered.
    pub sandbox: Option<Arc<Sandbox>>,
    /// When set, the `read_file` tool is offered over this directory.
    pub upload_dir: Option<String>,
    /// Anything else the caller wants offered.
    pub extra: Vec<Tool>,
}

/// The built-in set: browsing (unless `offline`), the calculator, and - when
/// given - `python` and `read_file`, in Python's order.
pub fn default_toolbox(o: ToolOptions) -> Result<ToolBox, String> {
    let mut toolbox = ToolBox::new();
    if !o.offline {
        let web = Arc::new(WebClient::new(o.web)?);
        toolbox.register(web_search_tool(web.clone()))?;
        toolbox.register(web_fetch_tool(web.clone()))?;
        toolbox.register(web_links_tool(web))?;
    }
    toolbox.register(calculator_tool())?;
    if let Some(sandbox) = o.sandbox {
        toolbox.register(python_tool(sandbox))?;
    }
    if let Some(dir) = o.upload_dir.filter(|d| !d.trim().is_empty()) {
        toolbox.register(read_file_tool(dir))?;
    }
    for tool in o.extra {
        toolbox.register(tool)?;
    }
    Ok(toolbox)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use std::time::Duration;

    fn obj(text: &str) -> Json {
        parse(text).unwrap()
    }

    #[test]
    fn the_default_set_is_python_s() {
        let offline = default_toolbox(ToolOptions {
            offline: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(offline.names(), ["calculator"]);
        let full = default_toolbox(ToolOptions::default()).unwrap();
        assert_eq!(full.names(), ["web_search", "web_fetch", "web_links", "calculator"]);
        assert!(full.get("web_fetch").unwrap().network);
        assert!(!full.get("calculator").unwrap().network);
        assert_eq!(
            full.get("web_fetch").unwrap().to_json().render(0),
            "{\"name\":\"web_fetch\",\"description\":\"Open a web page and read it as plain text (the title, then \
             the body).\",\"signature\":\"web_fetch(url: string, [max_chars: integer])\",\"network\":true,\
             \"params\":[{\"name\":\"url\",\"type\":\"string\",\"description\":\"the address of the page\",\
             \"required\":true,\"default\":null,\"enum\":null},{\"name\":\"max_chars\",\"type\":\"integer\",\
             \"description\":\"how much of the page to read\",\"required\":false,\"default\":2000,\"enum\":null}]}"
        );
        let calc = offline.call("calculator", &obj("{\"expression\": \"6*7\"}"));
        assert_eq!((calc.ok, calc.output.as_str()), (true, "42"));
        let zero = offline.call("calculator", &obj("{\"expression\": \"1/0\"}"));
        assert_eq!(zero.error.as_deref(), Some("division by zero"));
    }

    #[test]
    fn the_web_tools_refuse_a_private_address_readably() {
        let toolbox = default_toolbox(ToolOptions {
            web: WebOptions {
                timeout: Duration::from_secs(2),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
        let result = toolbox.call("web_fetch", &obj("{\"url\": \"http://127.0.0.1:1/\"}"));
        assert!(!result.ok);
        assert!(result.error.unwrap().contains("private"));
    }

    #[test]
    fn read_file_reads_an_upload_and_nothing_else() {
        let dir = std::env::temp_dir().join(format!("radixnet-read-file-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("notes.txt"), "\u{feff}the cat sat on the mat\n").unwrap();
        std::fs::write(dir.join("big.txt"), "y".repeat(5000)).unwrap();
        let toolbox = default_toolbox(ToolOptions {
            offline: true,
            upload_dir: Some(dir.to_string_lossy().into_owned()),
            ..Default::default()
        })
        .unwrap();
        let result = toolbox.call("read_file", &obj("{\"name\": \"notes.txt\"}"));
        assert_eq!(result.output, "the cat sat on the mat\n");
        assert_eq!(result.meta.at("name").as_str(), Some("notes.txt"));
        let missing = toolbox.call("read_file", &obj("{\"name\": \"nope.txt\"}"));
        assert_eq!(
            missing.error.as_deref(),
            Some("no uploaded file named 'nope.txt' (have: big.txt, notes.txt)")
        );
        let climbing = toolbox.call("read_file", &obj("{\"name\": \"../../etc/passwd\"}"));
        assert!(climbing.error.unwrap().contains("no uploaded file named 'passwd'"));
        assert!(!toolbox.call("read_file", &obj("{\"name\": \"  \"}")).ok);
        let clipped = toolbox.call("read_file", &obj("{\"name\": \"big.txt\", \"max_chars\": 100}"));
        assert_eq!(clipped.output, format!("{} ...", "y".repeat(100)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn python_runs_in_the_sandbox() {
        if !sandbox::python_available() {
            return;
        }
        let sandbox = Sandbox::new(sandbox::SandboxOptions {
            timeout: Duration::from_secs(20),
            isolate_network: false,
            ..Default::default()
        })
        .unwrap();
        let toolbox = default_toolbox(ToolOptions {
            offline: true,
            sandbox: Some(Arc::new(sandbox)),
            ..Default::default()
        })
        .unwrap();
        let result = toolbox.call("python", &obj("{\"code\": \"print(6 * 7)\"}"));
        assert!(result.ok, "{result:?}");
        assert_eq!(result.output, "42");
        assert_eq!(result.meta.at("exit_code"), &Json::Int(0));
        let failed = toolbox.call("python", &obj("{\"code\": \"raise ValueError('boom')\"}"));
        let error = failed.error.unwrap();
        assert!(error.starts_with("the program failed: ValueError: boom"), "{error}");
        let silent = toolbox.call("python", &obj("{\"code\": \"x = 1\"}"));
        assert_eq!(silent.output, "(the program printed nothing)");
    }
}
