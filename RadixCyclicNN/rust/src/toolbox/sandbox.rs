//! The sandbox a Python program runs in: the `python` tool's, and code
//! generation's (`radixnet/codegen.py` `Sandbox`, `go/radixnet/codegen.go`).
//!
//! Isolation from accidents, not from a hostile program - the same promise
//! the other two make, kept the same way, because the isolation is *the same
//! code*: the child is `python3 -I -B -c <bootstrap> mem cpu script`, and the
//! bootstrap is Python's own, character for character.  It sets the memory,
//! CPU, file-size and core-dump limits inside the child before the program
//! starts, so the three implementations limit a program identically instead
//! of each reimplementing `setrlimit`.  Around it:
//!
//! * a fresh scratch directory (`radixnet-sandbox-*`) as the working directory,
//!   `$HOME` and `$TMPDIR`, removed afterwards;
//! * an empty environment but for `PATH`, `LANG` and `PYTHONIOENCODING`;
//! * `-I` (no environment variables, no user site-packages) and `-B` (no
//!   `.pyc` written);
//! * no network, when `unshare -rn` can give an unprivileged user a network
//!   namespace of its own (probed once per process);
//! * a wall-clock timeout, after which the child's whole process group is
//!   killed - a program that forks does not outlive its budget.
//!
//! A program that fails is a [`RunResult`], never an `Err`: the failure is the
//! answer.  `Err` is kept for a sandbox that could not run at all (no Python,
//! no scratch directory).
//!
//! The API is the one code generation will call: [`Sandbox::new`] with
//! [`SandboxOptions`], then [`Sandbox::run`] with the code, optional tests
//! appended to it, an optional expected output, and the program's stdin.

use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::json::Json;

/// What the sandbox's own lines are filed under.
const LOG: &str = "tools";

/// How much of a program's stdout or stderr a result keeps, in characters.
pub const MAX_OUTPUT_CHARS: usize = 4000;

/// The Python the child runs: it sets its own limits, then runs the program.
/// Character for character `radixnet/codegen.py`'s `_BOOTSTRAP`.
pub const BOOTSTRAP: &str = "
import runpy, sys
try:
    import resource
except ImportError:
    resource = None
mem, cpu, script = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
if resource is not None:
    if mem:
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    if cpu:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 << 20, 16 << 20))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
sys.argv = [script]
runpy.run_path(script, run_name=\"__main__\")
";

/// The interpreter a sandbox runs: `$RADIXNET_PYTHON`, else the first
/// `python3` / `python` on the `PATH`.
pub fn python_binary() -> String {
    if let Ok(named) = std::env::var("RADIXNET_PYTHON") {
        if !named.trim().is_empty() {
            return named.trim().to_string();
        }
    }
    let path = std::env::var("PATH").unwrap_or_default();
    for name in ["python3", "python"] {
        for dir in path.split(':').filter(|d| !d.is_empty()) {
            let candidate = Path::new(dir).join(name);
            if candidate.is_file() {
                return candidate.to_string_lossy().into_owned();
            }
        }
    }
    "python3".to_string()
}

/// Whether there is a Python to run programs with.
pub fn python_available() -> bool {
    Command::new(python_binary())
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// The prefix that puts a child in a network namespace of its own, or `None`
/// when an unprivileged user cannot make one here.  Probed once.
fn unshare_prefix() -> Option<Vec<String>> {
    static PROBE: OnceLock<Option<Vec<String>>> = OnceLock::new();
    PROBE
        .get_or_init(|| {
            let mut child = Command::new("unshare")
                .args(["-rn", "true"])
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .ok()?;
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                match child.try_wait() {
                    Ok(Some(status)) => {
                        return status.success().then(|| vec!["unshare".to_string(), "-rn".to_string()])
                    }
                    Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(5)),
                    _ => {
                        let _ = child.kill();
                        let _ = child.wait();
                        return None;
                    }
                }
            }
        })
        .clone()
}

/// A float as C's `%g` writes it (`10`, `0.5`, `1e+06`) - how Python words a
/// timeout in the error it gives.
pub fn format_g(x: f64) -> String {
    if x == 0.0 {
        return "0".to_string();
    }
    if !x.is_finite() {
        return crate::calc::float_repr(x);
    }
    let sci = format!("{x:.5e}");
    let (mantissa, exponent) = sci.split_once('e').unwrap_or((&sci, "0"));
    let exp: i32 = exponent.parse().unwrap_or(0);
    let strip = |s: &str| -> String {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.').to_string()
        } else {
            s.to_string()
        }
    };
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        return format!("{}e{sign}{:02}", strip(mantissa), exp.abs());
    }
    strip(&format!("{:.*}", (5 - exp).max(0) as usize, x))
}

/// How a [`Sandbox`] limits a program.
#[derive(Clone, Debug)]
pub struct SandboxOptions {
    /// `None`: [`python_binary`].
    pub python: Option<String>,
    /// Wall-clock time a program may take (must be > 0).
    pub timeout: Duration,
    /// Address-space limit; 0 is unlimited.
    pub memory_mb: u64,
    /// CPU limit; `None` is the timeout plus one second.
    pub cpu_seconds: Option<u64>,
    /// Run the program in a network namespace of its own, where possible.
    pub isolate_network: bool,
}

impl Default for SandboxOptions {
    /// Python's defaults: ten seconds, 256 MB, no network.
    fn default() -> SandboxOptions {
        SandboxOptions {
            python: None,
            timeout: Duration::from_secs(10),
            memory_mb: 256,
            cpu_seconds: None,
            isolate_network: true,
        }
    }
}

/// What happened when a program ran.
#[derive(Clone, Debug, PartialEq)]
pub struct RunResult {
    pub ok: bool,
    /// `None` when the program was stopped at the timeout; `-N` when a signal
    /// `N` ended it (as Python reports it).
    pub exit_code: Option<i32>,
    pub stdout: String,
    /// The traceback, the bootstrap's own frames removed.
    pub stderr: String,
    /// The last line of the traceback, or why the program stopped.
    pub error: Option<String>,
    pub timed_out: bool,
    pub seconds: f64,
    /// Whether stdout matched the expected output, when one was given.
    pub expected_ok: Option<bool>,
    pub network_isolated: bool,
}

impl RunResult {
    /// The run as the API reports it, in Python's key order.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("ok", Json::Bool(self.ok)),
            (
                "exit_code",
                self.exit_code.map_or(Json::Null, |c| Json::Int(i64::from(c))),
            ),
            ("stdout", Json::str(self.stdout.clone())),
            ("stderr", Json::str(self.stderr.clone())),
            ("error", self.error.clone().map_or(Json::Null, Json::Str)),
            ("timed_out", Json::Bool(self.timed_out)),
            ("seconds", Json::Num(self.seconds)),
            ("expected_ok", self.expected_ok.map_or(Json::Null, Json::Bool)),
            ("network_isolated", Json::Bool(self.network_isolated)),
        ])
    }
}

/// Runs a Python program in a scratch directory with resource limits.
#[derive(Clone, Debug)]
pub struct Sandbox {
    pub python: String,
    pub timeout: Duration,
    pub memory_mb: u64,
    pub cpu_seconds: u64,
    unshare: Option<Vec<String>>,
}

/// A scratch directory, removed when it goes out of scope.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Result<Scratch, String> {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir();
        for _ in 0..16 {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |d| d.subsec_nanos());
            // letters and digits only, so `radixnet-sandbox-\w+` names it whole
            let name = format!(
                "radixnet-sandbox-{:x}{:x}{:x}",
                std::process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed),
                nanos
            );
            let path = base.join(name);
            match std::fs::DirBuilder::new().mode(0o700).create(&path) {
                Ok(()) => return Ok(Scratch(path)),
                Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(err) => return Err(format!("cannot make a scratch directory: {err}")),
            }
        }
        Err("cannot make a scratch directory".to_string())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Text as Python's text mode reads it: invalid UTF-8 replaced, and every
/// `\r\n` and `\r` a `\n`.
fn text_mode(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).replace("\r\n", "\n").replace('\r', "\n")
}

/// Keeps the first [`MAX_OUTPUT_CHARS`] characters and says how many more there were.
pub fn truncate_output(text: &str) -> String {
    let total = text.chars().count();
    if total <= MAX_OUTPUT_CHARS {
        return text.to_string();
    }
    let head: String = text.chars().take(MAX_OUTPUT_CHARS).collect();
    format!("{head}\n... ({} more characters)", total - MAX_OUTPUT_CHARS)
}

/// Drops the bootstrap's own frames from a traceback, so what is reported is
/// the program's failure rather than the harness's.
pub fn clean_traceback(stderr: &str) -> String {
    let mut out: Vec<&str> = Vec::new();
    let mut skip = false;
    for line in stderr.lines() {
        if line.starts_with("  File \"<string>\"") || line.contains("runpy.py") || line.contains("<frozen runpy>") {
            skip = true;
            continue;
        }
        if skip && line.starts_with("    ") {
            continue;
        }
        skip = false;
        out.push(line);
    }
    out.join("\n").trim().to_string()
}

/// Kills a process group - the child and anything it started.
fn kill_group(pid: u32) {
    let _ = Command::new("kill")
        .args(["-KILL", "--", &format!("-{pid}")])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
}

fn reader<R: Read + Send + 'static>(mut pipe: R) -> std::thread::JoinHandle<Vec<u8>> {
    std::thread::spawn(move || {
        let mut out = Vec::new();
        let _ = pipe.read_to_end(&mut out);
        out
    })
}

impl Sandbox {
    /// A sandbox, with network isolation probed when it is asked for.
    pub fn new(o: SandboxOptions) -> Result<Sandbox, String> {
        if o.timeout.is_zero() {
            return Err("timeout must be > 0".to_string());
        }
        let python = o.python.filter(|p| !p.trim().is_empty()).unwrap_or_else(python_binary);
        let cpu_seconds = o.cpu_seconds.filter(|c| *c > 0).unwrap_or(o.timeout.as_secs() + 1);
        Ok(Sandbox {
            python,
            timeout: o.timeout,
            memory_mb: o.memory_mb,
            cpu_seconds,
            unshare: if o.isolate_network { unshare_prefix() } else { None },
        })
    }

    /// Whether programs run without a network.
    pub fn network_isolated(&self) -> bool {
        self.unshare.is_some()
    }

    /// Runs `code` (with `tests` appended) and reports what happened; a program
    /// that fails is a result, not an error.
    pub fn run(
        &self,
        code: &str,
        tests: Option<&str>,
        expected_output: Option<&str>,
        stdin: &str,
    ) -> Result<RunResult, String> {
        let mut source = code.to_string();
        if !source.ends_with('\n') {
            source.push('\n');
        }
        if let Some(tests) = tests.filter(|t| !t.trim().is_empty()) {
            source.push_str("\n\n# --- tests ---\n");
            source.push_str(tests.trim_matches('\n'));
            source.push('\n');
        }
        let scratch = Scratch::new()?;
        let script = scratch.0.join("solution.py");
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&script)
            .and_then(|mut file| file.write_all(source.as_bytes()))
            .map_err(|err| format!("cannot write the program: {err}"))?;
        let mut command = match &self.unshare {
            Some(prefix) => {
                let mut c = Command::new(&prefix[0]);
                c.args(&prefix[1..]).arg(&self.python);
                c
            }
            None => Command::new(&self.python),
        };
        let path = std::env::var("PATH")
            .ok()
            .filter(|p| !p.trim().is_empty())
            .unwrap_or_else(|| "/usr/bin:/bin".to_string());
        command
            .args(["-I", "-B", "-c", BOOTSTRAP])
            .arg((self.memory_mb << 20).to_string())
            .arg(self.cpu_seconds.to_string())
            .arg(&script)
            .current_dir(&scratch.0)
            .env_clear()
            .env("PATH", path)
            .env("HOME", &scratch.0)
            .env("TMPDIR", &scratch.0)
            .env("LANG", "C.UTF-8")
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            // its own process group, so the timeout can kill everything it started
            .process_group(0);
        let started = Instant::now();
        let mut child = command
            .spawn()
            .map_err(|err| format!("cannot run {}: {err}", self.python))?;
        if let Some(mut pipe) = child.stdin.take() {
            let input = stdin.as_bytes().to_vec();
            std::thread::spawn(move || {
                let _ = pipe.write_all(&input);
            });
        }
        let out = reader(child.stdout.take().ok_or("no stdout")?);
        let err = reader(child.stderr.take().ok_or("no stderr")?);
        let deadline = started + self.timeout;
        let mut timed_out = false;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {}
                Err(e) => return Err(format!("cannot wait for the program: {e}")),
            }
            if Instant::now() >= deadline {
                timed_out = true;
                kill_group(child.id());
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            let waited = started.elapsed();
            std::thread::sleep(if waited < Duration::from_millis(200) {
                Duration::from_millis(1)
            } else {
                Duration::from_millis(10)
            });
        };
        let seconds = started.elapsed().as_secs_f64();
        let stdout = text_mode(&out.join().unwrap_or_default());
        let mut stderr = text_mode(&err.join().unwrap_or_default());
        let exit_code = if timed_out {
            None
        } else {
            status.and_then(|s| s.code().or_else(|| s.signal().map(|sig| -sig)))
        };
        if timed_out {
            stderr = format!(
                "{stderr}\nTimeoutError: the program did not finish within {} seconds",
                format_g(self.timeout.as_secs_f64())
            )
            .trim()
            .to_string();
        }
        let stderr = clean_traceback(&stderr);
        let ok = exit_code == Some(0) && !timed_out;
        let error = (!ok).then(|| {
            let last = stderr
                .lines()
                .rev()
                .find(|l| !l.trim().is_empty())
                .map(|l| l.trim().to_string());
            let error = last.unwrap_or_else(|| match exit_code {
                Some(code) => format!("exit code {code}"),
                None => "exit code None".to_string(),
            });
            if exit_code == Some(-9) && !timed_out {
                format!("killed (memory or CPU limit exceeded): {error}")
            } else {
                error
            }
        });
        let expected_ok = expected_output.map(|want| ok && stdout.trim() == want.trim());
        crate::log_debug!(
            LOG,
            "sandbox: {} in {seconds:.3}s{}",
            if ok { "ok" } else { "failed" },
            if timed_out { " (timed out)" } else { "" }
        );
        Ok(RunResult {
            ok,
            exit_code,
            stdout: truncate_output(&stdout),
            stderr: truncate_output(&stderr),
            error,
            timed_out,
            seconds,
            expected_ok,
            network_isolated: self.network_isolated(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sandbox(timeout: Duration) -> Option<Sandbox> {
        if !python_available() {
            return None;
        }
        Some(
            Sandbox::new(SandboxOptions {
                timeout,
                ..Default::default()
            })
            .unwrap(),
        )
    }

    #[test]
    fn a_program_runs_and_its_tests_with_it() {
        let Some(sandbox) = sandbox(Duration::from_secs(20)) else {
            return;
        };
        let run = sandbox
            .run("def f(x):\n    return x * 2", Some("print(f(21))"), Some("42\n"), "")
            .unwrap();
        assert!(run.ok, "{run:?}");
        assert_eq!(
            (run.stdout.as_str(), run.exit_code, run.expected_ok),
            ("42\n", Some(0), Some(true))
        );
        let echo = sandbox
            .run("import sys; print(sys.stdin.read().upper())", None, None, "hi")
            .unwrap();
        assert_eq!(echo.stdout, "HI\n");
        let json = run.to_json();
        let keys: Vec<&str> = match &json {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            keys,
            [
                "ok",
                "exit_code",
                "stdout",
                "stderr",
                "error",
                "timed_out",
                "seconds",
                "expected_ok",
                "network_isolated"
            ]
        );
    }

    #[test]
    fn a_failure_is_the_program_s_not_the_harness_s() {
        let Some(sandbox) = sandbox(Duration::from_secs(20)) else {
            return;
        };
        let run = sandbox.run("x = 1\nraise ValueError('boom')", None, None, "").unwrap();
        assert!(!run.ok);
        assert_eq!(run.exit_code, Some(1));
        assert_eq!(run.error.as_deref(), Some("ValueError: boom"));
        assert!(run.stderr.contains("solution.py"), "{}", run.stderr);
        assert!(
            !run.stderr.contains("<string>") && !run.stderr.contains("runpy"),
            "{}",
            run.stderr
        );
        let exit = sandbox.run("raise SystemExit(3)", None, None, "").unwrap();
        assert_eq!((exit.exit_code, exit.error.as_deref()), (Some(3), Some("exit code 3")));
        let env = sandbox
            .run("import os; print(sorted(os.environ))", None, None, "")
            .unwrap();
        assert!(
            !env.stdout.contains("HTTPS_PROXY"),
            "the environment is scrubbed: {}",
            env.stdout
        );
    }

    #[test]
    fn the_timeout_stops_the_program_and_what_it_started() {
        let Some(sandbox) = sandbox(Duration::from_millis(500)) else {
            return;
        };
        let started = Instant::now();
        let run = sandbox
            .run(
                "import subprocess, sys, time\n\
                 subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n\
                 print('started', flush=True)\ntime.sleep(30)",
                None,
                None,
                "",
            )
            .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the grandchild was killed too"
        );
        assert!(run.timed_out && !run.ok);
        assert_eq!(run.exit_code, None);
        assert_eq!(
            run.error.as_deref(),
            Some("TimeoutError: the program did not finish within 0.5 seconds")
        );
        assert_eq!(run.stdout, "started\n", "what it printed before the timeout is kept");
    }

    #[test]
    fn small_helpers_match_python() {
        assert_eq!(format_g(10.0), "10");
        assert_eq!(format_g(0.5), "0.5");
        assert_eq!(format_g(0.1), "0.1");
        assert_eq!(format_g(1234567.0), "1.23457e+06");
        assert_eq!(format_g(0.0001234), "0.0001234");
        assert_eq!(
            truncate_output(&"a".repeat(4001)),
            format!("{}\n... (1 more characters)", "a".repeat(4000))
        );
        assert!(Sandbox::new(SandboxOptions {
            timeout: Duration::ZERO,
            ..Default::default()
        })
        .is_err());
    }
}
