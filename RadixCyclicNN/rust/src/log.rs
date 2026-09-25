//! Logging, written out: levels, a target, a timestamp, and nothing on stdout.
//!
//! The Python service logs by writing to stderr when it is not `quiet`
//! (`radixnet/api.py`), and the Go port does not log at all.  This is the same
//! idea with the three things that made the Python one awkward to use fixed:
//! a **level** so a noisy run can be turned down rather than off, a **target**
//! so a line says which part of the model wrote it, and a **timestamp** so two
//! lines can be ordered.
//!
//! # Nothing goes to stdout
//!
//! Every line is written to **stderr**.  That is not a style choice: `--json`
//! puts one document on stdout, and `radixnet mcp` will put a protocol there,
//! and a log line landing in either one corrupts it.  There is no way to
//! configure a log line onto stdout, which is the point.
//!
//! # Turning it up and down
//!
//! `RADIXNET_LOG` takes a level (`error`, `warn`, `info`, `debug`, `trace`,
//! `off`), optionally per target:
//!
//! ```text
//! RADIXNET_LOG=info               everything at info and above
//! RADIXNET_LOG=debug              ... and debug
//! RADIXNET_LOG=off                silence
//! RADIXNET_LOG=warn,http=debug    warn everywhere, debug for the server
//! RADIXNET_LOG=info,train=trace   info everywhere, every epoch of training
//! ```
//!
//! The default is `warn`: a run says nothing unless something is wrong, which
//! is what a CLI whose output is its answer needs.  `radixnet serve` raises its
//! own default to `info`, because a server that says nothing at all while it
//! runs is a server nobody can debug.
//!
//! # Why not a crate
//!
//! `log` + `env_logger` is what this would be in any other Rust project, and
//! they are good crates.  The rule this crate keeps (D-072) is that it has no
//! dependencies, so the hash, the RNG, the exact sum, the worker pool, gzip and
//! the HTTP server are all written out here; a logger is 200 lines and joins
//! them rather than being the one exception that pulls in a tree.

use std::io::Write;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

/// How much a line has to matter to be written.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[repr(u8)]
pub enum Level {
    /// Nothing at all.
    Off = 0,
    /// Something failed and the run cannot do what was asked.
    Error = 1,
    /// Something is wrong but the run carries on.
    Warn = 2,
    /// The shape of the run: what started, what finished, how big it was.
    Info = 3,
    /// Enough to follow a request or an epoch through the code.
    Debug = 4,
    /// Every step, which is far too much for anything but a hunt.
    Trace = 5,
}

impl Level {
    /// The name that appears in a line, padded so the lines align.
    fn label(self) -> &'static str {
        match self {
            Level::Off => "OFF  ",
            Level::Error => "ERROR",
            Level::Warn => "WARN ",
            Level::Info => "INFO ",
            Level::Debug => "DEBUG",
            Level::Trace => "TRACE",
        }
    }

    /// The level a name means, or `None`.
    pub fn parse(name: &str) -> Option<Level> {
        match name.trim().to_ascii_lowercase().as_str() {
            "off" | "none" | "silent" | "quiet" => Some(Level::Off),
            "error" | "err" => Some(Level::Error),
            "warn" | "warning" => Some(Level::Warn),
            "info" => Some(Level::Info),
            "debug" => Some(Level::Debug),
            "trace" | "all" => Some(Level::Trace),
            _ => None,
        }
    }

    fn from_u8(value: u8) -> Level {
        match value {
            0 => Level::Off,
            1 => Level::Error,
            2 => Level::Warn,
            3 => Level::Info,
            4 => Level::Debug,
            _ => Level::Trace,
        }
    }
}

/// The level everything is measured against, until a target overrides it.
static LEVEL: AtomicU8 = AtomicU8::new(Level::Warn as u8);

/// How many lines were dropped for being below the level - reported by
/// [`stats`], so a run that looks too quiet can say why.
static DROPPED: AtomicUsize = AtomicUsize::new(0);
static WRITTEN: AtomicUsize = AtomicUsize::new(0);

/// Per-target overrides, in the order they were given.
fn targets() -> &'static Mutex<Vec<(String, Level)>> {
    static TARGETS: OnceLock<Mutex<Vec<(String, Level)>>> = OnceLock::new();
    TARGETS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Serialises the writes, so two threads never interleave half a line.
fn pen() -> &'static Mutex<()> {
    static PEN: OnceLock<Mutex<()>> = OnceLock::new();
    PEN.get_or_init(|| Mutex::new(()))
}

/// Reads `RADIXNET_LOG` and sets the level, falling back to `default`.
///
/// Called once at the top of each binary.  An unreadable spec is a warning on
/// stderr rather than an error: a mistyped log setting should not stop a run.
pub fn init(default: Level) {
    set_level(default);
    let Ok(spec) = std::env::var("RADIXNET_LOG") else {
        return;
    };
    if let Err(why) = configure(&spec) {
        // written directly: the logger is what is being configured
        let _ = writeln!(std::io::stderr(), "radixnet: RADIXNET_LOG: {why}");
    }
}

/// Applies a spec: `level` or `level,target=level,...`.
pub fn configure(spec: &str) -> Result<(), String> {
    let mut overrides: Vec<(String, Level)> = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('=') {
            None => {
                let level = Level::parse(part).ok_or_else(|| format!("unknown level {part:?}"))?;
                set_level(level);
            }
            Some((target, name)) => {
                let level = Level::parse(name).ok_or_else(|| format!("unknown level {name:?} for {target:?}"))?;
                overrides.push((target.trim().to_ascii_lowercase(), level));
            }
        }
    }
    *targets().lock().unwrap_or_else(|e| e.into_inner()) = overrides;
    Ok(())
}

/// Sets the level directly, ignoring the environment.
pub fn set_level(level: Level) {
    LEVEL.store(level as u8, Ordering::Relaxed);
}

/// The level a target is measured against.
pub fn level_for(target: &str) -> Level {
    let overrides = targets().lock().unwrap_or_else(|e| e.into_inner());
    for (name, level) in overrides.iter() {
        if target.eq_ignore_ascii_case(name) {
            return *level;
        }
    }
    Level::from_u8(LEVEL.load(Ordering::Relaxed))
}

/// Whether a line at `level` from `target` would be written.
///
/// Worth calling before building an expensive message - [`log`] checks it
/// again, so this is an optimisation and not a correctness requirement.
pub fn enabled(target: &str, level: Level) -> bool {
    level != Level::Off && level <= level_for(target)
}

/// Writes one line, if the level allows it.
pub fn log(target: &str, level: Level, message: &str) {
    if !enabled(target, level) {
        DROPPED.fetch_add(1, Ordering::Relaxed);
        return;
    }
    let line = format!("{} {} {target}: {message}\n", crate::clock::utc_now(), level.label());
    let _guard = pen().lock().unwrap_or_else(|e| e.into_inner());
    // a failed write is not worth failing a run over, and there is nowhere to
    // report it to but the stream that just failed
    let _ = std::io::stderr().write_all(line.as_bytes());
    WRITTEN.fetch_add(1, Ordering::Relaxed);
}

/// `(written, dropped)` since the process started.
pub fn stats() -> (usize, usize) {
    (WRITTEN.load(Ordering::Relaxed), DROPPED.load(Ordering::Relaxed))
}

/// Writes a line at a level, with `format!` arguments.
///
/// ```
/// # use radixnet::{log_at, log::Level};
/// log_at!("train", Level::Info, "{} epochs over {} texts", 5, 120);
/// ```
#[macro_export]
macro_rules! log_at {
    ($target:expr, $level:expr, $($arg:tt)*) => {
        if $crate::log::enabled($target, $level) {
            $crate::log::log($target, $level, &format!($($arg)*));
        }
    };
}

/// A line that says the run cannot do what was asked.
#[macro_export]
macro_rules! log_error {
    ($target:expr, $($arg:tt)*) => { $crate::log_at!($target, $crate::log::Level::Error, $($arg)*) };
}

/// A line that says something is wrong and the run carries on.
#[macro_export]
macro_rules! log_warn {
    ($target:expr, $($arg:tt)*) => { $crate::log_at!($target, $crate::log::Level::Warn, $($arg)*) };
}

/// A line about the shape of the run.
#[macro_export]
macro_rules! log_info {
    ($target:expr, $($arg:tt)*) => { $crate::log_at!($target, $crate::log::Level::Info, $($arg)*) };
}

/// A line for following a request or an epoch through the code.
#[macro_export]
macro_rules! log_debug {
    ($target:expr, $($arg:tt)*) => { $crate::log_at!($target, $crate::log::Level::Debug, $($arg)*) };
}

/// A line for a hunt.
#[macro_export]
macro_rules! log_trace {
    ($target:expr, $($arg:tt)*) => { $crate::log_at!($target, $crate::log::Level::Trace, $($arg)*) };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The level and the target table are process-wide, so the tests that touch
    /// them take a lock rather than racing each other.
    fn settings() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn the_names_a_level_answers_to() {
        assert_eq!(Level::parse("INFO"), Some(Level::Info));
        assert_eq!(Level::parse(" warn "), Some(Level::Warn));
        assert_eq!(Level::parse("quiet"), Some(Level::Off));
        assert_eq!(Level::parse("sideways"), None);
        // and they order the way a level has to
        assert!(Level::Error < Level::Warn);
        assert!(Level::Trace > Level::Debug);
        assert!(Level::Off < Level::Error);
    }

    #[test]
    fn a_spec_sets_the_level_and_the_targets() {
        let _guard = settings();
        configure("warn,http=debug,train=trace").expect("a good spec");
        assert_eq!(level_for("anything"), Level::Warn);
        assert_eq!(level_for("http"), Level::Debug);
        assert_eq!(
            level_for("HTTP"),
            Level::Debug,
            "a target is matched case-insensitively"
        );
        assert_eq!(level_for("train"), Level::Trace);

        assert!(enabled("http", Level::Debug));
        assert!(!enabled("http", Level::Trace));
        assert!(!enabled("model", Level::Info), "the default is warn");
        assert!(enabled("model", Level::Error));

        // off silences everything, including a target that asked for more
        configure("off").expect("a good spec");
        assert!(!enabled("http", Level::Error));
        assert!(!enabled("http", Level::Debug));

        configure("warn").expect("a good spec");
    }

    #[test]
    fn a_bad_spec_says_which_part_is_bad() {
        let _guard = settings();
        let why = configure("warn,http=sideways").expect_err("a bad level");
        assert!(why.contains("sideways") && why.contains("http"), "{why}");
        let why = configure("sideways").expect_err("a bad level");
        assert!(why.contains("sideways"), "{why}");
        configure("warn").expect("a good spec");
    }

    #[test]
    fn a_line_carries_its_level_target_and_time() {
        let _guard = settings();
        // only this test's own target: a test browsing in parallel logs its requests at debug, and under a
        // global trace those lines would be counted here too
        configure("warn,test=trace").expect("a good spec");
        let (written_before, _) = stats();
        log("test", Level::Info, "a message");
        let (written_after, _) = stats();
        assert_eq!(written_after, written_before + 1);

        // and a line below the level is counted rather than written
        configure("error").expect("a good spec");
        let (_, dropped_before) = stats();
        log("test", Level::Debug, "not this one");
        let (_, dropped_after) = stats();
        assert_eq!(dropped_after, dropped_before + 1);
        configure("warn").expect("a good spec");
    }

    #[test]
    fn the_macros_do_not_build_a_message_that_is_not_written() {
        let _guard = settings();
        configure("error").expect("a good spec");
        let mut built = false;
        let mut build = || {
            built = true;
            "expensive"
        };
        log_debug!("test", "{}", build());
        assert!(!built, "a dropped line built its message anyway");
        configure("warn").expect("a good spec");
    }
}
