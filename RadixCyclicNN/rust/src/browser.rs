//! A real browser behind the web tools: Chrome driven over the W3C WebDriver
//! protocol (`radixnet/browser.py`).
//!
//! The web tools fetch a page and read its HTML, which is enough for a
//! document and nothing at all for a page that draws itself with JavaScript -
//! a search engine, a single-page app, anything behind a consent banner.
//! `--browser` runs the page in **Chrome** instead and hands the tools the DOM
//! after the scripts have had their say, so `web_fetch`, `web_links` and
//! `web_search` see what a person would see.
//!
//! # Why WebDriver, and why no driver crate
//!
//! WebDriver is HTTP with JSON bodies: open a session, navigate, run a script,
//! read the title, close.  That is five requests through [`crate::fetch`] -
//! which the crate already has (D-072: no dependencies) - so the client here
//! is Python's `BrowserClient` request for request, and a fake WebDriver on a
//! local socket can stand in for Chrome in a test on either side.
//!
//! # What it drives
//!
//! A `chromedriver` and a Chrome or Chromium of the same major version, found
//! the way Python finds them (`$RADIXNET_CHROMEDRIVER` / `$RADIXNET_CHROME`,
//! then `PATH`, then a Playwright browser directory), and compared *before*
//! anything is started: a mismatch is by far the most common way this fails,
//! and [`describe`] says so in words a person can act on.  `$RADIXNET_WEBDRIVER`
//! names a WebDriver that is already running (a Selenium Grid, a
//! `selenium/standalone-chrome` container): it is used as it is, and nothing
//! is started or stopped here.
//!
//! One session per client, opened on the first page and reused for every page
//! after it, so the browser starts once per run.  A chromedriver this client
//! started is stopped when the client is dropped - Python stops it at exit -
//! and a session on somebody else's WebDriver is left alone, as Python leaves
//! it.  A browser executes whatever a page sends it, so this is not a sandbox:
//! the address guards of [`crate::web::WebClient`] run first either way.

use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::fetch;
use crate::json::Json;
use crate::llm::fields::{py_str_of, truthy};
use crate::tools::{dumps, py_repr_value};
use crate::web::Fetched;

const LOG: &str = "browser";

/// Seconds a page may take to load before the browser gives up on it.
pub const DEFAULT_PAGE_TIMEOUT: f64 = 30.0;
/// The viewport: sites lay out differently at 400px, so a real size matters.
pub const DEFAULT_WINDOW: (u32, u32) = (1280, 900);
/// How long a chromedriver this client started has to answer `/status`.
const STARTUP_TIMEOUT: f64 = 20.0;
/// How long a page may keep drawing after `readyState` says it is complete.
pub const SETTLE_SECONDS: f64 = 0.25;

/// Where Chrome is looked for on `PATH` (and the one absolute macOS path).
const CHROME_NAMES: [&str; 6] = [
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
    "chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
];

/// Chrome's flags, Python's: headless, quiet, and no images - the tools read
/// text, and images are bytes and time.
const CHROME_ARGS: [&str; 10] = [
    "--headless=new",
    "--disable-gpu",
    "--disable-dev-shm-usage",
    "--disable-extensions",
    "--disable-background-networking",
    "--disable-sync",
    "--no-first-run",
    "--no-default-browser-check",
    "--mute-audio",
    "--blink-settings=imagesEnabled=false",
];

// -- finding the binaries -----------------------------------------------------------------------

/// A non-empty environment variable, trimmed.
fn env_path(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// The `chromedriver` binary: `$RADIXNET_CHROMEDRIVER` (when it is a file),
/// else the one on `PATH`.
pub fn chromedriver_path() -> Option<String> {
    if let Some(explicit) = env_path("RADIXNET_CHROMEDRIVER") {
        return std::path::Path::new(&explicit).is_file().then_some(explicit);
    }
    crate::speech::asr::which("chromedriver")
}

/// The Chrome / Chromium to drive: `$RADIXNET_CHROME` (when it is a file),
/// then `PATH`, then the newest browser in `$PLAYWRIGHT_BROWSERS_PATH`.
pub fn chrome_path() -> Option<String> {
    if let Some(explicit) = env_path("RADIXNET_CHROME") {
        return std::path::Path::new(&explicit).is_file().then_some(explicit);
    }
    for name in CHROME_NAMES {
        let found = if name.contains('/') {
            std::path::Path::new(name).is_file().then(|| name.to_string())
        } else {
            crate::speech::asr::which(name)
        };
        if found.is_some() {
            return found;
        }
    }
    let root = env_path("PLAYWRIGHT_BROWSERS_PATH")?;
    for inner in [
        "chrome-linux/chrome",
        "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
        "chrome-win/chrome.exe",
    ] {
        // `glob(root/chromium*/inner)`, sorted, the last one
        let Ok(entries) = std::fs::read_dir(&root) else {
            return None;
        };
        let mut matches: Vec<String> = entries
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("chromium"))
            .map(|e| std::path::Path::new(&root).join(e.file_name()).join(inner))
            .filter(|p| std::fs::symlink_metadata(p).is_ok())
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        matches.sort();
        if let Some(last) = matches.pop() {
            return Some(last);
        }
    }
    None
}

/// `<binary> --version` with its whitespace collapsed, or `None` when it
/// cannot be run (or says nothing, or takes more than 15 seconds).
fn version_of(binary: Option<&str>) -> Option<String> {
    let binary = binary?;
    let mut child = Command::new(binary)
        .arg("--version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .ok()?;
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if Instant::now() < deadline => std::thread::sleep(Duration::from_millis(10)),
            _ => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }
    let out = child.wait_with_output().ok()?;
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let text = if stdout.is_empty() {
        String::from_utf8_lossy(&out.stderr).into_owned()
    } else {
        stdout
    };
    let joined = text.split_whitespace().collect::<Vec<_>>().join(" ");
    (!joined.is_empty()).then_some(joined)
}

/// The major version out of `Chromium 141.0.7390.37` or `ChromeDriver 147.0...`.
fn major(version: Option<&str>) -> Option<String> {
    version?.split_whitespace().find_map(|word| {
        let head = word.split('.').next().unwrap_or("");
        (!head.is_empty() && head.chars().all(|c| c.is_ascii_digit())).then(|| head.to_string())
    })
}

/// What browsing with Chrome would use, and why it would not work - Python's
/// `describe()`.
#[derive(Clone, Debug, PartialEq)]
pub struct Found {
    pub available: bool,
    pub chromedriver: Option<String>,
    pub chrome: Option<String>,
    pub chromedriver_version: Option<String>,
    pub chrome_version: Option<String>,
    pub headless: bool,
    pub error: Option<String>,
}

impl Found {
    /// `{"available", "chromedriver", "chrome", "chromedriver_version",
    /// "chrome_version", "headless", "error"}` - what `tools browser` prints
    /// and `/api/tools` reports as `browser`.
    pub fn to_json(&self) -> Json {
        let text = |v: &Option<String>| v.clone().map_or(Json::Null, Json::Str);
        Json::obj([
            ("available", Json::Bool(self.available)),
            ("chromedriver", text(&self.chromedriver)),
            ("chrome", text(&self.chrome)),
            ("chromedriver_version", text(&self.chromedriver_version)),
            ("chrome_version", text(&self.chrome_version)),
            ("headless", Json::Bool(self.headless)),
            ("error", text(&self.error)),
        ])
    }
}

/// The chromedriver and Chrome that would be driven, their versions, and why
/// they would not work - nothing is started to find out.
///
/// chromedriver refuses to drive a Chrome of another major version, which is
/// far and away the most common way this fails, so the versions are read and
/// compared here rather than left to a WebDriver error nobody can act on.
pub fn describe() -> Found {
    describe_with(chromedriver_path(), chrome_path())
}

fn describe_with(driver: Option<String>, chrome: Option<String>) -> Found {
    let missing: Vec<&str> = [("chromedriver", &driver), ("Chrome", &chrome)]
        .iter()
        .filter(|(_, found)| found.is_none())
        .map(|(name, _)| *name)
        .collect();
    let driver_version = version_of(driver.as_deref());
    let chrome_version = version_of(chrome.as_deref());
    let (driver_major, chrome_major) = (major(driver_version.as_deref()), major(chrome_version.as_deref()));
    let error = if !missing.is_empty() {
        let variables: Vec<&str> = missing
            .iter()
            .map(|m| {
                if *m == "chromedriver" {
                    "$RADIXNET_CHROMEDRIVER"
                } else {
                    "$RADIXNET_CHROME"
                }
            })
            .collect();
        Some(format!(
            "{} not found: install it, or point {} at it",
            missing.join(" and "),
            variables.join(" / ")
        ))
    } else {
        match (&driver_major, &chrome_major) {
            (Some(d), Some(c)) if d != c => Some(format!(
                "chromedriver {d} cannot drive Chrome {c}: they must share a major version. Install the matching \
                 chromedriver (chrome-for-testing has one per Chrome release) and point $RADIXNET_CHROMEDRIVER at \
                 it, or $RADIXNET_CHROME at a Chrome {d}"
            )),
            _ => None,
        }
    };
    Found {
        available: error.is_none(),
        chromedriver: driver,
        chrome,
        chromedriver_version: driver_version,
        chrome_version,
        headless: true,
        error,
    }
}

/// Whether both binaries are there and their major versions match.
pub fn browser_available() -> bool {
    describe().available
}

/// A port nothing listens on yet, for a chromedriver of our own.
fn free_port() -> Result<u16, String> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").map_err(|err| err.to_string())?;
    Ok(listener.local_addr().map_err(|err| err.to_string())?.port())
}

/// Whether this process runs as root, where Chrome refuses to start without
/// `--no-sandbox` (containers, CI).  Read from `/proc`, since the standard
/// library has no `geteuid`; elsewhere the answer is no.
fn is_root() -> bool {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|status| {
            status
                .lines()
                .find_map(|line| line.strip_prefix("Uid:"))
                .and_then(|ids| ids.split_whitespace().nth(1).map(|euid| euid == "0"))
        })
        .unwrap_or(false)
}

/// An OS error as Python's `str(OSError)` writes it: `[Errno 111] Connection refused`.
fn python_os_error(message: &str) -> String {
    let Some(open) = message.rfind(" (os error ") else {
        return message.to_string();
    };
    let code = message[open + 11..].trim_end_matches(')');
    format!("[Errno {code}] {}", &message[..open])
}

// -- the client ---------------------------------------------------------------------------------

/// How a [`BrowserClient`] browses.
#[derive(Clone, Debug)]
pub struct BrowserOptions {
    /// Seconds a navigation may take (> 0).
    pub page_timeout: f64,
    pub window: (u32, u32),
    /// Seconds a page is given to keep drawing after it reports itself complete.
    pub settle: f64,
    pub headless: bool,
    /// `None`: [`chromedriver_path`].
    pub driver: Option<String>,
    /// `None`: [`chrome_path`].
    pub chrome: Option<String>,
    /// A WebDriver already running; `None`: `$RADIXNET_WEBDRIVER`.
    pub endpoint: Option<String>,
}

impl Default for BrowserOptions {
    fn default() -> BrowserOptions {
        BrowserOptions {
            page_timeout: DEFAULT_PAGE_TIMEOUT,
            window: DEFAULT_WINDOW,
            settle: SETTLE_SECONDS,
            headless: true,
            driver: None,
            chrome: None,
            endpoint: None,
        }
    }
}

/// One page as the browser drew it.
#[derive(Clone, Debug, PartialEq)]
pub struct Page {
    /// Where the browser ended up (after any redirects the page made).
    pub url: String,
    pub title: String,
    /// The DOM after the scripts ran.
    pub html: String,
    /// WebDriver does not report the HTTP status: a page that loaded is 200.
    pub status: u16,
}

/// What changes while a client runs: where its WebDriver is, the process it
/// started, its session, and how many pages it has drawn.
#[derive(Default)]
struct Live {
    base: Option<String>,
    process: Option<Child>,
    session: Option<String>,
    pages: usize,
}

/// One Chrome, started on demand and reused for every page (Python's
/// `BrowserClient`).  Every method answers `Err` with the reason and never
/// leaves a process behind.
pub struct BrowserClient {
    pub page_timeout: f64,
    pub window: (u32, u32),
    pub settle: f64,
    pub headless: bool,
    /// The WebDriver used as it is, when one was named.
    pub endpoint: Option<String>,
    pub driver: Option<String>,
    pub chrome: Option<String>,
    live: Mutex<Live>,
}

impl std::fmt::Debug for BrowserClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BrowserClient")
            .field("endpoint", &self.endpoint)
            .field("driver", &self.driver)
            .field("chrome", &self.chrome)
            .field("headless", &self.headless)
            .field("page_timeout", &self.page_timeout)
            .finish()
    }
}

impl BrowserClient {
    /// A client; nothing is started until the first page.
    pub fn new(o: BrowserOptions) -> Result<BrowserClient, String> {
        if o.page_timeout.is_nan() || o.page_timeout <= 0.0 {
            return Err("page_timeout must be > 0".to_string());
        }
        let endpoint = o
            .endpoint
            .or_else(|| std::env::var("RADIXNET_WEBDRIVER").ok())
            .map(|e| e.trim().trim_end_matches('/').to_string())
            .filter(|e| !e.is_empty());
        let (driver, chrome) = if endpoint.is_some() {
            (None, None)
        } else {
            (o.driver.or_else(chromedriver_path), o.chrome.or_else(chrome_path))
        };
        Ok(BrowserClient {
            page_timeout: o.page_timeout,
            window: o.window,
            settle: o.settle.max(0.0),
            headless: o.headless,
            live: Mutex::new(Live {
                base: endpoint.clone(),
                ..Default::default()
            }),
            endpoint,
            driver,
            chrome,
        })
    }

    fn live(&self) -> MutexGuard<'_, Live> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether a session is open.
    pub fn running(&self) -> bool {
        self.live().session.is_some()
    }

    /// How many pages this client has drawn.
    pub fn pages(&self) -> usize {
        self.live().pages
    }

    // -- transport ----------------------------------------------------------------------------

    /// One WebDriver request: the answer's JSON, or why there is none - the
    /// WebDriver's own `error: message` when it refused, Python's wording
    /// otherwise.  Never through a proxy: the WebDriver is local.
    fn request(
        &self,
        live: &Live,
        method: &str,
        path: &str,
        body: Option<&Json>,
        timeout: Option<f64>,
    ) -> Result<Json, String> {
        let Some(base) = live.base.as_deref() else {
            return Err("the browser is not running".to_string());
        };
        let url = format!("{base}{path}");
        let seconds = timeout.unwrap_or(self.page_timeout + 5.0);
        let mut request = fetch::Request::new(method, &url)
            .header("Content-Type", "application/json; charset=utf-8")
            .header("Accept", "application/json")
            .timeout(Duration::from_secs_f64(seconds))
            .direct();
        if let Some(body) = body {
            request.body = dumps(body, false, true).into_bytes();
        }
        crate::log_debug!(LOG, "{method} {path}");
        let unreachable = |why: String| format!("cannot reach chromedriver: {why}");
        let response = request.send().map_err(|err| {
            if err.timeout {
                return unreachable("timed out".to_string());
            }
            let message = err.message.strip_prefix(&format!("{url}: ")).unwrap_or(&err.message);
            let message = match message.find(": ") {
                Some(at) if message.starts_with("cannot connect to ") => &message[at + 2..],
                _ => message,
            };
            unreachable(python_os_error(message))
        })?;
        let text = response.text();
        if !(200..300).contains(&response.status) {
            return Err(webdriver_error(&text, response.status));
        }
        crate::mcp::pyjson::loads(if text.is_empty() { "{}" } else { &text }).map_err(unreachable)
    }

    /// A request inside the session, which is opened first when there is none.
    fn session_request(&self, live: &mut Live, method: &str, path: &str, body: Option<&Json>) -> Result<Json, String> {
        self.start(live)?;
        let session = live.session.clone().unwrap_or_default();
        self.request(live, method, &format!("/session/{session}{path}"), body, None)
    }

    // -- lifecycle ----------------------------------------------------------------------------

    /// Starts chromedriver (unless a WebDriver was named) and opens one
    /// session; nothing when one is open.
    fn start(&self, live: &mut Live) -> Result<(), String> {
        if live.session.is_some() {
            return Ok(());
        }
        if self.endpoint.is_none() {
            let found = describe();
            if self.driver.is_none() || self.chrome.is_none() || found.error.is_some() {
                return Err(found.error.unwrap_or_else(|| "no browser available".to_string()));
            }
            self.spawn(live)?;
        }
        self.open_session(live)
    }

    /// Our own chromedriver on a free port, waited for.
    fn spawn(&self, live: &mut Live) -> Result<(), String> {
        let driver = self.driver.clone().unwrap_or_default();
        let port = free_port()?;
        let mut command = Command::new(&driver);
        command
            .arg(format!("--port={port}"))
            .arg("--silent")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        #[cfg(unix)]
        {
            // its own process group, as Python's `start_new_session`: a Ctrl-C
            // meant for this run does not reach the browser mid-page
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let child = command.spawn().map_err(|err| {
            let why = match err.raw_os_error() {
                Some(code) => format!(
                    "[Errno {code}] {}: {}",
                    python_os_error(&err.to_string())
                        .split_once("] ")
                        .map_or(err.to_string(), |(_, text)| text.to_string()),
                    crate::negative::python_repr(&driver)
                ),
                None => err.to_string(),
            };
            format!("cannot start {driver}: {why}")
        })?;
        crate::log_info!(LOG, "started {driver} on port {port}");
        live.process = Some(child);
        live.base = Some(format!("http://127.0.0.1:{port}"));
        self.await_driver(live)
    }

    /// Waits for chromedriver to answer `/status` - it binds its port a
    /// moment after it starts.
    fn await_driver(&self, live: &mut Live) -> Result<(), String> {
        let driver = self.driver.clone().unwrap_or_default();
        let deadline = Instant::now() + Duration::from_secs_f64(STARTUP_TIMEOUT);
        let mut last = "no answer".to_string();
        while Instant::now() < deadline {
            if let Some(process) = live.process.as_mut() {
                if let Ok(Some(status)) = process.try_wait() {
                    return Err(format!("{driver} exited with code {} on start-up", exit_code(status)));
                }
            }
            match self.request(live, "GET", "/status", None, Some(2.0)) {
                Err(why) => last = why,
                Ok(status) => {
                    let ready = match status.get("value") {
                        Some(value @ Json::Obj(_)) => value.get("ready").is_none_or(truthy),
                        _ => true,
                    };
                    if ready {
                        return Ok(());
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(50));
        }
        self.close_live(live);
        Err(format!(
            "{driver} did not become ready in {}s ({last})",
            crate::llm::format_g(STARTUP_TIMEOUT)
        ))
    }

    /// Asks the WebDriver - ours or somebody else's - for a session.
    fn open_session(&self, live: &mut Live) -> Result<(), String> {
        let mut args: Vec<String> = CHROME_ARGS.iter().map(|a| a.to_string()).collect();
        args.push(format!("--window-size={},{}", self.window.0, self.window.1));
        if !self.headless {
            args.retain(|a| a != "--headless=new");
        }
        if cfg!(unix) && is_root() {
            args.push("--no-sandbox".to_string());
        }
        let mut options = vec![("args".to_string(), Json::strs(args))];
        if let Some(chrome) = &self.chrome {
            // a remote WebDriver has its own browser
            options.push(("binary".to_string(), Json::str(chrome.clone())));
        }
        let millis = Json::Int((self.page_timeout * 1000.0) as i64);
        let body = Json::obj([(
            "capabilities",
            Json::obj([(
                "alwaysMatch",
                Json::obj([
                    ("browserName", Json::str("chrome")),
                    ("pageLoadStrategy", Json::str("normal")),
                    ("goog:chromeOptions", Json::Obj(options)),
                    (
                        "timeouts",
                        Json::obj([("pageLoad", millis.clone()), ("script", millis)]),
                    ),
                ]),
            )]),
        )]);
        let answer = match self.request(live, "POST", "/session", Some(&body), None) {
            Ok(answer) => answer,
            Err(why) => {
                self.close_live(live);
                return Err(why);
            }
        };
        let session = [answer.at("sessionId"), answer.at("value").at("sessionId")]
            .into_iter()
            .find(|id| truthy(id))
            .map(py_str_of);
        let Some(session) = session else {
            self.close_live(live);
            let shown: String = py_repr_value(&answer).chars().take(200).collect();
            return Err(format!("chromedriver did not open a session: {shown}"));
        };
        crate::log_debug!(LOG, "session {session}");
        live.session = Some(session);
        Ok(())
    }

    /// Closes the session and stops the chromedriver this client started
    /// (safe to call twice).
    pub fn close(&self) {
        let mut live = self.live();
        self.close_live(&mut live);
    }

    fn close_live(&self, live: &mut Live) {
        if let Some(session) = live.session.clone() {
            let _ = self.request(live, "DELETE", &format!("/session/{session}"), None, Some(5.0));
            live.session = None;
        }
        if let Some(mut process) = live.process.take() {
            if let Ok(None) = process.try_wait() {
                // the session is gone, so Chrome is too: the driver can go at once
                let _ = process.kill();
                let _ = process.wait();
            }
        }
        live.base = self.endpoint.clone();
    }

    // -- pages ----------------------------------------------------------------------------------

    /// Runs JavaScript in the page and returns what it gives back.
    pub fn script(&self, source: &str, args: &[Json]) -> Result<Json, String> {
        let mut live = self.live();
        self.script_in(&mut live, source, args)
    }

    fn script_in(&self, live: &mut Live, source: &str, args: &[Json]) -> Result<Json, String> {
        let body = Json::obj([("script", Json::str(source)), ("args", Json::Arr(args.to_vec()))]);
        let answer = self.session_request(live, "POST", "/execute/sync", Some(&body))?;
        Ok(answer.at("value").clone())
    }

    /// Opens `url` and returns the page once it has drawn itself.
    pub fn get(&self, url: &str) -> Result<Page, String> {
        let (html, last, title) = {
            let mut live = self.live();
            self.session_request(&mut live, "POST", "/url", Some(&Json::obj([("url", Json::str(url))])))?;
            self.settle_in(&mut live);
            let html = self.script_in(&mut live, "return document.documentElement.outerHTML;", &[])?;
            let last = self.script_in(&mut live, "return document.location.href;", &[])?;
            let title = self.session_request(&mut live, "GET", "/title", None)?;
            live.pages += 1;
            (html, last, title.at("value").clone())
        };
        let Json::Str(html) = html else {
            return Err(format!("the page at {url} returned no HTML"));
        };
        Ok(Page {
            url: if truthy(&last) {
                py_str_of(&last)
            } else {
                url.to_string()
            },
            title: if truthy(&title) {
                py_str_of(&title)
            } else {
                String::new()
            },
            html,
            status: 200,
        })
    }

    /// Waits for `document.readyState` to be complete, then a moment more for
    /// the scripts.
    fn settle_in(&self, live: &mut Live) {
        let deadline = Instant::now() + Duration::from_secs_f64(self.page_timeout);
        while Instant::now() < deadline {
            match self.script_in(live, "return document.readyState;", &[]) {
                Ok(Json::Str(state)) if state == "complete" => break,
                Err(_) => break,
                _ => std::thread::sleep(Duration::from_millis(50)),
            }
        }
        if self.settle > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(self.settle));
        }
    }

    /// The answer the web tools read, drawn by Chrome: what `WebClient`'s
    /// fetch answers, the body cut at `max_bytes` characters (Python's
    /// `_fetch_with_browser`).
    pub fn fetch(&self, target: &str, max_bytes: usize) -> Result<Fetched, String> {
        let page = self
            .get(target)
            .map_err(|why| format!("cannot open {target} in the browser: {why}"))?;
        let chars = page.html.chars().count();
        let body: String = page.html.chars().take(max_bytes).collect();
        Ok(Fetched {
            url: page.url,
            status: page.status,
            content_type: "text/html".to_string(),
            bytes: body.len(),
            truncated: chars > max_bytes,
            body,
        })
    }

    /// What this client is, and whether its browser is up.
    pub fn describe(&self) -> Json {
        let mut pairs = match &self.endpoint {
            Some(endpoint) => vec![
                ("available".to_string(), Json::Bool(true)),
                ("chromedriver".to_string(), Json::str(endpoint.clone())),
                ("chrome".to_string(), Json::str("(the endpoint's own)")),
                ("chromedriver_version".to_string(), Json::Null),
                ("chrome_version".to_string(), Json::Null),
                ("error".to_string(), Json::Null),
            ],
            None => match describe().to_json() {
                Json::Obj(pairs) => pairs,
                _ => Vec::new(),
            },
        };
        let live = self.live();
        for (key, value) in [
            ("endpoint", self.endpoint.clone().map_or(Json::Null, Json::Str)),
            ("headless", Json::Bool(self.headless)),
            ("running", Json::Bool(live.session.is_some())),
            ("pages", Json::Int(live.pages as i64)),
            ("page_timeout", Json::Num(self.page_timeout)),
            (
                "window",
                Json::Arr(vec![
                    Json::Int(i64::from(self.window.0)),
                    Json::Int(i64::from(self.window.1)),
                ]),
            ),
        ] {
            crate::codegen::set(&mut pairs, key, value);
        }
        Json::Obj(pairs)
    }
}

impl Drop for BrowserClient {
    /// A chromedriver this client started is stopped with it (Python stops it
    /// at exit); a WebDriver somebody else runs is left as it is.
    fn drop(&mut self) {
        let mut live = self.live();
        if live.process.is_some() {
            self.close_live(&mut live);
        }
    }
}

/// A process's exit code as Python's `returncode`: minus the signal that
/// ended it.
fn exit_code(status: std::process::ExitStatus) -> i32 {
    if let Some(code) = status.code() {
        return code;
    }
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        if let Some(signal) = status.signal() {
            return -signal;
        }
    }
    -1
}

/// Why the WebDriver refused, as Python reads it: its `value.error: message`,
/// else the body, first line, 300 characters - or `HTTP <code>`.
fn webdriver_error(body: &str, status: u16) -> String {
    let mut detail = body.to_string();
    if let Ok(Json::Obj(doc)) = crate::mcp::pyjson::loads(body) {
        let value = doc.iter().rev().find(|(k, _)| k == "value").map(|(_, v)| v.clone());
        let value = value.unwrap_or(Json::Obj(Vec::new()));
        if let Json::Obj(_) = value {
            let part = |key: &str| value.get(key).map_or(String::new(), py_str_of);
            let joined = format!("{}: {}", part("error"), part("message"));
            let stripped = joined.trim_matches(|c| c == ':' || c == ' ');
            if !stripped.is_empty() {
                detail = stripped.to_string();
            }
        }
    }
    let line: String = detail.split('\n').next().unwrap_or("").chars().take(300).collect();
    if line.is_empty() {
        format!("HTTP {status}")
    } else {
        line
    }
}

// -- the server's one browser -----------------------------------------------------------------------

/// The one browser of a server, started on first use and shared by every
/// request - a browser takes a second or two to start, so one per request
/// would be unusable (Python's `ModelService.browser`).  Clones share it.
#[derive(Clone, Default)]
pub struct Shared(Arc<Mutex<Option<Arc<BrowserClient>>>>);

impl Shared {
    /// A slot holding `browser` already.
    pub fn with(browser: Arc<BrowserClient>) -> Shared {
        Shared(Arc::new(Mutex::new(Some(browser))))
    }

    /// The browser, made on the first call: refused with Python's words when
    /// Chrome cannot be driven.
    pub fn get(&self, page_timeout: f64, headless: bool) -> Result<Arc<BrowserClient>, String> {
        let mut slot = self.0.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(browser) = slot.as_ref() {
            return Ok(browser.clone());
        }
        let found = describe();
        if !found.available {
            return Err(format!(
                "the browser is not available: {}",
                found.error.unwrap_or_default()
            ));
        }
        let browser = Arc::new(BrowserClient::new(BrowserOptions {
            page_timeout,
            headless,
            ..Default::default()
        })?);
        *slot = Some(browser.clone());
        Ok(browser)
    }

    /// Stops the browser, if one was started.
    pub fn close(&self) {
        if let Some(browser) = self.0.lock().unwrap_or_else(|e| e.into_inner()).take() {
            browser.close();
        }
    }
}

impl std::fmt::Debug for Shared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let started = self.0.lock().map(|slot| slot.is_some()).unwrap_or(false);
        f.debug_struct("Shared").field("started", &started).finish()
    }
}

impl PartialEq for Shared {
    /// Two settings are the same whatever browser either has started.
    fn eq(&self, _other: &Shared) -> bool {
        true
    }
}

/// The browser a command line asks for with `--browser` (unless `--offline`):
/// refused up front, in Python's words, when Chrome cannot be driven.
pub fn from_flags(page_timeout: f64, headless: bool) -> Result<Arc<BrowserClient>, String> {
    let found = describe();
    if !found.available {
        return Err(format!("--browser: {}", found.error.unwrap_or_default()));
    }
    Ok(Arc::new(BrowserClient::new(BrowserOptions {
        page_timeout,
        headless,
        ..Default::default()
    })?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex as StdMutex;

    /// A WebDriver on a local socket: every request is logged, and the
    /// answers are the ones a Chrome showing one small page would give.
    struct FakeDriver {
        url: String,
        log: Arc<StdMutex<Vec<(String, String, String)>>>,
    }

    fn fake_driver(html: &'static str, refuse_session: bool) -> FakeDriver {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://127.0.0.1:{}", listener.local_addr().unwrap().port());
        let log = Arc::new(StdMutex::new(Vec::new()));
        let seen = log.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut line = String::new();
                if reader.read_line(&mut line).is_err() {
                    continue;
                }
                let mut parts = line.split_whitespace();
                let (method, path) = (
                    parts.next().unwrap_or("").to_string(),
                    parts.next().unwrap_or("").to_string(),
                );
                let mut length = 0;
                loop {
                    let mut header = String::new();
                    reader.read_line(&mut header).unwrap();
                    if header.trim().is_empty() {
                        break;
                    }
                    if let Some((name, value)) = header.split_once(':') {
                        if name.eq_ignore_ascii_case("content-length") {
                            length = value.trim().parse().unwrap_or(0);
                        }
                    }
                }
                let mut body = vec![0; length];
                reader.read_exact(&mut body).unwrap();
                let body = String::from_utf8_lossy(&body).into_owned();
                seen.lock().unwrap().push((method.clone(), path.clone(), body.clone()));
                let (status, answer) = if method == "POST" && path == "/session" {
                    if refuse_session {
                        (
                            500,
                            "{\"value\": {\"error\": \"session not created\", \"message\": \"no Chrome\\nstack\"}}"
                                .to_string(),
                        )
                    } else {
                        (
                            200,
                            "{\"value\": {\"sessionId\": \"s1\", \"capabilities\": {}}}".to_string(),
                        )
                    }
                } else if path.ends_with("/execute/sync") {
                    let value = if body.contains("readyState") {
                        "\"complete\"".to_string()
                    } else if body.contains("outerHTML") {
                        crate::json::Json::str(html).render(0)
                    } else {
                        "\"http://site.test/final\"".to_string()
                    };
                    (200, format!("{{\"value\": {value}}}"))
                } else if path.ends_with("/title") {
                    (200, "{\"value\": \"Cats\"}".to_string())
                } else {
                    (200, "{\"value\": null}".to_string())
                };
                let _ = write!(
                    stream,
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: \
                     close\r\n\r\n{answer}",
                    answer.len()
                );
            }
        });
        FakeDriver { url, log }
    }

    fn client(url: &str) -> BrowserClient {
        BrowserClient::new(BrowserOptions {
            endpoint: Some(format!("{url}/")),
            settle: 0.0,
            page_timeout: 5.0,
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn a_page_is_drawn_through_the_webdriver() {
        let driver = fake_driver(
            "<html><head><title>Cats</title></head><body>A cat.</body></html>",
            false,
        );
        let browser = client(&driver.url);
        assert_eq!(
            browser.endpoint.as_deref(),
            Some(driver.url.as_str()),
            "the trailing slash goes"
        );
        assert!(!browser.running());
        let page = browser.get("http://site.test/cats").unwrap();
        assert_eq!(page.url, "http://site.test/final");
        assert_eq!(page.title, "Cats");
        assert!(page.html.contains("A cat."));
        assert_eq!((page.status, browser.pages()), (200, 1));
        // a second page reuses the session
        browser.get("http://site.test/dogs").unwrap();
        let log = driver.log.lock().unwrap().clone();
        let paths: Vec<&str> = log.iter().map(|(_, p, _)| p.as_str()).collect();
        assert_eq!(paths.iter().filter(|p| **p == "/session").count(), 1);
        assert_eq!(
            &paths[..6],
            [
                "/session",
                "/session/s1/url",
                "/session/s1/execute/sync",
                "/session/s1/execute/sync",
                "/session/s1/execute/sync",
                "/session/s1/title"
            ]
        );
        // the session body is Python's `json.dumps`: its separators, no binary for a remote browser
        let (_, _, session) = &log[0];
        assert!(session.starts_with("{\"capabilities\": {\"alwaysMatch\": {\"browserName\": \"chrome\""));
        assert!(session.contains("\"--window-size=1280,900\""));
        assert!(!session.contains("binary"));
        assert!(session.contains("\"timeouts\": {\"pageLoad\": 5000, \"script\": 5000}"));
        assert_eq!(log[1].2, "{\"url\": \"http://site.test/cats\"}");
        // a remote session is closed only when asked
        browser.close();
        assert!(!browser.running());
        let log = driver.log.lock().unwrap().clone();
        assert_eq!(
            log.last().map(|(m, p, _)| (m.as_str(), p.as_str())),
            Some(("DELETE", "/session/s1"))
        );
    }

    #[test]
    fn the_web_tools_read_the_drawn_page() {
        let driver = fake_driver(
            "<html><head><title>Cats</title></head><body><p>A cat has four legs.</p></body></html>",
            false,
        );
        let browser = Arc::new(client(&driver.url));
        let fetched = browser.fetch("http://site.test/cats", 1024).unwrap();
        assert_eq!(
            (fetched.status, fetched.content_type.as_str(), fetched.truncated),
            (200, "text/html", false)
        );
        let clipped = browser.fetch("http://site.test/cats", 10).unwrap();
        assert_eq!(
            (clipped.body.as_str(), clipped.bytes, clipped.truncated),
            ("<html><hea", 10, true)
        );
        let web = crate::web::WebClient::new(crate::web::WebOptions {
            allow_private: true,
            browser: Some(browser.clone()),
            ..Default::default()
        })
        .unwrap();
        let page = web.page("http://site.test/cats").unwrap();
        assert_eq!(page.title, "Cats");
        assert_eq!(page.text, "A cat has four legs.");
        assert_eq!(web.fetched(), 1);
    }

    #[test]
    fn a_refusal_reads_as_the_webdriver_s_own_words() {
        let driver = fake_driver("", true);
        let browser = client(&driver.url);
        assert_eq!(
            browser.get("http://site.test/").unwrap_err(),
            "session not created: no Chrome"
        );
        assert_eq!(
            browser.fetch("http://site.test/", 100).unwrap_err(),
            "cannot open http://site.test/ in the browser: session not created: no Chrome"
        );
        assert_eq!(webdriver_error("", 502), "HTTP 502");
        assert_eq!(webdriver_error("{\"value\": {}}", 500), "{\"value\": {}}");
        assert_eq!(webdriver_error("plain\ntext", 500), "plain");
        assert_eq!(webdriver_error("{\"value\": {\"error\": \"x\"}}", 500), "x");
        // nothing listening: Python's OS error
        let port = free_port().unwrap();
        let nobody = client(&format!("http://127.0.0.1:{port}"));
        assert_eq!(
            nobody.get("http://site.test/").unwrap_err(),
            "cannot reach chromedriver: [Errno 111] Connection refused"
        );
    }

    #[test]
    fn the_binaries_are_described_as_python_describes_them() {
        let none = describe_with(None, None);
        assert!(!none.available);
        assert_eq!(
            none.error.as_deref(),
            Some("chromedriver and Chrome not found: install it, or point $RADIXNET_CHROMEDRIVER / $RADIXNET_CHROME at it")
        );
        let dir = std::env::temp_dir().join(format!("radixnet-browser-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let script = |name: &str, says: &str| -> String {
            let path = dir.join(name);
            std::fs::write(&path, format!("#!/bin/sh\necho '{says}'\n")).unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
            }
            path.to_string_lossy().into_owned()
        };
        let driver = script("chromedriver", "ChromeDriver 147.0.1  (abc)");
        let chrome = script("chrome", "Chromium 141.0.2");
        let found = describe_with(Some(driver.clone()), Some(chrome.clone()));
        assert_eq!(
            found.chromedriver_version.as_deref(),
            Some("ChromeDriver 147.0.1 (abc)")
        );
        assert!(!found.available);
        assert!(found
            .error
            .unwrap()
            .starts_with("chromedriver 147 cannot drive Chrome 141: they must share"));
        let matching = script("chrome147", "Google Chrome 147.0.9");
        assert!(describe_with(Some(driver), Some(matching)).available);
        let only = describe_with(None, Some(chrome));
        assert_eq!(
            only.error.as_deref(),
            Some("chromedriver not found: install it, or point $RADIXNET_CHROMEDRIVER at it")
        );
        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(major(Some("Chromium 141.0.7390.37")).as_deref(), Some("141"));
        assert_eq!(major(Some("no digits")), None);
        assert_eq!(
            python_os_error("Connection refused (os error 111)"),
            "[Errno 111] Connection refused"
        );
        assert!(BrowserClient::new(BrowserOptions {
            page_timeout: 0.0,
            ..Default::default()
        })
        .is_err());
    }
}
