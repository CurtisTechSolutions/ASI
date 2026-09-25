//! Browsing: HTML reduced to readable text, and a client that refuses
//! everything which is not a plain public web page (`radixnet/tools.py`
//! `WebClient` / `html_to_text`, `go/radixnet/web.go`).
//!
//! The network browses by writing `<tool>web_fetch {"url": ...}</tool>`, and
//! what it writes is text it was never told to be careful with, so the client
//! is the one that is careful:
//!
//! * **`http` and `https` only**, no credentials in the URL, and - unless
//!   `allow_private` - no host that resolves to a loopback, private,
//!   link-local, multicast or reserved address (the SSRF guard: the cloud
//!   metadata endpoint is one `http://169.254.169.254/` away);
//! * **every redirect hop is checked again**, which is why [`crate::fetch`]
//!   follows redirects only when asked: here it never is, and the loop below
//!   follows them by hand;
//! * the address is vetted **again when the connection is made**
//!   ([`crate::fetch::PeerCheck`]): a name that resolved to a public address
//!   when the URL was checked and to a private one a moment later is still
//!   refused.  Python and Go resolve twice and trust the second answer; this
//!   is the one place the port is stricter than the original;
//! * a **byte cap** and a **timeout** per request.
//!
//! A host this process cannot resolve is still fetchable when a proxy would
//! carry the request, because then the proxy resolves it - behind a corporate
//! or sandbox proxy this process often has no DNS at all, and refusing there
//! would make browsing impossible rather than safer.
//!
//! HTTPS goes through `crate::fetch`, which hands it to the system's `curl`
//! (D-076); plain HTTP never leaves the process.
//!
//! The HTML reader is a small tokenizer rather than a parser: the text, the
//! title and the links, with `<script>` and `<style>` taken whole, what is
//! never shown and the page's own chrome (`<head>`, `<nav>`, `<footer>`,
//! hidden elements, ...) dropped with their content, and the text the page is
//! about put first ([`html_to_text`]).  `<meta>` and `<link>` are void
//! elements, so a `<meta charset="utf-8">` cannot open a drop that never
//! closes - which once made every page that has one read as empty, in all
//! three ports but this one.
//!
//! A search tries each of its endpoints in turn ([`DEFAULT_SEARCH_ENGINES`]),
//! passing over one that answers with an error, a `202` or a CAPTCHA.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, ToSocketAddrs};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use crate::fetch;
use crate::json::{parse, Json};
use crate::negative::python_repr;

/// What this module's own lines are filed under.
const LOG: &str = "tools";

/// The user agent a request identifies itself with (`$RADIXNET_USER_AGENT`
/// overrides it).
pub fn default_user_agent() -> String {
    match std::env::var("RADIXNET_USER_AGENT") {
        Ok(named) if !named.trim().is_empty() => named.trim().to_string(),
        _ => "radixnet/0.1 (+https://curtistechsolutions.com)".to_string(),
    }
}

/// The search endpoints tried in turn by default: DuckDuckGo's HTML page, its
/// lite page, and then Wikipedia's own search API, which is made for programs
/// and so still answers when a search engine takes this client for a bot (each
/// of its hits carries the first sentences of the article).
pub const DEFAULT_SEARCH_ENGINES: [&str; 3] = [
    "https://html.duckduckgo.com/html/?q={query}",
    "https://lite.duckduckgo.com/lite/?q={query}",
    "https://en.wikipedia.org/w/api.php?action=query&format=json&formatversion=2&generator=search&gsrlimit=10\
     &prop=info%7Cextracts&inprop=url&exintro=1&explaintext=1&exsentences=2&exlimit=10&gsrsearch={query}",
];

/// The search endpoints, separated by spaces and tried in turn until one has
/// results, `{query}` replaced by the URL-encoded query (`$RADIXNET_SEARCH_URL`
/// overrides them).  A JSON answer is understood too.
pub fn default_search_url() -> String {
    match std::env::var("RADIXNET_SEARCH_URL") {
        Ok(named) if !named.trim().is_empty() => named.trim().to_string(),
        _ => DEFAULT_SEARCH_ENGINES.join(" "),
    }
}

// -- URLs -------------------------------------------------------------------------------------------

/// A URL split the way Python's `urllib.parse.urlsplit` splits it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    /// Lower-cased; empty when there is none.
    pub scheme: String,
    pub netloc: String,
    pub path: String,
    pub query: String,
    pub fragment: String,
}

impl Split {
    /// The user name and password of `user:pw@host`, empty when absent.
    pub fn credentials(&self) -> (String, String) {
        match self.netloc.rsplit_once('@') {
            Some((info, _)) => match info.split_once(':') {
                Some((user, pw)) => (user.to_string(), pw.to_string()),
                None => (info.to_string(), String::new()),
            },
            None => (String::new(), String::new()),
        }
    }

    /// The host, lower-cased, without brackets, credentials or port.
    pub fn hostname(&self) -> String {
        let info = self.netloc.rsplit('@').next().unwrap_or("");
        let host = match info.split_once('[') {
            Some((_, bracketed)) => bracketed.split(']').next().unwrap_or(""),
            None => info.split(':').next().unwrap_or(""),
        };
        host.to_lowercase()
    }

    /// The URL put back together (`urlunsplit`).
    pub fn unsplit(&self) -> String {
        let mut url = self.path.clone();
        if !self.netloc.is_empty() || matches!(self.scheme.as_str(), "http" | "https") {
            if !url.is_empty() && !url.starts_with('/') {
                url.insert(0, '/');
            }
            url = format!("//{}{url}", self.netloc);
        }
        if !self.scheme.is_empty() {
            url = format!("{}:{url}", self.scheme);
        }
        if !self.query.is_empty() {
            url.push('?');
            url.push_str(&self.query);
        }
        if !self.fragment.is_empty() {
            url.push('#');
            url.push_str(&self.fragment);
        }
        url
    }
}

/// Splits a URL into scheme, network location, path, query and fragment.
pub fn urlsplit(url: &str) -> Split {
    // leading control characters and spaces go, and tabs and new lines go
    // wherever they are, as the WHATWG rules Python follows say
    let cleaned: String = url
        .trim_start_matches(|c: char| c <= ' ')
        .chars()
        .filter(|c| !matches!(c, '\t' | '\r' | '\n'))
        .collect();
    let mut rest = cleaned.as_str();
    let mut out = Split::default();
    if let Some(colon) = rest.find(':') {
        let candidate = &rest[..colon];
        let first = candidate.chars().next();
        if colon > 0
            && first.is_some_and(|c| c.is_ascii_alphabetic())
            && candidate
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
        {
            out.scheme = candidate.to_ascii_lowercase();
            rest = &rest[colon + 1..];
        }
    }
    if let Some(after) = rest.strip_prefix("//") {
        let end = after.find(['/', '?', '#']).unwrap_or(after.len());
        out.netloc = after[..end].to_string();
        rest = &after[end..];
    }
    if let Some((before, fragment)) = rest.split_once('#') {
        out.fragment = fragment.to_string();
        rest = before;
    }
    if let Some((before, query)) = rest.split_once('?') {
        out.query = query.to_string();
        rest = before;
    }
    out.path = rest.to_string();
    out
}

/// Resolves a link against the page it is on, as `urllib.parse.urljoin` does.
pub fn urljoin(base: &str, url: &str) -> String {
    if base.is_empty() {
        return url.to_string();
    }
    if url.is_empty() {
        return base.to_string();
    }
    let b = urlsplit(base);
    let mut u = urlsplit(url);
    if u.scheme.is_empty() {
        u.scheme = b.scheme.clone();
    }
    let relative = ["", "http", "https", "ftp", "file", "ws", "wss"];
    if u.scheme != b.scheme || !relative.contains(&u.scheme.as_str()) {
        return url.to_string();
    }
    if !u.netloc.is_empty() {
        return u.unsplit();
    }
    u.netloc = b.netloc.clone();
    if u.path.is_empty() {
        u.path = b.path.clone();
        if u.query.is_empty() {
            u.query = b.query.clone();
        }
        return u.unsplit();
    }
    let mut base_parts: Vec<&str> = b.path.split('/').collect();
    if base_parts.last().is_some_and(|last| !last.is_empty()) {
        base_parts.pop();
    }
    let mut segments: Vec<&str> = if u.path.starts_with('/') {
        u.path.split('/').collect()
    } else {
        base_parts.into_iter().chain(u.path.split('/')).collect()
    };
    // the empty segments in the middle go, as Python's filter(None, ...) drops them
    if segments.len() > 2 {
        let last = segments.len() - 1;
        let middle: Vec<&str> = segments[1..last].iter().copied().filter(|s| !s.is_empty()).collect();
        segments = std::iter::once(segments[0])
            .chain(middle)
            .chain(std::iter::once(segments[last]))
            .collect();
    }
    let mut resolved: Vec<&str> = Vec::new();
    for seg in &segments {
        match *seg {
            ".." => {
                resolved.pop();
            }
            "." => {}
            other => resolved.push(other),
        }
    }
    if matches!(segments.last(), Some(&".") | Some(&"..")) {
        resolved.push("");
    }
    let path = resolved.join("/");
    u.path = if path.is_empty() { "/".to_string() } else { path };
    u.unsplit()
}

/// `urllib.parse.quote_plus`: a query value with spaces as `+`.
pub fn quote_plus(text: &str) -> String {
    let mut out = String::new();
    for byte in text.bytes() {
        match byte {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_' | b'.' | b'-' | b'~' => out.push(byte as char),
            b' ' => out.push('+'),
            other => out.push_str(&format!("%{other:02X}")),
        }
    }
    out
}

/// `urllib.parse.unquote_plus`: `+` is a space, `%XX` a byte, and bytes that
/// are not UTF-8 are replaced.
pub fn unquote_plus(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => out.push(b' '),
            b'%' if i + 3 <= bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or("");
                match u8::from_str_radix(hex, 16) {
                    Ok(value) => {
                        out.push(value);
                        i += 3;
                        continue;
                    }
                    Err(_) => out.push(b'%'),
                }
            }
            other => out.push(other),
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// The first value of `key` in a query string (`parse_qs(...)[key][0]`).
fn query_value(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|pair| {
        let (k, v) = pair.split_once('=')?;
        (unquote_plus(k) == key && !v.is_empty()).then(|| unquote_plus(v))
    })
}

/// A URL made safe to put on a request line: spaces, control characters,
/// non-ASCII and the characters no server accepts raw are percent-encoded;
/// an existing `%XX` is left as it is.
fn request_target(url: &str) -> String {
    let mut out = String::with_capacity(url.len());
    for byte in url.bytes() {
        if byte <= b' ' || byte >= 0x7f || matches!(byte, b'"' | b'<' | b'>' | b'`' | b'{' | b'}' | b'|' | b'\\' | b'^')
        {
            out.push_str(&format!("%{byte:02X}"));
        } else {
            out.push(byte as char);
        }
    }
    out
}

// -- the address guard ------------------------------------------------------------------------------

/// What kind of address an IP is, when it is one the web tools refuse.
///
/// Python's `ipaddress` kinds, most diagnosable first (127.0.0.1 is loopback
/// *and* private), plus the ranges Go refuses that Python calls neither
/// (100.64/10, 192.88.99/24).
pub fn address_kind(ip: IpAddr) -> Option<&'static str> {
    match ip {
        IpAddr::V4(v4) => v4_kind(v4),
        IpAddr::V6(v6) => {
            if let Some(v4) = v6.to_ipv4_mapped() {
                return v4_kind(v4);
            }
            let seg = v6.segments();
            if v6.is_unspecified() {
                Some("an unspecified address")
            } else if v6.is_loopback() {
                Some("a loopback address")
            } else if seg[0] & 0xffc0 == 0xfe80 {
                Some("a link-local address")
            } else if seg[0] & 0xff00 == 0xff00 {
                Some("a multicast address")
            } else if seg[0] & 0xfe00 == 0xfc00 // unique local
                || seg[0] & 0xffc0 == 0xfec0 // the old site-local
                || (seg[0] == 0x2001 && seg[1] == 0x0db8) // documentation
                || (seg[0] == 0x2001 && seg[1] < 0x0200) // IETF protocol assignments
                || (seg[0] == 0x0064 && seg[1] == 0xff9b && seg[2] == 0x0001) // local NAT64
                || (seg[0] == 0x0100 && seg[1] == 0 && seg[2] == 0 && seg[3] == 0)
            // discard-only
            {
                Some("a private address")
            } else if seg[0] & 0xe000 != 0x2000 {
                // outside 2000::/3, the only range handed out as global unicast
                Some("a reserved address")
            } else {
                None
            }
        }
    }
}

fn v4_kind(ip: Ipv4Addr) -> Option<&'static str> {
    let [a, b, c, _] = ip.octets();
    if ip.is_unspecified() {
        Some("an unspecified address (0.0.0.0 usually means DNS is blocking the name)")
    } else if ip.is_loopback() {
        Some("a loopback address")
    } else if ip.is_link_local() {
        Some("a link-local address")
    } else if ip.is_multicast() {
        Some("a multicast address")
    } else if ip.is_private()
        || a == 0
        || (a == 100 && b & 0xc0 == 64)
        || (a == 192 && b == 0 && c == 0)
        || (a == 192 && b == 0 && c == 2)
        || (a == 198 && b & 0xfe == 18)
        || (a == 198 && b == 51 && c == 100)
        || (a == 203 && b == 0 && c == 113)
        || ip.is_broadcast()
    {
        Some("a private address")
    } else if a >= 240 || (a == 192 && b == 88 && c == 99) {
        Some("a reserved address")
    } else {
        None
    }
}

/// The connect-time half of the guard: the [`fetch::PeerCheck`] a request is
/// sent with when private addresses are refused.
fn public_peer(ip: IpAddr) -> Result<(), String> {
    match address_kind(ip) {
        Some(what) => Err(format!(
            "refusing {ip}: it is {what} - pass allow_private (--allow-private) to fetch it anyway"
        )),
        None => Ok(()),
    }
}

/// Why `host` must not be fetched, in words, or `None` for an ordinary public
/// host.  A name that does not resolve, and a name that resolves somewhere
/// internal, are kept apart: they are very different problems.
fn address_problem(host: &str) -> Option<String> {
    let addrs = match (host, 0).to_socket_addrs() {
        Ok(addrs) => addrs.collect::<Vec<_>>(),
        Err(err) => {
            let text = err.to_string();
            let why = text
                .strip_prefix("failed to lookup address information: ")
                .unwrap_or(&text);
            return Some(format!("cannot be resolved here ({why})"));
        }
    };
    if addrs.is_empty() {
        return Some("cannot be resolved here (no address)".to_string());
    }
    addrs
        .iter()
        .find_map(|addr| address_kind(addr.ip()).map(|what| format!("resolves to {}, {what}", addr.ip())))
}

/// A host name the URL grammar allows (an IPv6 literal is checked apart).
fn plausible_host(host: &str) -> Option<char> {
    if host.parse::<Ipv6Addr>().is_ok() {
        return None;
    }
    host.chars()
        .find(|c| c.is_whitespace() || c.is_control() || "<>\"'`{}|\\^%#?/@[]".contains(*c))
}

// -- the client -------------------------------------------------------------------------------------

/// How a [`WebClient`] behaves.
#[derive(Clone, Debug)]
pub struct WebOptions {
    /// Per request (must be > 0).
    pub timeout: Duration,
    /// The most bytes read from one page (at least 1024).
    pub max_bytes: usize,
    pub max_redirects: usize,
    /// `None`: [`default_user_agent`].
    pub user_agent: Option<String>,
    /// Let the client reach loopback / private addresses (a local test site).
    pub allow_private: bool,
    /// `None`: [`default_search_url`].
    pub search_url: Option<String>,
    /// Draw pages in Chrome instead of fetching them (`--browser`,
    /// [`crate::browser`]); the address guards run first either way.
    pub browser: Option<std::sync::Arc<crate::browser::BrowserClient>>,
}

impl Default for WebOptions {
    /// Python's defaults: 20 seconds, 2 MB, four redirects, public addresses only.
    fn default() -> WebOptions {
        WebOptions {
            timeout: Duration::from_secs(20),
            max_bytes: 2_000_000,
            max_redirects: 4,
            user_agent: None,
            allow_private: false,
            search_url: None,
            browser: None,
        }
    }
}

/// One fetched document.
#[derive(Clone, Debug)]
pub struct Fetched {
    pub url: String,
    pub status: u16,
    /// Lower-cased, without its parameters; `text/html` when the server sent none.
    pub content_type: String,
    pub bytes: usize,
    pub truncated: bool,
    pub body: String,
}

/// One link on a page: the text it shows and where it points.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub text: String,
    pub url: String,
}

impl Link {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("text", Json::str(self.text.clone())),
            ("url", Json::str(self.url.clone())),
        ])
    }
}

/// A fetched page reduced to text, its links made absolute.
#[derive(Clone, Debug)]
pub struct Page {
    pub url: String,
    pub title: String,
    pub text: String,
    pub links: Vec<Link>,
    pub status: u16,
    pub content_type: String,
    pub truncated: bool,
}

/// One hit of a search.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub title: String,
    pub url: String,
    pub snippet: String,
}

impl SearchResult {
    /// `{"title", "url", "snippet"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("title", Json::str(self.title.clone())),
            ("url", Json::str(self.url.clone())),
            ("snippet", Json::str(self.snippet.clone())),
        ])
    }
}

/// Fetches pages, refusing everything that is not a plain public web page.
#[derive(Debug)]
pub struct WebClient {
    pub timeout: Duration,
    pub max_bytes: usize,
    pub max_redirects: usize,
    pub user_agent: String,
    pub allow_private: bool,
    pub search_url: String,
    fetched: AtomicUsize,
    /// When set, pages are drawn by Chrome ([`crate::browser`]).
    pub browser: Option<std::sync::Arc<crate::browser::BrowserClient>>,
}

impl WebClient {
    /// A client, with Python's validation of the two limits.
    pub fn new(o: WebOptions) -> Result<WebClient, String> {
        if o.timeout.is_zero() {
            return Err("timeout must be > 0".to_string());
        }
        if o.max_bytes < 1024 {
            return Err("max_bytes must be >= 1024".to_string());
        }
        Ok(WebClient {
            timeout: o.timeout,
            max_bytes: o.max_bytes,
            max_redirects: o.max_redirects,
            user_agent: o
                .user_agent
                .filter(|u| !u.trim().is_empty())
                .unwrap_or_else(default_user_agent),
            allow_private: o.allow_private,
            search_url: o
                .search_url
                .map(|u| u.split_whitespace().collect::<Vec<_>>().join(" "))
                .filter(|u| !u.is_empty())
                .unwrap_or_else(default_search_url),
            fetched: AtomicUsize::new(0),
            browser: o.browser,
        })
    }

    /// How many documents this client has fetched.
    pub fn fetched(&self) -> usize {
        self.fetched.load(Ordering::Relaxed)
    }

    /// The normalised URL, or why it is refused.
    pub fn check(&self, url: &str) -> Result<String, String> {
        let text = url.trim().trim_matches(['<', '>', '"', '\'']);
        if text.is_empty() {
            return Err("the URL is empty".to_string());
        }
        let text = if text.contains("://") {
            text.to_string()
        } else {
            format!("https://{text}")
        };
        let parts = urlsplit(&text);
        if parts.scheme != "http" && parts.scheme != "https" {
            let scheme = if parts.scheme.is_empty() {
                "no scheme"
            } else {
                &parts.scheme
            };
            return Err(format!("only http and https are allowed (got {scheme})"));
        }
        let (user, password) = parts.credentials();
        if !user.is_empty() || !password.is_empty() {
            return Err("URLs with credentials are refused".to_string());
        }
        if parts.netloc.contains('[') != parts.netloc.contains(']') {
            return Err(format!("cannot read the URL {}: Invalid IPv6 URL", python_repr(&text)));
        }
        let host = parts.hostname();
        if host.is_empty() {
            return Err(format!("no host in {}", python_repr(&text)));
        }
        if let Some(bad) = plausible_host(&host) {
            return Err(format!(
                "cannot read the URL {}: invalid character {} in host name",
                python_repr(&text),
                python_repr(&bad.to_string())
            ));
        }
        if !self.allow_private {
            let mut problem = address_problem(&host);
            // a proxy resolves what this process cannot
            if problem.as_deref().is_some_and(|p| p.starts_with("cannot be resolved"))
                && fetch::proxy_for(&text).is_some()
            {
                problem = None;
            }
            if let Some(problem) = problem {
                let hint = if problem.starts_with("cannot be resolved") {
                    ""
                } else {
                    " - pass allow_private (--allow-private) to fetch it anyway"
                };
                return Err(format!("refusing {host}: it {problem}{hint}"));
            }
        }
        Ok(parts.unsplit())
    }

    /// One document, following redirects by hand and checking every hop.
    pub fn fetch(&self, url: &str) -> Result<Fetched, String> {
        let mut target = self.check(url)?;
        if let Some(browser) = &self.browser {
            // Python's `_fetch_with_browser`: the page as Chrome drew it
            let fetched = browser.fetch(&target, self.max_bytes)?;
            self.fetched.fetch_add(1, Ordering::Relaxed);
            return Ok(fetched);
        }
        let mut seen = vec![target.clone()];
        for _hop in 0..=self.max_redirects {
            let mut request = fetch::Request::get(&request_target(&target))
                .header("User-Agent", &self.user_agent)
                .header(
                    "Accept",
                    "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1",
                )
                .header("Accept-Language", "en")
                .timeout(self.timeout)
                .limit(self.max_bytes);
            if !self.allow_private {
                request = request.peer(public_peer);
            }
            crate::log_debug!(LOG, "GET {target}");
            let sent = request.url.clone();
            let response = request.send().map_err(|err| {
                // the transport names the URL itself; once is enough
                let why = err.message.strip_prefix(&format!("{sent}: ")).unwrap_or(&err.message);
                format!("cannot fetch {target}: {why}")
            })?;
            let status = response.status;
            if matches!(status, 301 | 302 | 303 | 307 | 308) {
                if let Some(location) = response.header("location").filter(|l| !l.is_empty()) {
                    let next = self.check(&urljoin(&target, location))?;
                    if seen.contains(&next) {
                        return Err(format!("redirect loop at {next}"));
                    }
                    crate::log_debug!(LOG, "{status}: {target} -> {next}");
                    seen.push(next.clone());
                    target = next;
                    continue;
                }
            }
            if !(200..300).contains(&status) {
                return Err(format!(
                    "{} answered HTTP {status} ({})",
                    urlsplit(&target).hostname(),
                    reason_phrase(status)
                ));
            }
            let content_type = response.header("content-type").unwrap_or("").to_string();
            let kind = content_type.split(';').next().unwrap_or("").trim().to_ascii_lowercase();
            let body = decode(&response.body, charset(&content_type).as_deref());
            self.fetched.fetch_add(1, Ordering::Relaxed);
            return Ok(Fetched {
                url: target,
                status,
                content_type: if kind.is_empty() { "text/html".to_string() } else { kind },
                bytes: response.body.len(),
                truncated: response.truncated,
                body,
            });
        }
        Err(format!("too many redirects starting at {url}"))
    }

    /// A fetched page reduced to text: HTML read as a page, JSON as it is,
    /// anything else with its entities unescaped.  The links are absolute
    /// `http(s)` addresses, each once, and none of them back into this same
    /// page (a table of contents, a "back to top").
    pub fn page(&self, url: &str) -> Result<Page, String> {
        let response = self.fetch(url)?;
        let (title, text, links) = match response.content_type.as_str() {
            "text/html" | "application/xhtml+xml" | "" => {
                let page = html_to_text(&response.body);
                let here = response.url.split('#').next().unwrap_or("");
                let mut seen = std::collections::HashSet::new();
                let mut links = Vec::new();
                for link in page.links {
                    let url = urljoin(&response.url, &link.url);
                    if !matches!(urlsplit(&url).scheme.as_str(), "http" | "https") || seen.contains(&url) {
                        continue;
                    }
                    if url.split('#').next().unwrap_or("") == here {
                        continue;
                    }
                    seen.insert(url.clone());
                    links.push(Link { text: link.text, url });
                }
                (page.title, page.text, links)
            }
            "application/json" => (String::new(), response.body.clone(), Vec::new()),
            _ => (String::new(), unescape(&response.body), Vec::new()),
        };
        Ok(Page {
            url: response.url,
            title,
            text: text.trim().to_string(),
            links,
            status: response.status,
            content_type: response.content_type,
            truncated: response.truncated,
        })
    }

    /// The results for a query from the first endpoint of `search_url` that
    /// has any.  `search_url` may list several endpoints separated by spaces,
    /// and the default does ([`DEFAULT_SEARCH_ENGINES`]).  An endpoint that
    /// refuses rather than answers - an HTTP error, a `202 Accepted`, a
    /// CAPTCHA where the results should be - is passed over for the next
    /// one, and when every one of them refused, the error says what each
    /// answered.  A JSON answer is read as such (SearxNG, MediaWiki,
    /// OpenSearch, ...), anything else as the engine's HTML.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
        let text = query.trim();
        if text.is_empty() {
            return Err("the search query is empty".to_string());
        }
        let quoted = quote_plus(text);
        let endpoints: Vec<&str> = self.search_url.split_whitespace().collect();
        let mut refusals = Vec::new();
        let mut answered = false;
        for endpoint in &endpoints {
            let url = if endpoint.contains("{query}") {
                endpoint.replace("{query}", &quoted)
            } else {
                let join = if endpoint.contains('?') { '&' } else { '?' };
                format!("{endpoint}{join}q={quoted}")
            };
            match self.search_at(&url) {
                Ok(results) if !results.is_empty() => return Ok(results.into_iter().take(limit).collect()),
                Ok(_) => answered = true,
                Err(err) if endpoints.len() == 1 => return Err(err),
                Err(err) => refusals.push(err),
            }
        }
        if answered || refusals.is_empty() {
            return Ok(Vec::new());
        }
        Err(format!("no search engine answered: {}", refusals.join("; ")))
    }

    /// The hits one search endpoint answers with; an error when it refuses
    /// instead.
    fn search_at(&self, url: &str) -> Result<Vec<SearchResult>, String> {
        let response = self.fetch(url)?;
        let host = match urlsplit(&response.url).hostname() {
            host if host.is_empty() => response.url.clone(),
            host => host,
        };
        if response.status == 202 {
            // "Accepted", and not answered: DuckDuckGo's way of saying no
            return Err(format!(
                "{host} answered HTTP {} instead of results - it may be turning automated searches away",
                response.status
            ));
        }
        let results = search_results(&response.body, &response.content_type, &response.url);
        if results.is_empty() && is_challenge(&response.body) {
            return Err(format!(
                "{host} answered with a CAPTCHA instead of results - it takes this client for a bot"
            ));
        }
        Ok(results)
    }
}

/// The charset a `Content-Type` names, lower-cased.
fn charset(content_type: &str) -> Option<String> {
    content_type.split(';').skip(1).find_map(|param| {
        let (key, value) = param.split_once('=')?;
        (key.trim().eq_ignore_ascii_case("charset"))
            .then(|| value.trim().trim_matches(['"', '\'']).to_ascii_lowercase())
    })
}

/// The Windows-1252 characters of 0x80..0x9f (the rest of its upper half is Latin-1).
const CP1252: [char; 32] = [
    '€', '\u{81}', '‚', 'ƒ', '„', '…', '†', '‡', 'ˆ', '‰', 'Š', '‹', 'Œ', '\u{8d}', 'Ž', '\u{8f}', '\u{90}', '‘', '’',
    '“', '”', '•', '–', '—', '˜', '™', 'š', '›', 'œ', '\u{9d}', 'ž', 'Ÿ',
];

/// A body as text in its charset: UTF-8 unless the page says Latin-1 or
/// Windows-1252; what does not decode is replaced.
fn decode(body: &[u8], charset: Option<&str>) -> String {
    match charset {
        Some("iso-8859-1" | "latin-1" | "latin1" | "l1" | "iso8859-1") => body.iter().map(|&b| b as char).collect(),
        Some("windows-1252" | "cp1252") => body
            .iter()
            .map(|&b| {
                if (0x80..0xa0).contains(&b) {
                    CP1252[(b - 0x80) as usize]
                } else {
                    b as char
                }
            })
            .collect(),
        _ => String::from_utf8_lossy(body).into_owned(),
    }
}

/// The reason phrase of a status, for the error a failed page reports.
fn reason_phrase(status: u16) -> &'static str {
    match status {
        300 => "Multiple Choices",
        301 => "Moved Permanently",
        302 => "Found",
        303 => "See Other",
        304 => "Not Modified",
        307 => "Temporary Redirect",
        308 => "Permanent Redirect",
        400 => "Bad Request",
        401 => "Unauthorized",
        402 => "Payment Required",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        406 => "Not Acceptable",
        407 => "Proxy Authentication Required",
        408 => "Request Timeout",
        409 => "Conflict",
        410 => "Gone",
        413 => "Request Entity Too Large",
        414 => "Request-URI Too Long",
        415 => "Unsupported Media Type",
        418 => "I'm a Teapot",
        429 => "Too Many Requests",
        451 => "Unavailable For Legal Reasons",
        500 => "Internal Server Error",
        501 => "Not Implemented",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

/// The hits in one answer of a search endpoint: a JSON API's results, else
/// the links out of the engine's page.
fn search_results(body: &str, content_type: &str, base: &str) -> Vec<SearchResult> {
    let head = body.trim_start();
    if content_type == "application/json" || head.starts_with('{') || head.starts_with('[') {
        let results = search_from_json(body);
        if !results.is_empty() {
            return results;
        }
    }
    let page = read_html(body, false); // the hits in the engine's own ranking
    let results = search_from_links(&page.links, base);
    if results.is_empty() && (page.text.starts_with('{') || page.text.starts_with('[')) {
        return search_from_json(&page.text); // a JSON answer that a browser drew as a page
    }
    results
}

/// OpenSearch suggestions, `[query, [titles], [descriptions], [urls]]`, as
/// the objects the other APIs answer with.
fn opensearch(list: &[Json]) -> Option<Vec<Json>> {
    if list.len() < 4 || !matches!(list[0], Json::Str(_)) {
        return None;
    }
    let (Json::Arr(titles), Json::Arr(descriptions), Json::Arr(urls)) = (&list[1], &list[2], &list[3]) else {
        return None;
    };
    Some(
        titles
            .iter()
            .zip(descriptions)
            .zip(urls)
            .map(|((title, description), url)| {
                Json::obj([
                    ("title", title.clone()),
                    ("description", description.clone()),
                    ("url", url.clone()),
                ])
            })
            .collect(),
    )
}

/// Results out of a JSON search API: SearxNG, Brave, MediaWiki
/// (`query.pages`), OpenSearch and the like.
fn search_from_json(body: &str) -> Vec<SearchResult> {
    let Ok(data) = parse(body) else { return Vec::new() };
    let mut items: Vec<Json> = match &data {
        Json::Obj(_) => {
            let mut found = Vec::new();
            for key in ["results", "items", "webPages", "web", "data", "query"] {
                let mut value = data.at(key).clone();
                if let Json::Obj(_) = value {
                    value = ["value", "results", "pages", "search"]
                        .iter()
                        .find_map(|k| value.get(k).cloned())
                        .unwrap_or(Json::Null);
                }
                if let Json::Arr(list) = value {
                    found = list;
                    break;
                }
            }
            found
        }
        Json::Arr(list) => opensearch(list).unwrap_or_else(|| list.clone()),
        _ => Vec::new(),
    };
    let ranked = |item: &Json| matches!(item, Json::Obj(_)) && matches!(item.at("index"), Json::Int(_) | Json::Num(_));
    if !items.is_empty() && items.iter().all(ranked) {
        // MediaWiki lists the pages of a search out of rank
        let rank = |item: &Json| item.at("index").as_f64().unwrap_or(0.0);
        items.sort_by(|a, b| rank(a).total_cmp(&rank(b)));
    }
    let first = |item: &Json, keys: &[&str]| keys.iter().find_map(|k| item.at(k).as_str().map(str::to_string));
    let mut out = Vec::new();
    for item in &items {
        if !matches!(item, Json::Obj(_)) {
            continue;
        }
        let Some(url) = first(item, &["url", "link", "href", "fullurl", "canonicalurl"]).filter(|u| !u.is_empty())
        else {
            continue;
        };
        let title = first(item, &["title", "name", "heading"]).unwrap_or_else(|| url.clone());
        let snippet = first(
            item,
            &["content", "snippet", "description", "body", "summary", "extract"],
        )
        .unwrap_or_default();
        out.push(SearchResult {
            title: collapse(&title),
            url,
            snippet: collapse(&snippet).chars().take(300).collect(),
        });
    }
    out
}

/// Whether a link is DuckDuckGo's redirect wrapper around a hit.
fn ddg_redirect(href: &str) -> bool {
    let lower = href.to_ascii_lowercase();
    let rest = lower
        .strip_prefix("https:")
        .or_else(|| lower.strip_prefix("http:"))
        .unwrap_or(&lower);
    rest.starts_with("//duckduckgo.com/l/?uddg=")
}

/// Results out of a search engine's page: the links that leave the engine,
/// each once, DuckDuckGo's redirect wrapper unwrapped.  An engine links a hit
/// more than once - its title, its address, a snippet of the page - so a
/// later link to a hit already listed lends it its text as the snippet: the
/// longest such text that has a space in it (an address has none), up to 300
/// characters.
fn search_from_links(links: &[Link], base: &str) -> Vec<SearchResult> {
    let engine = urlsplit(base).hostname();
    let mut out: Vec<SearchResult> = Vec::new();
    for link in links {
        let mut href = link.url.clone();
        if ddg_redirect(&href) {
            href = query_value(&urlsplit(&href).query, "uddg").unwrap_or_default();
        }
        let href = urljoin(base, &href);
        let parts = urlsplit(&href);
        let host = parts.hostname();
        if !matches!(parts.scheme.as_str(), "http" | "https") || host.is_empty() {
            continue;
        }
        if host == engine || engine.ends_with(&format!(".{host}")) || host.ends_with(&format!(".{engine}")) {
            continue;
        }
        if host == "duckduckgo.com" || host.ends_with(".duckduckgo.com") {
            continue; // its ads (y.js), its feedback page
        }
        if let Some(hit) = out.iter_mut().find(|r| r.url == href) {
            let snippet: String = link.text.chars().take(300).collect();
            let title: String = link.text.chars().take(200).collect();
            if snippet.contains(' ') && title != hit.title && snippet.chars().count() > hit.snippet.chars().count() {
                hit.snippet = snippet;
            }
            continue;
        }
        if link.text.chars().count() < 3 {
            continue;
        }
        out.push(SearchResult {
            title: link.text.chars().take(200).collect(),
            url: href,
            snippet: String::new(),
        });
    }
    out
}

/// What a search engine's "are you a human?" page says where its results
/// should be.
const CHALLENGE_MARKERS: [&str; 6] = [
    "captcha",
    "anomaly-modal",
    "unusual traffic",
    "are you a robot",
    "not a robot",
    "bots use duckduckgo",
];

fn is_challenge(body: &str) -> bool {
    let lower = body.to_lowercase();
    CHALLENGE_MARKERS.iter().any(|marker| lower.contains(marker))
}

// -- HTML to text -----------------------------------------------------------------------------------

/// A page reduced to text: its title, its lines, and its links as written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Html {
    pub title: String,
    pub text: String,
    pub links: Vec<Link>,
}

/// Tags that start a new line in the text.
const BLOCK_TAGS: [&str; 35] = [
    "p",
    "div",
    "br",
    "li",
    "tr",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "section",
    "article",
    "header",
    "footer",
    "nav",
    "aside",
    "blockquote",
    "pre",
    "table",
    "main",
    "ul",
    "ol",
    "dl",
    "dt",
    "dd",
    "figure",
    "figcaption",
    "form",
    "fieldset",
    "details",
    "summary",
    "caption",
    "hr",
    "address",
];

/// Table cells: set off from the cell before them by a space rather than
/// glued to it.
const CELL_TAGS: [&str; 2] = ["td", "th"];

/// Tags whose content goes with them: what is never shown, and the page's own
/// chrome and controls.
const DROP_TAGS: [&str; 19] = [
    "script", "style", "noscript", "template", "svg", "canvas", "head", "nav", "footer", "aside", "dialog", "button",
    "select", "textarea", "iframe", "object", "audio", "video", "datalist",
];

/// Elements that have no content and no end tag, so one of them can never
/// open a drop: a `<meta charset="utf-8">` once did, in Python and Go, and
/// every page that has one read as empty.
const VOID_TAGS: [&str; 14] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
];

/// Elements whose end tag may be left out: hiding one by its attributes could
/// hide the rest of the page.
const OPTIONAL_END_TAGS: [&str; 21] = [
    "html", "head", "body", "p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot", "caption", "colgroup",
    "option", "optgroup", "rb", "rt", "rtc", "rp",
];

/// ARIA roles of page chrome, dropped like the elements they stand for.
const DROP_ROLES: [&str; 11] = [
    "navigation",
    "banner",
    "contentinfo",
    "complementary",
    "search",
    "menu",
    "menubar",
    "toolbar",
    "dialog",
    "alertdialog",
    "tooltip",
];

/// `" ".join(text.split())`.
pub fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// How many letters and digits a text has: what the link density of a line is
/// measured in.  (`is_alphanumeric` also counts the combining vowel signs
/// Python's `isalnum` does not; they sit inside words, so a line reads as
/// link text or not the same way on both sides.)
fn letters(text: &str) -> usize {
    text.chars().filter(|c| c.is_alphanumeric()).count()
}

/// One tag: its lower-cased name, its attributes, whether it closes an
/// element, whether it closes itself, and where the markup continues.
struct Tag {
    name: String,
    attrs: Vec<(String, Option<String>)>,
    end: bool,
    self_closing: bool,
    next: usize,
}

/// Reads the tag that starts at `at` (a `<` followed by a letter, or `</`).
fn read_tag(markup: &str, at: usize) -> Option<Tag> {
    let bytes = markup.as_bytes();
    let mut i = at + 1;
    let end = bytes.get(i) == Some(&b'/');
    if end {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/' {
        i += 1;
    }
    let name = markup[start..i].to_ascii_lowercase();
    let mut attrs = Vec::new();
    let mut self_closing = false;
    loop {
        while i < bytes.len() && (bytes[i].is_ascii_whitespace() || bytes[i] == b'/') {
            if bytes[i] == b'/' {
                self_closing = true;
            }
            i += 1;
        }
        if i >= bytes.len() {
            return None;
        }
        if bytes[i] == b'>' {
            break;
        }
        self_closing = false;
        let key_start = i;
        while i < bytes.len() && !bytes[i].is_ascii_whitespace() && !matches!(bytes[i], b'=' | b'>' | b'/') {
            i += 1;
        }
        let key = markup[key_start..i].to_ascii_lowercase();
        while i < bytes.len() && bytes[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut value = None;
        if i < bytes.len() && bytes[i] == b'=' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && (bytes[i] == b'"' || bytes[i] == b'\'') {
                let quote = bytes[i];
                let value_start = i + 1;
                let close = bytes[value_start..].iter().position(|b| *b == quote)?;
                value = Some(unescape(&markup[value_start..value_start + close]));
                i = value_start + close + 1;
            } else {
                let value_start = i;
                while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' {
                    i += 1;
                }
                value = Some(unescape(&markup[value_start..i]));
            }
        }
        if !key.is_empty() {
            attrs.push((key, value));
        }
    }
    Some(Tag {
        name,
        attrs,
        end,
        self_closing,
        next: i + 1,
    })
}

/// Elements whose content is text rather than markup, taken whole up to their
/// end tag as Python's `html.parser` takes them (and as HTML has it): a `<`
/// inside a script or a title is not a tag.  The escapable ones have their
/// character references read; the others are taken as written.
const RAW_TEXT: [&str; 6] = ["script", "style", "xmp", "iframe", "noembed", "noframes"];
const ESCAPABLE_RAW_TEXT: [&str; 2] = ["title", "textarea"];

/// The offset of the `</name` that closes a raw-text element: case ignored,
/// and the name followed by a space, a `/` or a `>` (`</scripts>` closes no
/// script).
fn raw_end(markup: &str, from: usize, name: &str) -> usize {
    let bytes = markup.as_bytes();
    let mut i = from;
    while let Some(at) = markup[i..].find("</") {
        let at = i + at;
        let end = at + 2 + name.len();
        if end < bytes.len()
            && bytes[at + 2..end].eq_ignore_ascii_case(name.as_bytes())
            && matches!(bytes[end], b'\t' | b'\n' | b'\r' | b'\x0c' | b' ' | b'/' | b'>')
        {
            return at;
        }
        i = at + 2;
    }
    markup.len()
}

/// The value of an attribute (`""` for a bare one), `None` when the tag has
/// none; the last of the same name wins, as it does in Python's `dict(attrs)`.
fn attr<'a>(tag: &'a Tag, name: &str) -> Option<&'a str> {
    tag.attrs
        .iter()
        .rev()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_deref().unwrap_or(""))
}

/// An element's ARIA role: the first of the roles it lists, lower-cased.
fn role(tag: &Tag) -> String {
    attr(tag, "role")
        .unwrap_or("")
        .to_lowercase()
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_string()
}

/// One line of a page as it is read: its text, how many of its letters are
/// link text, and where it lies.
#[derive(Default)]
struct Line {
    text: String,
    linked: usize,
    main: bool,
    article: bool,
}

/// The page as a reader meets it rather than as its markup lists it (Python's
/// `_TextExtractor`):
///
/// * what is never shown (`<head>`, `<script>`, an element that is `hidden`
///   or `display: none`) and the page's own chrome (`<nav>`, `<footer>`,
///   `<aside>`, the page's `<header>`, buttons, menus, the ARIA navigation /
///   banner / search roles) go with everything inside them, links included;
/// * the lines follow the page's blocks: a paragraph written over several
///   lines of markup is one line, and a `<pre>` keeps its own;
/// * a run of lines made of nothing but link text (a menu, a list of
///   languages or of categories) moves after the rest, and so does what lies
///   outside the page's `<main>` (its `<article>` when it has no main).
#[derive(Default)]
struct Reader {
    lines: Vec<Line>,
    title: String,
    /// `(text, href, the line its text starts on)`.
    anchors: Vec<(String, String, usize)>,
    /// The element being dropped with everything inside it, and how many
    /// elements of its name are open, itself included.
    drop: Option<String>,
    drop_depth: usize,
    /// The element that is the page's main region, and the same count.
    main: Option<String>,
    main_depth: usize,
    article: usize,
    section: usize,
    pre: usize,
    in_title: bool,
    href: Option<String>,
    anchor: String,
    anchor_line: Option<usize>,
}

impl Reader {
    fn new() -> Reader {
        Reader {
            lines: vec![Line::default()],
            ..Default::default()
        }
    }

    fn last(&mut self) -> &mut Line {
        let at = self.lines.len() - 1;
        &mut self.lines[at]
    }

    fn line_break(&mut self) {
        if !self.last().text.is_empty() {
            self.lines.push(Line::default());
        }
    }

    fn space(&mut self) {
        let line = self.last();
        if !line.text.is_empty() {
            line.text.push(' ');
        }
    }

    fn add(&mut self, data: &str) {
        if data.is_empty() {
            return;
        }
        let at = self.lines.len() - 1;
        let (main, article, linking) = (self.main.is_some(), self.article > 0, self.href.is_some());
        let line = &mut self.lines[at];
        if line.text.is_empty() {
            line.main = main;
            line.article = article;
        }
        line.text.push_str(data);
        if linking {
            let count = letters(data);
            line.linked += count;
            if count > 0 && self.anchor_line.is_none() {
                self.anchor_line = Some(at);
            }
        }
    }

    fn data(&mut self, chunk: &str) {
        if !chunk.is_empty() {
            self.text(unescape(chunk));
        }
    }

    /// The content of a raw-text element, its references read only when it is
    /// an escapable one.
    fn raw(&mut self, chunk: &str, escapable: bool) {
        if !chunk.is_empty() {
            self.text(if escapable { unescape(chunk) } else { chunk.to_string() });
        }
    }

    fn text(&mut self, chunk: String) {
        if self.in_title {
            // the title lives inside <head>, which is otherwise dropped
            self.title.push_str(&chunk);
            return;
        }
        if self.drop.is_some() {
            return;
        }
        if self.href.is_some() {
            self.anchor.push_str(&chunk);
        }
        if self.pre == 0 {
            self.add(&chunk);
            return;
        }
        for (i, piece) in chunk.split('\n').enumerate() {
            // a <pre> keeps its lines
            if i > 0 {
                self.line_break();
            }
            self.add(piece);
        }
    }

    /// Whether an element (not a void one) goes, with everything inside it.
    fn drops(&self, tag: &Tag) -> bool {
        let name = tag.name.as_str();
        if DROP_TAGS.contains(&name) {
            return true;
        }
        if name == "header" && self.main.is_none() && self.article == 0 && self.section == 0 {
            return true; // the page's banner, not the header of an article or a section
        }
        if OPTIONAL_END_TAGS.contains(&name) {
            return false;
        }
        if attr(tag, "hidden").is_some_and(|value| value.trim().to_lowercase() != "until-found") {
            return true;
        }
        if attr(tag, "aria-hidden").is_some_and(|value| value.trim().to_lowercase() == "true") {
            return true;
        }
        if DROP_ROLES.contains(&role(tag).as_str()) {
            return true;
        }
        let style: String = attr(tag, "style")
            .unwrap_or("")
            .to_lowercase()
            .split_whitespace()
            .collect();
        style.contains("display:none") || style.contains("visibility:hidden")
    }

    fn start(&mut self, tag: &Tag) {
        let name = tag.name.as_str();
        let void = VOID_TAGS.contains(&name);
        match &self.main {
            // the region counts its own name everywhere, dropped or not
            Some(root) => {
                if root == name && !void {
                    self.main_depth += 1;
                }
            }
            None => {
                if self.drop.is_none() && !void && (name == "main" || role(tag) == "main") {
                    self.line_break();
                    self.main = Some(name.to_string());
                    self.main_depth = 1;
                }
            }
        }
        if let Some(root) = &self.drop {
            if root == name && !void {
                self.drop_depth += 1;
            } else if name == "body" && root == "head" {
                self.drop = None; // the head ends where the body begins, </head> or not
            } else if name == "title" && root == "head" {
                self.in_title = true;
            }
            return;
        }
        if !void && self.drops(tag) {
            if BLOCK_TAGS.contains(&name) {
                self.line_break();
            }
            self.drop = Some(name.to_string());
            self.drop_depth = 1;
            return;
        }
        match name {
            "title" => self.in_title = true,
            "a" => {
                self.href = tag
                    .attrs
                    .iter()
                    .rev()
                    .find(|(key, _)| key == "href")
                    .and_then(|(_, value)| value.clone());
                self.anchor.clear();
                self.anchor_line = None;
            }
            _ => {
                match name {
                    "article" => self.article += 1,
                    "section" => self.section += 1,
                    "pre" => self.pre += 1,
                    _ => {}
                }
                if BLOCK_TAGS.contains(&name) {
                    self.line_break();
                } else if CELL_TAGS.contains(&name) {
                    self.space();
                }
            }
        }
    }

    fn end(&mut self, name: &str) {
        let void = VOID_TAGS.contains(&name);
        if name == "title" {
            self.in_title = false;
        } else if name == "a" {
            self.close_anchor();
        }
        if let Some(root) = &self.drop {
            if root == name && !void {
                self.drop_depth -= 1;
                if self.drop_depth == 0 {
                    self.drop = None;
                    if BLOCK_TAGS.contains(&name) {
                        self.line_break();
                    }
                }
            }
        } else {
            match name {
                "article" => self.article = self.article.saturating_sub(1),
                "section" => self.section = self.section.saturating_sub(1),
                "pre" => self.pre = self.pre.saturating_sub(1),
                _ => {}
            }
            if BLOCK_TAGS.contains(&name) {
                self.line_break();
            }
        }
        if self.main.as_deref() == Some(name) && !void {
            self.main_depth -= 1;
            if self.main_depth == 0 {
                self.main = None;
                self.line_break();
            }
        }
    }

    fn close_anchor(&mut self) {
        let shown = collapse(&self.anchor);
        if let Some(target) = self.href.take() {
            if !target.is_empty() && !shown.is_empty() {
                let line = self.anchor_line.unwrap_or(self.lines.len() - 1);
                self.anchors.push((shown, target, line));
            }
        }
        self.anchor.clear();
        self.anchor_line = None;
    }

    /// The main text first, then the rest, and the links in the same order;
    /// or, without `reading_order`, both as the page has them.
    fn finish(self, reading_order: bool) -> Html {
        let texts: Vec<String> = self.lines.iter().map(|line| collapse(&line.text)).collect();
        let shown: Vec<usize> = (0..texts.len()).filter(|&i| !texts[i].is_empty()).collect();
        let mut order = shown.clone();
        let mut anchors: Vec<&(String, String, usize)> = self.anchors.iter().collect();
        if reading_order {
            let primary: Vec<bool> = if shown.iter().any(|&i| self.lines[i].main) {
                self.lines.iter().map(|line| line.main).collect()
            } else if shown.iter().any(|&i| self.lines[i].article) {
                self.lines.iter().map(|line| line.article).collect()
            } else {
                vec![true; self.lines.len()]
            };
            // a line of nothing but link text is a menu item when the line beside it is one too
            let weak: Vec<bool> = shown
                .iter()
                .map(|&i| self.lines[i].linked >= letters(&texts[i]))
                .collect();
            let mut menu = vec![false; self.lines.len()];
            for (k, &i) in shown.iter().enumerate() {
                if weak[k] && ((k > 0 && weak[k - 1]) || (k + 1 < shown.len() && weak[k + 1])) {
                    menu[i] = true;
                }
            }
            let tier: Vec<usize> = (0..self.lines.len())
                .map(|i| 2 * usize::from(menu[i]) + usize::from(!primary[i]))
                .collect();
            order.sort_by_key(|&i| (tier[i], i));
            anchors.sort_by_key(|anchor| tier[anchor.2]);
        }
        Html {
            title: collapse(&self.title),
            text: order.iter().map(|&i| texts[i].as_str()).collect::<Vec<_>>().join("\n"),
            links: anchors
                .into_iter()
                .map(|(text, url, _)| Link {
                    text: text.clone(),
                    url: url.clone(),
                })
                .collect(),
        }
    }
}

/// Reduces a page to readable text, its title and its links, the text that
/// the page is about first ([`Reader`]); broken markup yields whatever could
/// be read rather than an error.
pub fn html_to_text(markup: &str) -> Html {
    read_html(markup, true)
}

/// [`html_to_text`], or with `reading_order` off the lines and the links as
/// the page has them (a search engine's hits are ranked by it).
pub fn read_html(markup: &str, reading_order: bool) -> Html {
    let mut reader = Reader::new();
    let bytes = markup.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let Some(offset) = markup[i..].find('<') else {
            reader.data(&markup[i..]);
            break;
        };
        reader.data(&markup[i..i + offset]);
        i += offset;
        let rest = &markup[i..];
        if let Some(comment) = rest.strip_prefix("<!--") {
            i = match comment.find("-->") {
                Some(end) => i + 4 + end + 3,
                None => bytes.len(),
            };
            continue;
        }
        if rest.starts_with("<!") || rest.starts_with("<?") {
            i = match rest.find('>') {
                Some(end) => i + end + 1,
                None => bytes.len(),
            };
            continue;
        }
        let after = rest.as_bytes().get(1).copied();
        let is_tag = match after {
            Some(b'/') => rest.as_bytes().get(2).is_some_and(|b| b.is_ascii_alphabetic()),
            Some(b) => b.is_ascii_alphabetic(),
            None => false,
        };
        if !is_tag {
            if rest.starts_with("</>") {
                i += 3;
                continue;
            }
            // a stray '<' is text, as it is to Python's parser
            reader.data("<");
            i += 1;
            continue;
        }
        let Some(tag) = read_tag(markup, i) else {
            // a tag the document ends inside (a page cut off at the byte cap) is dropped, as Python's parser drops it
            break;
        };
        i = tag.next;
        let name = tag.name.as_str();
        if tag.end {
            reader.end(name);
            continue;
        }
        if tag.self_closing {
            if name == "br" && reader.drop.is_none() {
                reader.line_break();
            }
            continue;
        }
        reader.start(&tag);
        let escapable = ESCAPABLE_RAW_TEXT.contains(&name);
        if escapable || RAW_TEXT.contains(&name) {
            // text rather than markup, up to the end tag
            let end = raw_end(markup, i, name);
            reader.raw(&markup[i..end], escapable);
            i = end;
        }
    }
    reader.finish(reading_order)
}

/// The named character references a page is likely to use.
const ENTITIES: [(&str, &str); 106] = [
    ("amp", "&"),
    ("lt", "<"),
    ("gt", ">"),
    ("quot", "\""),
    ("apos", "'"),
    ("nbsp", "\u{a0}"),
    ("copy", "©"),
    ("reg", "®"),
    ("trade", "™"),
    ("mdash", "—"),
    ("ndash", "–"),
    ("hellip", "…"),
    ("lsquo", "‘"),
    ("rsquo", "’"),
    ("sbquo", "‚"),
    ("ldquo", "“"),
    ("rdquo", "”"),
    ("bdquo", "„"),
    ("laquo", "«"),
    ("raquo", "»"),
    ("lsaquo", "‹"),
    ("rsaquo", "›"),
    ("middot", "·"),
    ("bull", "•"),
    ("deg", "°"),
    ("plusmn", "±"),
    ("times", "×"),
    ("divide", "÷"),
    ("frac12", "½"),
    ("frac14", "¼"),
    ("frac34", "¾"),
    ("sup1", "¹"),
    ("sup2", "²"),
    ("sup3", "³"),
    ("micro", "µ"),
    ("para", "¶"),
    ("sect", "§"),
    ("euro", "€"),
    ("pound", "£"),
    ("yen", "¥"),
    ("cent", "¢"),
    ("curren", "¤"),
    ("iexcl", "¡"),
    ("iquest", "¿"),
    ("shy", "\u{ad}"),
    ("ordf", "ª"),
    ("ordm", "º"),
    ("not", "¬"),
    ("macr", "¯"),
    ("acute", "´"),
    ("cedil", "¸"),
    ("uml", "¨"),
    ("dagger", "†"),
    ("Dagger", "‡"),
    ("permil", "‰"),
    ("prime", "′"),
    ("Prime", "″"),
    ("larr", "←"),
    ("rarr", "→"),
    ("uarr", "↑"),
    ("darr", "↓"),
    ("harr", "↔"),
    ("hearts", "♥"),
    ("ensp", "\u{2002}"),
    ("emsp", "\u{2003}"),
    ("thinsp", "\u{2009}"),
    ("zwnj", "\u{200c}"),
    ("zwj", "\u{200d}"),
    ("minus", "−"),
    ("le", "≤"),
    ("ge", "≥"),
    ("ne", "≠"),
    ("asymp", "≈"),
    ("infin", "∞"),
    ("sum", "∑"),
    ("alpha", "α"),
    ("beta", "β"),
    ("gamma", "γ"),
    ("delta", "δ"),
    ("pi", "π"),
    ("sigma", "σ"),
    ("mu", "μ"),
    ("lambda", "λ"),
    ("omega", "ω"),
    ("Omega", "Ω"),
    ("agrave", "à"),
    ("aacute", "á"),
    ("acirc", "â"),
    ("auml", "ä"),
    ("aring", "å"),
    ("ccedil", "ç"),
    ("egrave", "è"),
    ("eacute", "é"),
    ("ecirc", "ê"),
    ("euml", "ë"),
    ("iacute", "í"),
    ("ntilde", "ñ"),
    ("oacute", "ó"),
    ("ouml", "ö"),
    ("uacute", "ú"),
    ("uuml", "ü"),
    ("szlig", "ß"),
    ("Eacute", "É"),
    ("Auml", "Ä"),
    ("Ouml", "Ö"),
    ("Uuml", "Ü"),
];

/// The references HTML still honours without their semicolon.
const LEGACY: [&str; 7] = ["amp", "lt", "gt", "quot", "nbsp", "copy", "reg"];

/// A numeric reference's character, with the replacements HTML prescribes.
fn numeric_reference(code: u32) -> char {
    if (0x80..0xa0).contains(&code) {
        return CP1252[(code - 0x80) as usize];
    }
    if code == 0 || (0xd800..=0xdfff).contains(&code) || code > 0x10ffff {
        return '\u{fffd}';
    }
    char::from_u32(code).unwrap_or('\u{fffd}')
}

/// Replaces character references (`&amp;`, `&#169;`, `&#xa9;`) with the
/// characters they stand for, as `html.unescape` does for the ones pages use.
pub fn unescape(text: &str) -> String {
    if !text.contains('&') {
        return text.to_string();
    }
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find('&') {
        out.push_str(&rest[..at]);
        let tail = &rest[at + 1..];
        if let Some(num) = tail.strip_prefix('#') {
            let (hex, digits) = match num.strip_prefix(['x', 'X']) {
                Some(h) => (true, h),
                None => (false, num),
            };
            let len = digits
                .find(|c: char| {
                    if hex {
                        !c.is_ascii_hexdigit()
                    } else {
                        !c.is_ascii_digit()
                    }
                })
                .unwrap_or(digits.len());
            if len > 0 {
                let code = u32::from_str_radix(&digits[..len.min(8)], if hex { 16 } else { 10 }).unwrap_or(0x110000);
                out.push(numeric_reference(code));
                let used = 1 + usize::from(hex) + len;
                rest = &tail[used..];
                if let Some(after) = rest.strip_prefix(';') {
                    rest = after;
                }
                continue;
            }
        } else {
            let len = tail.find(|c: char| !c.is_ascii_alphanumeric()).unwrap_or(tail.len());
            let name = &tail[..len];
            if tail[len..].starts_with(';') {
                if let Some((_, value)) = ENTITIES.iter().find(|(n, _)| *n == name) {
                    out.push_str(value);
                    rest = &tail[len + 1..];
                    continue;
                }
            } else if let Some(legacy) = LEGACY.iter().find(|l| name.starts_with(**l)) {
                let value = ENTITIES
                    .iter()
                    .find(|(n, _)| n == legacy)
                    .map(|(_, v)| *v)
                    .unwrap_or("");
                out.push_str(value);
                rest = &tail[legacy.len()..];
                continue;
            }
        }
        out.push('&');
        rest = tail;
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    const HOME: &str = "<html><head><title> Example Home </title><style>p{color:red}</style>\
        <script>alert('x')</script></head><body><h1>Welcome</h1><p>The cat sat on the mat.</p>\
        <a href='/cats'>All about cats</a> <a href='mailto:x@y.z'>mail</a>\
        <a href='http://127.0.0.1:1/other'>absolute</a></body></html>";
    const CATS: &str =
        "<html><head><title>Cats</title></head><body><p>A cat has four legs.</p><br><p>And a tail.</p></body></html>";
    /// A page as real sites write them: a banner, menus, a language list,
    /// hidden parts and a footer around the article.
    const ARTICLE: &str = "<html><head><meta charset=\"utf-8\"><title>Cat</title><link rel=stylesheet href=s.css>\
        </head><body><a href=\"#content\">Jump to content</a>\
        <header><form role=search><button>Search</button></form><nav><a href=\"/login\">Log in</a></nav></header>\
        <main id=\"content\"><header><h1>Cat</h1><ul><li><a href=\"https://de.example.org/\">Deutsch</a></li>\
        <li><a href=\"https://fr.example.org/\">Fran&ccedil;ais</a></li></ul></header>\
        <p>The cat is a small\n <a href=\"/wiki/Mammal\">mammal</a>; see <a href=\"#Legs\">below</a>.</p>\
        <table><tr><th>Legs:</th><td><a href=\"/wiki/Mammal\">4</a></td></tr></table>\
        <div hidden>x</div><input type=hidden hidden><pre>meow()\npurr()</pre>\
        <div role=\"navigation\"><a href=\"/wiki/Lion\">Lion</a></div></main>\
        <footer><a href=\"/privacy\">Privacy</a></footer></body></html>";
    /// DuckDuckGo's page: an ad, then a hit linked three times (its title, its
    /// address, a snippet) through the duckduckgo.com/l/ redirect.
    const DDG: &str = "<html><head><meta charset=\"utf-8\"><title>cats at DuckDuckGo</title></head><body>\
        <form><select name=kl><option>All Regions</option></select></form>\
        <h2><a href=\"https://duckduckgo.com/y.js?ad_domain=shop.example\">Cat food deals</a></h2>\
        <h2><a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">Cat - Encyclopedia</a></h2>\
        <a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">en.example.org/wiki/Cat</a>\
        <a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">The <b>cat</b> is a small \
        mammal.</a><a href=\"//duckduckgo.com/feedback.html\">Feedback</a></body></html>";
    const DDG_BLOCKED: &str = "<html><head><meta charset=\"utf-8\"><title>DuckDuckGo</title></head><body>\
        <div class=\"anomaly-modal__title\">Unfortunately, bots use DuckDuckGo too.</div></body></html>";
    const CAPTCHA: &str =
        "<html><body><p>Our systems have detected unusual traffic.</p><a href='/help'>Why?</a></body></html>";
    const NO_RESULTS: &str =
        "<html><head><meta charset=\"utf-8\"><title>No results</title></head><body>No results.</body></html>";
    const WIKI_API: &str = "{\"query\": {\"pages\": [{\"title\": \"Cat anatomy\", \"index\": 2, \"fullurl\": \
        \"https://en.example.org/wiki/Cat_anatomy\", \"extract\": \"Cat anatomy is the study of cats.\"}, \
        {\"title\": \"Cat\", \"index\": 1, \"fullurl\": \"https://en.example.org/wiki/Cat\", \"extract\": \"The cat is a \
        mammal.\"}]}}";
    const OPENSEARCH: &str = "[\"cat\", [\"Cat\", \"Catalonia\"], [\"\", \"A region\"], \
        [\"https://en.example.org/wiki/Cat\", \"https://en.example.org/wiki/Catalonia\"]]";
    const JSON_PAGE: &str =
        "<html><body><pre>{\"results\": [{\"title\": \"Cats\", \"url\": \"http://example.org/cats\"}]}\
        </pre></body></html>";
    const SEARCH_HTML: &str = "<html><body><a href='/internal'>engine link</a>\
        <a href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fhit'>A wrapped hit</a>\
        <a href='https://example.com/plain'>A plain hit</a></body></html>";

    /// The fake site of the Python and Go tests, on a real port.
    fn site() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let mut buf = vec![0u8; 8192];
                let n = stream.read(&mut buf).unwrap_or(0);
                let head = String::from_utf8_lossy(&buf[..n]).into_owned();
                let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
                let path = path.split('?').next().unwrap_or("/").to_string();
                let search = "{\"results\": [{\"title\": \"Cats\", \"url\": \"http://example.org/cats\", \
                     \"content\": \"cats have four legs\"}, {\"title\": \"Dogs\", \"url\": \
                     \"http://example.org/dogs\", \"content\": \"dogs bark\"}]}";
                let big = "x".repeat(200_000);
                let (status, kind, body, extra): (&str, &str, &str, String) = match path.as_str() {
                    "/" => ("200 OK", "text/html", HOME, String::new()),
                    "/cats" => ("200 OK", "text/html; charset=utf-8", CATS, String::new()),
                    "/search" => ("200 OK", "application/json", search, String::new()),
                    "/search-html" => ("200 OK", "text/html", SEARCH_HTML, String::new()),
                    "/article" => ("200 OK", "text/html", ARTICLE, String::new()),
                    "/ddg" => ("200 OK", "text/html", DDG, String::new()),
                    "/ddg-blocked" => ("202 Accepted", "text/html", DDG_BLOCKED, String::new()),
                    "/captcha" => ("200 OK", "text/html", CAPTCHA, String::new()),
                    "/no-results" => ("200 OK", "text/html", NO_RESULTS, String::new()),
                    "/wiki-api" => ("200 OK", "application/json", WIKI_API, String::new()),
                    "/opensearch" => ("200 OK", "application/x-suggestions+json", OPENSEARCH, String::new()),
                    "/json-page" => ("200 OK", "text/html", JSON_PAGE, String::new()),
                    "/plain" => ("200 OK", "text/plain", "just text", String::new()),
                    "/big" => ("200 OK", "text/html", big.as_str(), String::new()),
                    "/redirect" => ("302 Found", "text/html", "", "Location: /cats\r\n".to_string()),
                    "/loop" => ("302 Found", "text/html", "", "Location: /loop\r\n".to_string()),
                    "/away" => (
                        "302 Found",
                        "text/html",
                        "",
                        "Location: file:///etc/passwd\r\n".to_string(),
                    ),
                    _ => (
                        "404 Not Found",
                        "text/html",
                        "<html><body>not found</body></html>",
                        String::new(),
                    ),
                };
                let answer = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\n{extra}\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(answer.as_bytes());
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn local(url: &str) -> WebClient {
        WebClient::new(WebOptions {
            allow_private: true,
            search_url: Some(format!("{url}/search?q={{query}}")),
            timeout: Duration::from_secs(10),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn html_becomes_text_a_title_and_links() {
        let page = html_to_text(HOME);
        assert_eq!(page.title, "Example Home");
        assert!(page.text.contains("The cat sat on the mat."));
        assert!(!page.text.contains("alert") && !page.text.contains("color:red"));
        assert_eq!(page.links.len(), 3);
        assert_eq!(
            page.links[0],
            Link {
                text: "All about cats".into(),
                url: "/cats".into()
            }
        );
        assert_eq!(html_to_text("<p>a</p><p>b</p>").text, "a\nb");
        assert_eq!(html_to_text("a<br>b").text, "a\nb");
        assert_eq!(html_to_text("a<br/>b").text, "a\nb");
        assert!(html_to_text("<p>hello<<<>").text.contains("hello"));
        assert_eq!(html_to_text("<p>unclosed <b>bold").text, "unclosed bold");
        assert_eq!(html_to_text("").text, "");
        assert_eq!(html_to_text("<div>a<svg><text>svgtext</text></svg>b</div>").text, "ab");
        let script = html_to_text("<script>if (a < b) { x = \"</p>\" }</script><p>after</p>");
        assert_eq!(script.text, "after");
    }

    #[test]
    fn a_meta_tag_does_not_swallow_the_page() {
        let page = html_to_text(
            "<html><head><meta charset=\"utf-8\"><link rel=x href=y><title>T &amp; U</title></head>\
             <body><p>Hello &amp; bye&nbsp;now &#169; &copy x</p><a href=\"/q?a=1&amp;b=2\">Link &lt;1&gt;</a></body></html>",
        );
        assert_eq!(page.title, "T & U");
        assert_eq!(page.text, "Hello & bye now © © x\nLink <1>");
        assert_eq!(page.links[0].url, "/q?a=1&b=2");
    }

    #[test]
    fn the_main_text_comes_first() {
        let page = html_to_text(ARTICLE);
        assert_eq!(page.title, "Cat");
        assert_eq!(
            page.text,
            "Cat\nThe cat is a small mammal; see below.\nLegs: 4\nmeow()\npurr()\nJump to content\nDeutsch\nFrançais"
        );
        let hrefs: Vec<&str> = page.links.iter().map(|l| l.url.as_str()).collect();
        assert_eq!(
            hrefs,
            [
                "/wiki/Mammal",
                "#Legs",
                "/wiki/Mammal",
                "#content",
                "https://de.example.org/",
                "https://fr.example.org/"
            ]
        );
        // and as the page has them, for a search engine's ranking
        let written = read_html(ARTICLE, false);
        assert_eq!(
            written.text.split('\n').take(3).collect::<Vec<_>>(),
            ["Jump to content", "Cat", "Deutsch"]
        );
        assert_eq!(written.links[0].url, "#content");
    }

    #[test]
    fn what_is_never_shown_is_dropped() {
        let text = html_to_text(
            "<p>a</p><div hidden><div>nested</div>still hidden</div><p>b</p>\
             <span style='DISPLAY: none'>x</span><span style='visibility:hidden'>y</span><span aria-hidden=true>z</span>\
             <div role='navigation'>menu</div><div role='Search box'>find</div><dialog>cookies?</dialog>\
             <p>c<button>Click</button><select><option>One</option></select><textarea>typed</textarea></p>",
        )
        .text;
        assert_eq!(text, "a\nb\nc");
        // a void element cannot hide what follows it, nor can one whose end tag may be left out
        assert_eq!(
            html_to_text("<p>before</p><input name=\"m\" hidden><img hidden src=x><p>after</p>").text,
            "before\nafter"
        );
        assert_eq!(
            html_to_text("<ul><li hidden>one<li>two</ul><p>three").text,
            "one\ntwo\nthree"
        );
        assert_eq!(
            html_to_text("<div hidden=\"until-found\">findable</div>").text,
            "findable"
        );
        // the banner goes, the header of an article or a section stays
        assert_eq!(
            html_to_text("<header>Site name</header><article><header><h1>Story</h1></header><p>Body.</p></article>")
                .text,
            "Story\nBody."
        );
        assert_eq!(
            html_to_text("<section><header>Part one</header><p>x</p></section>").text,
            "Part one\nx"
        );
        // an unclosed head ends where the body begins; an icon's title is not the page's
        let page = html_to_text("<html><head><title>T</title><body><a href=/><svg><title>Icon</title></svg>Home</a>");
        assert_eq!((page.title.as_str(), page.text.as_str()), ("T", "Home"));
    }

    #[test]
    fn raw_text_is_taken_whole_and_a_cut_off_tag_dropped() {
        // a title is text, its references read; a script ends only at its own end tag
        let page = html_to_text("<title>Use <b> tags &amp; more</title><p>x</p>");
        assert_eq!((page.title.as_str(), page.text.as_str()), ("Use <b> tags & more", "x"));
        assert_eq!(
            html_to_text("<script>if (a) \"</scripts>\"</script><p>after</p>").text,
            "after"
        );
        assert_eq!(html_to_text("<xmp><b>raw</b> &amp;</xmp>").text, "<b>raw</b> &amp;");
        assert_eq!(html_to_text("<textarea><p>typed</p></textarea><p>b</p>").text, "b");
        // a page cut off at the byte cap in the middle of a tag
        let page = html_to_text("<p>cat</p><a href=\"/x");
        assert_eq!((page.text.as_str(), page.links.len()), ("cat", 0));
        assert_eq!(html_to_text("cat</").text, "cat</");
    }

    #[test]
    fn lines_follow_the_blocks() {
        // one link on its own line is part of the text; only a run of them is a menu
        assert_eq!(
            html_to_text("<p>First.</p><p><a href='/a'>A link</a></p><p>Last.</p>").text,
            "First.\nA link\nLast."
        );
        assert_eq!(
            html_to_text("<ul><li><a href='/a'>One</a><li><a href='/b'>Two</a></ul><p>Body.</p>").text,
            "Body.\nOne\nTwo"
        );
        assert_eq!(
            html_to_text(
                "<div>Promo</div><div role='main'><div>Body<div hidden>x</div></div>more</div><div>After</div>"
            )
            .text,
            "Body\nmore\nPromo\nAfter"
        );
        assert_eq!(
            html_to_text("<table><tr><td>Kingdom:</td><td><a href=/a>Animalia</a></td></tr></table>").text,
            "Kingdom: Animalia"
        );
    }

    #[test]
    fn a_page_links_each_address_once_and_never_back_to_itself() {
        let url = site();
        let page = local(&url).page(&format!("{url}/article")).unwrap();
        let hrefs: Vec<String> = page.links.into_iter().map(|l| l.url).collect();
        assert_eq!(
            hrefs,
            [
                format!("{url}/wiki/Mammal"),
                "https://de.example.org/".to_string(),
                "https://fr.example.org/".to_string()
            ]
        );
    }

    /// A client that searches the fake site's endpoints, in the order given.
    fn engines(url: &str, paths: &[&str]) -> WebClient {
        let chain: Vec<String> = paths.iter().map(|p| format!("{url}{p}?q={{query}}")).collect();
        WebClient::new(WebOptions {
            allow_private: true,
            search_url: Some(chain.join(" ")),
            timeout: Duration::from_secs(10),
            ..Default::default()
        })
        .unwrap()
    }

    #[test]
    fn a_search_reads_duckduckgo_and_falls_back_from_an_engine_that_refuses() {
        let url = site();
        let hits = engines(&url, &["/ddg"]).search("cats", 5).unwrap();
        assert_eq!(
            hits,
            [SearchResult {
                title: "Cat - Encyclopedia".into(),
                url: "https://en.example.org/wiki/Cat".into(),
                snippet: "The cat is a small mammal.".into()
            }]
        ); // the ad and the feedback link are DuckDuckGo's own, the address link is no snippet
        let hits = engines(&url, &["/ddg-blocked", "/wiki-api"])
            .search("cat legs", 5)
            .unwrap();
        let found: Vec<(&str, &str)> = hits.iter().map(|h| (h.title.as_str(), h.snippet.as_str())).collect();
        assert_eq!(
            found,
            [
                ("Cat", "The cat is a mammal."),
                ("Cat anatomy", "Cat anatomy is the study of cats.")
            ]
        ); // ranked by "index", not in the order the API lists them
        assert_eq!(
            engines(&url, &["/ddg-blocked"]).search("cats", 5).unwrap_err(),
            "127.0.0.1 answered HTTP 202 instead of results - it may be turning automated searches away"
        );
        assert!(engines(&url, &["/captcha"])
            .search("cats", 5)
            .unwrap_err()
            .contains("answered with a CAPTCHA instead of results"));
        let err = engines(&url, &["/ddg-blocked", "/captcha", "/nope"])
            .search("cats", 5)
            .unwrap_err();
        assert!(err.starts_with("no search engine answered: "), "{err}");
        for part in ["HTTP 202", "CAPTCHA", "HTTP 404"] {
            assert!(err.contains(part), "{err}");
        }
        assert!(engines(&url, &["/ddg-blocked", "/no-results"])
            .search("cats", 5)
            .unwrap()
            .is_empty());
        let hits = engines(&url, &["/opensearch"]).search("cat", 5).unwrap();
        let found: Vec<(&str, &str)> = hits.iter().map(|h| (h.title.as_str(), h.snippet.as_str())).collect();
        assert_eq!(found, [("Cat", ""), ("Catalonia", "A region")]);
        let hits = engines(&url, &["/json-page"]).search("cats", 5).unwrap();
        assert_eq!(hits[0].url, "http://example.org/cats"); // a JSON answer drawn as a page, as --browser has it
    }

    #[test]
    fn the_default_search_engines() {
        if std::env::var("RADIXNET_SEARCH_URL").is_ok_and(|v| !v.trim().is_empty()) {
            return; // the environment names its own
        }
        let web = WebClient::new(WebOptions::default()).unwrap();
        assert_eq!(web.search_url.split(' ').collect::<Vec<_>>(), DEFAULT_SEARCH_ENGINES);
        let spaced = WebClient::new(WebOptions {
            search_url: Some("  http://a/?q={query}\n http://b/  ".into()),
            ..Default::default()
        })
        .unwrap();
        assert_eq!(spaced.search_url, "http://a/?q={query} http://b/");
    }

    #[test]
    fn urls_split_and_join_as_python_s_do() {
        let s = urlsplit("HTTPS://user:pw@Example.com:8080/a/b?q=1#f");
        assert_eq!((s.scheme.as_str(), s.hostname().as_str()), ("https", "example.com"));
        assert_eq!(s.credentials(), ("user".to_string(), "pw".to_string()));
        assert_eq!(
            (s.path.as_str(), s.query.as_str(), s.fragment.as_str()),
            ("/a/b", "q=1", "f")
        );
        assert_eq!(urlsplit("http://[::1]:80/").hostname(), "::1");
        assert_eq!(urljoin("http://a/b/c/d?q", "../e"), "http://a/b/e");
        assert_eq!(urljoin("http://a/b/c", "/x"), "http://a/x");
        assert_eq!(urljoin("http://a/b/c", "?y"), "http://a/b/c?y");
        assert_eq!(urljoin("http://a/b/c", "//o/p"), "http://o/p");
        assert_eq!(urljoin("http://a/b/c", "mailto:x@y.z"), "mailto:x@y.z");
        assert_eq!(urljoin("http://a/b/", "./"), "http://a/b/");
        assert_eq!(urljoin("http://a", "cats"), "http://a/cats");
        assert_eq!(quote_plus("cats & dogs/ü"), "cats+%26+dogs%2F%C3%BC");
        assert_eq!(
            unquote_plus("https%3A%2F%2Fexample.net%2Fhit+x"),
            "https://example.net/hit x"
        );
        assert_eq!(request_target("http://h/a b?c=ü"), "http://h/a%20b?c=%C3%BC");
    }

    #[test]
    fn the_guard_refuses_what_is_not_a_public_page() {
        let web = WebClient::new(WebOptions::default()).unwrap();
        for url in [
            "http://127.0.0.1/",
            "http://10.0.0.1/",
            "http://192.168.1.1/",
            "http://localhost/",
        ] {
            let err = web.check(url).unwrap_err();
            assert!(err.contains("private"), "{url}: {err}");
        }
        assert!(web
            .check("http://169.254.169.254/")
            .unwrap_err()
            .contains("a link-local address"));
        assert!(web
            .check("http://0.0.0.0/")
            .unwrap_err()
            .contains("an unspecified address"));
        assert!(web.check("http://[::1]/").unwrap_err().contains("a loopback address"));
        assert_eq!(
            web.check("file:///etc/passwd").unwrap_err(),
            "only http and https are allowed (got file)"
        );
        assert!(web.check("ftp://example.com/x").is_err());
        assert_eq!(
            web.check("http://user:pw@example.com/").unwrap_err(),
            "URLs with credentials are refused"
        );
        assert_eq!(web.check("  ").unwrap_err(), "the URL is empty");
        assert!(web.check("not a url at all").is_err());
        let open = WebClient::new(WebOptions {
            allow_private: true,
            ..Default::default()
        })
        .unwrap();
        assert_eq!(
            open.check("<127.0.0.1/x>").unwrap(),
            "https://127.0.0.1/x",
            "a scheme is added"
        );
        assert!(WebClient::new(WebOptions {
            max_bytes: 10,
            ..Default::default()
        })
        .is_err());
        assert!(WebClient::new(WebOptions {
            timeout: Duration::ZERO,
            ..Default::default()
        })
        .is_err());
        assert!(public_peer("8.8.8.8".parse().unwrap()).is_ok());
        assert!(public_peer("100.64.0.1".parse().unwrap()).is_err());
        assert!(public_peer("fd00::1".parse().unwrap()).is_err());
        assert!(public_peer("2606:4700::1".parse().unwrap()).is_ok());
        assert!(public_peer("::ffff:127.0.0.1".parse().unwrap()).is_err());
    }

    #[test]
    fn it_browses_a_local_site() {
        let url = site();
        let web = local(&url);
        let page = web.page(&format!("{url}/")).unwrap();
        assert_eq!((page.title.as_str(), page.status), ("Example Home", 200));
        let hrefs: Vec<&str> = page.links.iter().map(|l| l.url.as_str()).collect();
        assert!(hrefs.contains(&format!("{url}/cats").as_str()), "{hrefs:?}");
        assert!(!hrefs.contains(&"mailto:x@y.z"));
        assert_eq!(web.page(&format!("{url}/plain")).unwrap().text, "just text");
        let err = web.page(&format!("{url}/nope")).unwrap_err();
        assert!(err.contains("404") && err.contains("Not Found"), "{err}");
        // a redirect is followed and checked again
        assert_eq!(web.page(&format!("{url}/redirect")).unwrap().title, "Cats");
        assert!(web.page(&format!("{url}/loop")).unwrap_err().contains("redirect loop"));
        assert!(web
            .page(&format!("{url}/away"))
            .unwrap_err()
            .contains("only http and https"));
        // the byte cap
        let small = WebClient::new(WebOptions {
            allow_private: true,
            max_bytes: 2048,
            ..Default::default()
        })
        .unwrap();
        let big = small.fetch(&format!("{url}/big")).unwrap();
        assert!(big.truncated);
        assert_eq!(big.bytes, 2048);
        assert_eq!(web.fetched(), 3, "a failed page and a redirect hop are not documents");
        // and the default guard refuses the same site outright
        let guarded = WebClient::new(WebOptions::default()).unwrap();
        assert!(guarded.page(&format!("{url}/cats")).unwrap_err().contains("loopback"));
    }

    #[test]
    fn it_searches_json_and_html() {
        let url = site();
        let web = local(&url);
        let results = web.search("cats", 5).unwrap();
        let titles: Vec<&str> = results.iter().map(|r| r.title.as_str()).collect();
        assert_eq!(titles, ["Cats", "Dogs"]);
        assert_eq!(results[0].snippet, "cats have four legs");
        assert_eq!(web.search("cats", 1).unwrap().len(), 1);
        let html = WebClient::new(WebOptions {
            allow_private: true,
            search_url: Some(format!("{url}/search-html?q={{query}}")),
            ..Default::default()
        })
        .unwrap();
        let urls: Vec<String> = html.search("cats", 5).unwrap().into_iter().map(|r| r.url).collect();
        assert_eq!(urls, ["https://example.net/hit", "https://example.com/plain"]);
        assert_eq!(web.search("  ", 5).unwrap_err(), "the search query is empty");
    }

    #[test]
    fn entities_and_charsets() {
        assert_eq!(
            unescape("a &amp; b &lt;c&gt; &#65;&#x42; &unknown; &"),
            "a & b <c> AB &unknown; &"
        );
        assert_eq!(unescape("&#128;"), "€");
        assert_eq!(decode(b"caf\xe9", Some("iso-8859-1")), "café");
        assert_eq!(decode("café".as_bytes(), None), "café");
        assert_eq!(charset("text/html; charset=\"UTF-8\""), Some("utf-8".to_string()));
    }
}
