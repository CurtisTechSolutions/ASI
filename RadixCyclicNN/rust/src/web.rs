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
//! title and the links, with `<script>` and `<style>` taken whole and
//! `<head>`, `<svg>` and the like dropped with their content.  It does one
//! thing Python's does not: `<meta>` and `<link>` are void elements, so a
//! `<meta charset="utf-8">` does not open a drop that never closes (in Python
//! and Go it does, and every page that has one reads as empty).

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

/// The search endpoint, `{query}` replaced by the URL-encoded query
/// (`$RADIXNET_SEARCH_URL` overrides it).  A JSON answer is understood too.
pub fn default_search_url() -> String {
    match std::env::var("RADIXNET_SEARCH_URL") {
        Ok(named) if !named.trim().is_empty() => named.trim().to_string(),
        _ => "https://html.duckduckgo.com/html/?q={query}".to_string(),
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
                .filter(|u| !u.trim().is_empty())
                .unwrap_or_else(default_search_url),
            fetched: AtomicUsize::new(0),
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
    /// anything else with its entities unescaped.
    pub fn page(&self, url: &str) -> Result<Page, String> {
        let response = self.fetch(url)?;
        let (title, text, links) = match response.content_type.as_str() {
            "text/html" | "application/xhtml+xml" | "" => {
                let page = html_to_text(&response.body);
                let links = page
                    .links
                    .into_iter()
                    .map(|link| Link {
                        text: link.text,
                        url: urljoin(&response.url, &link.url),
                    })
                    .filter(|link| matches!(urlsplit(&link.url).scheme.as_str(), "http" | "https"))
                    .collect();
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

    /// The results for a query from the search endpoint: a JSON answer (SearxNG,
    /// Brave and the like) is read as such, anything else as the engine's HTML.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, String> {
        let text = query.trim();
        if text.is_empty() {
            return Err("the search query is empty".to_string());
        }
        let quoted = quote_plus(text);
        let url = if self.search_url.contains("{query}") {
            self.search_url.replace("{query}", &quoted)
        } else {
            let join = if self.search_url.contains('?') { '&' } else { '?' };
            format!("{}{join}q={quoted}", self.search_url)
        };
        let response = self.fetch(&url)?;
        let head = response.body.trim_start();
        if response.content_type == "application/json" || head.starts_with('{') || head.starts_with('[') {
            let results = search_from_json(&response.body);
            if !results.is_empty() {
                return Ok(results.into_iter().take(limit).collect());
            }
        }
        Ok(search_from_html(&response.body, &response.url)
            .into_iter()
            .take(limit)
            .collect())
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

/// Results out of a JSON search API.
fn search_from_json(body: &str) -> Vec<SearchResult> {
    let Ok(data) = parse(body) else { return Vec::new() };
    let items: Vec<Json> = match &data {
        Json::Obj(_) => {
            let mut found = Vec::new();
            for key in ["results", "items", "webPages", "web", "data"] {
                let mut value = data.at(key).clone();
                if let Json::Obj(_) = value {
                    value = match value.get("value") {
                        Some(inner) => inner.clone(),
                        None => value.at("results").clone(),
                    };
                }
                if let Json::Arr(list) = value {
                    found = list;
                    break;
                }
            }
            found
        }
        Json::Arr(list) => list.clone(),
        _ => Vec::new(),
    };
    let first = |item: &Json, keys: &[&str]| keys.iter().find_map(|k| item.at(k).as_str().map(str::to_string));
    let mut out = Vec::new();
    for item in &items {
        if !matches!(item, Json::Obj(_)) {
            continue;
        }
        let Some(url) = first(item, &["url", "link", "href"]).filter(|u| !u.is_empty()) else {
            continue;
        };
        let title = first(item, &["title", "name", "heading"]).unwrap_or_else(|| url.clone());
        let snippet = first(item, &["content", "snippet", "description", "body", "summary"]).unwrap_or_default();
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

/// Results out of a search engine's HTML: the links that leave the engine,
/// de-duplicated, DuckDuckGo's redirect wrapper unwrapped.
fn search_from_html(body: &str, base: &str) -> Vec<SearchResult> {
    let page = html_to_text(body);
    let engine = urlsplit(base).hostname();
    let mut out: Vec<SearchResult> = Vec::new();
    for link in page.links {
        let mut href = link.url;
        if ddg_redirect(&href) {
            href = query_value(&urlsplit(&href).query, "uddg").unwrap_or_default();
        }
        let href = urljoin(base, &href);
        let parts = urlsplit(&href);
        let host = parts.hostname();
        if !matches!(parts.scheme.as_str(), "http" | "https")
            || host.is_empty()
            || out.iter().any(|r| r.url == href)
            || link.text.chars().count() < 3
        {
            continue;
        }
        if host == engine || engine.ends_with(&format!(".{host}")) || host.ends_with(&format!(".{engine}")) {
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

// -- HTML to text -----------------------------------------------------------------------------------

/// A page reduced to text: its title, its lines, and its links as written.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Html {
    pub title: String,
    pub text: String,
    pub links: Vec<Link>,
}

/// Tags that start a new line in the text.
const BLOCK_TAGS: [&str; 20] = [
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
];

/// Tags whose content is dropped with them.
const DROP_TAGS: [&str; 9] = [
    "script", "style", "noscript", "template", "svg", "canvas", "head", "meta", "link",
];

/// Elements that never have content or an end tag: counting one as an open
/// drop would never be undone.
const VOID_TAGS: [&str; 14] = [
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
];

/// `" ".join(text.split())`.
pub fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
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

/// The offset of the `</name` that closes a raw-text element, case ignored.
fn raw_end(markup: &str, from: usize, name: &str) -> usize {
    let needle = format!("</{name}");
    let lower = markup[from..].to_ascii_lowercase();
    match lower.find(&needle) {
        Some(at) => from + at,
        None => markup.len(),
    }
}

/// Reduces a page to readable text, its title and its links; broken markup
/// yields whatever could be read rather than an error.
pub fn html_to_text(markup: &str) -> Html {
    let mut text = String::new();
    let mut title = String::new();
    let mut links = Vec::new();
    let mut drop = 0usize;
    let mut in_title = false;
    let mut href: Option<String> = None;
    let mut anchor = String::new();
    let bytes = markup.as_bytes();
    let mut i = 0;
    let data = |chunk: &str,
                text: &mut String,
                title: &mut String,
                anchor: &mut String,
                drop: usize,
                in_title: bool,
                linking: bool| {
        if chunk.is_empty() {
            return;
        }
        let chunk = unescape(chunk);
        if in_title {
            // the title lives inside <head>, which is otherwise dropped
            title.push_str(&chunk);
            return;
        }
        if drop > 0 {
            return;
        }
        text.push_str(&chunk);
        if linking {
            anchor.push_str(&chunk);
        }
    };
    while i < bytes.len() {
        let Some(offset) = markup[i..].find('<') else {
            data(
                &markup[i..],
                &mut text,
                &mut title,
                &mut anchor,
                drop,
                in_title,
                href.is_some(),
            );
            break;
        };
        data(
            &markup[i..i + offset],
            &mut text,
            &mut title,
            &mut anchor,
            drop,
            in_title,
            href.is_some(),
        );
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
            data("<", &mut text, &mut title, &mut anchor, drop, in_title, href.is_some());
            i += 1;
            continue;
        }
        let Some(tag) = read_tag(markup, i) else {
            // an unterminated tag: the rest is text
            data(rest, &mut text, &mut title, &mut anchor, drop, in_title, href.is_some());
            break;
        };
        i = tag.next;
        let name = tag.name.as_str();
        if tag.end {
            if DROP_TAGS.contains(&name) {
                drop = drop.saturating_sub(1);
            } else if name == "title" {
                in_title = false;
            } else if BLOCK_TAGS.contains(&name) {
                text.push('\n');
            } else if name == "a" {
                let shown = collapse(&anchor);
                if let Some(target) = href.take() {
                    if !target.is_empty() && !shown.is_empty() {
                        links.push(Link {
                            text: shown,
                            url: target,
                        });
                    }
                }
                anchor.clear();
            }
            continue;
        }
        if tag.self_closing {
            if name == "br" {
                text.push('\n');
            }
            continue;
        }
        if DROP_TAGS.contains(&name) {
            if !VOID_TAGS.contains(&name) {
                drop += 1;
            }
            if name == "script" || name == "style" {
                // raw text: a '<' inside a script is not a tag
                i = raw_end(markup, i, name);
            }
            continue;
        }
        if name == "title" {
            in_title = true;
        } else if BLOCK_TAGS.contains(&name) {
            text.push('\n');
        } else if name == "a" {
            href = tag
                .attrs
                .iter()
                .rev()
                .find(|(k, _)| k == "href")
                .and_then(|(_, v)| v.clone());
            anchor.clear();
        }
    }
    let lines: Vec<String> = text.split('\n').map(collapse).filter(|l| !l.is_empty()).collect();
    Html {
        title: collapse(&title),
        text: lines.join("\n"),
        links,
    }
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
