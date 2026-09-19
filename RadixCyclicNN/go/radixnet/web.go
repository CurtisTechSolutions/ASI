package radixnet

// Browsing with the standard library: HTML reduced to readable text, and a
// client that refuses everything which is not a plain public web page.
//
// The Go twin of the browsing half of the Python radixnet.tools: http / https
// only, no credentials in the URL, no redirect to a private address, a byte
// cap, a timeout, and the same search-result extraction.

import (
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
	"unicode"

	"html"
)

// blockTags start a new line in the extracted text; dropTags take their
// content with them.
var (
	blockTags = map[string]bool{
		"p": true, "div": true, "br": true, "li": true, "tr": true, "h1": true, "h2": true, "h3": true,
		"h4": true, "h5": true, "h6": true, "section": true, "article": true, "header": true,
		"footer": true, "nav": true, "aside": true, "blockquote": true, "pre": true, "table": true,
	}
	dropTags = map[string]bool{
		"script": true, "style": true, "noscript": true, "template": true, "svg": true, "canvas": true,
		"head": true, "meta": true, "link": true,
	}
	// rawText elements hold text rather than markup, so their content is taken
	// whole rather than tokenised (a "<" inside a script is not a tag).
	rawText = map[string]bool{"script": true, "style": true, "title": true, "textarea": true}
)

// Link is one anchor of a page: the text it shows and where it points.
type Link struct {
	Text string `json:"text"`
	URL  string `json:"url"`
}

// Page is an HTML document reduced to text.
type Page struct {
	Title string `json:"title"`
	Text  string `json:"text"`
	Links []Link `json:"links"`
}

// HTMLToText reduces a page to readable text, its title and its links; broken
// markup yields whatever could be read rather than an error.
func HTMLToText(markup string) Page {
	parts := []string{}
	title := ""
	links := []Link{}
	drop := 0
	href := ""
	inAnchor := false
	anchor := []string{}
	data := func(text string) {
		if drop > 0 {
			return
		}
		parts = append(parts, html.UnescapeString(text))
		if inAnchor {
			anchor = append(anchor, html.UnescapeString(text))
		}
	}
	for i := 0; i < len(markup); {
		next := strings.IndexByte(markup[i:], '<')
		if next < 0 {
			data(markup[i:])
			break
		}
		if next > 0 {
			data(markup[i : i+next])
			i += next
		}
		if strings.HasPrefix(markup[i:], "<!--") {
			end := strings.Index(markup[i+4:], "-->")
			if end < 0 {
				break
			}
			i += 4 + end + 3
			continue
		}
		if strings.HasPrefix(markup[i:], "<!") || strings.HasPrefix(markup[i:], "<?") {
			end := strings.IndexByte(markup[i:], '>')
			if end < 0 {
				break
			}
			i += end + 1
			continue
		}
		end := strings.IndexByte(markup[i:], '>')
		if end < 0 { // an unterminated tag: the rest is text
			data(markup[i:])
			break
		}
		raw := markup[i+1 : i+end]
		i += end + 1
		closing := strings.HasPrefix(raw, "/")
		selfClosing := strings.HasSuffix(raw, "/")
		name, attrs := parseTag(strings.Trim(raw, "/"))
		if name == "" {
			continue
		}
		switch {
		case closing:
			if dropTags[name] {
				if drop > 0 {
					drop--
				}
				continue
			}
			if blockTags[name] {
				parts = append(parts, "\n")
			} else if name == "a" {
				text := collapse(strings.Join(anchor, ""))
				if href != "" && text != "" {
					links = append(links, Link{Text: text, URL: href})
				}
				href, inAnchor, anchor = "", false, nil
			}
		case rawText[name] && !selfClosing:
			// take the element's content whole: <script> may hold anything
			body, rest := rawContent(markup, i, name)
			i = rest
			if name == "title" {
				title += html.UnescapeString(body) // the title lives inside <head>, which is otherwise dropped
			}
		case dropTags[name]:
			if !selfClosing {
				drop++
			}
		case blockTags[name]:
			parts = append(parts, "\n")
		case name == "a":
			href, inAnchor, anchor = attrs["href"], true, nil
		}
	}
	lines := []string{}
	for _, line := range strings.Split(strings.Join(parts, ""), "\n") {
		if collapsed := collapse(line); collapsed != "" {
			lines = append(lines, collapsed)
		}
	}
	return Page{Title: collapse(title), Text: strings.Join(lines, "\n"), Links: links}
}

// rawContent is the text of a raw-text element and the offset just past its
// closing tag.
func rawContent(markup string, from int, name string) (string, int) {
	lower := strings.ToLower(markup[from:])
	end := strings.Index(lower, "</"+name)
	if end < 0 {
		return markup[from:], len(markup)
	}
	body := markup[from : from+end]
	rest := from + end
	if close := strings.IndexByte(markup[rest:], '>'); close >= 0 {
		rest += close + 1
	} else {
		rest = len(markup)
	}
	return body, rest
}

// parseTag reads a tag's lower-cased name and its attributes.
func parseTag(raw string) (string, map[string]string) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return "", nil
	}
	i := 0
	for i < len(raw) && !unicode.IsSpace(rune(raw[i])) {
		i++
	}
	name := strings.ToLower(raw[:i])
	attrs := map[string]string{}
	rest := raw[i:]
	for {
		rest = strings.TrimSpace(rest)
		if rest == "" {
			break
		}
		eq := strings.IndexByte(rest, '=')
		space := strings.IndexFunc(rest, unicode.IsSpace)
		if eq < 0 || (space >= 0 && space < eq) { // a bare attribute
			if space < 0 {
				break
			}
			rest = rest[space:]
			continue
		}
		key := strings.ToLower(strings.TrimSpace(rest[:eq]))
		value := strings.TrimSpace(rest[eq+1:])
		if value == "" {
			break
		}
		if quote := value[0]; quote == '"' || quote == '\'' {
			closing := strings.IndexByte(value[1:], quote)
			if closing < 0 {
				attrs[key] = html.UnescapeString(value[1:])
				break
			}
			attrs[key] = html.UnescapeString(value[1 : 1+closing])
			rest = value[closing+2:]
			continue
		}
		if space := strings.IndexFunc(value, unicode.IsSpace); space >= 0 {
			attrs[key] = html.UnescapeString(value[:space])
			rest = value[space:]
			continue
		}
		attrs[key] = html.UnescapeString(value)
		break
	}
	return name, attrs
}

// -- the HTTP client ---------------------------------------------------------

// WebError is a page that could not be fetched, or an address that was refused.
type WebError struct{ Message string }

func (e *WebError) Error() string { return e.Message }

func webErrorf(format string, args ...any) *WebError { return &WebError{fmt.Sprintf(format, args...)} }

// DefaultUserAgent and DefaultSearchURL mirror the Python defaults, the
// environment overrides included.
func DefaultUserAgent() string {
	if named := strings.TrimSpace(os.Getenv("RADIXNET_USER_AGENT")); named != "" {
		return named
	}
	return "radixnet/0.1 (+https://curtistechsolutions.com)"
}

// DefaultSearchURL is the search endpoint; {query} is replaced by the
// URL-encoded query.
func DefaultSearchURL() string {
	if named := strings.TrimSpace(os.Getenv("RADIXNET_SEARCH_URL")); named != "" {
		return named
	}
	return "https://html.duckduckgo.com/html/?q={query}"
}

// WebClient fetches pages, refusing everything that is not a plain public web
// page: http / https only, no credentials in the URL, at most MaxBytes read,
// Timeout per request, redirects followed by hand (at most MaxRedirects) so
// every hop is checked again, and - unless AllowPrivate - no address inside a
// private, loopback or link-local range.
type WebClient struct {
	Timeout      time.Duration
	MaxBytes     int
	MaxRedirects int
	UserAgent    string
	AllowPrivate bool
	SearchURL    string
	Fetched      int

	client *http.Client
}

// NewWebClient builds a client with the Python defaults.
func NewWebClient(timeout time.Duration, maxBytes int, allowPrivate bool, searchURL, userAgent string) (*WebClient, error) {
	if timeout <= 0 {
		timeout = 20 * time.Second
	}
	if maxBytes == 0 {
		maxBytes = 2_000_000
	}
	if maxBytes < 1024 {
		return nil, fmt.Errorf("max_bytes must be >= 1024")
	}
	if searchURL == "" {
		searchURL = DefaultSearchURL()
	}
	if userAgent == "" {
		userAgent = DefaultUserAgent()
	}
	web := &WebClient{Timeout: timeout, MaxBytes: maxBytes, MaxRedirects: 4, UserAgent: userAgent,
		AllowPrivate: allowPrivate, SearchURL: searchURL}
	web.client = &http.Client{
		Timeout: timeout,
		// redirects are followed by hand so every hop goes through Check again
		CheckRedirect: func(*http.Request, []*http.Request) error { return http.ErrUseLastResponse },
	}
	return web, nil
}

// isPrivate reports whether host resolves to a loopback / private /
// link-local / reserved address (the SSRF guard).
func isPrivate(host string) bool {
	if ip := net.ParseIP(host); ip != nil {
		return privateIP(ip)
	}
	addrs, err := net.LookupIP(host)
	if err != nil || len(addrs) == 0 {
		return true // cannot resolve it -> do not let the request out
	}
	for _, ip := range addrs {
		if privateIP(ip) {
			return true
		}
	}
	return false
}

func privateIP(ip net.IP) bool {
	if ip.IsLoopback() || ip.IsPrivate() || ip.IsLinkLocalUnicast() || ip.IsLinkLocalMulticast() ||
		ip.IsMulticast() || ip.IsUnspecified() || ip.IsInterfaceLocalMulticast() {
		return true
	}
	if four := ip.To4(); four != nil { // the reserved ranges Python's ipaddress calls is_reserved
		switch {
		case four[0] == 0, four[0] >= 240, four[0] == 100 && four[1]&0xc0 == 64,
			four[0] == 192 && four[1] == 0 && four[2] == 0,
			four[0] == 198 && four[1]&0xfe == 18,
			four[0] == 192 && four[1] == 88 && four[2] == 99:
			return true
		}
	}
	return false
}

// Check is the normalised URL, or a WebError explaining why it is refused.
func (w *WebClient) Check(raw string) (string, error) {
	text := strings.Trim(strings.TrimSpace(raw), "<>\"'")
	if text == "" {
		return "", webErrorf("the URL is empty")
	}
	if !strings.Contains(text, "://") {
		text = "https://" + text
	}
	parts, err := url.Parse(text)
	if err != nil {
		return "", webErrorf("cannot read the URL %s: %v", pythonRepr(text), err)
	}
	if parts.Scheme != "http" && parts.Scheme != "https" {
		scheme := parts.Scheme
		if scheme == "" {
			scheme = "no scheme"
		}
		return "", webErrorf("only http and https are allowed (got %s)", scheme)
	}
	if parts.User != nil {
		return "", webErrorf("URLs with credentials are refused")
	}
	if parts.Hostname() == "" {
		return "", webErrorf("no host in %s", pythonRepr(text))
	}
	if !w.AllowPrivate && isPrivate(parts.Hostname()) {
		return "", webErrorf("refusing %s: it is a private, loopback or unresolvable address", parts.Hostname())
	}
	return parts.String(), nil
}

// Response is one fetched document.
type Response struct {
	URL         string
	Status      int
	ContentType string
	Bytes       int
	Truncated   bool
	Body        string
}

// Fetch reads one page, following redirects by hand and checking every hop.
func (w *WebClient) Fetch(raw string) (*Response, error) {
	target, err := w.Check(raw)
	if err != nil {
		return nil, err
	}
	seen := map[string]bool{target: true}
	for hop := 0; hop <= w.MaxRedirects; hop++ {
		request, err := http.NewRequest(http.MethodGet, target, nil)
		if err != nil {
			return nil, webErrorf("cannot fetch %s: %v", target, err)
		}
		request.Header.Set("User-Agent", w.UserAgent)
		request.Header.Set("Accept", "text/html,application/xhtml+xml,text/plain;q=0.9,application/json;q=0.8,*/*;q=0.1")
		request.Header.Set("Accept-Language", "en")
		response, err := w.client.Do(request)
		if err != nil {
			return nil, webErrorf("cannot fetch %s: %v", target, unwrapURLError(err))
		}
		if location := response.Header.Get("Location"); location != "" && isRedirect(response.StatusCode) {
			response.Body.Close()
			base, _ := url.Parse(target)
			next, err := url.Parse(location)
			if err != nil {
				return nil, webErrorf("cannot fetch %s: %v", target, err)
			}
			target, err = w.Check(base.ResolveReference(next).String())
			if err != nil {
				return nil, err
			}
			if seen[target] {
				return nil, webErrorf("redirect loop at %s", target)
			}
			seen[target] = true
			continue
		}
		defer response.Body.Close()
		if response.StatusCode >= 400 {
			host := ""
			if parts, err := url.Parse(target); err == nil {
				host = parts.Hostname()
			}
			return nil, webErrorf("%s answered HTTP %d (%s)", host, response.StatusCode,
				strings.TrimSpace(strings.TrimPrefix(response.Status, fmt.Sprintf("%d", response.StatusCode))))
		}
		raw, err := io.ReadAll(io.LimitReader(response.Body, int64(w.MaxBytes)+1))
		if err != nil {
			return nil, webErrorf("cannot fetch %s: %v", target, err)
		}
		truncated := len(raw) > w.MaxBytes
		if truncated {
			raw = raw[:w.MaxBytes]
		}
		kind := strings.ToLower(strings.TrimSpace(strings.Split(response.Header.Get("Content-Type"), ";")[0]))
		if kind == "" {
			kind = "text/html"
		}
		final := target
		if response.Request != nil && response.Request.URL != nil {
			final = response.Request.URL.String()
		}
		w.Fetched++
		return &Response{URL: final, Status: response.StatusCode, ContentType: kind, Bytes: len(raw),
			Truncated: truncated, Body: string(raw)}, nil
	}
	return nil, webErrorf("too many redirects starting at %s", raw)
}

func isRedirect(status int) bool {
	switch status {
	case 301, 302, 303, 307, 308:
		return true
	}
	return false
}

// unwrapURLError strips the "Get \"url\":" prefix Go puts on a transport error,
// so the message reads like Python's.
func unwrapURLError(err error) error {
	if wrapped, ok := err.(*url.Error); ok {
		return wrapped.Err
	}
	return err
}

// FetchedPage is a page reduced to text, with its links made absolute.
type FetchedPage struct {
	URL         string
	Title       string
	Text        string
	Links       []Link
	Status      int
	ContentType string
	Truncated   bool
}

// GetPage fetches a page and reduces it to text.
func (w *WebClient) GetPage(raw string) (*FetchedPage, error) {
	response, err := w.Fetch(raw)
	if err != nil {
		return nil, err
	}
	out := &FetchedPage{URL: response.URL, Status: response.Status, ContentType: response.ContentType,
		Truncated: response.Truncated, Links: []Link{}}
	switch response.ContentType {
	case "text/html", "application/xhtml+xml", "":
		page := HTMLToText(response.Body)
		out.Title, out.Text = page.Title, page.Text
		base, _ := url.Parse(response.URL)
		for _, link := range page.Links {
			href, err := url.Parse(link.URL)
			if err != nil {
				continue
			}
			if base != nil {
				href = base.ResolveReference(href)
			}
			if href.Scheme != "http" && href.Scheme != "https" {
				continue
			}
			out.Links = append(out.Links, Link{Text: link.Text, URL: href.String()})
		}
	case "application/json":
		out.Text = response.Body
	default:
		out.Text = html.UnescapeString(response.Body)
	}
	out.Text = strings.TrimSpace(out.Text)
	return out, nil
}

// SearchResult is one hit of a search.
type SearchResult struct {
	Title   string `json:"title"`
	URL     string `json:"url"`
	Snippet string `json:"snippet"`
}

// Search returns the results for a query from SearchURL (a JSON answer is
// parsed as such, else the HTML).
func (w *WebClient) Search(query string, limit int) ([]SearchResult, error) {
	text := strings.TrimSpace(query)
	if text == "" {
		return nil, webErrorf("the search query is empty")
	}
	quoted := strings.ReplaceAll(url.QueryEscape(text), "%20", "+")
	target := w.SearchURL
	if strings.Contains(target, "{query}") {
		target = strings.ReplaceAll(target, "{query}", quoted)
	} else if strings.Contains(target, "?") {
		target += "&q=" + quoted
	} else {
		target += "?q=" + quoted
	}
	response, err := w.Fetch(target)
	if err != nil {
		return nil, err
	}
	head := strings.TrimLeft(response.Body, " \t\r\n")
	if response.ContentType == "application/json" || strings.HasPrefix(head, "{") || strings.HasPrefix(head, "[") {
		if results := searchFromJSON(response.Body); len(results) > 0 {
			return firstN(results, limit), nil
		}
	}
	return firstN(searchFromHTML(response.Body, response.URL), limit), nil
}

// searchFromJSON reads results out of a JSON search API (SearxNG, Brave and
// the like).
func searchFromJSON(body string) []SearchResult {
	var data any
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return nil
	}
	var items []any
	switch typed := data.(type) {
	case map[string]any:
		for _, key := range []string{"results", "items", "webPages", "web", "data"} {
			value := typed[key]
			if nested, ok := value.(map[string]any); ok {
				if inner, ok := nested["value"]; ok {
					value = inner
				} else {
					value = nested["results"]
				}
			}
			if list, ok := value.([]any); ok {
				items = list
				break
			}
		}
	case []any:
		items = typed
	}
	results := []SearchResult{}
	for _, item := range items {
		object, ok := item.(map[string]any)
		if !ok {
			continue
		}
		link := firstString(object, "url", "link", "href")
		if link == "" {
			continue
		}
		title := firstString(object, "title", "name", "heading")
		if title == "" {
			title = link
		}
		snippet := firstString(object, "content", "snippet", "description", "body", "summary")
		results = append(results, SearchResult{Title: collapse(title), URL: link,
			Snippet: clipRunes(collapse(snippet), 300)})
	}
	return results
}

func firstString(object map[string]any, keys ...string) string {
	for _, key := range keys {
		if text, ok := object[key].(string); ok {
			return text
		}
	}
	return ""
}

var ddgRedirect = regexp.MustCompile(`(?i)^(?:https?:)?//duckduckgo\.com/l/\?uddg=`)

// searchFromHTML reads results out of a search engine's HTML: the links that
// leave the engine, de-duplicated.
func searchFromHTML(body, base string) []SearchResult {
	page := HTMLToText(body)
	engine := ""
	baseURL, _ := url.Parse(base)
	if baseURL != nil {
		engine = baseURL.Hostname()
	}
	results := []SearchResult{}
	seen := map[string]bool{}
	for _, link := range page.Links {
		href := link.URL
		if ddgRedirect.MatchString(href) { // DuckDuckGo wraps every hit in a redirect
			if parsed, err := url.Parse(href); err == nil {
				if target := parsed.Query().Get("uddg"); target != "" {
					href = target
				}
			}
		}
		parsed, err := url.Parse(href)
		if err != nil {
			continue
		}
		if baseURL != nil {
			parsed = baseURL.ResolveReference(parsed)
		}
		href = parsed.String()
		host := parsed.Hostname()
		if (parsed.Scheme != "http" && parsed.Scheme != "https") || host == "" || seen[href] || runeLen(link.Text) < 3 {
			continue
		}
		if host == engine || strings.HasSuffix(engine, "."+host) || strings.HasSuffix(host, "."+engine) {
			continue
		}
		seen[href] = true
		results = append(results, SearchResult{Title: clipRunes(link.Text, 200), URL: href})
	}
	return results
}

// clipRunes is the first n runes of a text.
func clipRunes(text string, n int) string {
	runes := []rune(text)
	if len(runes) <= n {
		return text
	}
	return string(runes[:n])
}
