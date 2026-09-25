package radixnet

// Browsing with the standard library: HTML reduced to readable text, and a
// client that refuses everything which is not a plain public web page.
//
// The Go twin of the browsing half of the Python radixnet.tools: http / https
// only, no credentials in the URL, no redirect to a private address, a byte
// cap, a timeout, the same reading of a page and the same search-result
// extraction.  The HTML reader is the Rust port's tokenizer (rust/src/web.rs):
// a stray "<" is text, a ">" inside a quoted attribute does not end the tag,
// and <script> / <style> are taken whole.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"sort"
	"strings"
	"time"
	"unicode"

	"html"
)

func setOf(names ...string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}
	return set
}

var (
	// blockTags start a new line of the text.
	blockTags = setOf("p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "header",
		"footer", "nav", "aside", "blockquote", "pre", "table", "main", "ul", "ol", "dl", "dt", "dd", "figure",
		"figcaption", "form", "fieldset", "details", "summary", "caption", "hr", "address")
	// cellTags are set off from the cell before them by a space rather than
	// glued to it.
	cellTags = setOf("td", "th")
	// dropTags take their content with them: what is never shown, and the
	// page's own chrome and controls.
	dropTags = setOf("script", "style", "noscript", "template", "svg", "canvas", "head", "nav", "footer", "aside",
		"dialog", "button", "select", "textarea", "iframe", "object", "audio", "video", "datalist")
	// voidTags have no content and no end tag, so one of them can never open a
	// drop: a <meta charset="utf-8"> once did, and every page that has one read
	// as empty.
	voidTags = setOf("area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source",
		"track", "wbr")
	// optionalEndTags may be left unclosed: hiding one by its attributes could
	// hide the rest of the page.
	optionalEndTags = setOf("html", "head", "body", "p", "li", "dt", "dd", "tr", "td", "th", "thead", "tbody", "tfoot",
		"caption", "colgroup", "option", "optgroup", "rb", "rt", "rtc", "rp")
	// dropRoles are the ARIA roles of page chrome, dropped like the elements
	// they stand for.
	dropRoles = setOf("navigation", "banner", "contentinfo", "complementary", "search", "menu", "menubar", "toolbar",
		"dialog", "alertdialog", "tooltip")
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

// HTMLToText reduces a page to readable text, its title and its links, the
// text the page is about first; broken markup yields whatever could be read
// rather than an error.
//
// The text is the page as a reader meets it rather than as its markup lists
// it: what is never shown (<head>, <script>, an element that is hidden or
// display: none) and the page's own chrome (<nav>, <footer>, <aside>, the
// page's <header>, buttons, menus, the ARIA navigation / banner / search
// roles) go with everything inside them, links included; the lines follow the
// page's blocks, so a paragraph written over several lines of markup is one
// line and a <pre> keeps its own; and a run of lines made of nothing but link
// text (a menu, a list of languages or of categories) moves after the rest,
// as does what lies outside the page's <main> (its <article> when it has no
// main).  The links come in the same order, the ones in the running text first.
func HTMLToText(markup string) Page { return readHTML(markup, true) }

// readHTML is HTMLToText, or with readingOrder off the lines and the links as
// the page has them (a search engine's hits are ranked by it).
func readHTML(markup string, readingOrder bool) Page {
	r := newHTMLReader()
	for i := 0; i < len(markup); {
		offset := strings.IndexByte(markup[i:], '<')
		if offset < 0 {
			r.data(markup[i:])
			break
		}
		r.data(markup[i : i+offset])
		i += offset
		rest := markup[i:]
		if strings.HasPrefix(rest, "<!--") {
			if end := strings.Index(rest[4:], "-->"); end >= 0 {
				i += 4 + end + 3
			} else {
				i = len(markup)
			}
			continue
		}
		if strings.HasPrefix(rest, "<!") || strings.HasPrefix(rest, "<?") {
			if end := strings.IndexByte(rest, '>'); end >= 0 {
				i += end + 1
			} else {
				i = len(markup)
			}
			continue
		}
		isTag := false
		if len(rest) > 1 {
			if rest[1] == '/' {
				isTag = len(rest) > 2 && isASCIILetter(rest[2])
			} else {
				isTag = isASCIILetter(rest[1])
			}
		}
		if !isTag {
			if strings.HasPrefix(rest, "</>") {
				i += 3
				continue
			}
			r.data("<") // a stray '<' is text, as it is to Python's parser
			i++
			continue
		}
		tag, ok := readTag(markup, i)
		if !ok { // a tag the document ends inside (a page cut off at the byte cap) is dropped, as Python's parser drops it
			break
		}
		i = tag.next
		if tag.end {
			r.end(tag.name)
			continue
		}
		if tag.selfClosing {
			if tag.name == "br" && r.drop == "" {
				r.lineBreak()
			}
			continue
		}
		r.start(&tag)
		if escapable := escapableRawText[tag.name]; escapable || rawText[tag.name] { // text rather than markup
			end := rawEnd(markup, i, tag.name)
			r.raw(markup[i:end], escapable)
			i = end
		}
	}
	return r.finish(readingOrder)
}

func isASCIILetter(b byte) bool { return b >= 'a' && b <= 'z' || b >= 'A' && b <= 'Z' }

func isASCIISpace(b byte) bool { return b == ' ' || b == '\t' || b == '\n' || b == '\f' || b == '\r' }

// asciiLower lower-cases the ASCII letters only, so every offset into the text
// stays where it was.
func asciiLower(text string) string {
	out := []byte(text)
	for i, b := range out {
		if b >= 'A' && b <= 'Z' {
			out[i] = b + 'a' - 'A'
		}
	}
	return string(out)
}

// letters is how many letters and digits a text has: what the link density of
// a line is measured in.
func letters(text string) int {
	count := 0
	for _, r := range text {
		if unicode.IsLetter(r) || unicode.IsNumber(r) {
			count++
		}
	}
	return count
}

// htmlAttr is one attribute of a tag; a bare one has no value.
type htmlAttr struct {
	key, value string
	bare       bool
}

// htmlTag is one tag: its lower-cased name, its attributes, whether it closes
// an element, whether it closes itself, and where the markup continues.
type htmlTag struct {
	name        string
	attrs       []htmlAttr
	end         bool
	selfClosing bool
	next        int
}

// attr is the value of an attribute ("" for a bare one) and whether the tag
// has it at all; the last of the same name wins, as in Python's dict(attrs).
func (t *htmlTag) attr(name string) (string, bool) {
	for i := len(t.attrs) - 1; i >= 0; i-- {
		if t.attrs[i].key == name {
			return t.attrs[i].value, true
		}
	}
	return "", false
}

// href is where an anchor points; a bare href points nowhere.
func (t *htmlTag) href() (string, bool) {
	for i := len(t.attrs) - 1; i >= 0; i-- {
		if t.attrs[i].key == "href" {
			return t.attrs[i].value, !t.attrs[i].bare
		}
	}
	return "", false
}

// role is an element's ARIA role: the first of the roles it lists, lower-cased.
func (t *htmlTag) role() string {
	value, _ := t.attr("role")
	if fields := strings.Fields(strings.ToLower(value)); len(fields) > 0 {
		return fields[0]
	}
	return ""
}

// readTag reads the tag that starts at `at` (a '<' followed by a letter, or
// "</"); ok is false when the tag never ends.
func readTag(markup string, at int) (htmlTag, bool) {
	i := at + 1
	tag := htmlTag{}
	if i < len(markup) && markup[i] == '/' {
		tag.end = true
		i++
	}
	start := i
	for i < len(markup) && !isASCIISpace(markup[i]) && markup[i] != '>' && markup[i] != '/' {
		i++
	}
	tag.name = asciiLower(markup[start:i])
	for {
		for i < len(markup) && (isASCIISpace(markup[i]) || markup[i] == '/') {
			if markup[i] == '/' {
				tag.selfClosing = true
			}
			i++
		}
		if i >= len(markup) {
			return tag, false
		}
		if markup[i] == '>' {
			break
		}
		tag.selfClosing = false
		keyStart := i
		for i < len(markup) && !isASCIISpace(markup[i]) && markup[i] != '=' && markup[i] != '>' && markup[i] != '/' {
			i++
		}
		key := asciiLower(markup[keyStart:i])
		for i < len(markup) && isASCIISpace(markup[i]) {
			i++
		}
		attr := htmlAttr{key: key, bare: true}
		if i < len(markup) && markup[i] == '=' {
			i++
			for i < len(markup) && isASCIISpace(markup[i]) {
				i++
			}
			if i < len(markup) && (markup[i] == '"' || markup[i] == '\'') {
				quote := markup[i]
				closing := strings.IndexByte(markup[i+1:], quote)
				if closing < 0 {
					return tag, false
				}
				attr.value, attr.bare = html.UnescapeString(markup[i+1:i+1+closing]), false
				i += closing + 2
			} else {
				valueStart := i
				for i < len(markup) && !isASCIISpace(markup[i]) && markup[i] != '>' {
					i++
				}
				attr.value, attr.bare = html.UnescapeString(markup[valueStart:i]), false
			}
		}
		if key != "" {
			tag.attrs = append(tag.attrs, attr)
		}
	}
	tag.next = i + 1
	return tag, true
}

// rawText and escapableRawText are the elements whose content is text rather
// than markup, taken whole up to their end tag as Python's html.parser takes
// them (and as HTML has it): a '<' inside a script or a title is not a tag.
// The escapable ones have their character references read; the others are
// taken as written.
var (
	rawText          = setOf("script", "style", "xmp", "iframe", "noembed", "noframes")
	escapableRawText = setOf("title", "textarea")
)

// rawEnd is the offset of the "</name" that closes a raw-text element: case
// ignored, and the name followed by a space, a '/' or a '>' ("</scripts>"
// closes no script).
func rawEnd(markup string, from int, name string) int {
	for i := from; ; {
		at := strings.Index(markup[i:], "</")
		if at < 0 {
			return len(markup)
		}
		at += i
		if end := at + 2 + len(name); end < len(markup) && asciiLower(markup[at+2:end]) == name &&
			strings.IndexByte("\t\n\r\f />", markup[end]) >= 0 {
			return at
		}
		i = at + 2
	}
}

// htmlLine is one line of a page as it is read: its pieces, how many of its
// letters are link text, and where it lies.
type htmlLine struct {
	parts   []string
	linked  int
	main    bool
	article bool
}

// htmlAnchor is a link with the line its text starts on.
type htmlAnchor struct {
	link Link
	line int
}

// htmlReader is the state of a page being read (Python's _TextExtractor).
type htmlReader struct {
	lines   []*htmlLine
	title   strings.Builder
	anchors []htmlAnchor
	// the element being dropped with everything inside it ("" for none), and
	// how many elements of its name are open, itself included
	drop      string
	dropDepth int
	// the element that is the page's main region, and the same count
	main                  string
	mainDepth             int
	article, section, pre int
	inTitle               bool
	linking               bool
	href                  string
	anchor                strings.Builder
	anchorLine            int
}

func newHTMLReader() *htmlReader {
	return &htmlReader{lines: []*htmlLine{{}}, anchorLine: -1}
}

func (r *htmlReader) last() *htmlLine { return r.lines[len(r.lines)-1] }

func (r *htmlReader) lineBreak() {
	if len(r.last().parts) > 0 {
		r.lines = append(r.lines, &htmlLine{})
	}
}

func (r *htmlReader) space() {
	if line := r.last(); len(line.parts) > 0 {
		line.parts = append(line.parts, " ")
	}
}

func (r *htmlReader) add(data string) {
	if data == "" {
		return
	}
	line := r.last()
	if len(line.parts) == 0 {
		line.main, line.article = r.main != "", r.article > 0
	}
	line.parts = append(line.parts, data)
	if r.linking {
		count := letters(data)
		line.linked += count
		if count > 0 && r.anchorLine < 0 {
			r.anchorLine = len(r.lines) - 1
		}
	}
}

func (r *htmlReader) data(chunk string) {
	if chunk != "" {
		r.text(html.UnescapeString(chunk))
	}
}

// raw takes the content of a raw-text element, its references read only when
// it is an escapable one.
func (r *htmlReader) raw(chunk string, escapable bool) {
	if chunk == "" {
		return
	}
	if escapable {
		chunk = html.UnescapeString(chunk)
	}
	r.text(chunk)
}

func (r *htmlReader) text(chunk string) {
	if r.inTitle { // the title lives inside <head>, which is otherwise dropped
		r.title.WriteString(chunk)
		return
	}
	if r.drop != "" {
		return
	}
	if r.linking {
		r.anchor.WriteString(chunk)
	}
	if r.pre == 0 {
		r.add(chunk)
		return
	}
	for i, piece := range strings.Split(chunk, "\n") { // a <pre> keeps its lines
		if i > 0 {
			r.lineBreak()
		}
		r.add(piece)
	}
}

// drops reports whether an element (not a void one) goes, with everything
// inside it.
func (r *htmlReader) drops(tag *htmlTag) bool {
	if dropTags[tag.name] {
		return true
	}
	if tag.name == "header" && r.main == "" && r.article == 0 && r.section == 0 {
		return true // the page's banner, not the header of an article or a section
	}
	if optionalEndTags[tag.name] {
		return false
	}
	if value, ok := tag.attr("hidden"); ok && strings.ToLower(strings.TrimSpace(value)) != "until-found" {
		return true
	}
	if value, _ := tag.attr("aria-hidden"); strings.ToLower(strings.TrimSpace(value)) == "true" {
		return true
	}
	if dropRoles[tag.role()] {
		return true
	}
	style, _ := tag.attr("style")
	style = strings.Join(strings.Fields(strings.ToLower(style)), "")
	return strings.Contains(style, "display:none") || strings.Contains(style, "visibility:hidden")
}

func (r *htmlReader) start(tag *htmlTag) {
	name, void := tag.name, voidTags[tag.name]
	if r.main != "" {
		if name == r.main && !void { // the region counts its own name everywhere, dropped or not
			r.mainDepth++
		}
	} else if r.drop == "" && !void && (name == "main" || tag.role() == "main") {
		r.lineBreak()
		r.main, r.mainDepth = name, 1
	}
	if r.drop != "" {
		switch {
		case name == r.drop && !void:
			r.dropDepth++
		case name == "body" && r.drop == "head": // the head ends where the body begins, </head> or not
			r.drop = ""
		case name == "title" && r.drop == "head":
			r.inTitle = true
		}
		return
	}
	if !void && r.drops(tag) {
		if blockTags[name] {
			r.lineBreak()
		}
		r.drop, r.dropDepth = name, 1
		return
	}
	switch name {
	case "title":
		r.inTitle = true
	case "a":
		r.href, r.linking = tag.href()
		r.anchor.Reset()
		r.anchorLine = -1
	default:
		switch name {
		case "article":
			r.article++
		case "section":
			r.section++
		case "pre":
			r.pre++
		}
		if blockTags[name] {
			r.lineBreak()
		} else if cellTags[name] {
			r.space()
		}
	}
}

func (r *htmlReader) end(name string) {
	void := voidTags[name]
	if name == "title" {
		r.inTitle = false
	} else if name == "a" {
		r.closeAnchor()
	}
	if r.drop != "" {
		if name == r.drop && !void {
			r.dropDepth--
			if r.dropDepth == 0 {
				r.drop = ""
				if blockTags[name] {
					r.lineBreak()
				}
			}
		}
	} else {
		switch name {
		case "article":
			r.article = max(0, r.article-1)
		case "section":
			r.section = max(0, r.section-1)
		case "pre":
			r.pre = max(0, r.pre-1)
		}
		if blockTags[name] {
			r.lineBreak()
		}
	}
	if r.main != "" && name == r.main && !void {
		r.mainDepth--
		if r.mainDepth == 0 {
			r.main = ""
			r.lineBreak()
		}
	}
}

func (r *htmlReader) closeAnchor() {
	text := collapse(r.anchor.String())
	if r.linking && r.href != "" && text != "" {
		line := r.anchorLine
		if line < 0 {
			line = len(r.lines) - 1
		}
		r.anchors = append(r.anchors, htmlAnchor{Link{Text: text, URL: r.href}, line})
	}
	r.href, r.linking, r.anchorLine = "", false, -1
	r.anchor.Reset()
}

// finish is the main text first, then the rest, and the links in the same
// order; or, without readingOrder, both as the page has them.
func (r *htmlReader) finish(readingOrder bool) Page {
	texts := make([]string, len(r.lines))
	shown := []int{}
	for i, line := range r.lines {
		texts[i] = collapse(strings.Join(line.parts, ""))
		if texts[i] != "" {
			shown = append(shown, i)
		}
	}
	order := append([]int{}, shown...)
	anchors := append([]htmlAnchor{}, r.anchors...)
	if readingOrder {
		anyMain, anyArticle := false, false
		for _, i := range shown {
			anyMain = anyMain || r.lines[i].main
			anyArticle = anyArticle || r.lines[i].article
		}
		primary := make([]bool, len(r.lines))
		for i, line := range r.lines {
			switch {
			case anyMain:
				primary[i] = line.main
			case anyArticle:
				primary[i] = line.article
			default:
				primary[i] = true
			}
		}
		// a line of nothing but link text is a menu item when the line beside it is one too
		weak := make([]bool, len(shown))
		for k, i := range shown {
			weak[k] = r.lines[i].linked >= letters(texts[i])
		}
		tier := make([]int, len(r.lines))
		for k, i := range shown {
			if weak[k] && ((k > 0 && weak[k-1]) || (k+1 < len(shown) && weak[k+1])) {
				tier[i] = 2
			}
		}
		for i := range r.lines {
			if !primary[i] {
				tier[i]++
			}
		}
		sort.SliceStable(order, func(a, b int) bool { return tier[order[a]] < tier[order[b]] })
		sort.SliceStable(anchors, func(a, b int) bool { return tier[anchors[a].line] < tier[anchors[b].line] })
	}
	lines := make([]string, 0, len(order))
	for _, i := range order {
		lines = append(lines, texts[i])
	}
	links := make([]Link, 0, len(anchors))
	for _, anchor := range anchors {
		links = append(links, anchor.link)
	}
	return Page{Title: collapse(r.title.String()), Text: strings.Join(lines, "\n"), Links: links}
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

// DefaultSearchEngines are the search endpoints tried in turn by default:
// DuckDuckGo's HTML page, its lite page, and then Wikipedia's own search API,
// which is made for programs and so still answers when a search engine takes
// this client for a bot (each of its hits carries the first sentences of the
// article).
var DefaultSearchEngines = []string{
	"https://html.duckduckgo.com/html/?q={query}",
	"https://lite.duckduckgo.com/lite/?q={query}",
	"https://en.wikipedia.org/w/api.php?action=query&format=json&formatversion=2&generator=search&gsrlimit=10" +
		"&prop=info%7Cextracts&inprop=url&exintro=1&explaintext=1&exsentences=2&exlimit=10&gsrsearch={query}",
}

// DefaultSearchURL is the search endpoints, separated by spaces and tried in
// turn until one has results; {query} is replaced by the URL-encoded query.
func DefaultSearchURL() string {
	if named := strings.TrimSpace(os.Getenv("RADIXNET_SEARCH_URL")); named != "" {
		return named
	}
	return strings.Join(DefaultSearchEngines, " ")
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
	SearchURL    string // one endpoint, or several separated by spaces
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
	searchURL = strings.Join(strings.Fields(searchURL), " ")
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

// addressKind is what kind of address an IP is when it is one the web tools
// refuse, most diagnosable first (127.0.0.1 is loopback *and* private), and ""
// for an ordinary public address: Python's ipaddress kinds, and the Rust
// port's words for them.
func addressKind(ip net.IP) string {
	if four := ip.To4(); four != nil {
		a, b, c := four[0], four[1], four[2]
		switch {
		case four.Equal(net.IPv4zero):
			return "an unspecified address (0.0.0.0 usually means DNS is blocking the name)"
		case a == 127:
			return "a loopback address"
		case a == 169 && b == 254:
			return "a link-local address"
		case a >= 224 && a <= 239:
			return "a multicast address"
		case a == 10, a == 172 && b&0xf0 == 16, a == 192 && b == 168, a == 0, a == 100 && b&0xc0 == 64,
			a == 192 && b == 0 && (c == 0 || c == 2), a == 198 && b&0xfe == 18, a == 198 && b == 51 && c == 100,
			a == 203 && b == 0 && c == 113, four.Equal(net.IPv4bcast):
			return "a private address"
		case a >= 240, a == 192 && b == 88 && c == 99:
			return "a reserved address"
		}
		return ""
	}
	six := ip.To16()
	if six == nil {
		return "not an IP address"
	}
	seg := func(i int) uint16 { return uint16(six[2*i])<<8 | uint16(six[2*i+1]) }
	switch {
	case six.Equal(net.IPv6unspecified):
		return "an unspecified address"
	case six.Equal(net.IPv6loopback):
		return "a loopback address"
	case seg(0)&0xffc0 == 0xfe80:
		return "a link-local address"
	case seg(0)&0xff00 == 0xff00:
		return "a multicast address"
	case seg(0)&0xfe00 == 0xfc00, seg(0)&0xffc0 == 0xfec0, seg(0) == 0x2001 && seg(1) == 0x0db8,
		seg(0) == 0x2001 && seg(1) < 0x0200, seg(0) == 0x0064 && seg(1) == 0xff9b && seg(2) == 0x0001,
		seg(0) == 0x0100 && seg(1) == 0 && seg(2) == 0 && seg(3) == 0:
		return "a private address"
	case seg(0)&0xe000 != 0x2000: // outside 2000::/3, the only range handed out as global unicast
		return "a reserved address"
	}
	return ""
}

// lookupIP and environmentProxy are how a host is resolved and which proxy a
// request goes through: variables, so the tests can stand in for DNS and for a
// proxy (http.ProxyFromEnvironment reads the environment once).
var (
	lookupIP         = net.LookupIP
	environmentProxy = http.ProxyFromEnvironment
)

// addressProblem is why host must not be fetched, in words, or "" for an
// ordinary public host.  A name that does not resolve and a name that resolves
// somewhere internal are kept apart: they are very different problems.
func addressProblem(host string) string {
	ips := []net.IP{}
	if ip := net.ParseIP(host); ip != nil {
		ips = append(ips, ip)
	} else {
		found, err := lookupIP(host)
		if err != nil {
			reason := err.Error()
			var dnsErr *net.DNSError
			if errors.As(err, &dnsErr) && dnsErr.Err != "" {
				reason = dnsErr.Err
			}
			return fmt.Sprintf("cannot be resolved here (%s)", reason)
		}
		if len(found) == 0 {
			return "cannot be resolved here (no address)"
		}
		ips = found
	}
	for _, ip := range ips {
		if what := addressKind(ip); what != "" {
			return fmt.Sprintf("resolves to %s, %s", ip, what)
		}
	}
	return ""
}

// proxyFor reports whether a request to this URL goes through a proxy, which
// then resolves the host name itself.
func proxyFor(target *url.URL) bool {
	proxy, err := environmentProxy(&http.Request{URL: target})
	return err == nil && proxy != nil
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
	host := strings.ToLower(parts.Hostname())
	if host == "" {
		return "", webErrorf("no host in %s", pythonRepr(text))
	}
	if !w.AllowPrivate {
		problem := addressProblem(host)
		// A host this process cannot resolve is still fetchable when a proxy does the resolving: behind one
		// (Docker, a corporate network, a sandbox) there is often no DNS here at all, and refusing it there
		// would make browsing impossible rather than safer.
		if strings.HasPrefix(problem, "cannot be resolved") && proxyFor(parts) {
			problem = ""
		}
		if problem != "" {
			hint := ""
			if !strings.HasPrefix(problem, "cannot be resolved") {
				hint = " - pass allow_private (--allow-private) to fetch it anyway"
			}
			return "", webErrorf("refusing %s: it %s%s", host, problem, hint)
		}
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
				host = strings.ToLower(parts.Hostname())
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
		contentType := response.Header.Get("Content-Type")
		kind := strings.ToLower(strings.TrimSpace(strings.Split(contentType, ";")[0]))
		if kind == "" {
			kind = "text/html"
		}
		final := target
		if response.Request != nil && response.Request.URL != nil {
			final = response.Request.URL.String()
		}
		w.Fetched++
		return &Response{URL: final, Status: response.StatusCode, ContentType: kind, Bytes: len(raw),
			Truncated: truncated, Body: decodeBody(raw, contentType)}, nil
	}
	return nil, webErrorf("too many redirects starting at %s", raw)
}

// cp1252 is what Windows-1252 has at 0x80..0x9f (the rest of its upper half is
// Latin-1).
var cp1252 = []rune("€\u0081‚ƒ„…†‡ˆ‰Š‹Œ\u008dŽ\u008f\u0090‘’“”•–—˜™š›œ\u009džŸ")

// decodeBody is a body as text in the charset its Content-Type names: UTF-8
// unless that is Latin-1 or Windows-1252, what does not decode replaced.
func decodeBody(raw []byte, contentType string) string {
	charset := ""
	for _, param := range strings.Split(contentType, ";")[1:] {
		if key, value, ok := strings.Cut(param, "="); ok && strings.EqualFold(strings.TrimSpace(key), "charset") {
			charset = strings.ToLower(strings.Trim(strings.TrimSpace(value), "\"'"))
			break
		}
	}
	switch charset {
	case "iso-8859-1", "latin-1", "latin1", "l1", "iso8859-1", "windows-1252", "cp1252":
		runes := make([]rune, len(raw))
		for i, b := range raw {
			runes[i] = rune(b)
			if b >= 0x80 && b < 0xa0 && (charset == "windows-1252" || charset == "cp1252") {
				runes[i] = cp1252[b-0x80]
			}
		}
		return string(runes)
	}
	return strings.ToValidUTF8(string(raw), "�")
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

// GetPage fetches a page and reduces it to text.  The links are absolute
// http(s) addresses, each once, and none of them back into this same page (a
// table of contents, a "back to top").
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
		here, _, _ := strings.Cut(response.URL, "#")
		base, _ := url.Parse(response.URL)
		seen := map[string]bool{}
		for _, link := range page.Links {
			href, err := url.Parse(link.URL)
			if err != nil {
				continue
			}
			if base != nil {
				href = base.ResolveReference(href)
			}
			address := href.String()
			if (href.Scheme != "http" && href.Scheme != "https") || seen[address] {
				continue
			}
			if before, _, _ := strings.Cut(address, "#"); before == here {
				continue
			}
			seen[address] = true
			out.Links = append(out.Links, Link{Text: link.Text, URL: address})
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

// Search returns the results for a query from the first endpoint of SearchURL
// that has any.  SearchURL may list several endpoints separated by spaces, and
// the default does (DefaultSearchEngines).  An endpoint that refuses rather
// than answers - an HTTP error, a 202 Accepted, a CAPTCHA where the results
// should be - is passed over for the next one, and when every one of
// them refused, the error says what each answered.  A JSON answer is read as
// such (SearxNG, MediaWiki, OpenSearch, ...), anything else as the engine's
// HTML.
func (w *WebClient) Search(query string, limit int) ([]SearchResult, error) {
	text := strings.TrimSpace(query)
	if text == "" {
		return nil, webErrorf("the search query is empty")
	}
	quoted := strings.ReplaceAll(url.QueryEscape(text), "%20", "+")
	endpoints := strings.Fields(w.SearchURL)
	refusals := []string{}
	answered := false
	for _, endpoint := range endpoints {
		target := endpoint
		if strings.Contains(target, "{query}") {
			target = strings.ReplaceAll(target, "{query}", quoted)
		} else if strings.Contains(target, "?") {
			target += "&q=" + quoted
		} else {
			target += "?q=" + quoted
		}
		results, err := w.searchAt(target)
		if err != nil {
			if len(endpoints) == 1 {
				return nil, err
			}
			refusals = append(refusals, err.Error())
			continue
		}
		if len(results) > 0 {
			return firstN(results, limit), nil
		}
		answered = true
	}
	if answered || len(refusals) == 0 {
		return []SearchResult{}, nil
	}
	return nil, webErrorf("no search engine answered: %s", strings.Join(refusals, "; "))
}

// searchAt is the hits one search endpoint answers with; an error when it
// refuses instead.
func (w *WebClient) searchAt(target string) ([]SearchResult, error) {
	response, err := w.Fetch(target)
	if err != nil {
		return nil, err
	}
	host := response.URL
	if parts, err := url.Parse(response.URL); err == nil && parts.Hostname() != "" {
		host = strings.ToLower(parts.Hostname())
	}
	if response.Status == 202 { // "Accepted", and not answered: DuckDuckGo's way of saying no
		return nil, webErrorf("%s answered HTTP %d instead of results - it may be turning automated searches away",
			host, response.Status)
	}
	results := searchResults(response.Body, response.ContentType, response.URL)
	if len(results) == 0 && isChallenge(response.Body) {
		return nil, webErrorf("%s answered with a CAPTCHA instead of results - it takes this client for a bot", host)
	}
	return results, nil
}

// searchResults are the hits in one answer of a search endpoint: a JSON API's
// results, else the links out of the engine's page.
func searchResults(body, contentType, base string) []SearchResult {
	head := strings.TrimLeft(body, " \t\r\n")
	if contentType == "application/json" || strings.HasPrefix(head, "{") || strings.HasPrefix(head, "[") {
		if results := searchFromJSON(body); len(results) > 0 {
			return results
		}
	}
	page := readHTML(body, false) // the hits in the engine's own ranking
	results := searchFromLinks(page.Links, base)
	if len(results) == 0 && (strings.HasPrefix(page.Text, "{") || strings.HasPrefix(page.Text, "[")) {
		return searchFromJSON(page.Text) // a JSON answer that a browser drew as a page
	}
	return results
}

// searchFromJSON reads results out of a JSON search API: SearxNG, Brave,
// MediaWiki (query.pages), OpenSearch and the like.
func searchFromJSON(body string) []SearchResult {
	var data any
	if err := json.Unmarshal([]byte(body), &data); err != nil {
		return nil
	}
	var items []any
	switch typed := data.(type) {
	case map[string]any:
		for _, key := range []string{"results", "items", "webPages", "web", "data", "query"} {
			value := typed[key]
			if nested, ok := value.(map[string]any); ok {
				value = nil
				for _, inner := range []string{"value", "results", "pages", "search"} {
					if found, present := nested[inner]; present {
						value = found
						break
					}
				}
			}
			if list, ok := value.([]any); ok {
				items = list
				break
			}
		}
	case []any:
		items = typed
		if list := openSearch(typed); list != nil {
			items = list
		}
	}
	ranked := len(items) > 0
	for _, item := range items {
		object, ok := item.(map[string]any)
		if _, number := object["index"].(float64); !ok || !number {
			ranked = false
			break
		}
	}
	if ranked { // MediaWiki lists the pages of a search out of rank
		sort.SliceStable(items, func(a, b int) bool {
			return items[a].(map[string]any)["index"].(float64) < items[b].(map[string]any)["index"].(float64)
		})
	}
	results := []SearchResult{}
	for _, item := range items {
		object, ok := item.(map[string]any)
		if !ok {
			continue
		}
		link := firstString(object, "url", "link", "href", "fullurl", "canonicalurl")
		if link == "" {
			continue
		}
		title := firstString(object, "title", "name", "heading")
		if title == "" {
			title = link
		}
		snippet := firstString(object, "content", "snippet", "description", "body", "summary", "extract")
		results = append(results, SearchResult{Title: collapse(title), URL: link,
			Snippet: clipRunes(collapse(snippet), 300)})
	}
	return results
}

// openSearch reads OpenSearch suggestions, [query, [titles], [descriptions],
// [urls]], as the objects the other APIs answer with (nil when it is not that).
func openSearch(list []any) []any {
	if len(list) < 4 {
		return nil
	}
	if _, ok := list[0].(string); !ok {
		return nil
	}
	titles, ok1 := list[1].([]any)
	descriptions, ok2 := list[2].([]any)
	urls, ok3 := list[3].([]any)
	if !ok1 || !ok2 || !ok3 {
		return nil
	}
	items := []any{}
	for i := 0; i < len(titles) && i < len(descriptions) && i < len(urls); i++ {
		items = append(items, map[string]any{"title": titles[i], "description": descriptions[i], "url": urls[i]})
	}
	return items
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

// searchFromLinks reads results out of a search engine's page: the links that
// leave the engine, each once.  An engine links a hit more than once - its
// title, its address, a snippet of the page - so a later link to a hit already
// listed lends it its text as the snippet: the longest such text that has a
// space in it (an address has none), up to 300 characters.
func searchFromLinks(links []Link, base string) []SearchResult {
	engine := ""
	baseURL, _ := url.Parse(base)
	if baseURL != nil {
		engine = strings.ToLower(baseURL.Hostname())
	}
	results := []SearchResult{}
	listed := map[string]int{}
	for _, link := range links {
		href := link.URL
		if ddgRedirect.MatchString(href) { // DuckDuckGo wraps every hit in a redirect
			target := ""
			if parsed, err := url.Parse(href); err == nil {
				target = parsed.Query().Get("uddg")
			}
			href = target
		}
		parsed, err := url.Parse(href)
		if err != nil {
			continue
		}
		if baseURL != nil {
			parsed = baseURL.ResolveReference(parsed)
		}
		href = parsed.String()
		host := strings.ToLower(parsed.Hostname())
		if (parsed.Scheme != "http" && parsed.Scheme != "https") || host == "" {
			continue
		}
		if host == engine || strings.HasSuffix(engine, "."+host) || strings.HasSuffix(host, "."+engine) {
			continue
		}
		if host == "duckduckgo.com" || strings.HasSuffix(host, ".duckduckgo.com") { // its ads (y.js), its feedback page
			continue
		}
		if at, seen := listed[href]; seen {
			snippet := clipRunes(link.Text, 300)
			if strings.Contains(snippet, " ") && clipRunes(link.Text, 200) != results[at].Title &&
				runeLen(snippet) > runeLen(results[at].Snippet) {
				results[at].Snippet = snippet
			}
			continue
		}
		if runeLen(link.Text) < 3 {
			continue
		}
		listed[href] = len(results)
		results = append(results, SearchResult{Title: clipRunes(link.Text, 200), URL: href})
	}
	return results
}

// challengeMarkers are what a search engine's "are you a human?" page says
// where its results should be.
var challengeMarkers = []string{"captcha", "anomaly-modal", "unusual traffic", "are you a robot", "not a robot",
	"bots use duckduckgo"}

func isChallenge(body string) bool {
	lower := strings.ToLower(body)
	for _, marker := range challengeMarkers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

// clipRunes is the first n runes of a text.
func clipRunes(text string, n int) string {
	runes := []rune(text)
	if len(runes) <= n {
		return text
	}
	return string(runes[:n])
}
