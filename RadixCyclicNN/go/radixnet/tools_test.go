package radixnet

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// A fake website stands in for the internet: a couple of HTML pages, a JSON
// search endpoint, a redirect chain and a page larger than the byte cap.
// Nothing here reaches the network - the client is pointed at 127.0.0.1 with
// AllowPrivate, which is exactly the guard the tests also check refuses by
// default.

const homePage = "<html><head><title> Example Home </title><style>p{color:red}</style>" +
	"<script>alert('x')</script></head><body><h1>Welcome</h1><p>The cat sat on the mat.</p>" +
	"<a href='/cats'>All about cats</a> <a href='mailto:x@y.z'>mail</a>" +
	"<a href='http://127.0.0.1:1/other'>absolute</a></body></html>"

const catsPage = "<html><head><title>Cats</title></head><body><p>A cat has four legs.</p><br><p>And a tail.</p></body></html>"

const searchHTML = "<html><body><a href='/internal'>engine link</a>" +
	"<a href='https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fhit'>A wrapped hit</a>" +
	"<a href='https://example.com/plain'>A plain hit</a></body></html>"

// articlePage is a page as real sites write them: a <meta charset> before the
// title (which once made every page read as empty), a banner, menus, a
// language list, hidden parts and a footer around the article.
const articlePage = "<html><head><meta charset=\"utf-8\"><title>Cat</title><link rel=stylesheet href=s.css>" +
	"</head><body><a href=\"#content\">Jump to content</a>" +
	"<header><form role=search><button>Search</button></form><nav><a href=\"/login\">Log in</a></nav></header>" +
	"<main id=\"content\"><header><h1>Cat</h1><ul><li><a href=\"https://de.example.org/\">Deutsch</a></li>" +
	"<li><a href=\"https://fr.example.org/\">Fran&ccedil;ais</a></li></ul></header>" +
	"<p>The cat is a small\n <a href=\"/wiki/Mammal\">mammal</a>; see <a href=\"#Legs\">below</a>.</p>" +
	"<table><tr><th>Legs:</th><td><a href=\"/wiki/Mammal\">4</a></td></tr></table>" +
	"<div hidden>x</div><input type=hidden hidden><pre>meow()\npurr()</pre>" +
	"<div role=\"navigation\"><a href=\"/wiki/Lion\">Lion</a></div></main>" +
	"<footer><a href=\"/privacy\">Privacy</a></footer></body></html>"

// ddgPage is DuckDuckGo's page: an ad, then a hit linked three times (its
// title, its address, a snippet) through the duckduckgo.com/l/ redirect.
const ddgPage = "<html><head><meta charset=\"utf-8\"><title>cats at DuckDuckGo</title></head><body>" +
	"<form><select name=kl><option>All Regions</option></select></form>" +
	"<h2><a href=\"https://duckduckgo.com/y.js?ad_domain=shop.example\">Cat food deals</a></h2>" +
	"<h2><a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">Cat - Encyclopedia</a></h2>" +
	"<a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">en.example.org/wiki/Cat</a>" +
	"<a href=\"//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.example.org%2Fwiki%2FCat&amp;rut=1\">The <b>cat</b> is a small " +
	"mammal.</a><a href=\"//duckduckgo.com/feedback.html\">Feedback</a></body></html>"

const wikiAPI = `{"query": {"pages": [{"title": "Cat anatomy", "index": 2, "fullurl": "https://en.example.org/wiki/Cat_anatomy",
 "extract": "Cat anatomy is the study of cats."}, {"title": "Cat", "index": 1, "fullurl": "https://en.example.org/wiki/Cat",
 "extract": "The cat is a mammal."}]}}`

func fakeSite(t *testing.T) *httptest.Server {
	t.Helper()
	searchJSON, _ := json.Marshal(map[string]any{"results": []map[string]any{
		{"title": "Cats", "url": "http://example.org/cats", "content": "cats have four legs"},
		{"title": "Dogs", "url": "http://example.org/dogs", "content": "dogs bark"},
	}})
	site := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch strings.SplitN(r.URL.Path, "?", 2)[0] {
		case "/":
			fmt.Fprint(w, homePage)
		case "/cats":
			fmt.Fprint(w, catsPage)
		case "/search":
			w.Header().Set("Content-Type", "application/json")
			w.Write(searchJSON)
		case "/search-html":
			w.Write([]byte(searchHTML))
		case "/article":
			fmt.Fprint(w, articlePage)
		case "/ddg":
			w.Write([]byte(ddgPage))
		case "/ddg-blocked":
			w.WriteHeader(http.StatusAccepted)
			fmt.Fprint(w, "<html><body><div class=\"anomaly-modal__title\">Unfortunately, bots use DuckDuckGo too.</div></body></html>")
		case "/search-203": // a proxy that rewrote the answer: still an answer
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusNonAuthoritativeInfo)
			w.Write(searchJSON)
		case "/captcha":
			fmt.Fprint(w, "<html><body><p>Our systems have detected unusual traffic.</p><a href='/help'>Why?</a></body></html>")
		case "/no-results":
			fmt.Fprint(w, "<html><head><meta charset=\"utf-8\"><title>No results</title></head><body>No results.</body></html>")
		case "/wiki-api":
			w.Header().Set("Content-Type", "application/json")
			fmt.Fprint(w, wikiAPI)
		case "/opensearch":
			w.Header().Set("Content-Type", "application/x-suggestions+json")
			fmt.Fprint(w, `["cat", ["Cat", "Catalonia"], ["", "A region"], ["https://en.example.org/wiki/Cat", "https://en.example.org/wiki/Catalonia"]]`)
		case "/json-page":
			fmt.Fprint(w, `<html><body><pre>{"results": [{"title": "Cats", "url": "http://example.org/cats"}]}</pre></body></html>`)
		case "/latin1":
			w.Header().Set("Content-Type", "text/html; charset=ISO-8859-1")
			w.Write([]byte("<p>caf\xe9 \x80</p>"))
		case "/cp1252":
			w.Header().Set("Content-Type", "text/html; charset=windows-1252")
			w.Write([]byte("<p>\x93quoted\x94 \x80</p>"))
		case "/plain":
			w.Header().Set("Content-Type", "text/plain")
			fmt.Fprint(w, "just text")
		case "/big":
			fmt.Fprint(w, strings.Repeat("x", 200000))
		case "/redirect":
			http.Redirect(w, r, "/cats", http.StatusFound)
		case "/loop":
			http.Redirect(w, r, "/loop", http.StatusFound)
		case "/away":
			w.Header().Set("Location", "file:///etc/passwd")
			w.WriteHeader(http.StatusFound)
		default:
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprint(w, "<html><body>not found</body></html>")
		}
	}))
	t.Cleanup(site.Close)
	return site
}

func localWeb(t *testing.T, site *httptest.Server) *WebClient {
	t.Helper()
	web, err := NewWebClient(10*time.Second, 0, true, site.URL+"/search?q={query}", "")
	if err != nil {
		t.Fatal(err)
	}
	return web
}

// -- the text format ---------------------------------------------------------

func TestCallTextIsOneJSONLine(t *testing.T) {
	if out := CallText("web_fetch", map[string]any{"url": "u"}); out != `<tool>web_fetch {"url": "u"}</tool>`+"\n" {
		t.Fatalf("the call is one JSON line, spaced as Python writes it: %q", out)
	}
	if out := CallText("now", nil); out != "<tool>now {}</tool>\n" {
		t.Fatalf("no arguments: %q", out)
	}
}

func TestResultTextCollapsesAndClips(t *testing.T) {
	if out := ResultText("a\n  b\tc", DefaultObservationChars); out != "<result>a b c</result>\n" {
		t.Fatalf("whitespace is collapsed: %q", out)
	}
	if out := ResultText(strings.Repeat("x", 50), 10); out != "<result>"+strings.Repeat("x", 10)+" ...</result>\n" {
		t.Fatalf("clipped: %q", out)
	}
	if out := ResultText(strings.Repeat("x", 50), -1); out != "<result>"+strings.Repeat("x", 50)+"</result>\n" {
		t.Fatalf("no limit: %q", out)
	}
}

func TestAnswerHeaderAndFindAnswer(t *testing.T) {
	if out := AnswerText(" four\nlegs "); out != "<answer>four legs</answer>\n" {
		t.Fatalf("answer: %q", out)
	}
	if out := TaskHeader("  how   many?\n"); out != "TASK: how many?\n" {
		t.Fatalf("header: %q", out)
	}
	if out := FindAnswer("noise <answer> 42 </answer> more"); out != "42" {
		t.Fatalf("found: %q", out)
	}
	if out := FindAnswer("<answer>truncated"); out != "truncated" { // the generator stopped early
		t.Fatalf("truncated: %q", out)
	}
	if out := FindAnswer("nothing here"); out != "" {
		t.Fatalf("no answer: %q", out)
	}
	if out := FindAnswer("<answer>   </answer>"); out != "" {
		t.Fatalf("an empty answer is none: %q", out)
	}
}

func TestTranscriptIsOneTrainingText(t *testing.T) {
	steps := []TranscriptStep{{
		Name: "calculator", Arguments: map[string]any{"expression": "1+1"},
		Result: &ToolResult{Tool: "calculator", OK: true, Output: "2"},
	}}
	answer := "two"
	text := TranscriptText("What is 1+1?", steps, &answer, DefaultObservationChars)
	want := "TASK: What is 1+1?\n" + `<tool>calculator {"expression": "1+1"}</tool>` + "\n" +
		"<result>2</result>\n<answer>two</answer>\n"
	if text != want {
		t.Fatalf("the attempt is one text:\n%q\nwant\n%q", text, want)
	}
	if out := TranscriptText("q", nil, nil, DefaultObservationChars); out != "TASK: q\n" {
		t.Fatalf("no calls, no answer: %q", out)
	}
}

func TestFormatObservationMarksFailures(t *testing.T) {
	ok := &ToolResult{Tool: "t", OK: true, Output: "fine"}
	bad := &ToolResult{Tool: "t", OK: false, Error: "boom"}
	if out := FormatObservation(ok, DefaultObservationChars); out != "<result>fine</result>\n" {
		t.Fatalf("ok: %q", out)
	}
	if out := FormatObservation(bad, DefaultObservationChars); out != "<result>ERROR: boom</result>\n" {
		t.Fatalf("failed: %q", out)
	}
}

// -- parsing -----------------------------------------------------------------

func offlineBox(t *testing.T) *ToolBox {
	t.Helper()
	box, err := DefaultToolBox(ToolOptions{Offline: true})
	if err != nil {
		t.Fatal(err)
	}
	return box
}

func TestParseCallReadsWhateverItCan(t *testing.T) {
	box := offlineBox(t)
	for _, row := range []struct {
		text string
		want string
	}{
		{`<tool>calculator {"expression": "1+1"}</tool>`, "1+1"},
		{"<tool>calculator 5*5</tool>", "5*5"},
		{"<tool>calculator expression=2+2</tool>", "2+2"},
		{`<tool>calculator {"expression": "9"}`, "9"}, // unterminated: read to the end
	} {
		call := ParseCall(row.text, box)
		if call == nil || !call.OK() {
			t.Fatalf("%q: %+v", row.text, call)
		}
		if call.Name != "calculator" || call.Arguments["expression"] != row.want {
			t.Fatalf("%q: %+v", row.text, call.Arguments)
		}
	}
	if call := ParseCall("nothing here", box); call != nil {
		t.Fatalf("no call: %+v", call)
	}
}

func TestParseArgumentsRepairsTruncatedJSON(t *testing.T) {
	out, err := ParseArguments(`{"a": "b"} trailing junk`, nil)
	if err != nil || out["a"] != "b" {
		t.Fatalf("the last brace wins: %+v %v", out, err)
	}
	out, err = ParseArguments(`{"url": "http://example.com`, nil) // cut off mid-string
	if err != nil || out["url"] != "http://example.com" {
		t.Fatalf("the missing quote and brace are added: %+v %v", out, err)
	}
	if _, err := ParseArguments("!!!", nil); err == nil {
		t.Fatal("rubbish with no tool to guess from is refused")
	}
}

func TestParseCallReportsWhatItCannotUse(t *testing.T) {
	box := offlineBox(t)
	call := ParseCall("<tool>nosuchtool {}</tool>", box)
	if call == nil || call.OK() || !strings.Contains(call.Error, "unknown tool") {
		t.Fatalf("an unknown tool is reported, not dropped: %+v", call)
	}
	call = ParseCall("<tool>{}</tool>", box)
	if call == nil || call.OK() || !strings.Contains(call.Error, "no tool name") {
		t.Fatalf("a nameless call is reported: %+v", call)
	}
}

// -- the registry ------------------------------------------------------------

func TestToolCoercionAndDefaults(t *testing.T) {
	tool := &Tool{Name: "t", Params: []Param{
		{Name: "text", Type: "string", Required: true},
		{Name: "count", Type: "integer", Default: 3},
		{Name: "on", Type: "boolean", Default: false},
		{Name: "ratio", Type: "number", Default: 0.5},
	}}
	out, err := tool.Coerce(map[string]any{"TEXT": 12, "count": "7", "on": "yes", "extra": "dropped"})
	if err != nil {
		t.Fatal(err)
	}
	if out["text"] != "12" || out["count"] != 7 || out["on"] != true || out["ratio"] != 0.5 {
		t.Fatalf("coerced, aliased, defaulted and pruned: %+v", out)
	}
	if _, ok := out["extra"]; ok {
		t.Fatalf("unknown names are dropped: %+v", out)
	}
	if _, err := tool.Coerce(map[string]any{}); err == nil {
		t.Fatal("a missing required argument is refused")
	}
}

func TestToolBoxCallNeverPanics(t *testing.T) {
	box := NewToolBox(&Tool{
		Name: "boom", Params: []Param{{Name: "x", Type: "string", Required: true}},
		Handler: func(map[string]any) (string, map[string]any, error) { return "", nil, fmt.Errorf("it went wrong") },
	})
	result := box.Call("boom", map[string]any{"x": "1"})
	if result.OK || result.Error != "it went wrong" {
		t.Fatalf("a failing handler is an observation: %+v", result)
	}
	result = box.Call("missing", nil)
	if result.OK || !strings.Contains(result.Error, "unknown tool") {
		t.Fatalf("an unknown tool is an observation too: %+v", result)
	}
	if box.Calls != 2 {
		t.Fatalf("both calls counted: %d", box.Calls)
	}
}

func TestToolBoxSchemasAndCatalogue(t *testing.T) {
	box := offlineBox(t)
	if box.Names()[0] != "calculator" || box.Len() != 1 {
		t.Fatalf("offline: the calculator alone: %+v", box.Names())
	}
	schema := box.Schemas()[0]
	function := schema["function"].(map[string]any)
	if schema["type"] != "function" || function["name"] != "calculator" {
		t.Fatalf("Ollama's tools format: %+v", schema)
	}
	parameters := function["parameters"].(map[string]any)
	if len(parameters["required"].([]string)) != 1 {
		t.Fatalf("one required argument: %+v", parameters)
	}
	if !strings.HasPrefix(box.Catalogue(), "- calculator(expression: string) — ") {
		t.Fatalf("the prompt catalogue: %q", box.Catalogue())
	}
}

// -- the calculator ----------------------------------------------------------

func TestSafeEvalMatchesPython(t *testing.T) {
	for _, row := range []struct{ in, want string }{
		{"2 * (3 + 4)", "14"}, {"sqrt(841)", "29.0"}, {"2 ** 10", "1024"}, {"7 / 2", "3.5"},
		{"7 // 2", "3"}, {"-7 % 3", "2"}, {"round(2.5)", "2"}, {"round(3.14159, 2)", "3.14"},
		{"min(3, 1, 2)", "1"}, {"max([3, 1, 2])", "3"}, {"sum([1, 2, 3])", "6"}, {"2 ^ 8", "256"},
		{"1 < 2", "True"}, {"not 0", "True"}, {"1 and 2", "2"}, {"abs(-3)", "3"}, {"int(3.9)", "3"},
		{"float(3)", "3.0"}, {"log(100, 10)", "2.0"}, {"hypot(3, 4)", "5.0"}, {"1e3", "1000.0"},
		{"0.1 + 0.2", "0.30000000000000004"}, {"pi", "3.141592653589793"},
	} {
		out, err := SafeEval(row.in)
		if err != nil {
			t.Fatalf("%s: %v", row.in, err)
		}
		if out != row.want {
			t.Fatalf("%s = %s, want %s", row.in, out, row.want)
		}
	}
}

func TestSafeEvalRefusesEverythingElse(t *testing.T) {
	for _, expression := range []string{
		"", "open('x')", "__import__('os')", "1 +", "10 / 0", "[].__class__", "nosuchname",
		"nosuchfunction(1)", strings.Repeat("1+", 300) + "1",
	} {
		if out, err := SafeEval(expression); err == nil {
			t.Fatalf("%q was evaluated to %s", expression, out)
		}
	}
}

// -- browsing ----------------------------------------------------------------

func TestHTMLToText(t *testing.T) {
	page := HTMLToText(homePage)
	if page.Title != "Example Home" {
		t.Fatalf("the title comes out of <head>: %q", page.Title)
	}
	if strings.Contains(page.Text, "alert") || strings.Contains(page.Text, "color:red") {
		t.Fatalf("script and style are dropped: %q", page.Text)
	}
	if !strings.Contains(page.Text, "Welcome") || !strings.Contains(page.Text, "The cat sat on the mat.") {
		t.Fatalf("the body is kept: %q", page.Text)
	}
	if len(page.Links) != 3 || page.Links[0].Text != "All about cats" || page.Links[0].URL != "/cats" {
		t.Fatalf("the links are kept as written: %+v", page.Links)
	}
	if broken := HTMLToText("<p>unclosed <b>bold"); !strings.Contains(broken.Text, "unclosed bold") {
		t.Fatalf("broken markup still yields its text: %q", broken.Text)
	}
}

func TestWebClientFetchesAndRefuses(t *testing.T) {
	site := fakeSite(t)
	web := localWeb(t, site)
	page, err := web.GetPage(site.URL + "/cats")
	if err != nil {
		t.Fatal(err)
	}
	if page.Title != "Cats" || !strings.Contains(page.Text, "four legs") {
		t.Fatalf("the page: %+v", page)
	}
	// a redirect is followed, and every hop is checked again
	page, err = web.GetPage(site.URL + "/redirect")
	if err != nil || !strings.Contains(page.Text, "four legs") {
		t.Fatalf("redirect: %+v %v", page, err)
	}
	if _, err := web.GetPage(site.URL + "/loop"); err == nil {
		t.Fatal("a redirect loop is refused")
	}
	if _, err := web.GetPage(site.URL + "/away"); err == nil {
		t.Fatal("a redirect to a file:// URL is refused")
	}
	if _, err := web.GetPage(site.URL + "/missing"); err == nil {
		t.Fatal("a 404 is an error, not a page")
	}
	// the byte cap
	web.MaxBytes = 5000
	page, err = web.GetPage(site.URL + "/big")
	if err != nil || !page.Truncated {
		t.Fatalf("the cap: %+v %v", page, err)
	}
	// a plain-text page keeps its body
	page, err = web.GetPage(site.URL + "/plain")
	if err != nil || page.Text != "just text" {
		t.Fatalf("text/plain: %+v %v", page, err)
	}
}

func TestWebClientRefusesPrivateAddressesByDefault(t *testing.T) {
	site := fakeSite(t)
	web, err := NewWebClient(5*time.Second, 0, false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := web.Check(site.URL); err == nil {
		t.Fatal("127.0.0.1 is refused unless allow_private")
	}
	for _, bad := range []string{"", "ftp://example.com", "https://user:pw@example.com", "not a url at all"} {
		if _, err := web.Check(bad); err == nil {
			t.Fatalf("%q was allowed", bad)
		}
	}
}

func TestSearchReadsJSONAndHTML(t *testing.T) {
	site := fakeSite(t)
	web := localWeb(t, site)
	results, err := web.Search("cats", 5)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 || results[0].Title != "Cats" || results[0].Snippet != "cats have four legs" {
		t.Fatalf("the JSON answer: %+v", results)
	}
	web.SearchURL = site.URL + "/search-html?q={query}"
	results, err = web.Search("cats", 5)
	if err != nil {
		t.Fatal(err)
	}
	urls := []string{}
	for _, result := range results {
		urls = append(urls, result.URL)
	}
	if len(urls) != 2 || urls[0] != "https://example.net/hit" || urls[1] != "https://example.com/plain" {
		t.Fatalf("the engine's own links are skipped and the wrapper unwrapped: %+v", urls)
	}
}

func TestHTMLToTextPutsTheMainTextFirst(t *testing.T) {
	page := HTMLToText(articlePage)
	if page.Title != "Cat" {
		t.Fatalf("the title: %q", page.Title)
	}
	want := "Cat\nThe cat is a small mammal; see below.\nLegs: 4\nmeow()\npurr()\nJump to content\nDeutsch\nFrançais"
	if page.Text != want {
		t.Fatalf("the article first, the chrome gone, the menus last:\n%q\nwant\n%q", page.Text, want)
	}
	hrefs := []string{}
	for _, link := range page.Links {
		hrefs = append(hrefs, link.URL)
	}
	wantLinks := []string{"/wiki/Mammal", "#Legs", "/wiki/Mammal", "#content", "https://de.example.org/", "https://fr.example.org/"}
	if strings.Join(hrefs, " ") != strings.Join(wantLinks, " ") {
		t.Fatalf("the links in the running text first: %v", hrefs)
	}
	written := readHTML(articlePage, false) // as the page has them, for a search engine's ranking
	if lines := strings.Split(written.Text, "\n"); lines[0] != "Jump to content" || lines[1] != "Cat" || lines[2] != "Deutsch" {
		t.Fatalf("page order: %q", written.Text)
	}
	if written.Links[0].URL != "#content" {
		t.Fatalf("page order: %+v", written.Links)
	}
	// the <meta> that once hid every page, and entities in a title
	meta := HTMLToText(`<html><head><meta charset="utf-8"><link rel=x href=y><title>T &amp; U</title></head>` +
		`<body><p>Hello &amp; bye&nbsp;now</p><a href="/q?a=1&amp;b=2">Link &lt;1&gt;</a></body></html>`)
	if meta.Title != "T & U" || meta.Text != "Hello & bye now\nLink <1>" || meta.Links[0].URL != "/q?a=1&b=2" {
		t.Fatalf("a meta tag swallowed the page: %+v", meta)
	}
}

func TestHTMLToTextDropsWhatIsNeverShown(t *testing.T) {
	cases := map[string]string{
		"<p>a</p><div hidden><div>nested</div>still hidden</div><p>b</p>" +
			"<span style='DISPLAY: none'>x</span><span style='visibility:hidden'>y</span><span aria-hidden=true>z</span>" +
			"<div role='navigation'>menu</div><div role='Search box'>find</div><dialog>cookies?</dialog>" +
			"<p>c<button>Click</button><select><option>One</option></select><textarea>typed</textarea></p>": "a\nb\nc",
		// a void element cannot hide what follows it, nor can one whose end tag may be left out
		`<p>before</p><input name="m" hidden><img hidden src=x><p>after</p>`: "before\nafter",
		"<ul><li hidden>one<li>two</ul><p>three":                             "one\ntwo\nthree",
		`<div hidden="until-found">findable</div>`:                           "findable",
		// the banner goes, the header of an article or a section stays
		"<header>Site name</header><article><header><h1>Story</h1></header><p>Body.</p></article>": "Story\nBody.",
		"<section><header>Part one</header><p>x</p></section>":                                     "Part one\nx",
		// one link on its own line is text; a run of them is a menu
		"<p>First.</p><p><a href='/a'>A link</a></p><p>Last.</p>":                                       "First.\nA link\nLast.",
		"<ul><li><a href='/a'>One</a><li><a href='/b'>Two</a></ul><p>Body.</p>":                         "Body.\nOne\nTwo",
		"<div>Promo</div><div role='main'><div>Body<div hidden>x</div></div>more</div><div>After</div>": "Body\nmore\nPromo\nAfter",
		"<table><tr><td>Kingdom:</td><td><a href=/a>Animalia</a></td></tr></table>":                     "Kingdom: Animalia",
		// the tokenizer: a stray '<' is text, a '>' inside quotes does not end a tag, a title may hold '<'
		"<p>1 < 2</p><a title=\"a>b\" href=/x>link</a>":                               "1 < 2\nlink",
		"<script>if (a < b) { x = \"</p>\" }</script><p>after</p><SCRIPT>y</Script>z": "after\nz",
	}
	for markup, want := range cases {
		if got := HTMLToText(markup).Text; got != want {
			t.Errorf("%q:\n got %q\nwant %q", markup, got, want)
		}
	}
	// an unclosed head ends where the body begins; an icon's title is not the page's
	page := HTMLToText("<html><head><title>T</title><body><a href=/><svg><title>Icon</title></svg>Home</a>")
	if page.Title != "T" || page.Text != "Home" {
		t.Fatalf("head and titles: %+v", page)
	}
	if page := HTMLToText("<title>Tom &amp; Jerry <3</title><p>x</p>"); page.Title != "Tom & Jerry <3" || page.Text != "x" {
		t.Fatalf("a '<' in a title: %+v", page)
	}
}

func TestHTMLToTextTakesRawTextWholeAndDropsACutOffTag(t *testing.T) {
	// a title is text, its references read; a script ends only at its own end tag
	if page := HTMLToText("<title>Use <b> tags &amp; more</title><p>x</p>"); page.Title != "Use <b> tags & more" || page.Text != "x" {
		t.Fatalf("a title: %+v", page)
	}
	for markup, want := range map[string]string{
		`<script>if (a) "</scripts>"</script><p>after</p>`: "after",
		"<xmp><b>raw</b> &amp;</xmp>":                      "<b>raw</b> &amp;",
		"<textarea><p>typed</p></textarea><p>b</p>":        "b",
		"cat</": "cat</",
	} {
		if got := HTMLToText(markup).Text; got != want {
			t.Errorf("%q: got %q, want %q", markup, got, want)
		}
	}
	// a page cut off at the byte cap in the middle of a tag
	if page := HTMLToText(`<p>cat</p><a href="/x`); page.Text != "cat" || len(page.Links) != 0 {
		t.Fatalf("a cut-off tag: %+v", page)
	}
}

func TestGetPageLinksEachAddressOnceAndNeverBackToItself(t *testing.T) {
	site := fakeSite(t)
	page, err := localWeb(t, site).GetPage(site.URL + "/article")
	if err != nil {
		t.Fatal(err)
	}
	hrefs := []string{}
	for _, link := range page.Links {
		hrefs = append(hrefs, link.URL)
	}
	want := []string{site.URL + "/wiki/Mammal", "https://de.example.org/", "https://fr.example.org/"}
	if strings.Join(hrefs, " ") != strings.Join(want, " ") {
		t.Fatalf("absolute, each once, none back into the page: %v", hrefs)
	}
	for path, want := range map[string]string{"/latin1": "café \u0080", "/cp1252": "“quoted” €"} {
		page, err := localWeb(t, site).GetPage(site.URL + path)
		if err != nil || page.Text != want {
			t.Fatalf("%s is read in its charset: %+v %v", path, page, err)
		}
	}
}

// engines is a client that searches the fake site's endpoints, in the order given.
func engines(t *testing.T, site *httptest.Server, paths ...string) *WebClient {
	t.Helper()
	chain := []string{}
	for _, path := range paths {
		chain = append(chain, site.URL+path+"?q={query}")
	}
	web, err := NewWebClient(10*time.Second, 0, true, strings.Join(chain, " "), "")
	if err != nil {
		t.Fatal(err)
	}
	return web
}

func TestSearchFallsBackFromAnEngineThatRefuses(t *testing.T) {
	site := fakeSite(t)
	hits, err := engines(t, site, "/ddg").Search("cats", 5)
	if err != nil || len(hits) != 1 || hits[0] != (SearchResult{"Cat - Encyclopedia", "https://en.example.org/wiki/Cat", "The cat is a small mammal."}) {
		t.Fatalf("DuckDuckGo's page, the ad and the feedback link skipped, the snippet kept: %+v %v", hits, err)
	}
	hits, err = engines(t, site, "/ddg-blocked", "/wiki-api").Search("cat legs", 5)
	if err != nil || len(hits) != 2 || hits[0].Title != "Cat" || hits[0].Snippet != "The cat is a mammal." || hits[1].Title != "Cat anatomy" {
		t.Fatalf("DuckDuckGo refused, so Wikipedia's API answered, ranked by index: %+v %v", hits, err)
	}
	if _, err := engines(t, site, "/ddg-blocked").Search("cats", 5); err == nil ||
		err.Error() != "127.0.0.1 answered HTTP 202 instead of results - it may be turning automated searches away" {
		t.Fatalf("one engine that refuses: %v", err)
	}
	if _, err := engines(t, site, "/captcha").Search("cats", 5); err == nil || !strings.Contains(err.Error(), "answered with a CAPTCHA instead of results") {
		t.Fatalf("a CAPTCHA: %v", err)
	}
	_, err = engines(t, site, "/ddg-blocked", "/captcha", "/nope").Search("cats", 5)
	if err == nil || !strings.HasPrefix(err.Error(), "no search engine answered: ") {
		t.Fatalf("every engine refusing: %v", err)
	}
	for _, part := range []string{"HTTP 202", "CAPTCHA", "HTTP 404"} {
		if !strings.Contains(err.Error(), part) {
			t.Fatalf("each engine's answer is named: %v", err)
		}
	}
	if hits, err := engines(t, site, "/ddg-blocked", "/no-results").Search("cats", 5); err != nil || len(hits) != 0 {
		t.Fatalf("an engine that finds nothing is no results: %+v %v", hits, err)
	}
	hits, err = engines(t, site, "/opensearch").Search("cat", 5)
	if err != nil || len(hits) != 2 || hits[0].Title != "Cat" || hits[1].Snippet != "A region" {
		t.Fatalf("OpenSearch: %+v %v", hits, err)
	}
	if hits, err := engines(t, site, "/search-203").Search("cats", 5); err != nil || len(hits) != 2 {
		t.Fatalf("only a 202 is a refusal: %+v %v", hits, err)
	}
	hits, err = engines(t, site, "/json-page").Search("cats", 5)
	if err != nil || len(hits) != 1 || hits[0].URL != "http://example.org/cats" {
		t.Fatalf("a JSON answer drawn as a page, as a browser has it: %+v %v", hits, err)
	}
}

func TestDefaultSearchEngines(t *testing.T) {
	if strings.TrimSpace(os.Getenv("RADIXNET_SEARCH_URL")) != "" {
		t.Skip("$RADIXNET_SEARCH_URL names its own")
	}
	web, err := NewWebClient(0, 0, false, "", "")
	if err != nil {
		t.Fatal(err)
	}
	if strings.Join(strings.Fields(web.SearchURL), "|") != strings.Join(DefaultSearchEngines, "|") {
		t.Fatalf("the default engines: %q", web.SearchURL)
	}
	web, _ = NewWebClient(0, 0, false, "  http://a/?q={query}\n http://b/  ", "")
	if web.SearchURL != "http://a/?q={query} http://b/" {
		t.Fatalf("the endpoints are split on spaces: %q", web.SearchURL)
	}
}

func TestCheckSaysWhyAnAddressIsRefused(t *testing.T) {
	web, _ := NewWebClient(5*time.Second, 0, false, "", "")
	for raw, want := range map[string]string{
		"http://127.0.0.1/":       "refusing 127.0.0.1: it resolves to 127.0.0.1, a loopback address - pass allow_private (--allow-private) to fetch it anyway",
		"http://169.254.169.254/": "a link-local address",
		"http://0.0.0.0/":         "an unspecified address",
		"http://10.1.2.3/":        "a private address",
		"http://[::1]/":           "a loopback address",
		"http://[fd00::1]/":       "a private address",
	} {
		if _, err := web.Check(raw); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v (want %q)", raw, err, want)
		}
	}
	// a name this process cannot resolve is refused - unless a proxy carries the request and resolves it
	savedLookup, savedProxy := lookupIP, environmentProxy
	t.Cleanup(func() { lookupIP, environmentProxy = savedLookup, savedProxy })
	lookupIP = func(host string) ([]net.IP, error) {
		return nil, &net.DNSError{Err: "no such host", Name: host, IsNotFound: true}
	}
	environmentProxy = func(*http.Request) (*url.URL, error) { return nil, nil }
	if _, err := web.Check("https://example.com/"); err == nil || err.Error() != "refusing example.com: it cannot be resolved here (no such host)" {
		t.Fatalf("no DNS, no proxy: %v", err)
	}
	environmentProxy = func(*http.Request) (*url.URL, error) { return url.Parse("http://127.0.0.1:3128") }
	if got, err := web.Check("https://example.com/"); err != nil || got != "https://example.com/" {
		t.Fatalf("no DNS, but a proxy that resolves it: %q %v", got, err)
	}
	lookupIP = func(string) ([]net.IP, error) { return []net.IP{net.ParseIP("10.0.0.7")}, nil }
	if _, err := web.Check("https://example.com/"); err == nil || !strings.Contains(err.Error(), "resolves to 10.0.0.7, a private address") {
		t.Fatalf("a name that resolves somewhere internal is refused, proxy or not: %v", err)
	}
}

// -- the built-in tools ------------------------------------------------------

func TestDefaultToolBoxTools(t *testing.T) {
	site := fakeSite(t)
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("the cat sat on the mat"), 0o644); err != nil {
		t.Fatal(err)
	}
	box, err := DefaultToolBox(ToolOptions{AllowPrivate: true, SearchURL: site.URL + "/search?q={query}", UploadDir: dir})
	if err != nil {
		t.Fatal(err)
	}
	if names := strings.Join(box.Names(), ","); names != "web_search,web_fetch,web_links,calculator,read_file" {
		t.Fatalf("the built-in set: %s", names)
	}
	result := box.Call("calculator", map[string]any{"expression": "6*7"})
	if !result.OK || result.Output != "42" {
		t.Fatalf("calculator: %+v", result)
	}
	result = box.Call("web_fetch", map[string]any{"url": site.URL + "/cats", "max_chars": 100})
	if !result.OK || !strings.Contains(result.Output, "Cats — ") {
		t.Fatalf("web_fetch: %+v", result)
	}
	result = box.Call("web_search", map[string]any{"query": "cats"})
	if !result.OK || !strings.HasPrefix(result.Output, "1. Cats — http://example.org/cats") {
		t.Fatalf("web_search: %+v", result)
	}
	result = box.Call("web_links", map[string]any{"url": site.URL + "/"})
	if !result.OK || !strings.Contains(result.Output, "All about cats") {
		t.Fatalf("web_links: %+v", result)
	}
	result = box.Call("read_file", map[string]any{"name": "notes.txt"})
	if !result.OK || result.Output != "the cat sat on the mat" {
		t.Fatalf("read_file: %+v", result)
	}
	result = box.Call("read_file", map[string]any{"name": "../etc/passwd"})
	if result.OK || !strings.Contains(result.Error, "no uploaded file named") {
		t.Fatalf("read_file cannot leave the upload directory: %+v", result)
	}
}

func TestPythonToolRunsInTheSandbox(t *testing.T) {
	sandbox, err := NewSandbox("", 20*time.Second, 256, 0, true)
	if err != nil {
		t.Fatal(err)
	}
	box, err := DefaultToolBox(ToolOptions{Offline: true, Sandbox: sandbox})
	if err != nil {
		t.Fatal(err)
	}
	result := box.Call("python", map[string]any{"code": "print(6 * 7)"})
	if !result.OK || result.Output != "42" {
		t.Fatalf("python: %+v", result)
	}
	result = box.Call("python", map[string]any{"code": "raise ValueError('boom')"})
	if result.OK || !strings.Contains(result.Error, "ValueError") {
		t.Fatalf("a failing program is an observation: %+v", result)
	}
}
