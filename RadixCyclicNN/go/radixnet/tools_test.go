package radixnet

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
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
