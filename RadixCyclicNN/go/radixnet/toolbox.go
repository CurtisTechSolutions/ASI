package radixnet

// The built-in tools: browsing, the calculator, the sandbox and the uploads.
//
// The Go twin of the tool set Python's radixnet.tools.default_toolbox builds,
// with the same names, the same parameters and the same descriptions - a
// transcript written on one side is a transcript the other can read.

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"
)

// intArg reads an integer argument a handler was given (they arrive coerced).
func intArg(arguments map[string]any, name string, fallback int) int {
	if value, ok := arguments[name]; ok {
		if number, err := toNumber(value); err == nil {
			return int(number)
		}
	}
	return fallback
}

// textArg reads a string argument a handler was given.
func textArg(arguments map[string]any, name string) string {
	if value, ok := arguments[name].(string); ok {
		return value
	}
	return ""
}

func clampInt(value, low, high int) int {
	if value < low {
		return low
	}
	if value > high {
		return high
	}
	return value
}

// WebSearchTool searches the web and reports a numbered list of hits.
func WebSearchTool(web *WebClient) *Tool {
	return &Tool{
		Name:        "web_search",
		Description: "Search the web and get back a numbered list of titles with their URLs.",
		Params: []Param{
			{Name: "query", Type: "string", Description: "what to search for", Required: true},
			{Name: "limit", Type: "integer", Description: "how many results (1-10)", Default: 5},
		},
		Network: true,
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			query := textArg(arguments, "query")
			limit := clampInt(intArg(arguments, "limit", 5), 1, 10)
			results, err := web.Search(query, limit)
			if err != nil {
				return "", nil, err
			}
			if len(results) == 0 {
				return fmt.Sprintf("no results for %s", pythonRepr(query)), map[string]any{"results": []any{}}, nil
			}
			lines := make([]string, 0, len(results))
			rows := make([]map[string]any, 0, len(results))
			for i, result := range results {
				line := fmt.Sprintf("%d. %s — %s", i+1, result.Title, result.URL)
				if result.Snippet != "" {
					line += " — " + result.Snippet
				}
				lines = append(lines, line)
				rows = append(rows, map[string]any{"title": result.Title, "url": result.URL, "snippet": result.Snippet})
			}
			return strings.Join(lines, "\n"), map[string]any{"results": rows}, nil
		},
	}
}

// WebFetchTool opens a page and reads it as plain text.
func WebFetchTool(web *WebClient) *Tool {
	return &Tool{
		Name:        "web_fetch",
		Description: "Open a web page and read it as plain text (the title, then the body).",
		Params: []Param{
			{Name: "url", Type: "string", Description: "the address of the page", Required: true},
			{Name: "max_chars", Type: "integer", Description: "how much of the page to read", Default: 2000},
		},
		Network: true,
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			page, err := web.GetPage(textArg(arguments, "url"))
			if err != nil {
				return "", nil, err
			}
			limit := clampInt(intArg(arguments, "max_chars", 2000), 100, 20000)
			text := page.Text
			body := clipRunes(text, limit)
			if runeLen(text) > limit {
				body += " ..."
			}
			header := page.URL + "\n"
			if page.Title != "" {
				header = page.Title + " — " + page.URL + "\n"
			}
			links := make([]map[string]any, 0, 40)
			for i, link := range page.Links { // the pages this one leads to: what an exploration follows next
				if i >= 40 {
					break
				}
				links = append(links, map[string]any{"text": link.Text, "url": link.URL})
			}
			return header + body, map[string]any{
				"url": page.URL, "title": page.Title, "status": page.Status, "chars": runeLen(text),
				"truncated": page.Truncated || runeLen(text) > limit, "link_count": len(page.Links), "links": links,
			}, nil
		},
	}
}

// WebLinksTool lists the links on a page.
func WebLinksTool(web *WebClient) *Tool {
	return &Tool{
		Name:        "web_links",
		Description: "List the links on a web page, so another page can be opened from it.",
		Params: []Param{
			{Name: "url", Type: "string", Description: "the address of the page", Required: true},
			{Name: "limit", Type: "integer", Description: "how many links (1-100)", Default: 20},
		},
		Network: true,
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			page, err := web.GetPage(textArg(arguments, "url"))
			if err != nil {
				return "", nil, err
			}
			limit := clampInt(intArg(arguments, "limit", 20), 1, 100)
			links := page.Links
			if len(links) > limit {
				links = links[:limit]
			}
			if len(links) == 0 {
				return "no links on " + page.URL, map[string]any{"links": []any{}}, nil
			}
			lines := make([]string, 0, len(links))
			rows := make([]map[string]any, 0, len(links))
			for i, link := range links {
				lines = append(lines, fmt.Sprintf("%d. %s — %s", i+1, link.Text, link.URL))
				rows = append(rows, map[string]any{"text": link.Text, "url": link.URL})
			}
			return strings.Join(lines, "\n"), map[string]any{"links": rows, "url": page.URL}, nil
		},
	}
}

// CalculatorTool works out an arithmetic expression.
func CalculatorTool() *Tool {
	return &Tool{
		Name:        "calculator",
		Description: "Work out an arithmetic expression, for example 2 * (3 + 4) or sqrt(841).",
		Params:      []Param{{Name: "expression", Type: "string", Description: "the expression to evaluate", Required: true}},
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			out, err := SafeEval(textArg(arguments, "expression"))
			if err != nil {
				return "", nil, err
			}
			return out, nil, nil
		},
	}
}

// PythonTool runs a short program in the codegen sandbox.
func PythonTool(sandbox *Sandbox) *Tool {
	return &Tool{
		Name:        "python",
		Description: "Run a short Python program in a sandbox and get back what it printed.",
		Params:      []Param{{Name: "code", Type: "string", Description: "the program to run", Required: true}},
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			run, err := sandbox.Run(textArg(arguments, "code"), nil, nil, "")
			if err != nil {
				return "", nil, &ToolError{err.Error()}
			}
			if run.OK {
				out := strings.TrimSpace(run.Stdout)
				if out == "" {
					out = "(the program printed nothing)"
				}
				meta := map[string]any{"seconds": run.Seconds, "exit_code": run.ExitCode}
				return out, meta, nil
			}
			tail := run.Stderr
			if runeLen(tail) > 400 {
				tail = lastN(tail, 400)
			}
			return "", nil, &ToolError{strings.TrimSpace(fmt.Sprintf("the program failed: %s\n%s", run.Error, tail))}
		},
	}
}

// ReadFileTool reads one of the files uploaded to the server.
func ReadFileTool(uploadDir string) *Tool {
	return &Tool{
		Name:        "read_file",
		Description: "Read one of the files uploaded to the server.",
		Params: []Param{
			{Name: "name", Type: "string", Description: "the file name", Required: true},
			{Name: "max_chars", Type: "integer", Description: "how much to read", Default: 2000},
		},
		Handler: func(arguments map[string]any) (string, map[string]any, error) {
			raw := strings.TrimSpace(textArg(arguments, "name"))
			safe := filepath.Base(strings.ReplaceAll(raw, "\\", "/"))
			if safe == "" || safe == "." || safe == ".." || safe == "/" {
				return "", nil, toolErrorf("%s is not a file name", pythonRepr(raw))
			}
			path := filepath.Join(uploadDir, safe)
			info, err := os.Stat(path)
			if err != nil || info.IsDir() {
				have := []string{}
				if entries, err := os.ReadDir(uploadDir); err == nil {
					for _, entry := range entries {
						if !entry.IsDir() {
							have = append(have, entry.Name())
						}
					}
				}
				sort.Strings(have)
				have = firstN(have, 20)
				listed := strings.Join(have, ", ")
				if listed == "" {
					listed = "none"
				}
				return "", nil, toolErrorf("no uploaded file named %s (have: %s)", pythonRepr(safe), listed)
			}
			limit := clampInt(intArg(arguments, "max_chars", 2000), 100, 100000)
			handle, err := os.Open(path)
			if err != nil {
				return "", nil, &ToolError{err.Error()}
			}
			defer handle.Close()
			blob := make([]byte, limit+1)
			read, _ := handle.Read(blob)
			text := strings.TrimPrefix(string(blob[:read]), "\ufeff")
			out := clipRunes(text, limit)
			if runeLen(text) > limit {
				out += " ..."
			}
			return out, map[string]any{"name": safe, "chars": runeLen(text)}, nil
		},
	}
}

// ToolOptions are what a caller may change about the built-in tools.
type ToolOptions struct {
	Offline      bool     // no web tools at all
	AllowPrivate bool     // let the web tools reach private addresses (the tests do)
	SearchURL    string   // a different search endpoint
	Timeout      float64  // seconds per web request
	MaxBytes     int      // cap on a fetched page
	Sandbox      *Sandbox // when set, the `python` tool is registered
	UploadDir    string   // when set, the `read_file` tool is registered
	Extra        []*Tool  // anything else the caller wants offered
}

// DefaultToolOptions mirror the Python defaults.
func DefaultToolOptions() ToolOptions {
	return ToolOptions{Timeout: 20, MaxBytes: 2_000_000}
}

// DefaultToolBox is the built-in set: browsing (unless Offline), the
// calculator, and - when given - `python` and `read_file`.
func DefaultToolBox(o ToolOptions) (*ToolBox, error) {
	box := NewToolBox()
	if !o.Offline {
		timeout := o.Timeout
		if timeout <= 0 {
			timeout = 20
		}
		web, err := NewWebClient(time.Duration(timeout*float64(time.Second)), o.MaxBytes, o.AllowPrivate, o.SearchURL, "")
		if err != nil {
			return nil, err
		}
		box.Register(WebSearchTool(web))
		box.Register(WebFetchTool(web))
		box.Register(WebLinksTool(web))
	}
	box.Register(CalculatorTool())
	if o.Sandbox != nil {
		box.Register(PythonTool(o.Sandbox))
	}
	if strings.TrimSpace(o.UploadDir) != "" {
		box.Register(ReadFileTool(o.UploadDir))
	}
	for _, tool := range o.Extra {
		if err := box.Register(tool); err != nil {
			return nil, err
		}
	}
	return box, nil
}
