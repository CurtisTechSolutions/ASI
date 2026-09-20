package main

// `radixnet-count mcp`: speak the Model Context Protocol on stdin / stdout, so
// any MCP client - Claude Desktop, an editor, another agent - can use this
// instance: the external tools (browsing, the calculator, optionally the
// sandbox and the uploaded files) and the network itself (predict, generate,
// score, stats, solve a task through the agent loop, and judge a text against
// the negative network).
//
// Nothing is printed on stdout but the protocol: everything a run would
// normally say goes to stderr, because one stray line would corrupt the
// stream.  The Go twin of the Python CLI's `mcp` command.

import (
	"os"
	"strings"
	"time"

	"github.com/CurtisTechSolutions/ASI/RadixCyclicNN/go/radixnet"
)

func cmdMCP(args []string) {
	cfg := radixnet.DefaultAgentConfig()
	fs := subFlagSet("mcp")
	noModel := fs.Bool("no-model", false, "offer the external tools only, without the network")
	noSolve := fs.Bool("no-solve", false, "do not offer radixnet_solve (which needs an LLM for the criteria and the judging)")
	blame := fs.Bool("blame", false, "load the negative network even when its file does not exist yet (radixnet_judge)")
	addNegativeFlag(fs)
	fs.StringVar(&cfg.AgentModel, "agent-model", "", "the model radixnet_solve asks for criteria and judging")
	url := fs.String("url", "", "the provider's base URL (default: its own)")
	timeout := fs.Float64("timeout", 0, "seconds to wait for one LLM answer")
	addToolFlags(fs)
	addUploadFlag(fs)
	_ = fs.Parse(permute(fs, args))

	box := openToolBox()
	options := radixnet.MCPModelOptions{Config: cfg}
	if !*noModel {
		options.Model = openModel(false)
		// the negative network is offered when it is there, or when --blame asks
		// for an empty one: radixnet_judge on a network that has been taught
		// nothing answers "nothing like this has gone wrong before", which is an
		// answer rather than an error
		if *blame || fileExists(negativeFile()) {
			options.Negative = openNegative(false)
		}
		if !*noSolve {
			client, err := radixnet.NewLLMClient(cfg.Provider, *url, cfg.AgentModel,
				time.Duration(*timeout*float64(time.Second)))
			if err == nil {
				options.Box, options.Client = box, client
			} else {
				// no LLM: every other tool still works, so this is a note, not a failure
				note("mcp: radixnet_solve is not offered (%v)", err)
			}
		}
	}

	extra := radixnet.MCPModelTools(options)
	names := make([]string, 0, len(extra))
	for _, tool := range extra {
		names = append(names, tool.Name)
	}
	note("mcp: %s + %s", strings.Join(box.Names(), ", "), strings.Join(names, ", "))
	note("mcp: speaking MCP %s on stdin / stdout; nothing but the protocol goes to stdout",
		radixnet.MCPProtocolVersion)
	if err := radixnet.ServeMCPStdio(box, extra, os.Stdin, os.Stdout, os.Stderr); err != nil {
		fail("%v", err)
	}
}

// fileExists is true when path is there to be read.
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
