//! Provider-independent plumbing for the LLMs the teaching loops talk to (`radixnet/llm.py`, `go/radixnet/llm.go`).
//!
//! Not ported yet: a stub the port fills in.

/// What the server keeps for this area between requests.
#[derive(Default)]
pub struct State {}

impl State {
    /// Reads the `serve` command's flags for this area.
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }
}
