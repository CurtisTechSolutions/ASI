//! The negative network feeding itself: an LLM reviewer on a loop (`radixnet/critic.py`, `go/radixnet/critic.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::http::Server;
use crate::service::Service;

/// What the server keeps for this area between requests.
#[derive(Default)]
pub struct State {}

impl State {
    /// Reads the `serve` command's flags for this area.
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }
}

/// This area's routes:
/// `POST /api/negative/auto`
/// `GET /api/negative/auto/history`
pub fn routes(_server: &mut Server<Service>) {}
