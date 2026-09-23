//! An LLM converses with the model and marks every reply (`radixnet/chat.py`, `go/radixnet/chat.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
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

/// `radixnet chat`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("chat"))
}

/// This area's routes:
/// `POST /api/chat/start`
/// `GET /api/chat/history`
pub fn routes(_server: &mut Server<Service>) {}
