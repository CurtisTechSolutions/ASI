//! External tools the network can call, and the text format it learns them in (`radixnet/tools.py`, `go/radixnet/tools.go`).
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

/// `radixnet tools`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("tools"))
}

/// This area's routes:
/// `GET /api/tools`
/// `POST /api/tools/call`
pub fn routes(_server: &mut Server<Service>) {}
