//! Programs written by the network, run in a sandbox and judged (`radixnet/codegen.py`, `go/radixnet/codegen.go`).
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

/// `radixnet codegen`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("codegen"))
}

/// This area's routes:
/// `POST /api/codegen/start`
/// `POST /api/codegen/solve`
/// `POST /api/codegen/run`
/// `GET /api/codegen/history`
pub fn routes(_server: &mut Server<Service>) {}
