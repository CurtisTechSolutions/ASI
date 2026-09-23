//! Solving tasks with tools (`radixnet/agent.py`, `go/radixnet/agent.go`).
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

/// `radixnet agent`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("agent"))
}

/// `radixnet explore`.
pub fn explore_cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("explore"))
}

/// This area's routes:
/// `POST /api/agent/start`
/// `POST /api/agent/solve`
/// `POST /api/agent/explore`
/// `POST /api/agent/criteria`
/// `GET /api/agent/history`
pub fn routes(_server: &mut Server<Service>) {}
