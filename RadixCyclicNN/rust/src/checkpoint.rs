//! Checkpoints in the Python `CheckpointManager` layout (`radixnet/checkpoint.py`, `go/server/checkpoints.go`).
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

/// `radixnet checkpoint`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("checkpoints"))
}

/// This area's routes:
/// `GET /api/checkpoints`
/// `POST /api/checkpoints/save`
/// `POST /api/checkpoints/restore`
pub fn routes(_server: &mut Server<Service>) {}
