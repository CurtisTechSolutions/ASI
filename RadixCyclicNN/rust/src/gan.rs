//! The self-upgrade loop: the model is the generator, a second network the discriminator (`radixnet/gan.py`, `go/radixnet/gan.go`).
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

/// `radixnet gan`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("evolve"))
}

/// This area's routes:
/// `POST /api/evolve/start`
/// `POST /api/evolve/stop`
/// `GET /api/evolve/history`
pub fn routes(_server: &mut Server<Service>) {}
