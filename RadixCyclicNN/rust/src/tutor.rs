//! Automated English lessons (`radixnet/tutor.py`, `go/radixnet/tutor.go`).
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

/// `radixnet tutor`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("tutor"))
}

/// This area's routes:
/// `GET /api/tutor`
/// `POST /api/tutor/start`
/// `POST /api/tutor/lesson`
/// `POST /api/tutor/plan`
/// `GET /api/tutor/history`
pub fn routes(_server: &mut Server<Service>) {}
