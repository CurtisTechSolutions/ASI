//! The model converses with itself (`radixnet/dialogue.py`, `go/radixnet/dialogue.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
use crate::http::Server;
use crate::service::Service;

/// `radixnet dialogue`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("converse"))
}

/// This area's routes:
/// `POST /api/converse`
pub fn routes(_server: &mut Server<Service>) {}
