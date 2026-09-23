//! Images as text (`radixnet/vision.py`, `go/radixnet/vision.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
use crate::http::Server;
use crate::service::Service;

/// `radixnet vision`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("image"))
}

/// This area's routes:
/// `GET /api/images`
/// `POST /api/images/encode`
/// `POST /api/images/decode`
/// `POST /api/images/tutor`
pub fn routes(_server: &mut Server<Service>) {}
