//! Teach one correction: only what changed moves (`go/radixnet/correct.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};

/// `radixnet correct`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("correct"))
}
