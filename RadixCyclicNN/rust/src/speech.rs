//! Speech as text: the transcript and the waveform behind one unique token (`radixnet/speech.py`, `go/radixnet/speech.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
use crate::http::Server;
use crate::service::Service;

/// `radixnet speech`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("speech"))
}

/// This area's routes:
/// `GET /api/speech`
/// `POST /api/speech/teach`
/// `POST /api/speech/decode`
/// `POST /api/speech/tutor`
pub fn routes(_server: &mut Server<Service>) {}
