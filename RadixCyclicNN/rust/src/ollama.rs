//! The Ollama client, the prompt-driven corpus and the adversarial review (`radixnet/ollama.py`, `go/radixnet/ollama.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
use crate::http::Server;
use crate::service::Service;

/// `radixnet ollama`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("ollama"))
}

/// This area's routes:
/// `GET /api/ollama/models`
/// `POST /api/ollama/corpus`
/// `POST /api/ollama/review`
pub fn routes(_server: &mut Server<Service>) {}
