//! The ChatGPT (OpenAI) client (`radixnet/chatgpt.py`, `go/radixnet/chatgpt.go`).
//!
//! Not ported yet: a stub the port fills in.

use crate::cli::{not_ported, Ctx};
use crate::http::Server;
use crate::service::Service;

/// `radixnet chatgpt`.
pub fn cli(_ctx: &Ctx) -> Result<(), String> {
    Err(not_ported("chatgpt"))
}

/// This area's routes:
/// `GET /api/chatgpt/models`
pub fn routes(_server: &mut Server<Service>) {}
