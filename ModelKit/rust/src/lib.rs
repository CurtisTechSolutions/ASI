//! `modelkit` - everything around the RadixCyclicNN model, built on the
//! [`radixnet`] crate: every teaching loop (the tutor, the chat, the critic,
//! evolve, the recall tutor), the LLM clients ([`llm`], [`ollama`],
//! [`chatgpt`]), the agent and its tools, code generation, images and speech,
//! the negative network's guard on every answer ([`duo`]), MCP, the CLI
//! ([`cli`], `src/bin/radixnet.rs`) and the HTTP API the frontend talks to
//! ([`http`], [`service`]).
//!
//! The model itself - every kind, the JSON model files byte for byte what
//! Python writes, the search and the training - is `radixnet`
//! (`../../RadixCyclicNN/rust`).  This crate depends on it, and it never on
//! this one.  The binaries keep their names: `radixnet`, the CLI, and
//! `radixnet-bench`, the benchmark.
//!
//! Like `radixnet` it has no third-party dependency: HTTP/1.1 (server and
//! client), multipart, PNG / JPEG / GIF and base64 are written out here.
//! `../../RadixCyclicNN/rust/README.md` lists every module of both crates, and
//! what is deliberately not ported (torch, the Stable Diffusion encoder, local
//! Whisper) and why.

pub mod agent;
pub mod assistant;
pub mod bench;
pub mod blame;
pub mod calc;
pub mod chat;
pub mod chatgpt;
pub mod checkpoint;
pub mod cli;
pub mod codegen;
pub mod critic;
pub mod dialogue;
pub mod duo;
pub mod fetch;
pub mod gan;
pub mod http;
pub mod llm;
pub mod mcp;
pub mod multipart;
pub mod ollama;
pub mod plan;
pub mod recall;
pub mod review;
pub mod service;
pub mod speech;
pub mod thinking;
pub mod toolbox;
pub mod tools;
pub mod tutor;
pub mod vision;
pub mod voice;
pub mod voicechat;
pub mod web;
// Chrome over WebDriver behind the web tools (`--browser`)
pub mod browser;
// the teaching loops of a model of sounds show the LLM words
#[cfg(test)]
mod teaching_in_words;
