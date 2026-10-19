//! Base64: everything as one 64-letter alphabet, so one N-gram model can learn text, images, audio and video alike.
//!
//! The same as `latticefsm/b64.py`, letter for letter: the standard alphabet (RFC 4648, `=` padding), a lenient
//! decoding, the prefix whose letters are all known (what a prompt must be for the model to continue it), and every
//! file under some paths cut into sequences of `width` letters at byte boundaries that never split a group of three.

use std::fs;
use std::path::Path;

pub const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
/// Bytes per sequence when a file is cut up: 76 letters.
pub const CHUNK_BYTES: usize = 57;

fn letter(n: u32) -> char {
    ALPHABET[(n & 63) as usize] as char
}

/// Standard base64, padded.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    let full = data.len() - data.len() % 3;
    for c in data[..full].chunks(3) {
        let n = (c[0] as u32) << 16 | (c[1] as u32) << 8 | c[2] as u32;
        out.extend([letter(n >> 18), letter(n >> 12), letter(n >> 6), letter(n)]);
    }
    match data.len() % 3 {
        1 => {
            let n = (data[full] as u32) << 16;
            out.extend([letter(n >> 18), letter(n >> 12), '=', '=']);
        }
        2 => {
            let n = (data[full] as u32) << 16 | (data[full + 1] as u32) << 8;
            out.extend([letter(n >> 18), letter(n >> 12), letter(n >> 6), '=']);
        }
        _ => {}
    }
    out
}

/// The letters whose six bits are all known (`8 * len / 6` of them): a prefix of the encoding of every continuation.
pub fn encode_prefix(data: &[u8]) -> String {
    encode(data).chars().take(8 * data.len() / 6).collect()
}

fn value(c: char) -> Option<u32> {
    ALPHABET.iter().position(|&a| a as char == c).map(|i| i as u32)
}

/// Base64 back to bytes, leniently: letters outside the alphabet (and `=`) ignored, a last group of two or three
/// letters read as one or two bytes, a last lone letter dropped.
pub fn decode(text: &str) -> Vec<u8> {
    let values: Vec<u32> = text.chars().filter_map(value).collect();
    let mut out = Vec::with_capacity(values.len() / 4 * 3 + 2);
    let full = values.len() - values.len() % 4;
    for v in values[..full].chunks(4) {
        let n = v[0] << 18 | v[1] << 12 | v[2] << 6 | v[3];
        out.extend([(n >> 16) as u8, (n >> 8 & 255) as u8, (n & 255) as u8]);
    }
    let rest = &values[full..];
    if rest.len() >= 2 {
        let n = rest[0] << 18 | rest[1] << 12 | if rest.len() == 3 { rest[2] << 6 } else { 0 };
        out.push((n >> 16) as u8);
        if rest.len() == 3 {
            out.push((n >> 8 & 255) as u8);
        }
    }
    out
}

fn walk(dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    let (mut files, mut dirs) = (Vec::new(), Vec::new());
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            dirs.push((name, p));
        } else if name != "README.md" {
            files.push((name, p));
        }
    }
    files.sort();
    dirs.sort();
    out.extend(files.into_iter().map(|(_, p)| p.to_string_lossy().into_owned()));
    for (_, d) in dirs {
        walk(&d, out)?;
    }
    Ok(())
}

/// Every file under `paths` - any kind - directories walked in name order (hidden files and `README.md` left out).
pub fn files_under(paths: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, &mut out)?;
        } else if path.exists() {
            out.push(p.clone());
        } else {
            return Err(format!("{p}: no such file or directory"));
        }
    }
    Ok(out)
}

/// Every file under `paths` as (label, sequences): its bytes base64-encoded, `width` letters a sequence; an
/// archive's members each on their own (`corpus::byte_documents`).
pub fn file_sequences(paths: &[String], width: usize) -> Result<Vec<(String, Vec<String>)>, String> {
    let step = (width / 4).max(1) * 3;
    let mut out = Vec::new();
    crate::corpus::byte_documents(paths, &mut |_, label, r| {
        let mut seqs = Vec::new();
        crate::corpus::read_chunks(r, step, &mut |c| {
            seqs.push(encode(c));
            Ok(())
        })?;
        out.push((label.to_string(), seqs));
        Ok(())
    })
    .map(|_| out)
}
