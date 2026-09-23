//! Base64, written out: how bytes travel inside text here.
//!
//! Two things need it.  The media texts (`img:tiny:128x128:AAEC...`,
//! `aud:mu:8000x1:gOXv...`) carry their payload as base64, and that text is
//! what the network trains on and *predicts* - so the reader has to take back
//! a payload that a prediction cut off, padded with junk or broke with
//! whitespace ([`repair`], Python's `encoding.repair_base64`).  And the JSON
//! upload forms carry a file as `content_base64`, which is read strictly
//! ([`decode`], Python's `base64.b64decode(..., validate=True)`).
//!
//! The standard alphabet with `=` padding, the one Python's `base64` module and
//! Go's `base64.StdEncoding` write, so a text encoded by any of the three reads
//! back byte for byte in the other two.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The value of one alphabet character, or `None`.
fn value(c: u8) -> Option<u32> {
    match c {
        b'A'..=b'Z' => Some((c - b'A') as u32),
        b'a'..=b'z' => Some((c - b'a') as u32 + 26),
        b'0'..=b'9' => Some((c - b'0') as u32 + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}

/// `data` as padded base64.
pub fn encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n =
            (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        let digits = [(n >> 18) & 63, (n >> 12) & 63, (n >> 6) & 63, n & 63];
        for (i, &d) in digits.iter().enumerate() {
            if i <= chunk.len() {
                out.push(ALPHABET[d as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

/// Strict base64: the alphabet and trailing padding only, in whole quads.
///
/// What Python's `b64decode(text, validate=True)` accepts: whitespace and any
/// other character outside the alphabet is an error rather than something to
/// skip, because an upload that arrives damaged should say so.  Unused
/// trailing bits are ignored, as both Python and Go ignore them.
pub fn decode(text: &str) -> Result<Vec<u8>, String> {
    let bytes = text.as_bytes();
    let body_len = bytes.len() - bytes.iter().rev().take_while(|&&c| c == b'=').count();
    let padding = bytes.len() - body_len;
    if let Some(bad) = bytes[..body_len].iter().find(|&&c| value(c).is_none()) {
        return Err(if *bad == b'=' {
            "padding in the middle of the data".to_string()
        } else {
            format!("invalid base64 character {:?}", *bad as char)
        });
    }
    if body_len % 4 == 1 {
        return Err(format!(
            "number of data characters ({body_len}) cannot be 1 more than a multiple of 4"
        ));
    }
    if padding > 2 || (padding > 0 && bytes.len() % 4 != 0) || (padding == 0 && body_len % 4 != 0) {
        return Err("incorrect padding".to_string());
    }
    Ok(decode_clean(&bytes[..body_len]))
}

/// Decodes alphabet characters with no padding (the caller checked them).
fn decode_clean(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len() / 4 * 3 + 2);
    for quad in body.chunks(4) {
        let mut n = 0u32;
        for (i, &c) in quad.iter().enumerate() {
            n |= value(c).unwrap_or(0) << (18 - 6 * i as u32);
        }
        // two characters are one byte, three are two, four are three
        let keep = quad.len().saturating_sub(1);
        for i in 0..keep {
            out.push((n >> (16 - 8 * i as u32)) as u8);
        }
    }
    out
}

/// Python's whitespace: `str.isspace()`, which is Unicode `White_Space` plus
/// the four ASCII separators `\x1c`-`\x1f`.
///
/// It matters because Python's `strip()` and `split()` decide what a
/// transcript's words are and whether a payload was repaired, and those end up
/// in the text the model trains on.
pub fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Decodes the base64 tail of a media text, repairing it first:
/// `(payload, repaired)`.
///
/// Characters outside the alphabet are dropped, a single dangling character
/// (which can never decode) goes with them, the padding is completed, and
/// `repaired` says whether any of that changed the text.  The payload comes
/// back as it decodes; the caller pads or truncates it to the length its
/// format needs.
///
/// Padding *inside* the text is not repaired: Python refuses it
/// ("discontinuous padding") and so does Go, so a completion that wrote `=`
/// and carried on is unreadable on all three sides.
pub fn repair(body: &str) -> Result<(Vec<u8>, bool), String> {
    let mut clean: Vec<u8> = body.bytes().filter(|&c| value(c).is_some() || c == b'=').collect();
    while clean.last() == Some(&b'=') {
        clean.pop();
    }
    if clean.len() % 4 == 1 {
        clean.pop(); // a single dangling character can never decode
    }
    if let Some(at) = clean.iter().position(|&c| c == b'=') {
        let why = if at == 0 {
            "leading padding not allowed"
        } else {
            "discontinuous padding not allowed"
        };
        return Err(format!("the base64 part cannot be decoded: {why}"));
    }
    let pad = (4 - clean.len() % 4) % 4;
    let mut padded = String::from_utf8_lossy(&clean).into_owned();
    padded.push_str(&"=".repeat(pad));
    let repaired = padded != body.trim_matches(is_python_space);
    Ok((decode_clean(&clean), repaired))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_round_trips_every_tail_length() {
        for n in 0..10usize {
            let data: Vec<u8> = (0..n as u8).map(|b| b.wrapping_mul(37)).collect();
            let text = encode(&data);
            assert_eq!(text.len() % 4, 0);
            assert_eq!(decode(&text).unwrap(), data);
            assert_eq!(repair(&text).unwrap(), (data.clone(), false));
        }
        assert_eq!(encode(b"hello"), "aGVsbG8=");
        assert_eq!(encode(&[0xfb, 0xff]), "+/8=");
    }

    #[test]
    fn a_strict_decode_refuses_what_python_refuses() {
        assert!(decode("aGVsbG8").is_err(), "missing padding");
        assert!(decode("aGVs bG8=").is_err(), "whitespace");
        assert!(decode("aGVsb").is_err(), "a dangling character");
        assert!(decode("aG=sbG8=").is_err(), "padding in the middle");
        assert_eq!(decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn a_predicted_tail_is_repaired() {
        // the Go and Python tests of RepairBase64 / repair_base64
        assert_eq!(repair("AAECAwQF").unwrap(), (vec![0, 1, 2, 3, 4, 5], false));
        assert!(repair("AAEC AwQF").unwrap().1, "whitespace is a repair");
        let (payload, repaired) = repair("AAECA").unwrap();
        assert!(repaired, "a dangling character is dropped");
        assert_eq!(payload, vec![0, 1, 2]);
        assert_eq!(repair("").unwrap(), (Vec::new(), false));
        // a cut-off tail is padded and still decodes
        let (payload, repaired) = repair("AAECAwQ").unwrap();
        assert!(repaired);
        assert_eq!(payload, vec![0, 1, 2, 3, 4]);
        // surrounding whitespace alone is not a repair: the text is clean inside it
        assert!(!repair("  AAECAwQF\n").unwrap().1);
        // padding in the middle is refused, as Python and Go refuse it
        for broken in ["AB=CDE", "AAAA==BB", "AAA=BBBB", "=AAA"] {
            assert!(repair(broken).is_err(), "{broken}");
        }
        assert_eq!(repair("AA==").unwrap(), (vec![0], false));
    }
}
