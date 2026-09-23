//! Speech as text: what was said *and* the waveform that said it, both behind
//! one unique token (`radixnet/speech.py`, `go/radixnet/speech.go`).
//!
//! Teaching the network by talking to it turns one utterance into texts that
//! all start with the **same unique token**:
//!
//! ```text
//! <speech:9f2a1c7d> the cat sat on the mat
//! <speech:9f2a1c7d> aud:mu:8000x1:f39ta3t7e4R0...
//! ```
//!
//! The token is `<speech>` with a short BLAKE2b digest of the waveform text
//! folded in ([`blake2b`]), so it is unique to that recording and identical in
//! both texts: in the cyclic graph the words and the sound leave the *same*
//! node, which is what ties one to the other.  The waveform text is the
//! recording itself - mixed to mono, resampled and quantised to one byte per
//! sample - as base64, so the model trains on it like any other text, and
//! [`decode_text`] turns an encoded or *predicted* one back into a WAV file.
//!
//! # Byte for byte with Python
//!
//! The token is a digest of the waveform text and the token is in the text the
//! model trains on, so the same file must encode to the same bytes on every
//! side or one recording teaches three models three different things.  Python
//! carries samples in an `array("f")`, rounding every value to single precision
//! as it is stored; so do these, with the arithmetic in `f64` and the rounding
//! to `f32` at exactly the points Python rounds, and Python's round-half-even
//! wherever it rounds to an integer.
//!
//! One place follows Python rather than Go: a WAV file stored as G.711 mu-law
//! (format tag 7) is expanded with the telephony curve, as Python expands it;
//! Go reads those bytes with this module's own mu-law codec, which is a
//! different curve.
//!
//! # What transcribes
//!
//! [`asr`]: a transcript given with the audio, or an OpenAI-compatible
//! transcription server over HTTP - both of which Python has and Go does not
//! serve.  The local Whisper backends are Python packages and are refused with
//! the reason.  Audio that is not a WAV goes through `ffmpeg`, and `speech
//! listen` records with `arecord` / `sox` / `ffmpeg`, as Python does.

pub mod asr;
pub mod blake2b;

use std::sync::Arc;

use crate::cli::Ctx;
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::multipart::base64::{self, is_python_space};
use crate::multipart::Form;
use crate::negative::python_repr;
use crate::service::Service;

/// What this module's lines are filed under.
const LOG: &str = "media";

/// The header every waveform text carries.
pub const HEADER: &str = "aud";

/// The marker every spoken text starts with; [`utterance_token`] makes it
/// unique per utterance.
pub const SPEECH_TOKEN: &str = "<speech>";

/// The waveform codecs (`auto` is mu-law).
pub const CODECS: &[&str] = &["auto", "mu", "pcm8"];

/// The codec an unspecified encode uses.
pub const DEFAULT_CODEC: &str = "mu";

/// Samples per second the waveform is resampled to (8 kHz keeps speech intelligible).
pub const DEFAULT_RATE: i64 = 8000;

/// The sample rates a waveform text may have.
pub const MIN_RATE: i64 = 1000;
pub const MAX_RATE: i64 = 48000;

/// The mu-law compression parameter (256 quantisation levels).
pub const MU: f64 = 255.0;

/// Hex digits of the waveform digest in a unique token.
const TOKEN_DIGITS: usize = 8;

const WAVE_PCM: u16 = 0x0001;
const WAVE_FLOAT: u16 = 0x0003;
const WAVE_ALAW: u16 = 0x0006;
const WAVE_MULAW: u16 = 0x0007;
const WAVE_EXTENSIBLE: u16 = 0xFFFE;

// -- small Python semantics -------------------------------------------------------------------

/// `round(x, digits)`: round-half-even on the exact binary value, which is
/// what Rust's fixed-precision formatting does too.
pub(crate) fn py_round(x: f64, digits: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.digits$}").parse().unwrap_or(x)
}

/// `" ".join(text.split())`: the words of a text, one space apart.
pub fn squash(text: &str) -> String {
    text.split(is_python_space)
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

// -- the text format --------------------------------------------------------------------------

/// `aud:<codec>:<rate>x<channels>:<base64 of one byte per sample>`.
pub fn pack_text(codec: &str, rate: i64, channels: i64, payload: &[u8]) -> String {
    format!("{HEADER}:{codec}:{rate}x{channels}:{}", base64::encode(payload))
}

/// A waveform text, taken apart.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedWaveform {
    pub codec: String,
    pub rate: i64,
    pub channels: i64,
    pub payload: Vec<u8>,
    /// Whether the base64 had to be repaired to be read.
    pub repaired: bool,
}

/// The first `aud:<codec>:<rate>x<channels>:` header in `text`: the codec, the
/// two numbers and where the payload starts (`re.search`).
fn find_header(text: &str) -> Option<(String, i64, i64, usize)> {
    // saturating: Python's integers do not overflow, and a huge rate is
    // refused later by the range check rather than here
    let number = |digits: &str| {
        digits
            .bytes()
            .fold(0i64, |n, d| n.saturating_mul(10).saturating_add((d - b'0') as i64))
    };
    let mut from = 0;
    while let Some(found) = text[from..].find("aud:") {
        let start = from + found;
        from = start + 1;
        let rest = &text[start + 4..];
        let codec_len = rest
            .bytes()
            .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
            .count();
        if codec_len == 0 || rest.as_bytes().get(codec_len) != Some(&b':') {
            continue;
        }
        let after = &rest[codec_len + 1..];
        let rate_len = after.bytes().take_while(u8::is_ascii_digit).count();
        if rate_len == 0 || after.as_bytes().get(rate_len) != Some(&b'x') {
            continue;
        }
        let tail = &after[rate_len + 1..];
        let channels_len = tail.bytes().take_while(u8::is_ascii_digit).count();
        if channels_len == 0 || tail.as_bytes().get(channels_len) != Some(&b':') {
            continue;
        }
        let end = start + 4 + codec_len + 1 + rate_len + 1 + channels_len + 1;
        return Some((
            rest[..codec_len].to_string(),
            number(&after[..rate_len]),
            number(&tail[..channels_len]),
            end,
        ));
    }
    None
}

/// Reads an encoded - or predicted - waveform text (`speech.parse_text`).
///
/// The header is looked for *anywhere*, because a spoken text carries its
/// token first and a paired one its transcript last; the payload runs from
/// the header to the next whitespace, and its base64 is repaired when it is
/// not clean.
pub fn parse_text(text: &str) -> Result<ParsedWaveform, String> {
    let Some((codec, rate, channels, end)) = find_header(text) else {
        return Err("not an encoded waveform: expected 'aud:<codec>:<rate>x<channels>:<base64>'".to_string());
    };
    let body = text[end..].split(is_python_space).find(|w| !w.is_empty()).unwrap_or("");
    let (payload, repaired) = base64::repair(body)?;
    if rate <= 0 || channels <= 0 {
        return Err(format!("invalid waveform header {rate}x{channels}"));
    }
    Ok(ParsedWaveform {
        codec,
        rate,
        channels,
        payload,
        repaired,
    })
}

/// The token both texts of one utterance start with: `<speech:9f2a1c7d>`, a
/// digest of the waveform text - or the plain [`SPEECH_TOKEN`] when `unique`
/// is off, which merges every spoken text into one entry node.
pub fn utterance_token(payload: &str, unique: bool) -> String {
    if !unique {
        return SPEECH_TOKEN.to_string();
    }
    let digest = blake2b::hexdigest(payload.as_bytes(), TOKEN_DIGITS.div_ceil(2));
    format!(
        "{}:{}>",
        &SPEECH_TOKEN[..SPEECH_TOKEN.len() - 1],
        &digest[..TOKEN_DIGITS]
    )
}

/// A token as given: trimmed, non-empty, no whitespace inside.
pub fn check_token(token: &str) -> Result<String, String> {
    let token = token.trim_matches(is_python_space);
    if token.is_empty() {
        return Err("the token must not be empty".to_string());
    }
    if token.chars().any(is_python_space) {
        return Err(format!("the token must not contain whitespace: {}", python_repr(token)));
    }
    Ok(token.to_string())
}

/// The texts one utterance trains on, each starting with `token`: the
/// transcript, the waveform, and with `pair` the waveform followed by its
/// transcript (so the search can run from the sound into the words).
pub fn speech_texts(token: &str, transcript: &str, audio_text: &str, pair: bool) -> Result<Vec<String>, String> {
    let token = check_token(token)?;
    let transcript = squash(transcript);
    let audio_text = audio_text.trim_matches(is_python_space);
    let mut texts = Vec::new();
    if !transcript.is_empty() {
        texts.push(format!("{token} {transcript}"));
    }
    if !audio_text.is_empty() {
        texts.push(format!("{token} {audio_text}"));
        if pair && !transcript.is_empty() {
            texts.push(format!("{token} {audio_text} {transcript}"));
        }
    }
    Ok(texts)
}

// -- the codecs -------------------------------------------------------------------------------

/// A waveform codec: how a sample in `[-1, 1]` becomes one byte and back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    /// mu-law companding: `sign(x) * ln(1 + 255|x|) / ln(256)` on 256 levels -
    /// the classic 8-bit speech quantisation, which spends its levels where
    /// speech lives instead of on the peaks.
    Mu,
    /// Plain linear 8-bit, one signed byte per sample.
    Pcm8,
}

/// `min(1.0, x)` then `max(-1.0, ...)` in Python's order, which reads NaN as 1.
fn clamp_unit(x: f64) -> f64 {
    let high = if x < 1.0 { x } else { 1.0 };
    if high > -1.0 {
        high
    } else {
        -1.0
    }
}

impl Codec {
    pub fn name(self) -> &'static str {
        match self {
            Codec::Mu => "mu",
            Codec::Pcm8 => "pcm8",
        }
    }

    /// Samples -> one byte each.
    pub fn encode(self, samples: &[f32]) -> Vec<u8> {
        match self {
            Codec::Mu => {
                let scale = MU.ln_1p();
                samples
                    .iter()
                    .map(|&s| {
                        // clamp() lets NaN through, as Python's two comparisons do
                        let value = (s as f64).clamp(-1.0, 1.0);
                        let magnitude = (MU * value.abs()).ln_1p() / scale;
                        let companded = if value >= 0.0 { magnitude } else { -magnitude };
                        ((companded + 1.0) * 127.5).round_ties_even().clamp(0.0, 255.0) as u8
                    })
                    .collect()
            }
            Codec::Pcm8 => samples
                .iter()
                .map(|&s| {
                    let level = (clamp_unit(s as f64) * 127.0).round_ties_even().clamp(-127.0, 127.0);
                    (level as i64 & 0xFF) as u8
                })
                .collect(),
        }
    }

    /// One byte each -> samples.
    pub fn decode(self, payload: &[u8]) -> Vec<f32> {
        match self {
            Codec::Mu => payload
                .iter()
                .map(|&b| {
                    let companded = b as f64 / 127.5 - 1.0;
                    let magnitude = ((1.0 + MU).powf(companded.abs()) - 1.0) / MU;
                    (if companded >= 0.0 { magnitude } else { -magnitude }) as f32
                })
                .collect(),
            // -128, which only a *predicted* text holds, clamps to -1
            Codec::Pcm8 => payload
                .iter()
                .map(|&b| (b as i8 as f64 / 127.0).max(-1.0) as f32)
                .collect(),
        }
    }
}

/// The codec `name` names; `auto` (or nothing) is mu-law.
pub fn get_codec(name: &str) -> Result<Codec, String> {
    let raw = if name.is_empty() { "auto" } else { name };
    match raw.trim_matches(is_python_space).to_lowercase().as_str() {
        "auto" | "mu" => Ok(Codec::Mu),
        "pcm8" => Ok(Codec::Pcm8),
        _ => Err(format!(
            "unknown waveform codec {}; expected one of: {}",
            python_repr(raw),
            CODECS.join(", ")
        )),
    }
}

/// A sample rate a waveform may be resampled to.
pub fn check_rate(rate: i64) -> Result<i64, String> {
    if !(MIN_RATE..=MAX_RATE).contains(&rate) {
        return Err(format!(
            "the sample rate must be between {MIN_RATE} and {MAX_RATE} Hz, got {rate}"
        ));
    }
    Ok(rate)
}

// -- audio ------------------------------------------------------------------------------------

/// Decoded audio: interleaved samples in `[-1, 1]`, the rate and the channel count.
#[derive(Clone, Debug, PartialEq)]
pub struct Audio {
    pub samples: Vec<f32>,
    pub rate: i64,
    pub channels: i64,
}

impl Audio {
    pub fn frames(&self) -> usize {
        if self.channels > 0 {
            self.samples.len() / self.channels as usize
        } else {
            0
        }
    }

    pub fn seconds(&self) -> f64 {
        if self.rate > 0 {
            self.frames() as f64 / self.rate as f64
        } else {
            0.0
        }
    }

    /// `{"rate", "channels", "frames", "seconds"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("rate", Json::Int(self.rate)),
            ("channels", Json::Int(self.channels)),
            ("frames", Json::Int(self.frames() as i64)),
            ("seconds", Json::Num(py_round(self.seconds(), 4))),
        ])
    }
}

/// One G.711 mu-law byte -> a sample (the telephony curve, not [`Codec::Mu`]).
fn ulaw_to_linear(byte: u8) -> f64 {
    let byte = !byte;
    let magnitude = ((((byte & 0x0F) as i32) << 3) + 0x84) << ((byte >> 4) & 0x07);
    let value = magnitude - 0x84;
    (if byte & 0x80 != 0 { -value } else { value }) as f64 / 32768.0
}

/// One G.711 A-law byte -> a sample.
fn alaw_to_linear(byte: u8) -> f64 {
    let byte = byte ^ 0x55;
    let exponent = ((byte >> 4) & 0x07) as i32;
    let mantissa = (byte & 0x0F) as i32;
    let magnitude = if exponent > 0 {
        ((mantissa << 4) + 0x108) << (exponent - 1)
    } else {
        (mantissa << 4) + 8
    };
    (if byte & 0x80 != 0 { -magnitude } else { magnitude }) as f64 / 32768.0
}

/// A WAVE data chunk -> samples in `[-1, 1]`.
fn samples_from_pcm(data: &[u8], format: u16, bits: u16) -> Result<Vec<f32>, String> {
    let words = |width: usize| data.chunks_exact(width);
    match format {
        WAVE_MULAW => return Ok(data.iter().map(|&b| ulaw_to_linear(b) as f32).collect()),
        WAVE_ALAW => return Ok(data.iter().map(|&b| alaw_to_linear(b) as f32).collect()),
        WAVE_FLOAT => {
            return match bits {
                32 => Ok(words(4)
                    .map(|w| clamp_unit(f32::from_le_bytes([w[0], w[1], w[2], w[3]]) as f64) as f32)
                    .collect()),
                64 => Ok(words(8)
                    .map(|w| {
                        let mut b = [0u8; 8];
                        b.copy_from_slice(w);
                        clamp_unit(f64::from_le_bytes(b)) as f32
                    })
                    .collect()),
                _ => Err(format!("unsupported float WAV sample size: {bits} bits")),
            };
        }
        WAVE_PCM => {}
        other => {
            return Err(format!(
                "unsupported WAV format tag 0x{other:04x} (PCM, IEEE float, A-law and mu-law are read)"
            ))
        }
    }
    match bits {
        // 8-bit PCM is unsigned, everything wider is signed
        8 => Ok(data.iter().map(|&b| ((b as f64 - 128.0) / 128.0) as f32).collect()),
        16 => Ok(words(2)
            .map(|w| (i16::from_le_bytes([w[0], w[1]]) as f64 / 32768.0) as f32)
            .collect()),
        24 => Ok(words(3)
            .map(|w| {
                let raw = (w[0] as i32) | (w[1] as i32) << 8 | (w[2] as i32) << 16;
                let value = if raw & 0x80_0000 != 0 { raw - 0x100_0000 } else { raw };
                (value as f64 / 8_388_608.0) as f32
            })
            .collect()),
        32 => Ok(words(4)
            .map(|w| (i32::from_le_bytes([w[0], w[1], w[2], w[3]]) as f64 / 2_147_483_648.0) as f32)
            .collect()),
        _ => Err(format!("unsupported PCM sample size: {bits} bits")),
    }
}

fn u16_at(data: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([data[at], data[at + 1]])
}

fn u32_at(data: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([data[at], data[at + 1], data[at + 2], data[at + 3]])
}

/// A RIFF/WAVE file (`speech.parse_wav`): PCM 8 (unsigned) / 16 / 24 / 32,
/// IEEE float 32 / 64, A-law and mu-law, `WAVE_FORMAT_EXTENSIBLE` resolved
/// through its sub-format, chunks walked in order with their odd-length
/// padding, and a streamed data chunk of declared size 0 read to the end.
pub fn parse_wav(data: &[u8]) -> Result<Audio, String> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a WAV file (no RIFF/WAVE header)".to_string());
    }
    let (mut format, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
    let mut payload: &[u8] = &[];
    let mut offset = 12usize;
    while offset + 8 <= data.len() {
        let id = &data[offset..offset + 4];
        let size = u32_at(data, offset + 4) as usize;
        let body = &data[offset + 8..(offset + 8).saturating_add(size).min(data.len())];
        if id == b"fmt " && body.len() >= 16 {
            format = u16_at(body, 0);
            channels = u16_at(body, 2);
            rate = u32_at(body, 4);
            bits = u16_at(body, 14);
            if format == WAVE_EXTENSIBLE && body.len() >= 26 {
                format = u16_at(body, 24); // the sub-format GUID starts with the real tag
            }
        } else if id == b"data" {
            if size == 0 {
                payload = &data[offset + 8..]; // a streamed WAV: the rest of the file is audio
                break;
            }
            payload = body;
        }
        offset = offset.saturating_add(8 + size + (size & 1)); // chunks are padded to an even length
    }
    if channels == 0 || rate == 0 {
        return Err("the WAV file has no usable 'fmt ' chunk".to_string());
    }
    if payload.is_empty() {
        return Err("the WAV file has no audio data".to_string());
    }
    let mut samples = samples_from_pcm(payload, format, bits)?;
    samples.truncate(samples.len() - samples.len() % channels as usize);
    Ok(Audio {
        samples,
        rate: rate as i64,
        channels: channels as i64,
    })
}

/// Audio bytes -> [`Audio`]: a WAV is read directly, anything else through
/// ffmpeg (`speech.load_audio`).
pub fn load_audio(data: &[u8]) -> Result<Audio, String> {
    if data.is_empty() {
        return Err("the audio is empty".to_string());
    }
    if data.len() >= 12 && &data[..4] == b"RIFF" && &data[8..12] == b"WAVE" {
        return parse_wav(data);
    }
    parse_wav(&asr::convert_with_ffmpeg(data, None)?)
}

/// Averages interleaved channels down to one.
pub fn to_mono(samples: &[f32], channels: usize) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks_exact(channels)
        .map(|frame| {
            let total: f64 = frame.iter().fold(0.0, |sum, &s| sum + s as f64);
            (total / channels as f64) as f32
        })
        .collect()
}

/// Linear interpolation to `dst` samples per second (the waveform is
/// quantised to a byte afterwards anyway).
pub fn resample(samples: &[f32], src: i64, dst: i64) -> Result<Vec<f32>, String> {
    if src == dst || samples.is_empty() {
        return Ok(samples.to_vec());
    }
    if src <= 0 {
        return Err(format!("invalid source sample rate {src}"));
    }
    let count = ((samples.len() as f64 * dst as f64 / src as f64).round_ties_even() as usize).max(1);
    let step = src as f64 / dst as f64; // the rate ratio, not a stretch to the last sample
    let last = samples.len() - 1;
    Ok((0..count)
        .map(|i| {
            let position = i as f64 * step;
            let left = position as usize;
            if left >= last {
                return samples[last];
            }
            let fraction = position - left as f64;
            (samples[left] as f64 * (1.0 - fraction) + samples[left + 1] as f64 * fraction) as f32
        })
        .collect())
}

/// Scales the waveform so its loudest sample sits at `headroom`; silence is
/// left alone.
pub fn normalise_samples(samples: &[f32], headroom: f64) -> Vec<f32> {
    let peak = samples.iter().fold(0.0f64, |peak, &v| {
        let magnitude = (v as f64).abs();
        if magnitude > peak {
            magnitude
        } else {
            peak
        }
    });
    if peak <= 1e-9 {
        return samples.to_vec();
    }
    let gain = headroom / peak;
    samples.iter().map(|&v| (v as f64 * gain) as f32).collect()
}

/// Samples in `[-1, 1]` -> a 16-bit PCM WAV file.
pub fn wav_bytes(samples: &[f32], rate: i64, channels: i64) -> Vec<u8> {
    let mut body = Vec::with_capacity(samples.len() * 2);
    for &value in samples {
        let level = (value as f64 * 32767.0).round_ties_even().clamp(-32768.0, 32767.0) as i16;
        body.extend_from_slice(&level.to_le_bytes());
    }
    let (rate, channels) = (rate as u32, channels as u16);
    let mut out = Vec::with_capacity(44 + body.len());
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + body.len() as u32).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&WAVE_PCM.to_le_bytes());
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate.wrapping_mul(channels as u32).wrapping_mul(2)).to_le_bytes());
    out.extend_from_slice(&(channels.wrapping_mul(2)).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&(body.len() as u32).to_le_bytes());
    out.extend_from_slice(&body);
    out
}

// -- encode / decode --------------------------------------------------------------------------

/// A recording as the waveform text, with what went into it.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedAudio {
    pub text: String,
    pub codec: Codec,
    pub rate: i64,
    pub samples: usize,
    pub normalised: bool,
    /// The recording as it arrived: `{"rate", "channels", "frames", "seconds"}`.
    pub source: Json,
}

impl EncodedAudio {
    /// `{"text", "codec", "rate", "channels", "samples", "seconds", "bytes",
    /// "chars", "normalised", "source"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("text", Json::str(self.text.clone())),
            ("codec", Json::str(self.codec.name())),
            ("rate", Json::Int(self.rate)),
            ("channels", Json::Int(1)),
            ("samples", Json::Int(self.samples as i64)),
            (
                "seconds",
                Json::Num(py_round(self.samples as f64 / self.rate as f64, 4)),
            ),
            ("bytes", Json::Int(self.samples as i64)),
            ("chars", Json::Int(self.text.chars().count() as i64)),
            ("normalised", Json::Bool(self.normalised)),
            ("source", self.source.clone()),
        ])
    }
}

/// Audio bytes -> the waveform text: mixed to mono, resampled to `rate`,
/// optionally peak-normalised, one byte per sample, base64 (`encode_audio`).
pub fn encode_audio(data: &[u8], rate: i64, codec: &str, normalise: bool) -> Result<EncodedAudio, String> {
    let rate = check_rate(rate)?;
    let codec = get_codec(codec)?;
    let audio = load_audio(data)?;
    let mut samples = resample(&to_mono(&audio.samples, audio.channels as usize), audio.rate, rate)?;
    if normalise {
        samples = normalise_samples(&samples, 0.99);
    }
    let payload = codec.encode(&samples);
    Ok(EncodedAudio {
        text: pack_text(codec.name(), rate, 1, &payload),
        codec,
        rate,
        samples: payload.len(),
        normalised: normalise,
        source: audio.to_json(),
    })
}

/// A waveform text decoded back to audio.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedAudio {
    pub wav: Vec<u8>,
    pub codec: Codec,
    pub rate: i64,
    pub channels: i64,
    pub samples: usize,
    pub bytes: usize,
    pub repaired: bool,
}

impl DecodedAudio {
    /// `{"codec", "rate", "channels", "samples", "seconds", "bytes", "repaired"}`
    /// (the WAV itself goes wherever the caller sends it).
    pub fn to_json(&self) -> Json {
        let frames = self.samples / self.channels.max(1) as usize;
        Json::obj([
            ("codec", Json::str(self.codec.name())),
            ("rate", Json::Int(self.rate)),
            ("channels", Json::Int(self.channels)),
            ("samples", Json::Int(self.samples as i64)),
            ("seconds", Json::Num(py_round(frames as f64 / self.rate as f64, 4))),
            ("bytes", Json::Int(self.bytes as i64)),
            ("repaired", Json::Bool(self.repaired)),
        ])
    }
}

/// An encoded - or predicted - waveform text -> a 16-bit PCM WAV, so what the
/// network says can be listened to.  `codec` overrides the one in the text.
pub fn decode_text(text: &str, codec: Option<&str>) -> Result<DecodedAudio, String> {
    let parsed = parse_text(text)?;
    let chosen = codec.filter(|c| !c.is_empty()).unwrap_or(&parsed.codec);
    let codec = get_codec(chosen)?;
    let rate = check_rate(parsed.rate)?;
    if parsed.channels > u16::MAX as i64 {
        return Err(format!(
            "invalid waveform header {}x{}: a WAV holds at most {} channels",
            parsed.rate,
            parsed.channels,
            u16::MAX
        ));
    }
    let samples = codec.decode(&parsed.payload);
    Ok(DecodedAudio {
        wav: wav_bytes(&samples, rate, parsed.channels),
        codec,
        rate,
        channels: parsed.channels,
        samples: samples.len(),
        bytes: parsed.payload.len(),
        repaired: parsed.repaired,
    })
}

// -- teaching ---------------------------------------------------------------------------------

/// How one utterance becomes text (`speech.teach`'s keywords).
#[derive(Clone, Debug)]
pub struct TeachOptions {
    /// The words, when the caller has them.
    pub transcript: String,
    /// How to transcribe when it does not.
    pub asr: asr::AsrOptions,
    pub rate: i64,
    pub codec: String,
    pub normalise: bool,
    /// Learn the waveform too (off: the transcript only).
    pub waveform: bool,
    /// Also one text of the waveform followed by its transcript.
    pub pair: bool,
    /// A token to use instead of `<speech:digest>`.
    pub token: Option<String>,
    /// A token of its own per utterance (off: the plain `<speech>`).
    pub unique: bool,
}

impl Default for TeachOptions {
    /// The Python defaults.
    fn default() -> TeachOptions {
        TeachOptions {
            transcript: String::new(),
            asr: asr::AsrOptions {
                backend: "auto".to_string(),
                ..Default::default()
            },
            rate: DEFAULT_RATE,
            codec: "auto".to_string(),
            normalise: false,
            waveform: true,
            pair: false,
            token: None,
            unique: true,
        }
    }
}

/// What transcribed an utterance, or why nothing did.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AsrReport {
    pub backend: Option<String>,
    pub model: Option<String>,
    pub language: Option<String>,
    pub seconds: f64,
    pub error: Option<String>,
}

impl AsrReport {
    pub fn to_json(&self) -> Json {
        let opt = |v: &Option<String>| v.clone().map(Json::Str).unwrap_or(Json::Null);
        Json::obj([
            ("backend", opt(&self.backend)),
            ("model", opt(&self.model)),
            ("language", opt(&self.language)),
            ("seconds", Json::Num(self.seconds)),
            ("error", opt(&self.error)),
        ])
    }
}

/// One utterance as the texts the network learns.
#[derive(Clone, Debug, PartialEq)]
pub struct Taught {
    pub token: String,
    pub transcript: String,
    pub asr: AsrReport,
    pub audio: Option<EncodedAudio>,
    pub texts: Vec<String>,
    pub pair: bool,
}

impl Taught {
    /// `{"token", "transcript", "asr", "audio", "texts", "chars", "pair"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("token", Json::str(self.token.clone())),
            ("transcript", Json::str(self.transcript.clone())),
            ("asr", self.asr.to_json()),
            ("audio", self.audio.as_ref().map(|a| a.to_json()).unwrap_or(Json::Null)),
            ("texts", Json::strs(self.texts.clone())),
            (
                "chars",
                Json::Int(self.texts.iter().map(|t| t.chars().count()).sum::<usize>() as i64),
            ),
            ("pair", Json::Bool(self.pair)),
        ])
    }

    /// The waveform text, when there is one.
    pub fn waveform(&self) -> Option<&String> {
        self.texts.iter().find(|t| t.contains("aud:"))
    }
}

/// One utterance -> the texts the network learns, all behind the same unique
/// token (`speech.teach`).
///
/// The audio is transcribed unless the words were given, and encoded as a
/// waveform text; the token is derived from the waveform, so the transcript
/// and the sound share it.  A transcription that fails is *recorded*, not
/// raised: the waveform alone is still worth training on.
pub fn teach(data: &[u8], o: &TeachOptions) -> Result<Taught, String> {
    let mut words = squash(&o.transcript);
    if data.is_empty() && words.is_empty() {
        return Err("nothing to learn: pass audio, a transcript, or both".to_string());
    }
    let audio = if !data.is_empty() && o.waveform {
        Some(encode_audio(data, o.rate, &o.codec, o.normalise)?)
    } else {
        None
    };
    let mut report = AsrReport {
        language: o.asr.language.clone(),
        ..Default::default()
    };
    if words.is_empty() && !data.is_empty() {
        let asked = asr::AsrOptions {
            text: String::new(),
            ..o.asr.clone()
        };
        match asr::transcribe(data, &asked) {
            Ok(heard) => {
                words = squash(&heard.transcript);
                report.backend = Some(heard.backend);
                report.model = heard.model;
                report.language = heard.language;
                report.seconds = py_round(heard.seconds, 3);
            }
            Err(why) => {
                crate::log_info!(LOG, "not transcribed: {why}");
                report.error = Some(why);
            }
        }
    } else if !words.is_empty() {
        report.backend = Some("given".to_string());
    }
    let audio_text = audio.as_ref().map(|a| a.text.clone()).unwrap_or_default();
    let seed = if audio.is_some() { &audio_text } else { &words };
    let token = match &o.token {
        Some(token) if !token.is_empty() => check_token(token)?,
        _ => utterance_token(seed, o.unique),
    };
    let texts = speech_texts(&token, &words, &audio_text, o.pair)?;
    let pair = o.pair && !words.is_empty() && audio.is_some();
    Ok(Taught {
        token,
        transcript: words,
        asr: report,
        audio,
        texts,
        pair,
    })
}

/// What this side can do with audio (`speech.describe`), in the keys the
/// Speech tab reads.
pub fn describe() -> Json {
    let url = asr::default_asr_url();
    Json::obj([
        ("engine", Json::str(crate::service::ENGINE)),
        ("backends", Json::strs(asr::ASR_BACKENDS.iter().copied())),
        ("faster_whisper", Json::Bool(false)),
        ("whisper", Json::Bool(false)),
        ("whisper_model", Json::str(asr::default_whisper_model())),
        (
            "server_url",
            if url.is_empty() {
                Json::Null
            } else {
                Json::str(url.clone())
            },
        ),
        ("server_model", Json::str(asr::default_asr_model())),
        (
            "auto",
            if url.is_empty() {
                Json::Null
            } else {
                Json::str("server")
            },
        ),
        ("given_always", Json::Bool(true)),
        ("ffmpeg", Json::Bool(asr::ffmpeg_path().is_some())),
        ("recorders", Json::strs(asr::recorders())),
        ("codecs", Json::strs(CODECS.iter().copied())),
        ("default_codec", Json::str(DEFAULT_CODEC)),
        ("default_rate", Json::Int(DEFAULT_RATE)),
        ("token", Json::str(SPEECH_TOKEN)),
        ("token_example", Json::str(utterance_token("example", true))),
        (
            "backends_note",
            Json::str(
                "the local Whisper backends (faster-whisper, openai-whisper) are Python packages: send the words \
                 with the audio, as the browser's dictation does, or set $RADIXNET_ASR_URL to an \
                 OpenAI-compatible /v1/audio/transcriptions server",
            ),
        ),
        (
            "text_format",
            Json::str(format!(
                "{HEADER}:<codec>:<rate>x<channels>:<base64 of one byte per sample>"
            )),
        ),
        (
            "formats",
            Json::str("WAV natively (PCM, IEEE float, A-law, mu-law); other formats need ffmpeg"),
        ),
    ])
}

// -- the command line -------------------------------------------------------------------------

/// The files named after the action.
fn files(ctx: &Ctx) -> Vec<String> {
    ctx.args.rest.iter().skip(1).cloned().collect()
}

/// An audio file's bytes, or the error Python gives for a missing one.
fn read_audio_file(path: &str) -> Result<Vec<u8>, String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("audio file not found: {path}"));
    }
    std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))
}

/// The transcription flags every speech action shares.
fn asr_options(ctx: &Ctx, text: &str) -> asr::AsrOptions {
    let a = &ctx.args;
    asr::AsrOptions {
        backend: a.str("backend", "auto"),
        text: text.to_string(),
        language: a.get("language").map(str::to_string),
        model: a.get("asr-model").map(str::to_string),
        url: a.get("asr-url").map(str::to_string),
    }
}

/// How an utterance becomes text, from the command line.
fn teach_options(ctx: &Ctx) -> Result<TeachOptions, String> {
    let a = &ctx.args;
    let transcript = a.str("text", "");
    let codec = a.str("codec", "auto");
    if !CODECS.contains(&codec.as_str()) {
        return Err(format!("--codec must be one of {}, got {codec:?}", CODECS.join(", ")));
    }
    Ok(TeachOptions {
        asr: asr_options(ctx, ""),
        transcript,
        rate: a.int("rate", DEFAULT_RATE)?,
        codec,
        normalise: a.on("normalise"),
        waveform: !a.on("no-waveform"),
        pair: a.on("pair"),
        token: a.get("token").map(str::to_string),
        unique: !a.on("shared-token"),
    })
}

/// `{"path", "bytes"}` of a model just saved (`cli.save_model`).
pub(crate) fn save_model(model: &mut crate::model::Model, path: &str) -> Result<Json, String> {
    model.save(path)?;
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(Json::obj([
        ("path", Json::str(path)),
        ("bytes", Json::Int(bytes as i64)),
    ]))
}

/// Trains the model at `--model` on `texts` and saves it to `--model-out` (or
/// back): the `trained` document of `image encode --train` and `speech teach --train`.
///
/// The model learns the way its kind learns (`model.train(texts, epochs=,
/// lr=, batch_size=)`): `--lr` (0.5 - one long text, so it is high) and
/// `--batch-size` (8) are the sine model's; the count and phase models count.
pub(crate) fn train_on(ctx: &Ctx, texts: &[String]) -> Result<Json, String> {
    let mut model = ctx.open(false)?;
    let a = &ctx.args;
    let (epochs, lr, batch_size) = (a.usize("epochs", 3)?, a.float("lr", 0.5)?, a.usize("batch-size", 8)?);
    let records = crate::kinds::train_at(&mut model, texts, epochs, lr, batch_size)?;
    let out = ctx.args.str("model-out", &ctx.model_path);
    let saved = save_model(&mut model, &out)?;
    crate::log_info!(
        LOG,
        "trained {} epoch(s) on {} text(s); saved to {out}",
        records.len(),
        texts.len()
    );
    Ok(Json::obj([
        ("epochs", Json::Int(records.len() as i64)),
        ("loss", records.last().map(|r| Json::Num(r.loss)).unwrap_or(Json::Null)),
        ("texts", Json::Int(texts.len() as i64)),
        ("saved", saved),
        ("kind", Json::str(model.kind())),
    ]))
}

/// Appends `pairs` to a JSON object.
pub(crate) fn extend(doc: &mut Json, pairs: Vec<(&str, Json)>) {
    if let Json::Obj(fields) = doc {
        for (key, value) in pairs {
            match fields.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => fields.push((key.to_string(), value)),
            }
        }
    }
}

/// `speech teach` and `speech listen`: transcribe, encode the waveform, train.
fn teach_from_audio(ctx: &Ctx, data: &[u8], source: &str) -> Result<Json, String> {
    let taught = teach(data, &teach_options(ctx)?)?;
    if let Some(error) = &taught.asr.error {
        crate::log_warn!(LOG, "not transcribed ({error}); only the waveform is learned");
    }
    let mut doc = taught.to_json();
    extend(
        &mut doc,
        vec![
            ("source", Json::str(source)),
            ("out", Json::Null),
            ("saved_audio", Json::Null),
            ("trained", Json::Null),
        ],
    );
    if let Some(path) = save_path(ctx) {
        std::fs::write(&path, data).map_err(|err| format!("cannot write {path}: {err}"))?;
        extend(&mut doc, vec![("saved_audio", Json::str(path))]);
    }
    if let Some(out) = ctx.args.get("out") {
        std::fs::write(out, format!("{}\n", taught.texts.join("\n")))
            .map_err(|err| format!("cannot write {out}: {err}"))?;
        extend(&mut doc, vec![("out", Json::str(out))]);
    }
    if ctx.args.on("train") {
        if taught.texts.is_empty() {
            return Err("nothing to train on: no transcript and no waveform".to_string());
        }
        let trained = train_on(ctx, &taught.texts)?;
        extend(&mut doc, vec![("trained", trained)]);
    }
    Ok(doc)
}

/// `speech listen --save WAV`.  `--save` is a switch elsewhere on the command
/// line, so `--save=WAV` and `--save WAV` (the path left over as a word) both
/// name the file.
fn save_path(ctx: &Ctx) -> Option<String> {
    if ctx.command != "speech" || ctx.args.action() != "listen" {
        return None;
    }
    match ctx.args.get("save") {
        Some(path) if !matches!(path, "true" | "1" | "yes" | "on") => Some(path.to_string()),
        _ if ctx.args.on("save") => ctx.args.rest.get(1).cloned(),
        _ => None,
    }
}

/// `--text`, else the contents of `--data`: the text a decode reads.
pub(crate) fn text_or_data(ctx: &Ctx) -> Result<String, String> {
    if let Some(text) = ctx.args.get("text") {
        return Ok(text.to_string());
    }
    let Some(path) = ctx.args.get("data") else {
        return Err("give the text with --text, or a file holding it with --data".to_string());
    };
    if !std::path::Path::new(path).is_file() {
        return Err(format!("data file not found: {path}"));
    }
    std::fs::read_to_string(path).map_err(|err| format!("{path} is not valid UTF-8 text: {err}"))
}

/// `radixnet speech <action>`: info, transcribe, teach, listen, tutor, decode.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let a = &ctx.args;
    let doc = match a.action() {
        "info" => describe(),
        "transcribe" => {
            let path = files(ctx)
                .into_iter()
                .next()
                .ok_or("an audio file is required: radixnet speech transcribe FILE")?;
            let data = read_audio_file(&path)?;
            let heard = asr::transcribe(&data, &asr_options(ctx, &a.str("text", "")))?;
            let mut doc = heard.to_json();
            extend(&mut doc, vec![("file", Json::str(path)), ("out", Json::Null)]);
            if let Some(out) = a.get("out") {
                std::fs::write(out, format!("{}\n", heard.transcript))
                    .map_err(|err| format!("cannot write {out}: {err}"))?;
                extend(&mut doc, vec![("out", Json::str(out))]);
            }
            doc
        }
        "teach" => {
            let path = files(ctx)
                .into_iter()
                .next()
                .ok_or("an audio file is required: radixnet speech teach FILE")?;
            let data = read_audio_file(&path)?;
            teach_from_audio(ctx, &data, &path)?
        }
        "listen" => {
            let seconds = a.float("seconds", 5.0)?;
            let rate = a.int("record-rate", 16_000)?;
            crate::log_warn!(
                LOG,
                "recording {}s from the microphone - speak now",
                crate::json::py_repr(seconds)
            );
            let data = asr::record(seconds, rate, a.get("recorder"))?;
            crate::log_warn!(LOG, "recording finished");
            teach_from_audio(ctx, &data, &format!("microphone ({}s)", crate::json::py_repr(seconds)))?
        }
        "tutor" => {
            let paths = files(ctx);
            if paths.is_empty() {
                return Err("at least one audio file is required: radixnet speech tutor FILE...".to_string());
            }
            let mut options = teach_options(ctx)?;
            options.waveform = true;
            options.pair = false;
            let (mut texts, mut labels, mut said) = (Vec::new(), Vec::new(), Vec::new());
            for path in &paths {
                let taught = teach(&read_audio_file(path)?, &options)?;
                let waveform = taught
                    .waveform()
                    .cloned()
                    .ok_or_else(|| format!("{path} produced no waveform to remember"))?;
                texts.push(waveform);
                labels.push(format!("{} {}", base_name(path), taught.token));
                said.push(taught.transcript);
            }
            crate::recall::tutor_cli(ctx, &texts, &labels, "speech", &said)?
        }
        "decode" => {
            let text = text_or_data(ctx)?;
            let out = a.get("out").ok_or("--out is required: where to write the WAV")?;
            let decoded = decode_text(&text, a.get("codec"))?;
            std::fs::write(out, &decoded.wav).map_err(|err| format!("cannot write {out}: {err}"))?;
            let mut doc = decoded.to_json();
            extend(&mut doc, vec![("out", Json::str(out))]);
            doc
        }
        "" => return Err("speech needs an action: info, transcribe, teach, listen, tutor, decode".to_string()),
        other => {
            return Err(format!(
                "unknown speech action {other:?}; expected info, transcribe, teach, listen, tutor or decode"
            ))
        }
    };
    ctx.emit(doc);
    Ok(())
}

/// The last component of a path.
pub(crate) fn base_name(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

// -- the routes -------------------------------------------------------------------------------

/// How an utterance becomes text, from a request (`_r_speech_teach`'s options).
fn teach_options_from(form: &Form) -> Result<TeachOptions, ApiError> {
    let rate = form
        .int("rate", 0)
        .map_err(|_| ApiError::bad_request("'rate' must be an integer"))?;
    Ok(TeachOptions {
        transcript: form.text("transcript", ""),
        asr: asr::AsrOptions {
            backend: form.text("backend", "auto"),
            text: String::new(),
            language: form.maybe_text("language"),
            model: form.maybe_text("asr_model"),
            url: form.maybe_text("asr_url"),
        },
        rate: if rate == 0 { DEFAULT_RATE } else { rate },
        codec: form.text("codec", "auto"),
        normalise: form.flag("normalise", false),
        waveform: form.flag("waveform", true),
        pair: form.flag("pair", false),
        token: form.maybe_text("token"),
        unique: form.flag("unique", true),
    })
}

/// `GET /api/speech`.
fn r_speech(_svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(describe())
}

/// `POST /api/speech/transcribe`: audio -> the words, with the backend that did it.
fn r_transcribe(_svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::read(r, Some("speech"))?;
    let (name, data) = form.bytes("audio")?;
    let o = asr::AsrOptions {
        backend: form.text("backend", "auto"),
        text: form.text("transcript", ""),
        language: form.maybe_text("language"),
        model: form.maybe_text("asr_model"),
        url: form.maybe_text("asr_url"),
    };
    let heard = asr::transcribe(&data, &o).map_err(ApiError::bad_request)?;
    let mut doc = heard.to_json();
    extend(&mut doc, vec![("name", Json::str(name))]);
    Ok(doc)
}

/// `POST /api/speech/teach`: one utterance -> the words and the waveform behind
/// one unique token; `save_as` keeps the texts, `train` learns them (202).
fn r_teach(svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::read(r, Some("speech"))?;
    let (name, data) = form.bytes("recording")?;
    let options = teach_options_from(&form)?;
    let train = form.flag("train", false);
    if train {
        svc.ensure_idle()?;
    }
    let taught = teach(&data, &options).map_err(ApiError::bad_request)?;
    if train && taught.texts.is_empty() {
        return Err(ApiError::bad_request(
            "nothing to train on: the audio produced neither a transcript nor a waveform",
        ));
    }
    let mut doc = taught.to_json();
    extend(
        &mut doc,
        vec![("name", Json::str(name)), ("upload", Json::Null), ("job", Json::Null)],
    );
    crate::multipart::save_and_train(svc, &form, doc, &taught.texts)
}

/// `POST /api/speech/decode`: `{text, codec}` -> `{wav_base64, ...}`.
fn r_decode(_svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::json(r)?;
    let text = match form.field("text") {
        Some(Json::Str(text)) => text.clone(),
        Some(_) => return Err(ApiError::bad_request("'text' must be a string")),
        None => return Err(ApiError::bad_request("missing field 'text' (string)")),
    };
    let codec = form.field("codec").and_then(|c| c.as_str()).filter(|c| !c.is_empty());
    let decoded = decode_text(&text, codec).map_err(ApiError::bad_request)?;
    let mut doc = decoded.to_json();
    extend(&mut doc, vec![("wav_base64", Json::str(base64::encode(&decoded.wav)))]);
    Ok(doc)
}

/// `POST /api/speech/tutor`: ask the network to say back an utterance it was
/// taught (sent as a recording, or already encoded as `texts`), mark what
/// comes back, and with `blame` teach the negative network why it failed.
fn r_tutor(svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::read(r, Some("speech"))?;
    let mut texts: Vec<String> = form
        .texts("texts", "text")?
        .into_iter()
        .filter(|t| !t.trim_matches(is_python_space).is_empty())
        .collect();
    let mut labels: Vec<String> = (1..=texts.len()).map(|i| format!("text {i}")).collect();
    let mut said = vec![String::new(); texts.len()];
    if texts.is_empty() {
        let (name, data) = form.bytes("recording")?;
        let mut options = teach_options_from(&form)?;
        options.waveform = true;
        options.pair = false;
        let taught = teach(&data, &options).map_err(ApiError::bad_request)?;
        let waveform = taught
            .waveform()
            .cloned()
            .ok_or_else(|| ApiError::bad_request("nothing to remember: the recording produced no waveform"))?;
        texts = vec![waveform];
        labels = vec![format!("{name} {}", taught.token)];
        said = vec![taught.transcript];
    }
    crate::recall::quiz_route(svc, &form, texts, labels, said, "speech")
}

/// This area's routes:
/// `GET /api/speech`
/// `POST /api/speech/transcribe`
/// `POST /api/speech/teach`
/// `POST /api/speech/decode`
/// `POST /api/speech/tutor`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/speech", r_speech);
    server.route("POST", "/api/speech/transcribe", r_transcribe);
    server.route("POST", "/api/speech/teach", r_teach);
    server.route("POST", "/api/speech/decode", r_decode);
    server.route("POST", "/api/speech/tutor", r_tutor);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `amplitude * sin(2 pi f t)`, stored as `array("f")` stores it.
    pub(crate) fn tone(seconds: f64, rate: i64, freq: f64, amplitude: f64) -> Vec<f32> {
        let n = (rate as f64 * seconds) as usize;
        (0..n)
            .map(|i| (amplitude * (2.0 * std::f64::consts::PI * freq * i as f64 / rate as f64).sin()) as f32)
            .collect()
    }

    #[test]
    fn a_waveform_text_packs_and_parses() {
        let payload = [0u8, 64, 128, 192, 255];
        let text = pack_text("mu", 8000, 1, &payload);
        assert!(text.starts_with("aud:mu:8000x1:"));
        let parsed = parse_text(&text).unwrap();
        assert_eq!((parsed.codec.as_str(), parsed.rate, parsed.channels), ("mu", 8000, 1));
        assert_eq!(
            (parsed.payload.as_slice(), parsed.repaired),
            (payload.as_slice(), false)
        );
        // behind a token, and before a transcript
        for spoken in [
            format!("<speech:9f2a1c7d> {text}"),
            format!("<speech:9f2a1c7d> {text} the cat sat"),
        ] {
            let parsed = parse_text(&spoken).unwrap();
            assert_eq!(parsed.payload.len(), 5, "{spoken}");
        }
        assert!(parse_text("the cat sat on the mat").is_err());
        assert!(parse_text("aud:mu:0x1:AAAA").is_err(), "a zero rate");
        // a false start before the real header is skipped, as re.search skips it
        assert_eq!(parse_text("aud:Mu aud:pcm8:4000x2:AAAA").unwrap().codec, "pcm8");
    }

    #[test]
    fn the_token_is_unique_stable_and_pythons() {
        let first = utterance_token("aud:mu:8000x1:AAAA", true);
        assert_eq!(first, utterance_token("aud:mu:8000x1:AAAA", true));
        assert_ne!(first, utterance_token("aud:mu:8000x1:AAAB", true));
        assert!(first.starts_with("<speech:") && first.ends_with('>'));
        assert_eq!(utterance_token("anything", false), SPEECH_TOKEN);
        // hashlib.blake2b(b"example", digest_size=4).hexdigest()
        assert_eq!(utterance_token("example", true), "<speech:0da00195>");
        assert!(check_token("  ").is_err());
        assert!(check_token("two words").is_err());
        assert_eq!(check_token(" <speech:abc> ").unwrap(), "<speech:abc>");
    }

    #[test]
    fn the_texts_of_an_utterance() {
        let texts = speech_texts("<speech:a>", "the cat  sat", "aud:mu:8000x1:AAAA", false).unwrap();
        assert_eq!(texts, vec!["<speech:a> the cat sat", "<speech:a> aud:mu:8000x1:AAAA"]);
        let paired = speech_texts("<speech:a>", "the cat sat", "aud:mu:8000x1:AAAA", true).unwrap();
        assert_eq!(paired[2], "<speech:a> aud:mu:8000x1:AAAA the cat sat");
        assert_eq!(
            speech_texts("<speech:a>", "", "aud:mu:8000x1:AAAA", true)
                .unwrap()
                .len(),
            1
        );
    }

    #[test]
    fn the_codecs_are_stable_and_mu_law_is_kind_to_a_whisper() {
        for codec in [Codec::Mu, Codec::Pcm8] {
            let samples = tone(0.01, 8000, 220.0, 0.6);
            let once = codec.encode(&samples);
            assert_eq!(codec.decode(&once).len(), samples.len());
            assert_eq!(
                codec.encode(&codec.decode(&once)),
                once,
                "{codec:?} is not a stable code"
            );
        }
        let quiet = tone(0.01, 8000, 220.0, 0.02);
        let err = |codec: Codec| -> f64 {
            let back = codec.decode(&codec.encode(&quiet));
            quiet.iter().zip(&back).map(|(a, b)| (a - b).abs() as f64).sum()
        };
        assert!(err(Codec::Mu) < err(Codec::Pcm8));
        assert_eq!(get_codec("auto").unwrap(), Codec::Mu);
        assert!(get_codec("flac").is_err());
        // the ends of the byte range
        assert_eq!(Codec::Mu.encode(&[1.0, -1.0, 0.0, 2.0]), vec![255, 0, 128, 255]);
        assert_eq!(Codec::Pcm8.encode(&[1.0, -1.0, 0.5, -0.5]), vec![127, 129, 64, 192]);
        assert_eq!(Codec::Pcm8.decode(&[128]), vec![-1.0]);
    }

    #[test]
    fn a_wav_round_trips() {
        let samples = tone(0.05, 16000, 440.0, 0.5);
        let audio = parse_wav(&wav_bytes(&samples, 16000, 1)).unwrap();
        assert_eq!(
            (audio.rate, audio.channels, audio.samples.len()),
            (16000, 1, samples.len())
        );
        for (a, b) in audio.samples.iter().zip(&samples) {
            assert!((a - b).abs() < 1e-3);
        }
        assert!((audio.seconds() - 0.05).abs() < 1e-9);
        assert!(parse_wav(b"not a wav file at all").is_err());
        assert!(parse_wav(b"RIFF____WAVE").is_err());
    }

    #[test]
    fn every_sample_format_python_reads_is_read() {
        let wav = |format: u16, bits: u16, body: &[u8]| {
            let mut out = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
            out.extend_from_slice(&16u32.to_le_bytes());
            out.extend_from_slice(&format.to_le_bytes());
            out.extend_from_slice(&1u16.to_le_bytes());
            out.extend_from_slice(&8000u32.to_le_bytes());
            out.extend_from_slice(&[0; 6]);
            out.extend_from_slice(&bits.to_le_bytes());
            out.extend_from_slice(b"data");
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
            out
        };
        let read = |format, bits, body: &[u8]| parse_wav(&wav(format, bits, body)).unwrap().samples;
        assert_eq!(read(WAVE_PCM, 8, &[0, 128, 255]), vec![-1.0, 0.0, 0.9921875]);
        assert_eq!(
            read(WAVE_PCM, 24, &[0, 0, 0x80, 0xff, 0xff, 0x7f]),
            vec![-1.0, (8_388_607.0 / 8_388_608.0) as f32]
        );
        assert_eq!(read(WAVE_PCM, 32, &i32::MIN.to_le_bytes()), vec![-1.0]);
        let mut floats = 2.0f32.to_le_bytes().to_vec();
        floats.extend_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            read(WAVE_FLOAT, 32, &floats),
            vec![1.0, 1.0],
            "NaN reads as 1, as min() reads it"
        );
        assert_eq!(read(WAVE_FLOAT, 64, &(-0.25f64).to_le_bytes()), vec![-0.25]);
        // G.711: the telephony curves, as Python expands them
        assert_eq!(
            read(WAVE_MULAW, 8, &[0x00, 0xff, 0x80]),
            vec![-32124.0 / 32768.0, 0.0, 32124.0 / 32768.0]
        );
        assert_eq!(read(WAVE_ALAW, 8, &[0xd5, 0x55]), vec![-8.0 / 32768.0, 8.0 / 32768.0]);
        assert!(parse_wav(&wav(WAVE_PCM, 12, &[0, 0])).is_err());
        assert!(parse_wav(&wav(0x55, 8, &[0])).is_err());
        assert!(parse_wav(&wav(WAVE_PCM, 16, &[])).is_err(), "no audio data");
    }

    #[test]
    fn mono_resampling_and_normalising() {
        assert_eq!(to_mono(&[1.0, -1.0, 0.5, -0.5], 2), vec![0.0, 0.0]);
        assert_eq!(to_mono(&[1.0, -1.0], 1).len(), 2);
        assert_eq!(resample(&[0.0, 1.0], 1, 2).unwrap(), vec![0.0, 0.5, 1.0, 1.0]);
        assert_eq!(resample(&[0.0, 1.0], 8000, 8000).unwrap().len(), 2);
        let loud = normalise_samples(&[0.1, -0.05], 0.99);
        assert!((loud[0] - 0.99).abs() < 1e-6);
        assert_eq!(normalise_samples(&[0.0, 0.0], 0.99), vec![0.0, 0.0]);
    }

    #[test]
    fn a_recording_encodes_and_decodes_back() {
        let wav = wav_bytes(&tone(0.05, 16000, 440.0, 0.5), 16000, 1);
        let encoded = encode_audio(&wav, 8000, "auto", false).unwrap();
        assert_eq!((encoded.codec, encoded.rate, encoded.samples), (Codec::Mu, 8000, 400));
        let doc = encoded.to_json();
        assert_eq!(doc.at("source").at("rate").as_i64(), Some(16000));
        assert_eq!(doc.at("source").at("frames").as_i64(), Some(800));
        assert_eq!(doc.at("seconds").as_f64(), Some(0.05));
        let decoded = decode_text(&encoded.text, None).unwrap();
        assert_eq!((decoded.samples, decoded.repaired), (400, false));
        assert_eq!(parse_wav(&decoded.wav).unwrap().samples.len(), 400);
        let cut = decode_text(&encoded.text[..encoded.text.len() - 25], None).unwrap();
        assert!(cut.repaired);
        assert!(encode_audio(&wav, 10, "auto", false)
            .unwrap_err()
            .contains("sample rate"));
        assert!(encode_audio(&wav, 8000, "flac", false).is_err());
        assert!(decode_text("hello", None).is_err());
    }

    #[test]
    fn teaching_an_utterance() {
        let wav = wav_bytes(&tone(0.02, 8000, 220.0, 0.5), 8000, 1);
        let o = TeachOptions {
            transcript: "the cat sat on the mat".into(),
            ..Default::default()
        };
        let taught = teach(&wav, &o).unwrap();
        assert_eq!(taught.texts.len(), 2);
        assert!(taught
            .texts
            .iter()
            .all(|t| t.starts_with(&format!("{} ", taught.token))));
        assert_eq!(taught.asr.backend.as_deref(), Some("given"));
        let paired = teach(
            &wav,
            &TeachOptions {
                pair: true,
                ..o.clone()
            },
        )
        .unwrap();
        assert_eq!(paired.texts.len(), 3);
        assert!(paired.pair);
        let shared = teach(
            &wav,
            &TeachOptions {
                unique: false,
                ..o.clone()
            },
        )
        .unwrap();
        assert_eq!(shared.token, SPEECH_TOKEN);
        // the words alone, the waveform alone (the failed transcription is recorded), and nothing
        let words = teach(&[], &o).unwrap();
        assert_eq!((words.texts.len(), words.audio.is_none()), (1, true));
        let sound = teach(&wav, &TeachOptions::default()).unwrap();
        assert_eq!(sound.texts.len(), 1);
        assert!(sound.asr.error.is_some());
        assert!(teach(&[], &TeachOptions::default()).is_err());
        let doc = taught.to_json();
        assert_eq!(
            doc.at("chars").as_i64(),
            Some(taught.texts.iter().map(|t| t.len() as i64).sum())
        );
    }

    #[test]
    fn python_rounds_the_way_it_rounds() {
        assert_eq!(py_round(0.00125, 4), 0.0013);
        assert_eq!(py_round(0.00375, 4), 0.0037);
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(squash("  the\tcat \x1c sat\n"), "the cat sat");
    }
}
