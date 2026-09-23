//! Images as text: a thumbnail encoder, the text format and its decoder
//! (`radixnet/vision.py`, `go/radixnet/vision.go`).
//!
//! Stable Diffusion generates an image by decoding a latent - 4 x H/8 x W/8
//! numbers - with its VAE, and Python runs that backwards: an image becomes
//! its latent, one signed byte per number, base64:
//!
//! ```text
//! img:sd:128x128:AAECAwQF...
//! ```
//!
//! That text is what the network trains on and predicts.  The VAE needs
//! torch and the diffusion weights, so what is here is Go's encoder, `tiny`:
//! an RGB thumbnail at 1/8 of the size (3 channels, one byte each), the same
//! 8x spatial reduction in the same text format.  `auto` resolves to it.
//!
//! # Where this follows Go rather than Python
//!
//! The *pixel reduction*.  Python's thumbnail goes through Pillow's Lanczos
//! filter and Go's through a box filter, because reproducing Pillow's
//! coefficients without Pillow to check against would be a guess; so the same
//! picture gives two different - equally valid - texts on the two sides.  This
//! port is Go's twin here: the same box filter over the same pixels, which is
//! why the decoders ([`png`], [`jpeg`], [`gif`]) are Go's decoders translated,
//! down to the way Go reads a pixel (`At(x, y).RGBA() >> 8`, premultiplied by
//! alpha).  The same file gives the same text here as in Go.  Likewise a text
//! is decoded back to a picture by Go's nearest-neighbour upscale, not
//! Pillow's bicubic one.  Everything else is shared by all three: each side
//! parses, trains on, predicts and decodes the others' texts, and an `sd` text
//! can be parsed, measured and marked here - it just cannot be drawn without
//! the weights.

pub mod gif;
pub mod jpeg;
pub mod png;

use std::sync::Arc;

use crate::cli::Ctx;
use crate::http::{Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::multipart::base64::{self, is_python_space};
use crate::multipart::Form;
use crate::negative::python_repr;
use crate::service::Service;
use crate::speech::{base_name, extend, text_or_data, train_on};

/// The header every image text starts with.
pub const HEADER: &str = "img";

/// The encoders this side has; `auto` is `tiny` here, because `sd` needs the
/// diffusion weights and therefore the Python side.
pub const ENCODERS: &[&str] = &["auto", "tiny"];

/// The square an image is resized to before encoding (a multiple of 8; 128 is
/// a 3 x 16 x 16 thumbnail).
pub const DEFAULT_SIZE: i64 = 128;

/// The sizes an image may be encoded at.
pub const MIN_SIZE: i64 = 8;
pub const MAX_SIZE: i64 = 1024;

/// The largest side a text is drawn at: a predicted header can claim any
/// size, and drawing it is memory the server would give away.
pub const MAX_DECODE_SIDE: i64 = 8192;

const SD_NEEDS: &str = "the Stable Diffusion encoder needs torch and diffusers: use the Python side for it, or \
                        --encoder tiny here";

// -- the picture a decoder hands back ---------------------------------------------------------

/// A decoded image as Go's `image.Image` presents it: its bounds and, for
/// every pixel, the 8-bit RGB that `At(x, y).RGBA() >> 8` reads.
#[derive(Clone, Debug, PartialEq)]
pub struct Picture {
    /// The top-left corner (a GIF frame may start inside its screen).
    pub min_x: i64,
    pub min_y: i64,
    pub width: usize,
    pub height: usize,
    /// Row by row, three bytes a pixel.
    pub rgb: Vec<u8>,
    /// What a read outside the bounds gives: a paletted image's first colour,
    /// black for the rest.
    pub outside: [u8; 3],
}

impl Picture {
    pub fn new(min_x: i64, min_y: i64, width: usize, height: usize, outside: [u8; 3]) -> Picture {
        Picture {
            min_x,
            min_y,
            width,
            height,
            rgb: vec![0; width * height * 3],
            outside,
        }
    }

    /// Sets the pixel at (`x`, `y`) from the top-left corner.
    pub fn set(&mut self, x: usize, y: usize, rgb: [u8; 3]) {
        if x < self.width && y < self.height {
            let at = (y * self.width + x) * 3;
            self.rgb[at..at + 3].copy_from_slice(&rgb);
        }
    }

    /// The pixel at absolute (`x`, `y`), as `At` reads it.
    fn at(&self, x: i64, y: i64) -> [u8; 3] {
        let (dx, dy) = (x - self.min_x, y - self.min_y);
        if dx < 0 || dy < 0 || dx as usize >= self.width || dy as usize >= self.height {
            return self.outside;
        }
        let at = (dy as usize * self.width + dx as usize) * 3;
        [self.rgb[at], self.rgb[at + 1], self.rgb[at + 2]]
    }
}

/// Reads image bytes the way Go's `image.Decode` does: the format is decided
/// by the first bytes, and PNG, JPEG and GIF are read.
pub fn load_image(data: &[u8]) -> Result<Picture, String> {
    let decoded = if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        png::decode(data)
    } else if data.starts_with(b"\xff\xd8") {
        jpeg::decode(data)
    } else if data.len() >= 6 && data.starts_with(b"GIF8") && data[5] == b'a' {
        gif::decode(data)
    } else {
        Err("image: unknown format".to_string())
    };
    decoded.map_err(|err| format!("not a readable image: {err}"))
}

/// Box-filters a picture down to `width` x `height` and returns it as
/// interleaved RGB (Go's `resampleRGB`): each target cell is the truncated
/// mean of the source pixels under it, so the same image always gives the
/// same text.
pub fn resample_rgb(src: &Picture, width: usize, height: usize) -> Vec<u8> {
    let (sw, sh) = (src.width, src.height);
    let mut out = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        let y0 = src.min_y + (y * sh / height) as i64;
        let mut y1 = src.min_y + ((y + 1) * sh / height) as i64;
        if y1 <= y0 {
            y1 = y0 + 1;
        }
        for x in 0..width {
            let x0 = src.min_x + (x * sw / width) as i64;
            let mut x1 = src.min_x + ((x + 1) * sw / width) as i64;
            if x1 <= x0 {
                x1 = x0 + 1;
            }
            let (mut r, mut g, mut b, mut n) = (0u64, 0u64, 0u64, 0u64);
            for py in y0..y1 {
                for px in x0..x1 {
                    let [pr, pg, pb] = src.at(px, py);
                    r += pr as u64;
                    g += pg as u64;
                    b += pb as u64;
                    n += 1;
                }
            }
            let n = n.max(1);
            out.extend_from_slice(&[(r / n) as u8, (g / n) as u8, (b / n) as u8]);
        }
    }
    out
}

// -- the text format --------------------------------------------------------------------------

/// `img:<encoder>:<w>x<h>:<base64 of the quantised latent>`.
pub fn pack_text(encoder: &str, width: i64, height: i64, payload: &[u8]) -> String {
    format!("{HEADER}:{encoder}:{width}x{height}:{}", base64::encode(payload))
}

/// An image text, taken apart.
#[derive(Clone, Debug, PartialEq)]
pub struct ParsedImage {
    pub encoder: String,
    pub width: i64,
    pub height: i64,
    pub payload: Vec<u8>,
    /// Whether the base64 had to be repaired to be read.
    pub repaired: bool,
}

/// Reads an encoded - or predicted - image text (`vision.parse_text`): the
/// header must open the text, and everything after it is the payload, whose
/// base64 is repaired when it is not clean.  The payload is not padded here:
/// the decoder fits it to the latent it needs.
pub fn parse_text(text: &str) -> Result<ParsedImage, String> {
    let bad = || "not an encoded image: expected 'img:<encoder>:<w>x<h>:<base64>'".to_string();
    let rest = text
        .trim_start_matches(is_python_space)
        .strip_prefix("img:")
        .ok_or_else(bad)?;
    let encoder_len = rest
        .bytes()
        .take_while(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || *c == b'_')
        .count();
    if encoder_len == 0 || rest.as_bytes().get(encoder_len) != Some(&b':') {
        return Err(bad());
    }
    let encoder = &rest[..encoder_len];
    let dims = &rest[encoder_len + 1..];
    let width_len = dims.bytes().take_while(u8::is_ascii_digit).count();
    if width_len == 0 || dims.as_bytes().get(width_len) != Some(&b'x') {
        return Err(bad());
    }
    let tail = &dims[width_len + 1..];
    let height_len = tail.bytes().take_while(u8::is_ascii_digit).count();
    if height_len == 0 || tail.as_bytes().get(height_len) != Some(&b':') {
        return Err(bad());
    }
    let number = |digits: &str| {
        digits
            .bytes()
            .fold(0i64, |n, d| n.saturating_mul(10).saturating_add((d - b'0') as i64))
    };
    let (width, height) = (number(&dims[..width_len]), number(&tail[..height_len]));
    let (payload, repaired) = base64::repair(&tail[height_len + 1..])?;
    if width <= 0 || height <= 0 {
        return Err(format!("invalid image size {width}x{height}"));
    }
    Ok(ParsedImage {
        encoder: encoder.to_string(),
        width,
        height,
        payload,
        repaired,
    })
}

/// A size an image may be encoded at.
pub fn check_size(size: i64) -> Result<i64, String> {
    if !(MIN_SIZE..=MAX_SIZE).contains(&size) || size % 8 != 0 {
        return Err(format!(
            "size must be a multiple of 8 between {MIN_SIZE} and {MAX_SIZE}, got {size}"
        ));
    }
    Ok(size)
}

/// The encoder `name` names here: `auto` (or nothing) is `tiny`, and `sd` says
/// where to go for it.
pub fn normalise_encoder(name: &str) -> Result<&'static str, String> {
    match name.trim_matches(is_python_space).to_lowercase().as_str() {
        "" | "auto" | "tiny" => Ok("tiny"),
        "sd" => Err(SD_NEEDS.to_string()),
        _ => Err(format!(
            "unknown image encoder {}; expected one of: {}",
            python_repr(name),
            ENCODERS.join(", ")
        )),
    }
}

/// An image as text, with what went into it.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedImage {
    pub text: String,
    pub encoder: String,
    pub width: i64,
    pub height: i64,
    pub latent_shape: [i64; 3],
    pub bytes: usize,
    pub source_size: [usize; 2],
}

impl EncodedImage {
    /// `{"text", "encoder", "width", "height", "latent_shape", "bytes",
    /// "chars", "source_size"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("text", Json::str(self.text.clone())),
            ("encoder", Json::str(self.encoder.clone())),
            ("width", Json::Int(self.width)),
            ("height", Json::Int(self.height)),
            ("latent_shape", Json::ints(self.latent_shape)),
            ("bytes", Json::Int(self.bytes as i64)),
            ("chars", Json::Int(self.text.chars().count() as i64)),
            ("source_size", Json::ints(self.source_size.iter().map(|&s| s as i64))),
        ])
    }
}

/// Image bytes (PNG, JPEG, GIF) -> the text the network trains on: resized to
/// `size` x `size`, reduced to a 1/8 RGB thumbnail, one byte per channel
/// value, base64 (`vision.encode_image`).
pub fn encode_image(data: &[u8], size: i64, encoder: &str) -> Result<EncodedImage, String> {
    let size = check_size(size)?;
    let picture = load_image(data)?;
    let name = normalise_encoder(encoder)?;
    let small = (size / 8) as usize;
    let payload = resample_rgb(&picture, small, small);
    Ok(EncodedImage {
        text: pack_text(name, size, size, &payload),
        encoder: name.to_string(),
        width: size,
        height: size,
        latent_shape: [3, small as i64, small as i64],
        bytes: payload.len(),
        source_size: [picture.width, picture.height],
    })
}

/// A text drawn back as a picture.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedImage {
    pub png: Vec<u8>,
    pub encoder: String,
    pub width: i64,
    pub height: i64,
    pub bytes: usize,
    pub repaired: bool,
}

impl DecodedImage {
    /// `{"encoder", "width", "height", "bytes", "repaired"}` (the PNG goes
    /// wherever the caller sends it).
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("encoder", Json::str(self.encoder.clone())),
            ("width", Json::Int(self.width)),
            ("height", Json::Int(self.height)),
            ("bytes", Json::Int(self.bytes as i64)),
            ("repaired", Json::Bool(self.repaired)),
        ])
    }
}

/// Pads with zeros or truncates to `length`: `(payload, changed)`.
fn fit_payload(mut payload: Vec<u8>, length: usize) -> (Vec<u8>, bool) {
    let changed = payload.len() != length;
    payload.resize(length, 0);
    (payload, changed)
}

/// An encoded - or predicted - text back into a PNG (`vision.decode_text`).
///
/// A payload that is too short or too long (a cut-off or rambling prediction)
/// is padded with zeros or truncated, and `repaired` says so; `encoder`
/// overrides the one in the header.
pub fn decode_text(text: &str, encoder: Option<&str>) -> Result<DecodedImage, String> {
    let parsed = parse_text(text)?;
    let chosen = encoder.filter(|e| !e.is_empty()).unwrap_or(&parsed.encoder);
    let name = chosen.trim_matches(is_python_space).to_lowercase();
    match name.as_str() {
        "tiny" => {}
        "sd" => return Err(SD_NEEDS.to_string()),
        _ => {
            return Err(format!(
                "unknown image encoder {} in the text; expected 'sd' or 'tiny'",
                python_repr(&name)
            ))
        }
    }
    let (width, height) = (parsed.width, parsed.height);
    if width % 8 != 0 || height % 8 != 0 {
        return Err(format!("the image size {width}x{height} is not a multiple of 8"));
    }
    if width > MAX_DECODE_SIDE || height > MAX_DECODE_SIDE {
        return Err(format!(
            "the image size {width}x{height} is larger than this decoder draws ({MAX_DECODE_SIDE}x{MAX_DECODE_SIDE})"
        ));
    }
    let (sw, sh) = ((width / 8) as usize, (height / 8) as usize);
    let length = 3 * sw * sh;
    let (payload, changed) = fit_payload(parsed.payload, length);
    // the thumbnail upscaled by nearest neighbour, as Go draws it
    let (w, h) = (width as usize, height as usize);
    let mut rgb = vec![0u8; w * h * 3];
    for y in 0..h {
        let sy = y * sh / h;
        for x in 0..w {
            let sx = x * sw / w;
            let from = (sy * sw + sx) * 3;
            rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&payload[from..from + 3]);
        }
    }
    Ok(DecodedImage {
        png: png::encode(w, h, &rgb),
        encoder: name,
        width,
        height,
        bytes: length,
        repaired: parsed.repaired || changed,
    })
}

/// What this side can do with images, in the keys the Images tab reads
/// (Go's `DescribeVision`).
pub fn describe() -> Json {
    Json::obj([
        ("engine", Json::str(crate::service::ENGINE)),
        ("encoders", Json::strs(ENCODERS.iter().copied())),
        ("auto", Json::str("tiny")),
        ("default_size", Json::Int(DEFAULT_SIZE)),
        // this build decodes PNG / JPEG / GIF itself, and has no diffusion weights to load
        ("pillow", Json::Bool(true)),
        ("torch", Json::Bool(false)),
        ("diffusers", Json::Bool(false)),
        ("sd_model", Json::str("")),
        ("sd_loaded", Json::Bool(false)),
        (
            "sd_error",
            Json::str("the Stable Diffusion encoder needs torch and diffusers: use the Python server for it"),
        ),
        ("sd", Json::Bool(false)),
        ("resampling", Json::str("box")),
        (
            "resampling_note",
            Json::str(
                "Python's thumbnail uses Pillow's Lanczos filter and this one (like Go's) a box filter, so the same \
                 image gives a different (equally valid) text on the Python side and the same one as Go's; the \
                 format itself is shared, and every side reads, trains on and decodes the others' texts",
            ),
        ),
        (
            "sd_note",
            Json::str(
                "the Stable Diffusion encoder needs torch and diffusers: use the Python server for it (the text \
                 format is the same, so a model trained on sd texts by any side is read by all three)",
            ),
        ),
        (
            "text_format",
            Json::str(format!(
                "{HEADER}:<encoder>:<w>x<h>:<base64 of one byte per latent number>"
            )),
        ),
        ("formats", Json::str("PNG, JPEG and GIF")),
    ])
}

// -- the command line -------------------------------------------------------------------------

/// An image file's bytes, or the error Python gives for a missing one.
fn read_image_file(path: &str) -> Result<Vec<u8>, String> {
    if !std::path::Path::new(path).is_file() {
        return Err(format!("image file not found: {path}"));
    }
    std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))
}

/// `--encoder`, checked against Python's choices.
fn encoder_flag(ctx: &Ctx) -> Result<String, String> {
    let encoder = ctx.args.str("encoder", "auto");
    if !["auto", "sd", "tiny"].contains(&encoder.as_str()) {
        return Err(format!("--encoder must be one of auto, sd, tiny, got {encoder:?}"));
    }
    Ok(encoder)
}

/// `radixnet image <action>`: info, encode, tutor, decode.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let a = &ctx.args;
    let doc = match a.action() {
        "info" => describe(),
        "encode" => {
            let path = a
                .rest
                .get(1)
                .cloned()
                .ok_or("an image file is required: radixnet image encode FILE")?;
            let encoded = encode_image(
                &read_image_file(&path)?,
                a.int("size", DEFAULT_SIZE)?,
                &encoder_flag(ctx)?,
            )?;
            let mut doc = encoded.to_json();
            extend(
                &mut doc,
                vec![("file", Json::str(path)), ("out", Json::Null), ("trained", Json::Null)],
            );
            if let Some(out) = a.get("out") {
                std::fs::write(out, format!("{}\n", encoded.text))
                    .map_err(|err| format!("cannot write {out}: {err}"))?;
                extend(&mut doc, vec![("out", Json::str(out))]);
            }
            if a.on("train") {
                let trained = train_on(ctx, std::slice::from_ref(&encoded.text))?;
                extend(&mut doc, vec![("trained", trained)]);
            }
            doc
        }
        "tutor" => {
            let paths: Vec<String> = a.rest.iter().skip(1).cloned().collect();
            if paths.is_empty() {
                return Err("at least one image file is required: radixnet image tutor FILE...".to_string());
            }
            let (size, encoder) = (a.int("size", DEFAULT_SIZE)?, encoder_flag(ctx)?);
            let mut texts = Vec::new();
            for path in &paths {
                texts.push(encode_image(&read_image_file(path)?, size, &encoder)?.text);
            }
            let labels: Vec<String> = paths.iter().map(|p| base_name(p)).collect();
            crate::recall::tutor_cli(ctx, &texts, &labels, "image", &[])?
        }
        "decode" => {
            let text = text_or_data(ctx)?;
            let out = a.get("out").ok_or("--out is required: where to write the PNG")?;
            let decoded = decode_text(&text, a.get("encoder"))?;
            std::fs::write(out, &decoded.png).map_err(|err| format!("cannot write {out}: {err}"))?;
            let mut doc = decoded.to_json();
            extend(&mut doc, vec![("out", Json::str(out))]);
            doc
        }
        "" => return Err("image needs an action: info, encode, tutor, decode".to_string()),
        other => {
            return Err(format!(
                "unknown image action {other:?}; expected info, encode, tutor or decode"
            ))
        }
    };
    ctx.emit(doc);
    Ok(())
}

// -- the routes -------------------------------------------------------------------------------

/// `size` and `encoder` of a request (`_r_image_encode`).
fn size_and_encoder(form: &Form) -> Result<(i64, String), ApiError> {
    let size = form
        .int("size", 0)
        .map_err(|_| ApiError::bad_request("'size' must be an integer"))?;
    let size = if size == 0 { DEFAULT_SIZE } else { size };
    Ok((size, form.text("encoder", "auto")))
}

/// `GET /api/images`.
fn r_images(_svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(describe())
}

/// `POST /api/images/encode`: an image (multipart, raw or `content_base64`)
/// -> its text; `save_as` keeps it as an upload, `train` learns it (202).
fn r_encode(svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::read(r, Some("image"))?;
    let (name, data) = form.bytes("image")?;
    let (size, encoder) = size_and_encoder(&form)?;
    if form.flag("train", false) {
        svc.ensure_idle()?;
    }
    let encoded = encode_image(&data, size, &encoder).map_err(ApiError::bad_request)?;
    let mut doc = encoded.to_json();
    extend(
        &mut doc,
        vec![("name", Json::str(name)), ("upload", Json::Null), ("job", Json::Null)],
    );
    crate::multipart::save_and_train(svc, &form, doc, std::slice::from_ref(&encoded.text))
}

/// `POST /api/images/decode`: `{text, encoder}` -> `{png_base64, ...}`.
fn r_decode(_svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::json(r)?;
    let text = match form.field("text") {
        Some(Json::Str(text)) => text.clone(),
        Some(_) => return Err(ApiError::bad_request("'text' must be a string")),
        None => return Err(ApiError::bad_request("missing field 'text' (string)")),
    };
    let encoder = form.field("encoder").and_then(|e| e.as_str()).filter(|e| !e.is_empty());
    let decoded = decode_text(&text, encoder).map_err(ApiError::bad_request)?;
    let mut doc = decoded.to_json();
    extend(&mut doc, vec![("png_base64", Json::str(base64::encode(&decoded.png)))]);
    Ok(doc)
}

/// `POST /api/images/tutor`: ask the network to draw back an image it was
/// shown (sent as a picture, or already encoded as `texts`), mark what comes
/// back, and with `blame` teach the negative network why it failed.
fn r_tutor(svc: &Arc<Service>, r: &Request) -> Answer {
    let form = Form::read(r, Some("image"))?;
    let mut texts: Vec<String> = form
        .texts("texts", "text")?
        .into_iter()
        .filter(|t| !t.trim_matches(is_python_space).is_empty())
        .collect();
    let mut labels: Vec<String> = (1..=texts.len()).map(|i| format!("text {i}")).collect();
    if texts.is_empty() {
        let (name, data) = form.bytes("image")?;
        let (size, encoder) = size_and_encoder(&form)?;
        texts = vec![encode_image(&data, size, &encoder).map_err(ApiError::bad_request)?.text];
        labels = vec![name];
    }
    let said = vec![String::new(); texts.len()];
    crate::recall::quiz_route(svc, &form, texts, labels, said, "image")
}

/// This area's routes:
/// `GET /api/images`
/// `POST /api/images/encode`
/// `POST /api/images/decode`
/// `POST /api/images/tutor`
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/images", r_images);
    server.route("POST", "/api/images/encode", r_encode);
    server.route("POST", "/api/images/decode", r_decode);
    server.route("POST", "/api/images/tutor", r_tutor);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The parity suite's `_gradient_png`: RGB, built with this crate's own writer.
    pub(crate) fn gradient_png(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, 128]);
            }
        }
        png::encode(width, height, &rgb)
    }

    #[test]
    fn an_image_text_packs_and_parses() {
        let payload = [0u8, 1, 2, 3, 250, 251];
        let text = pack_text("tiny", 32, 32, &payload);
        assert!(text.starts_with("img:tiny:32x32:"));
        let parsed = parse_text(&text).unwrap();
        assert_eq!(
            (parsed.encoder.as_str(), parsed.width, parsed.height, parsed.repaired),
            ("tiny", 32, 32, false)
        );
        assert_eq!(parsed.payload, payload);
        let cut = parse_text(&text[..text.len() - 3]).unwrap();
        assert!(cut.repaired && !cut.payload.is_empty(), "a predicted tail is repaired");
        assert!(parse_text("the cat sat on the mat").is_err());
        assert!(parse_text("img:tiny:0x0:AAAA").is_err());
        assert!(parse_text("x img:tiny:8x8:AAAA").is_err(), "the header opens the text");
        assert_eq!(parse_text("\n img:sd:64x64:AAAA\n").unwrap().encoder, "sd");
    }

    #[test]
    fn an_image_encodes_and_decodes_back() {
        let encoded = encode_image(&gradient_png(64, 40), 64, "auto").unwrap();
        assert_eq!(
            (encoded.encoder.as_str(), encoded.width, encoded.height),
            ("tiny", 64, 64)
        );
        assert_eq!(encoded.bytes, 3 * 8 * 8);
        assert_eq!(encoded.source_size, [64, 40]);
        assert_eq!(
            encoded,
            encode_image(&gradient_png(64, 40), 64, "tiny").unwrap(),
            "deterministic"
        );
        let decoded = decode_text(&encoded.text, None).unwrap();
        assert_eq!((decoded.width, decoded.height, decoded.repaired), (64, 64, false));
        let back = png::decode(&decoded.png).unwrap();
        assert_eq!((back.width, back.height), (64, 64));
        // every 8x8 block of the drawing is one thumbnail pixel
        let payload = parse_text(&encoded.text).unwrap().payload;
        assert_eq!(&back.rgb[..3], &payload[..3]);
        assert_eq!(
            &back.rgb[(9 * 64 + 17) * 3..(9 * 64 + 17) * 3 + 3],
            &payload[(8 + 2) * 3..(8 + 2) * 3 + 3]
        );
    }

    #[test]
    fn bad_input_is_refused() {
        let png = gradient_png(32, 32);
        assert!(encode_image(&png, 30, "tiny").is_err());
        assert!(encode_image(&png, 64, "sd").unwrap_err().contains("Python"));
        assert!(encode_image(&png, 64, "sideways").is_err());
        assert!(encode_image(b"not an image", 64, "tiny")
            .unwrap_err()
            .starts_with("not a readable image"));
        assert!(decode_text(&pack_text("sd", 64, 64, &[1, 2, 3]), None)
            .unwrap_err()
            .contains("Python"));
        assert!(decode_text(&pack_text("tiny", 60, 64, &[1]), None).is_err());
        assert!(decode_text(&pack_text("tiny", 1 << 20, 8, &[1]), None).is_err());
    }

    #[test]
    fn a_short_or_long_payload_is_fitted() {
        let short = decode_text(&pack_text("tiny", 64, 64, &[1, 2, 3]), None).unwrap();
        assert!(short.repaired);
        assert_eq!(short.bytes, 3 * 8 * 8);
        let long = decode_text(&pack_text("tiny", 64, 64, &[0; 3 * 8 * 8 + 50]), None).unwrap();
        assert!(long.repaired);
    }

    #[test]
    fn the_box_filter_is_gos() {
        // a 3x1 picture averaged to one pixel truncates, as Go's integer mean does
        let mut picture = Picture::new(0, 0, 3, 1, [0, 0, 0]);
        picture.set(0, 0, [1, 10, 255]);
        picture.set(1, 0, [2, 10, 255]);
        picture.set(2, 0, [2, 11, 254]);
        assert_eq!(resample_rgb(&picture, 1, 1), vec![1, 10, 254]);
        // upscaling repeats pixels rather than reading outside the picture
        assert_eq!(resample_rgb(&picture, 6, 1).len(), 18);
        // an empty frame reads its outside colour
        let empty = Picture::new(5, 5, 0, 0, [9, 8, 7]);
        assert_eq!(resample_rgb(&empty, 1, 1), vec![9, 8, 7]);
    }

    #[test]
    fn describe_names_the_encoder() {
        let info = describe();
        assert_eq!(info.at("auto").as_str(), Some("tiny"));
        assert_eq!(info.at("sd").as_bool(), Some(false));
    }
}
