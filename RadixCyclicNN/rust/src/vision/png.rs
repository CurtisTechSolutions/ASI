//! PNG, written out: the decoder Go's `image/png` is, and an encoder for the
//! pictures [`super::decode_text`] draws.
//!
//! **Reading** is Go's reader, translated: every colour type and bit depth
//! the format has (grey 1/2/4/8/16, grey + alpha, RGB, RGBA, palette), `tRNS`
//! transparency, Adam7 interlacing, the five row filters, and the chunk-order
//! rules and checksums Go enforces.  Each pixel is then read the way Go's
//! `At(x, y).RGBA() >> 8` reads it - alpha-premultiplied, so a transparent
//! pixel is black - because that is the value Go's thumbnail averages, and the
//! texts the two sides write must be the same bytes.  The zlib stream is
//! inflated by [`crate::gzip::inflate`].
//!
//! **Writing** emits 8-bit RGB with a fixed-Huffman deflate block that only
//! ever copies the byte before (run-length coding): a decoded thumbnail is a
//! grid of flat 8x8 blocks, so rows that repeat are filtered to zeros and a
//! run of zeros costs a few bits, and the result is a fraction of the size of
//! stored blocks for no real compressor.

use super::Picture;
use crate::gzip::{crc32, inflate};

const SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// The pixels a PNG may hold before it is refused rather than decoded:
/// 2^27 (a 11585 x 11585 square), well past any photograph, and short of the
/// memory a decompression bomb would ask for.
const MAX_PIXELS: u64 = 1 << 27;

/// Adam7: (x step, y step, x offset, y offset) of each of the seven passes.
const ADAM7: [(usize, usize, usize, usize); 7] = [
    (8, 8, 0, 0),
    (8, 8, 4, 0),
    (4, 8, 0, 4),
    (4, 4, 2, 0),
    (2, 4, 0, 2),
    (2, 2, 1, 0),
    (1, 2, 0, 1),
];

/// A PNG error, worded as Go's `png` package words it.
fn invalid(what: &str) -> String {
    format!("png: invalid format: {what}")
}

fn unsupported(what: &str) -> String {
    format!("png: unsupported feature: {what}")
}

/// What the header says the pixels are.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Gray(u8),
    GrayAlpha(u8),
    Rgb(u8),
    Rgba(u8),
    Paletted(u8),
}

impl Layout {
    fn from(depth: u8, color: u8) -> Option<Layout> {
        Some(match (color, depth) {
            (0, 1 | 2 | 4 | 8 | 16) => Layout::Gray(depth),
            (2, 8 | 16) => Layout::Rgb(depth),
            (3, 1 | 2 | 4 | 8) => Layout::Paletted(depth),
            (4, 8 | 16) => Layout::GrayAlpha(depth),
            (6, 8 | 16) => Layout::Rgba(depth),
            _ => return None,
        })
    }

    fn bits_per_pixel(self) -> usize {
        match self {
            Layout::Gray(d) | Layout::Paletted(d) => d as usize,
            Layout::GrayAlpha(d) => 2 * d as usize,
            Layout::Rgb(d) => 3 * d as usize,
            Layout::Rgba(d) => 4 * d as usize,
        }
    }

    fn paletted(self) -> bool {
        matches!(self, Layout::Paletted(_))
    }

    fn true_color(self) -> bool {
        matches!(self, Layout::Rgb(_))
    }
}

/// `color.NRGBA{r, g, b, a}.RGBA() >> 8`: 8-bit, non-premultiplied.
fn nrgba8(v: u8, a: u8) -> u8 {
    ((v as u32 * 0x101 * a as u32 / 0xff) >> 8) as u8
}

/// `color.NRGBA64{r, g, b, a}.RGBA() >> 8`: 16-bit, non-premultiplied.
fn nrgba16(v: u16, a: u16) -> u8 {
    ((v as u32 * a as u32 / 0xffff) >> 8) as u8
}

/// A palette entry as Go holds it: opaque `RGBA`, or `NRGBA` once `tRNS` gave it an alpha.
#[derive(Clone, Copy)]
struct Entry {
    rgb: [u8; 3],
    alpha: Option<u8>,
}

impl Entry {
    fn read(self) -> [u8; 3] {
        match self.alpha {
            None => self.rgb,
            Some(a) => [nrgba8(self.rgb[0], a), nrgba8(self.rgb[1], a), nrgba8(self.rgb[2], a)],
        }
    }
}

struct Header {
    width: usize,
    height: usize,
    layout: Layout,
    interlaced: bool,
    palette: Vec<Entry>,
    /// The `tRNS` key of a grey or RGB image: the samples that are transparent.
    transparent: Option<[u16; 3]>,
}

/// Decodes a PNG into what Go's reader would hand its caller.
pub fn decode(data: &[u8]) -> Result<Picture, String> {
    if data.len() < 8 || &data[..8] != SIGNATURE {
        return Err(invalid("not a PNG file"));
    }
    let mut at = 8;
    let mut header: Option<Header> = None;
    let mut picture: Option<Picture> = None;
    // the stages of Go's decoder: start, IHDR, PLTE, tRNS, IDAT, IEND
    let mut stage = 0u8;
    loop {
        let (kind, body, next) = chunk(data, at)?;
        at = next;
        match &kind {
            b"IHDR" => {
                if stage != 0 {
                    return Err(invalid("chunk out of order"));
                }
                stage = 1;
                header = Some(parse_ihdr(body)?);
            }
            b"PLTE" => {
                if stage != 1 {
                    return Err(invalid("chunk out of order"));
                }
                stage = 2;
                let h = header.as_mut().ok_or_else(|| invalid("chunk out of order"))?;
                parse_plte(h, body)?;
            }
            b"tRNS" => {
                let h = header.as_mut().ok_or_else(|| invalid("chunk out of order"))?;
                let allowed = if h.layout.paletted() {
                    stage == 2
                } else if h.layout.true_color() {
                    stage == 1 || stage == 2
                } else {
                    stage == 1
                };
                if !allowed {
                    return Err(invalid("chunk out of order"));
                }
                stage = 3;
                parse_trns(h, body)?;
            }
            b"IDAT" => {
                let h = header.as_ref().ok_or_else(|| invalid("chunk out of order"))?;
                if stage > 4 || (stage == 1 && h.layout.paletted()) {
                    return Err(invalid("chunk out of order"));
                }
                if stage == 4 {
                    continue; // trailing zero-length or garbage IDAT chunks are ignored, as Go ignores them
                }
                stage = 4;
                // the zlib stream runs on through the IDAT chunks that follow this one
                let mut stream = body.to_vec();
                let mut boundaries = vec![stream.len()];
                while let Ok((next_kind, more, after)) = chunk(data, at) {
                    if &next_kind != b"IDAT" {
                        break;
                    }
                    stream.extend_from_slice(more);
                    boundaries.push(stream.len());
                    at = after;
                }
                picture = Some(decode_pixels(h, &stream, &boundaries)?);
            }
            b"IEND" => {
                if stage != 4 {
                    return Err(invalid("chunk out of order"));
                }
                if !body.is_empty() {
                    return Err(invalid("bad IEND length"));
                }
                break;
            }
            _ => {} // ancillary chunks are skipped, their checksum checked
        }
    }
    picture.ok_or_else(|| invalid("not enough pixel data"))
}

/// The chunk at `at`: its type, its body (checksum verified) and where the next starts.
fn chunk(data: &[u8], at: usize) -> Result<([u8; 4], &[u8], usize), String> {
    let head = data.get(at..at + 8).ok_or("unexpected EOF")?;
    let length = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
    if length > 0x7fff_ffff {
        return Err(invalid(&format!("Bad chunk length: {length}")));
    }
    let kind = [head[4], head[5], head[6], head[7]];
    let end = at + 8 + length;
    let body = data.get(at + 8..end).ok_or("unexpected EOF")?;
    let crc = data.get(end..end + 4).ok_or("unexpected EOF")?;
    let mut covered = Vec::with_capacity(length + 4);
    covered.extend_from_slice(&kind);
    covered.extend_from_slice(body);
    if u32::from_be_bytes([crc[0], crc[1], crc[2], crc[3]]) != crc32(&covered) {
        return Err(invalid("invalid checksum"));
    }
    Ok((kind, body, end + 4))
}

fn parse_ihdr(body: &[u8]) -> Result<Header, String> {
    if body.len() != 13 {
        return Err(invalid("bad IHDR length"));
    }
    if body[10] != 0 {
        return Err(unsupported("compression method"));
    }
    if body[11] != 0 {
        return Err(unsupported("filter method"));
    }
    if body[12] > 1 {
        return Err(invalid("invalid interlace method"));
    }
    let width = i32::from_be_bytes([body[0], body[1], body[2], body[3]]);
    let height = i32::from_be_bytes([body[4], body[5], body[6], body[7]]);
    if width <= 0 || height <= 0 {
        return Err(invalid("non-positive dimension"));
    }
    if width as u64 * height as u64 > MAX_PIXELS {
        return Err(unsupported(&format!(
            "{width}x{height} is more pixels than this reader decodes"
        )));
    }
    let layout = Layout::from(body[8], body[9])
        .ok_or_else(|| unsupported(&format!("bit depth {}, color type {}", body[8], body[9])))?;
    Ok(Header {
        width: width as usize,
        height: height as usize,
        layout,
        interlaced: body[12] == 1,
        palette: Vec::new(),
        transparent: None,
    })
}

fn parse_plte(h: &mut Header, body: &[u8]) -> Result<(), String> {
    let entries = body.len() / 3;
    let depth = match h.layout {
        Layout::Paletted(d) | Layout::Gray(d) | Layout::GrayAlpha(d) | Layout::Rgb(d) | Layout::Rgba(d) => d,
    };
    if body.len() % 3 != 0 || entries == 0 || entries > 256 || entries > 1 << depth.min(8) {
        return Err(invalid("bad PLTE length"));
    }
    match h.layout {
        Layout::Paletted(_) => {
            h.palette = body
                .chunks_exact(3)
                .map(|c| Entry {
                    rgb: [c[0], c[1], c[2]],
                    alpha: None,
                })
                .collect();
            Ok(())
        }
        // optional, and ignorable, for the true-colour types
        Layout::Rgb(_) | Layout::Rgba(_) => Ok(()),
        _ => Err(invalid("PLTE, color type mismatch")),
    }
}

fn parse_trns(h: &mut Header, body: &[u8]) -> Result<(), String> {
    let word = |i: usize| u16::from_be_bytes([body[i], body[i + 1]]);
    match h.layout {
        Layout::Gray(depth) => {
            if body.len() != 2 {
                return Err(invalid("bad tRNS length"));
            }
            // Go scales the low byte of a low-depth key up to 8 bits (and lets it wrap)
            let scale: u8 = match depth {
                1 => 0xff,
                2 => 0x55,
                4 => 0x11,
                _ => 1,
            };
            let key = if depth < 8 {
                body[1].wrapping_mul(scale) as u16
            } else if depth == 8 {
                body[1] as u16
            } else {
                word(0)
            };
            h.transparent = Some([key, key, key]);
        }
        Layout::Rgb(depth) => {
            if body.len() != 6 {
                return Err(invalid("bad tRNS length"));
            }
            h.transparent = Some(if depth == 8 {
                [body[1] as u16, body[3] as u16, body[5] as u16]
            } else {
                [word(0), word(2), word(4)]
            });
        }
        Layout::Paletted(_) => {
            if body.len() > 256 {
                return Err(invalid("bad tRNS length"));
            }
            // a key longer than the palette reaches the opaque black Go pads it with
            while h.palette.len() < body.len() {
                h.palette.push(Entry {
                    rgb: [0, 0, 0],
                    alpha: None,
                });
            }
            for (entry, &alpha) in h.palette.iter_mut().zip(body) {
                entry.alpha = Some(alpha);
            }
        }
        _ => return Err(invalid("tRNS, color type mismatch")),
    }
    Ok(())
}

/// The zlib stream of the IDAT chunks -> the picture.
fn decode_pixels(h: &Header, stream: &[u8], boundaries: &[usize]) -> Result<Picture, String> {
    if stream.len() < 2 {
        return Err("unexpected EOF".to_string());
    }
    let (cmf, flg) = (stream[0], stream[1]);
    if cmf & 0x0f != 8 || cmf >> 4 > 7 || (((cmf as u16) << 8) | flg as u16) % 31 != 0 {
        return Err("zlib: invalid header".to_string());
    }
    if flg & 0x20 != 0 {
        return Err("zlib: invalid dictionary".to_string());
    }
    let raw = inflate(&stream[2..]).map_err(|err| invalid(&format!("flate: {err}")))?;
    // the Adler-32 closes the stream, and Go wants it to close at the end of an
    // IDAT chunk: the IDAT chunks after that one are ignored
    let adler = adler32(&raw).to_be_bytes();
    if !boundaries.iter().any(|&b| b >= 6 && stream[b - 4..b] == adler) {
        return Err(invalid("zlib: invalid checksum"));
    }
    let mut picture = Picture::new(0, 0, h.width, h.height, [0, 0, 0]);
    let passes: Vec<(usize, usize, usize, usize)> = if h.interlaced {
        ADAM7.to_vec()
    } else {
        vec![(1, 1, 0, 0)]
    };
    let mut offset = 0;
    for (xs, ys, xo, yo) in passes {
        // a pass that falls outside a small image has no rows at all, not even filter bytes
        let width = (h.width + xs - 1).saturating_sub(xo) / xs;
        let height = (h.height + ys - 1).saturating_sub(yo) / ys;
        if width == 0 || height == 0 {
            continue;
        }
        let bpp = h.layout.bits_per_pixel();
        let row = (bpp * width).div_ceil(8);
        let step = bpp.div_ceil(8);
        let mut previous = vec![0u8; row];
        for y in 0..height {
            let Some(line) = raw.get(offset..offset + 1 + row) else {
                return Err(invalid("not enough pixel data"));
            };
            offset += 1 + row;
            let mut current = line[1..].to_vec();
            unfilter(line[0], &mut current, &previous, step)?;
            for x in 0..width {
                let rgb = pixel(h, &current, x);
                picture.set(x * xs + xo, y * ys + yo, rgb);
            }
            previous = current;
        }
    }
    if offset != raw.len() {
        return Err(invalid("too much pixel data"));
    }
    Ok(picture)
}

/// Undoes one row's filter in place.
fn unfilter(kind: u8, row: &mut [u8], previous: &[u8], step: usize) -> Result<(), String> {
    match kind {
        0 => {}
        1 => {
            for i in step..row.len() {
                row[i] = row[i].wrapping_add(row[i - step]);
            }
        }
        2 => {
            for (b, &p) in row.iter_mut().zip(previous) {
                *b = b.wrapping_add(p);
            }
        }
        3 => {
            for i in 0..row.len() {
                let left = if i >= step { row[i - step] as u16 } else { 0 };
                row[i] = row[i].wrapping_add(((left + previous[i] as u16) / 2) as u8);
            }
        }
        4 => {
            for i in 0..row.len() {
                let a = if i >= step { row[i - step] as i16 } else { 0 };
                let b = previous[i] as i16;
                let c = if i >= step { previous[i - step] as i16 } else { 0 };
                let p = a + b - c;
                let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                let predictor = if pa <= pb && pa <= pc {
                    a
                } else if pb <= pc {
                    b
                } else {
                    c
                };
                row[i] = row[i].wrapping_add(predictor as u8);
            }
        }
        _ => return Err(invalid("bad filter type")),
    }
    Ok(())
}

/// Pixel `x` of an unfiltered row, as Go's `At(x, y).RGBA() >> 8` reads it.
fn pixel(h: &Header, row: &[u8], x: usize) -> [u8; 3] {
    let word = |i: usize| u16::from_be_bytes([row[i], row[i + 1]]);
    match h.layout {
        Layout::Gray(depth) if depth < 8 => {
            let per_byte = 8 / depth as usize;
            let shift = 8 - depth as usize * (x % per_byte + 1);
            let level = (row[x / per_byte] >> shift) & ((1u8 << depth) - 1);
            let scale: u8 = match depth {
                1 => 0xff,
                2 => 0x55,
                _ => 0x11,
            };
            let y = level * scale;
            match h.transparent {
                Some(key) if y as u16 == key[0] => [0, 0, 0],
                _ => [y, y, y],
            }
        }
        Layout::Gray(8) => {
            let y = row[x];
            match h.transparent {
                Some(key) if y as u16 == key[0] => [0, 0, 0],
                _ => [y, y, y],
            }
        }
        Layout::Gray(_) => {
            let y = word(2 * x);
            match h.transparent {
                Some(key) if y == key[0] => [0, 0, 0],
                _ => {
                    let v = (y >> 8) as u8;
                    [v, v, v]
                }
            }
        }
        Layout::GrayAlpha(8) => {
            let v = nrgba8(row[2 * x], row[2 * x + 1]);
            [v, v, v]
        }
        Layout::GrayAlpha(_) => {
            let v = nrgba16(word(4 * x), word(4 * x + 2));
            [v, v, v]
        }
        Layout::Rgb(8) => {
            let rgb = [row[3 * x], row[3 * x + 1], row[3 * x + 2]];
            match h.transparent {
                Some(key) if key == [rgb[0] as u16, rgb[1] as u16, rgb[2] as u16] => [0, 0, 0],
                _ => rgb,
            }
        }
        Layout::Rgb(_) => {
            let rgb = [word(6 * x), word(6 * x + 2), word(6 * x + 4)];
            match h.transparent {
                Some(key) if key == rgb => [0, 0, 0],
                _ => [(rgb[0] >> 8) as u8, (rgb[1] >> 8) as u8, (rgb[2] >> 8) as u8],
            }
        }
        Layout::Rgba(8) => {
            let a = row[4 * x + 3];
            [
                nrgba8(row[4 * x], a),
                nrgba8(row[4 * x + 1], a),
                nrgba8(row[4 * x + 2], a),
            ]
        }
        Layout::Rgba(_) => {
            let a = word(8 * x + 6);
            [
                nrgba16(word(8 * x), a),
                nrgba16(word(8 * x + 2), a),
                nrgba16(word(8 * x + 4), a),
            ]
        }
        Layout::Paletted(depth) => {
            let per_byte = 8 / depth as usize;
            let shift = 8 - depth as usize * (x % per_byte + 1);
            let index = ((row[x / per_byte] >> shift) & (((1u16 << depth) - 1) as u8)) as usize;
            // an index past the palette is opaque black, as it is to Go and libpng
            h.palette.get(index).map(|e| e.read()).unwrap_or([0, 0, 0])
        }
    }
}

/// The Adler-32 of `data`, zlib's checksum.
pub fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for chunk in data.chunks(5552) {
        for &byte in chunk {
            a += byte as u32;
            b += a;
        }
        a %= 65521;
        b %= 65521;
    }
    (b << 16) | a
}

// -- writing ----------------------------------------------------------------------------------

/// Bits into bytes, least significant first, as deflate packs them.
struct BitWriter {
    out: Vec<u8>,
    held: u64,
    count: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        self.held |= (value as u64) << self.count;
        self.count += bits;
        while self.count >= 8 {
            self.out.push(self.held as u8);
            self.held >>= 8;
            self.count -= 8;
        }
    }

    /// A Huffman code, which deflate stores most significant bit first.
    fn code(&mut self, code: u32, bits: u32) {
        self.put(code.reverse_bits() >> (32 - bits), bits);
    }

    fn literal(&mut self, symbol: u32) {
        match symbol {
            0..=143 => self.code(0x30 + symbol, 8),
            144..=255 => self.code(0x190 + symbol - 144, 9),
            256..=279 => self.code(symbol - 256, 7),
            _ => self.code(0xc0 + symbol - 280, 8),
        }
    }

    fn finish(mut self) -> Vec<u8> {
        if self.count > 0 {
            self.out.push(self.held as u8);
        }
        self.out
    }
}

const LENGTH_BASE: [u32; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];

/// One fixed-Huffman deflate block whose only matches are runs of the byte
/// before (distance 1).
pub fn deflate_runs(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter {
        out: Vec::with_capacity(data.len() / 8 + 16),
        held: 0,
        count: 0,
    };
    w.put(1, 1); // the final block
    w.put(1, 2); // fixed Huffman codes
    let mut i = 0;
    while i < data.len() {
        let run = if i > 0 {
            data[i..].iter().take(258).take_while(|&&b| b == data[i - 1]).count()
        } else {
            0
        };
        if run >= 3 {
            let code = LENGTH_BASE.iter().rposition(|&base| base <= run as u32).unwrap_or(0);
            w.literal(257 + code as u32);
            w.put(run as u32 - LENGTH_BASE[code], LENGTH_EXTRA[code]);
            w.code(0, 5); // distance 1
            i += run;
        } else {
            w.literal(data[i] as u32);
            i += 1;
        }
    }
    w.literal(256);
    w.finish()
}

fn write_chunk(out: &mut Vec<u8>, kind: &[u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    let start = out.len();
    out.extend_from_slice(kind);
    out.extend_from_slice(body);
    let crc = crc32(&out[start..]);
    out.extend_from_slice(&crc.to_be_bytes());
}

/// An 8-bit RGB PNG of `width` x `height` pixels (`rgb` row by row).
///
/// A row that repeats the one above is filtered `Up` (all zeros), any other
/// row `Sub` (zeros wherever a colour carries on): both are what the
/// run-length block compresses.
pub fn encode(width: usize, height: usize, rgb: &[u8]) -> Vec<u8> {
    let stride = width * 3;
    let mut filtered = Vec::with_capacity(height * (stride + 1));
    for y in 0..height {
        let row = &rgb[y * stride..(y + 1) * stride];
        if y > 0 && row == &rgb[(y - 1) * stride..y * stride] {
            filtered.push(2);
            filtered.extend(std::iter::repeat_n(0u8, stride));
        } else {
            filtered.push(1);
            for i in 0..stride {
                let left = if i >= 3 { row[i - 3] } else { 0 };
                filtered.push(row[i].wrapping_sub(left));
            }
        }
    }
    let mut zlib = vec![0x78, 0x01];
    zlib.extend_from_slice(&deflate_runs(&filtered));
    zlib.extend_from_slice(&adler32(&filtered).to_be_bytes());
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&(width as u32).to_be_bytes());
    ihdr.extend_from_slice(&(height as u32).to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB, deflate, no filter method, not interlaced
    let mut out = SIGNATURE.to_vec();
    write_chunk(&mut out, b"IHDR", &ihdr);
    write_chunk(&mut out, b"IDAT", &zlib);
    write_chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(width: usize, height: usize) -> Vec<u8> {
        let mut rgb = Vec::new();
        for y in 0..height {
            for x in 0..width {
                rgb.extend_from_slice(&[(x * 255 / width) as u8, (y * 255 / height) as u8, 128]);
            }
        }
        rgb
    }

    #[test]
    fn what_it_writes_it_reads_back() {
        for (w, h) in [(1, 1), (7, 3), (64, 40), (130, 9)] {
            let rgb = gradient(w, h);
            let png = encode(w, h, &rgb);
            let back = decode(&png).unwrap();
            assert_eq!((back.width, back.height), (w, h));
            assert_eq!(back.rgb, rgb, "{w}x{h}");
        }
    }

    #[test]
    fn a_blocky_picture_compresses() {
        // what decode_text draws: flat 8x8 blocks
        let (w, h) = (128, 128);
        let mut rgb = Vec::new();
        for y in 0..h {
            for x in 0..w {
                rgb.extend_from_slice(&[(x / 8 * 16) as u8, (y / 8 * 16) as u8, 7]);
            }
        }
        let png = encode(w, h, &rgb);
        assert!(png.len() < rgb.len() / 20, "{} bytes for {} pixels", png.len(), w * h);
        assert_eq!(decode(&png).unwrap().rgb, rgb);
    }

    #[test]
    fn a_run_longer_than_a_match_is_split() {
        let data: Vec<u8> = std::iter::once(9).chain(std::iter::repeat_n(0, 1000)).collect();
        assert_eq!(inflate(&deflate_runs(&data)).unwrap(), data);
        assert_eq!(inflate(&deflate_runs(&[])).unwrap(), Vec::<u8>::new());
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }

    #[test]
    fn damage_is_refused() {
        let png = encode(4, 4, &gradient(4, 4));
        assert!(decode(b"GIF89a").is_err());
        let mut bad = png.clone();
        bad[20] ^= 1; // inside IHDR: the checksum no longer matches
        assert!(decode(&bad).is_err());
        assert!(decode(&png[..png.len() - 12]).is_err(), "no IEND");
    }
}
