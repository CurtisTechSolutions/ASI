//! GIF, written out: the first frame, as Go's `image/gif` decodes it.
//!
//! A GIF is a palette and LZW-compressed indices, cut into sub-blocks; this
//! is Go's reader translated - the header and colour tables, the extensions
//! it skips, the graphic control block's transparent index (a transparent
//! colour reads as black, as Go's `color.RGBA{}` does), the LZW decoder
//! (`compress/lzw`, least significant bit first), the leniency Go allows at
//! the end of the data, and interlacing.  Like `gif.Decode` it answers the
//! *first* frame, at the place in the logical screen the frame says.

use super::Picture;

const COLOR_TABLE: u8 = 1 << 7;
const INTERLACE: u8 = 1 << 6;

/// The pixels a GIF frame may declare before it is refused rather than decoded.
const MAX_PIXELS: u64 = 1 << 26;

/// The sub-blocks of image data, read a byte at a time (Go's `blockReader`).
struct Blocks<'a> {
    data: &'a [u8],
    at: usize,
    /// The current sub-block's bytes, and how many of them are used.
    block: &'a [u8],
    used: usize,
    /// Set once the terminator (a zero-length block) or the end of the data is reached.
    ended: Option<BlockEnd>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum BlockEnd {
    Terminator,
    Truncated,
}

impl<'a> Blocks<'a> {
    fn fill(&mut self) {
        if self.ended.is_some() {
            return;
        }
        let Some(&len) = self.data.get(self.at) else {
            self.ended = Some(BlockEnd::Truncated);
            return;
        };
        self.at += 1;
        if len == 0 {
            self.ended = Some(BlockEnd::Terminator);
            return;
        }
        match self.data.get(self.at..self.at + len as usize) {
            Some(block) => {
                self.block = block;
                self.used = 0;
                self.at += len as usize;
            }
            None => {
                self.block = &[];
                self.used = 0;
                self.ended = Some(BlockEnd::Truncated);
            }
        }
    }

    fn byte(&mut self) -> Option<u8> {
        if self.used == self.block.len() {
            self.fill();
            if self.ended.is_some() {
                return None;
            }
        }
        let b = self.block[self.used];
        self.used += 1;
        Some(b)
    }

    /// Go's `blockReader.close`: after the LZW data at most one more sub-block
    /// (of one byte, when the data filled its last block) may come before the
    /// terminator.
    fn close(&mut self) -> Result<(), String> {
        match self.ended {
            Some(BlockEnd::Terminator) => return Ok(()),
            Some(BlockEnd::Truncated) => return Err("gif: reading image data: unexpected EOF".to_string()),
            None => {}
        }
        if self.used == self.block.len() {
            self.fill();
            match self.ended {
                Some(BlockEnd::Terminator) => return Ok(()),
                Some(BlockEnd::Truncated) => return Err("gif: reading image data: unexpected EOF".to_string()),
                None if self.block.len() > 1 => return Err("gif: too much image data".to_string()),
                None => {}
            }
        }
        self.fill();
        match self.ended {
            Some(BlockEnd::Terminator) => Ok(()),
            Some(BlockEnd::Truncated) => Err("gif: reading image data: unexpected EOF".to_string()),
            None => Err("gif: too much image data".to_string()),
        }
    }
}

/// Why the LZW decoder stopped.
enum LzwEnd {
    /// The end-of-information code.
    Eof,
    /// The data ran out first (which Go accepts once every pixel is there).
    Truncated,
}

/// `compress/lzw`, least significant bit first: every index the stream holds.
fn lzw(blocks: &mut Blocks, lit_width: u32) -> Result<(Vec<u8>, LzwEnd), String> {
    const MAX_WIDTH: u32 = 12;
    const INVALID: u16 = 0xffff;
    let clear: u16 = 1 << lit_width;
    let eof = clear + 1;
    let mut width = lit_width + 1;
    let mut hi = eof;
    let mut overflow: u16 = 1 << width;
    let mut last = INVALID;
    let mut suffix = [0u8; 1 << MAX_WIDTH];
    let mut prefix = [0u16; 1 << MAX_WIDTH];
    let mut out: Vec<u8> = Vec::new();
    let mut scratch = vec![0u8; 1 << MAX_WIDTH];
    let (mut bits, mut n_bits) = (0u32, 0u32);
    loop {
        while n_bits < width {
            let Some(x) = blocks.byte() else {
                return Ok((out, LzwEnd::Truncated));
            };
            bits |= (x as u32) << n_bits;
            n_bits += 8;
        }
        let code = (bits & ((1 << width) - 1)) as u16;
        bits >>= width;
        n_bits -= width;
        if code < clear {
            out.push(code as u8);
            if last != INVALID {
                suffix[hi as usize] = code as u8;
                prefix[hi as usize] = last;
            }
        } else if code == clear {
            width = lit_width + 1;
            hi = eof;
            overflow = 1 << width;
            last = INVALID;
            continue;
        } else if code == eof {
            return Ok((out, LzwEnd::Eof));
        } else if code <= hi {
            // the string of `code`, written backwards from the end of the scratch
            let mut i = scratch.len() - 1;
            let mut c = code;
            if code == hi && last != INVALID {
                c = last;
                while c >= clear {
                    c = prefix[c as usize];
                }
                scratch[i] = c as u8;
                i -= 1;
                c = last;
            }
            while c >= clear {
                scratch[i] = suffix[c as usize];
                i -= 1;
                c = prefix[c as usize];
            }
            scratch[i] = c as u8;
            out.extend_from_slice(&scratch[i..]);
            if last != INVALID {
                suffix[hi as usize] = c as u8;
                prefix[hi as usize] = last;
            }
        } else {
            return Err("gif: reading image data: lzw: invalid code".to_string());
        }
        last = code;
        hi += 1;
        if hi >= overflow {
            if width == MAX_WIDTH {
                last = INVALID;
                hi -= 1;
            } else {
                width += 1;
                overflow = 1 << width;
            }
        }
    }
}

/// The colour table that follows a descriptor with the colour-table flag.
fn color_table(data: &[u8], at: &mut usize, fields: u8) -> Result<Vec<[u8; 3]>, String> {
    let n = 1usize << (1 + (fields & 7));
    let bytes = data
        .get(*at..*at + 3 * n)
        .ok_or("gif: reading color table: unexpected EOF")?;
    *at += 3 * n;
    Ok(bytes.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

/// Skips data sub-blocks up to and including the terminator.
fn skip_blocks(data: &[u8], at: &mut usize) -> Result<(), String> {
    loop {
        let n = *data.get(*at).ok_or("gif: reading extension: unexpected EOF")? as usize;
        *at += 1;
        if n == 0 {
            return Ok(());
        }
        if *at + n > data.len() {
            return Err("gif: reading extension: unexpected EOF".to_string());
        }
        *at += n;
    }
}

/// Decodes the first frame of a GIF.
pub fn decode(data: &[u8]) -> Result<Picture, String> {
    let header = data.get(..13).ok_or("gif: reading header: unexpected EOF")?;
    let version = &header[..6];
    if version != b"GIF87a" && version != b"GIF89a" {
        return Err(format!(
            "gif: can't recognize format {}",
            crate::negative::python_repr(&String::from_utf8_lossy(version))
        ));
    }
    let screen_w = header[6] as usize | (header[7] as usize) << 8;
    let screen_h = header[8] as usize | (header[9] as usize) << 8;
    let mut at = 13;
    let global = if header[10] & COLOR_TABLE != 0 {
        Some(color_table(data, &mut at, header[10])?)
    } else {
        None
    };
    let mut transparent: Option<u8> = None;
    loop {
        let kind = *data.get(at).ok_or("gif: reading frames: unexpected EOF")?;
        at += 1;
        match kind {
            0x21 => {
                let extension = *data.get(at).ok_or("gif: reading extension: unexpected EOF")?;
                at += 1;
                match extension {
                    0xf9 => {
                        let block = data
                            .get(at..at + 6)
                            .ok_or("gif: can't read graphic control: unexpected EOF")?;
                        at += 6;
                        if block[0] != 4 {
                            return Err(format!(
                                "gif: invalid graphic control extension block size: {}",
                                block[0]
                            ));
                        }
                        if block[5] != 0 {
                            return Err(format!(
                                "gif: invalid graphic control extension block terminator: {}",
                                block[5]
                            ));
                        }
                        transparent = (block[1] & 1 != 0).then_some(block[4]);
                    }
                    0x01 | 0xfe | 0xff => {
                        let size = match extension {
                            0x01 => 13,
                            0xff => {
                                let b = *data.get(at).ok_or("gif: reading extension: unexpected EOF")? as usize;
                                at += 1;
                                b
                            }
                            _ => 0,
                        };
                        if at + size > data.len() {
                            return Err("gif: reading extension: unexpected EOF".to_string());
                        }
                        at += size;
                        skip_blocks(data, &mut at)?;
                    }
                    other => return Err(format!("gif: unknown extension 0x{other:02x}")),
                }
            }
            0x2c => return frame(data, at, screen_w, screen_h, global.as_deref(), transparent),
            0x3b => return Err("gif: missing image data".to_string()),
            other => return Err(format!("gif: unknown block type: 0x{other:02x}")),
        }
    }
}

/// The frame whose descriptor starts at `at`.
fn frame(
    data: &[u8],
    mut at: usize,
    screen_w: usize,
    screen_h: usize,
    global: Option<&[[u8; 3]]>,
    transparent: Option<u8>,
) -> Result<Picture, String> {
    let d = data
        .get(at..at + 9)
        .ok_or("gif: can't read image descriptor: unexpected EOF")?;
    at += 9;
    let left = d[0] as usize | (d[1] as usize) << 8;
    let top = d[2] as usize | (d[3] as usize) << 8;
    let width = d[4] as usize | (d[5] as usize) << 8;
    let height = d[6] as usize | (d[7] as usize) << 8;
    let fields = d[8];
    if left + width > screen_w || top + height > screen_h {
        return Err("gif: frame bounds larger than image bounds".to_string());
    }
    if width as u64 * height as u64 > MAX_PIXELS {
        return Err("gif: frame larger than this reader decodes".to_string());
    }
    let mut palette: Vec<[u8; 3]> = if fields & COLOR_TABLE != 0 {
        color_table(data, &mut at, fields)?
    } else {
        global.ok_or("gif: no color table")?.to_vec()
    };
    // the transparent colour is color.RGBA{}; an index past the table grows it, as browsers allow
    if let Some(t) = transparent {
        let t = t as usize;
        if t >= palette.len() {
            palette.resize(t + 1, [0, 0, 0]);
        }
        palette[t] = [0, 0, 0];
    }
    let lit_width = *data.get(at).ok_or("gif: reading image data: unexpected EOF")? as u32;
    at += 1;
    if !(2..=8).contains(&lit_width) {
        return Err(format!("gif: pixel size in decode out of range: {lit_width}"));
    }
    let mut blocks = Blocks {
        data,
        at,
        block: &[],
        used: 0,
        ended: None,
    };
    let (pixels, _end) = lzw(&mut blocks, lit_width)?;
    let need = width * height;
    if pixels.len() < need {
        return Err("gif: not enough image data".to_string());
    }
    if pixels.len() > need {
        return Err("gif: too much image data".to_string());
    }
    blocks.close()?;
    if palette.len() < 256 && pixels.iter().any(|&p| p as usize >= palette.len()) {
        return Err("gif: invalid pixel value".to_string());
    }
    let order: Vec<usize> = if fields & INTERLACE != 0 {
        // rows 0, 8, 16, ...; then 4, 12, ...; then 2, 6, ...; then 1, 3, ...
        [(8, 0), (8, 4), (4, 2), (2, 1)]
            .iter()
            .flat_map(|&(skip, start)| (start..height).step_by(skip))
            .collect()
    } else {
        (0..height).collect()
    };
    let outside = palette.first().copied().unwrap_or([0, 0, 0]);
    let mut picture = Picture::new(left as i64, top as i64, width, height, outside);
    for (source_row, &y) in order.iter().enumerate() {
        for x in 0..width {
            picture.set(x, y, palette[pixels[source_row * width + x] as usize]);
        }
    }
    Ok(picture)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 2x2 GIF: a red, green, blue, white palette and one pixel of each,
    /// LZW-coded by hand (clear, 0, 1, 2, 3, end, at 3 bits then 4).
    fn tiny() -> Vec<u8> {
        let mut gif = b"GIF89a\x02\x00\x02\x00\x81\x00\x00".to_vec();
        gif.extend_from_slice(&[255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
        gif.extend_from_slice(b"\x2c\x00\x00\x00\x00\x02\x00\x02\x00\x00\x02");
        // codes LSB first: 4 (clear, 3 bits), 0, 1 (3 bits), 2, 3, 5 (4 bits from hi = 8)
        let codes: [(u32, u32); 6] = [(4, 3), (0, 3), (1, 3), (2, 3), (3, 4), (5, 4)];
        let (mut acc, mut n, mut bytes) = (0u32, 0u32, Vec::new());
        for (code, width) in codes {
            acc |= code << n;
            n += width;
            while n >= 8 {
                bytes.push(acc as u8);
                acc >>= 8;
                n -= 8;
            }
        }
        if n > 0 {
            bytes.push(acc as u8);
        }
        gif.push(bytes.len() as u8);
        gif.extend_from_slice(&bytes);
        gif.extend_from_slice(&[0, 0x3b]);
        gif
    }

    #[test]
    fn a_hand_made_gif_decodes() {
        let picture = decode(&tiny()).unwrap();
        assert_eq!((picture.width, picture.height), (2, 2));
        assert_eq!(picture.rgb, vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255]);
    }

    #[test]
    fn rubbish_and_truncation_are_refused() {
        assert!(decode(b"GIF90a........").is_err());
        let gif = tiny();
        assert!(decode(&gif[..gif.len() - 6]).is_err());
    }
}
