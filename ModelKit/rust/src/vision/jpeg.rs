//! JPEG, written out: Go's `image/jpeg` decoder, translated.
//!
//! Go reads baseline and progressive JPEGs - grey, YCbCr at every chroma
//! subsampling it supports (4:4:4, 4:4:0, 4:2:2, 4:2:0, 4:1:1, 4:1:0), RGB,
//! CMYK and YCbCrK - and the thumbnail it averages is made of the values its
//! decoder produces.  A JPEG decoder is not one algorithm but a family of
//! them (the inverse DCT, the chroma upsampling and the colour conversion all
//! vary between libraries), so this is Go's own, operation for operation: the
//! same integer IDCT, chroma read at the nearest subsampled position the way
//! `image.YCbCr.At` reads it, and the same fixed-point YCbCr, CMYK and YCCK
//! conversions.  That is what makes a photograph encode to the same text here
//! as in Go.
//!
//! Go's arithmetic is 32-bit and wraps; it is kept 32-bit and wrapping here
//! (the IDCT computes in 64 bits and wraps before every shift, which is the
//! same thing), so a corrupt file decodes to the same garbage rather than to a
//! panic.

use super::Picture;

const BLOCK: usize = 64;
type Block = [i32; BLOCK];

/// Zig-zag order to natural order.
const UNZIG: [usize; BLOCK] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54,
    47, 55, 62, 63,
];

const SOF0: u8 = 0xc0;
const SOF1: u8 = 0xc1;
const SOF2: u8 = 0xc2;
const DHT: u8 = 0xc4;
const RST0: u8 = 0xd0;
const RST7: u8 = 0xd7;
const SOI: u8 = 0xd8;
const EOI: u8 = 0xd9;
const SOS: u8 = 0xda;
const DQT: u8 = 0xdb;
const DRI: u8 = 0xdd;
const COM: u8 = 0xfe;
const APP0: u8 = 0xe0;
const APP14: u8 = 0xee;
const APP15: u8 = 0xef;

const MAX_TH: u8 = 3;
const MAX_TQ: u8 = 3;

/// The pixels a JPEG may declare before it is refused rather than decoded.
const MAX_PIXELS: u64 = 1 << 27;

fn format_error(what: &str) -> Error {
    Error::Other(format!("invalid JPEG format: {what}"))
}

fn unsupported(what: &str) -> Error {
    Error::Other(format!("unsupported JPEG feature: {what}"))
}

/// The errors whose identity the decoder acts on, and the rest.
#[derive(Debug, PartialEq)]
enum Error {
    /// The data ran out.
    Eof,
    /// An `0xff` in entropy-coded data that was not `0xff 0x00`: a marker.
    MissingFF00,
    /// The data ran out inside Huffman-coded data.
    ShortHuffman,
    Other(String),
}

impl Error {
    fn message(self) -> String {
        match self {
            Error::Eof => "unexpected EOF".to_string(),
            Error::MissingFF00 => "invalid JPEG format: missing 0xff00 sequence".to_string(),
            Error::ShortHuffman => "invalid JPEG format: short Huffman data".to_string(),
            Error::Other(message) => message,
        }
    }
}

type Result<T> = std::result::Result<T, Error>;

#[derive(Clone, Copy, Default)]
struct Component {
    h: usize,
    v: usize,
    c: u8,
    tq: u8,
}

#[derive(Clone)]
struct Huffman {
    n_codes: i32,
    /// The next 8 bits -> the value (high byte) and 1 + the code length (low byte).
    lut: [u16; 256],
    vals: [u8; 256],
    min_codes: [i32; 16],
    max_codes: [i32; 16],
    vals_indices: [i32; 16],
}

impl Default for Huffman {
    fn default() -> Huffman {
        Huffman {
            n_codes: 0,
            lut: [0; 256],
            vals: [0; 256],
            min_codes: [0; 16],
            max_codes: [0; 16],
            vals_indices: [0; 16],
        }
    }
}

/// The unread bits: the `n` low bits of `a`, read most significant first.
#[derive(Clone, Copy, Default)]
struct Bits {
    a: u32,
    m: u32,
    n: i32,
}

/// Chroma subsampling, as `image.YCbCrSubsampleRatio` names it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Ratio {
    R444,
    R440,
    R422,
    R420,
    R411,
    R410,
}

/// The decoded planes of a colour JPEG (`image.YCbCr`, with its origin at 0, 0).
struct Planes {
    y: Vec<u8>,
    cb: Vec<u8>,
    cr: Vec<u8>,
    y_stride: usize,
    c_stride: usize,
    ratio: Ratio,
}

impl Planes {
    /// `image.YCbCr.COffset`.
    fn c_offset(&self, x: usize, y: usize) -> usize {
        match self.ratio {
            Ratio::R422 => y * self.c_stride + x / 2,
            Ratio::R420 => (y / 2) * self.c_stride + x / 2,
            Ratio::R440 => (y / 2) * self.c_stride + x,
            Ratio::R411 => y * self.c_stride + x / 4,
            Ratio::R410 => (y / 2) * self.c_stride + x / 4,
            Ratio::R444 => y * self.c_stride + x,
        }
    }
}

struct Decoder<'a> {
    data: &'a [u8],
    at: usize,
    /// How far `at` backs up after an overshoot: 0, 1 or 2.
    n_unreadable: usize,
    bits: Bits,
    width: usize,
    height: usize,
    gray: Option<(Vec<u8>, usize)>,
    planes: Option<Planes>,
    black: Vec<u8>,
    black_stride: usize,
    ri: usize,
    n_comp: usize,
    baseline: bool,
    progressive: bool,
    jfif: bool,
    adobe_valid: bool,
    adobe_transform: u8,
    eob_run: u16,
    comp: [Component; 4],
    prog_coeffs: [Vec<Block>; 4],
    huff: [[Huffman; 4]; 2],
    quant: [Block; 4],
    tmp: [u8; 2 * BLOCK],
}

/// Decodes a JPEG into what Go's `jpeg.Decode` would hand its caller.
pub fn decode(data: &[u8]) -> std::result::Result<Picture, String> {
    let mut d = Decoder {
        data,
        at: 0,
        n_unreadable: 0,
        bits: Bits::default(),
        width: 0,
        height: 0,
        gray: None,
        planes: None,
        black: Vec::new(),
        black_stride: 0,
        ri: 0,
        n_comp: 0,
        baseline: false,
        progressive: false,
        jfif: false,
        adobe_valid: false,
        adobe_transform: 0,
        eob_run: 0,
        comp: [Component::default(); 4],
        prog_coeffs: Default::default(),
        huff: Default::default(),
        quant: [[0; BLOCK]; 4],
        tmp: [0; 2 * BLOCK],
    };
    d.decode().map_err(Error::message)
}

impl Decoder<'_> {
    // -- the byte stream ----------------------------------------------------------------------

    fn read_byte(&mut self) -> Result<u8> {
        let byte = *self.data.get(self.at).ok_or(Error::Eof)?;
        self.at += 1;
        self.n_unreadable = 0;
        Ok(byte)
    }

    /// A byte of entropy-coded data, with its `0xff 0x00` stuffing undone.
    fn read_stuffed_byte(&mut self) -> Result<u8> {
        if self.at + 2 <= self.data.len() {
            let x = self.data[self.at];
            self.at += 1;
            self.n_unreadable = 1;
            if x != 0xff {
                return Ok(x);
            }
            if self.data[self.at] != 0x00 {
                return Err(Error::MissingFF00);
            }
            self.at += 1;
            self.n_unreadable = 2;
            return Ok(0xff);
        }
        self.n_unreadable = 0;
        let x = self.read_byte()?;
        self.n_unreadable = 1;
        if x != 0xff {
            return Ok(x);
        }
        let x = self.read_byte()?;
        self.n_unreadable = 2;
        if x != 0x00 {
            return Err(Error::MissingFF00);
        }
        Ok(0xff)
    }

    /// Gives back the byte the Huffman decoder read one too many of.
    fn unread_stuffed_byte(&mut self) {
        self.at -= self.n_unreadable;
        self.n_unreadable = 0;
        if self.bits.n >= 8 {
            self.bits.a >>= 8;
            self.bits.n -= 8;
            self.bits.m >>= 8;
        }
    }

    /// Undoes an overshoot before reading plain bytes.
    fn settle(&mut self) {
        if self.n_unreadable != 0 {
            if self.bits.n >= 8 {
                self.unread_stuffed_byte();
            }
            self.n_unreadable = 0;
        }
    }

    /// The next `n` plain bytes into `tmp`.
    fn read_full(&mut self, n: usize) -> Result<()> {
        self.settle();
        let bytes = self.data.get(self.at..self.at + n).ok_or(Error::Eof)?;
        self.tmp[..n].copy_from_slice(bytes);
        self.at += n;
        Ok(())
    }

    fn ignore(&mut self, n: usize) -> Result<()> {
        self.settle();
        if self.at + n > self.data.len() {
            self.at = self.data.len();
            return Err(Error::Eof);
        }
        self.at += n;
        Ok(())
    }

    // -- the bit stream -----------------------------------------------------------------------

    fn ensure_bits(&mut self, n: i32) -> Result<()> {
        loop {
            let c = match self.read_stuffed_byte() {
                Ok(c) => c,
                Err(Error::Eof) => return Err(Error::ShortHuffman),
                Err(err) => return Err(err),
            };
            self.bits.a = self.bits.a << 8 | c as u32;
            self.bits.n += 8;
            self.bits.m = if self.bits.m == 0 { 1 << 7 } else { self.bits.m << 8 };
            if self.bits.n >= n {
                return Ok(());
            }
        }
    }

    fn receive_extend(&mut self, t: u8) -> Result<i32> {
        if self.bits.n < t as i32 {
            self.ensure_bits(t as i32)?;
        }
        self.bits.n -= t as i32;
        self.bits.m >>= t;
        let s = 1i32 << t;
        let mut x = ((self.bits.a >> self.bits.n) as i32) & (s - 1);
        if x < s >> 1 {
            x += ((-1i32) << t) + 1;
        }
        Ok(x)
    }

    fn decode_huffman(&mut self, class: usize, index: usize) -> Result<u8> {
        if self.huff[class][index].n_codes == 0 {
            return Err(format_error("uninitialized Huffman table"));
        }
        let mut fast = true;
        if self.bits.n < 8 {
            if let Err(err) = self.ensure_bits(8) {
                if err != Error::MissingFF00 && err != Error::ShortHuffman {
                    return Err(err);
                }
                // no more bytes in this segment, but the bits already read may hold the next symbol
                if self.n_unreadable != 0 {
                    self.unread_stuffed_byte();
                }
                fast = false;
            }
        }
        let h = &self.huff[class][index];
        if fast {
            let v = h.lut[((self.bits.a >> (self.bits.n - 8)) & 0xff) as usize];
            if v != 0 {
                let n = (v & 0xff) as i32 - 1;
                self.bits.n -= n;
                self.bits.m >>= n;
                return Ok((v >> 8) as u8);
            }
        }
        let (min_codes, max_codes, vals_indices) = (h.min_codes, h.max_codes, h.vals_indices);
        let mut code = 0i32;
        for i in 0..16 {
            if self.bits.n == 0 {
                self.ensure_bits(1)?;
            }
            if self.bits.a & self.bits.m != 0 {
                code |= 1;
            }
            self.bits.n -= 1;
            self.bits.m >>= 1;
            if code <= max_codes[i] {
                let at = vals_indices[i] + code - min_codes[i];
                return self.huff[class][index]
                    .vals
                    .get(at as usize)
                    .copied()
                    .filter(|_| at >= 0)
                    .ok_or_else(|| format_error("bad Huffman code"));
            }
            code <<= 1;
        }
        Err(format_error("bad Huffman code"))
    }

    fn decode_bit(&mut self) -> Result<bool> {
        if self.bits.n == 0 {
            self.ensure_bits(1)?;
        }
        let bit = self.bits.a & self.bits.m != 0;
        self.bits.n -= 1;
        self.bits.m >>= 1;
        Ok(bit)
    }

    fn decode_bits(&mut self, n: i32) -> Result<u32> {
        if self.bits.n < n {
            self.ensure_bits(n)?;
        }
        let value = (self.bits.a >> (self.bits.n - n)) & ((1u32 << n) - 1);
        self.bits.n -= n;
        self.bits.m >>= n;
        Ok(value)
    }

    // -- the segments -------------------------------------------------------------------------

    fn process_sof(&mut self, n: usize) -> Result<()> {
        if self.n_comp != 0 {
            return Err(format_error("multiple SOF markers"));
        }
        self.n_comp = match n {
            9 => 1,
            15 => 3,
            18 => 4,
            _ => return Err(unsupported("number of components")),
        };
        self.read_full(n)?;
        if self.tmp[0] != 8 {
            return Err(unsupported("precision"));
        }
        self.height = (self.tmp[1] as usize) << 8 | self.tmp[2] as usize;
        self.width = (self.tmp[3] as usize) << 8 | self.tmp[4] as usize;
        if self.tmp[5] as usize != self.n_comp {
            return Err(format_error("SOF has wrong length"));
        }
        if self.width as u64 * self.height as u64 > MAX_PIXELS {
            return Err(unsupported("more pixels than this reader decodes"));
        }
        let ratio = || unsupported("luma/chroma subsampling ratio");
        for i in 0..self.n_comp {
            self.comp[i].c = self.tmp[6 + 3 * i];
            if (0..i).any(|j| self.comp[j].c == self.comp[i].c) {
                return Err(format_error("repeated component identifier"));
            }
            self.comp[i].tq = self.tmp[8 + 3 * i];
            if self.comp[i].tq > MAX_TQ {
                return Err(format_error("bad Tq value"));
            }
            let hv = self.tmp[7 + 3 * i];
            let (mut h, mut v) = ((hv >> 4) as usize, (hv & 0x0f) as usize);
            if !(1..=4).contains(&h) || !(1..=4).contains(&v) {
                return Err(format_error("luma/chroma subsampling ratio"));
            }
            if h == 3 || v == 3 {
                return Err(ratio());
            }
            match self.n_comp {
                // one component is non-interleaved by definition: its (h, v) is effectively (1, 1)
                1 => (h, v) = (1, 1),
                3 => match i {
                    0 if v == 4 => return Err(ratio()),
                    1 if self.comp[0].h % h != 0 || self.comp[0].v % v != 0 => return Err(ratio()),
                    2 if self.comp[1].h != h || self.comp[1].v != v => return Err(ratio()),
                    _ => {}
                },
                _ => match i {
                    0 if hv != 0x11 && hv != 0x22 => return Err(ratio()),
                    1 | 2 if hv != 0x11 => return Err(ratio()),
                    3 if self.comp[0].h != h || self.comp[0].v != v => return Err(ratio()),
                    _ => {}
                },
            }
            self.comp[i].h = h;
            self.comp[i].v = v;
        }
        Ok(())
    }

    fn process_dht(&mut self, mut n: i64) -> Result<()> {
        while n > 0 {
            if n < 17 {
                return Err(format_error("DHT has wrong length"));
            }
            self.read_full(17)?;
            let tc = self.tmp[0] >> 4;
            if tc > 1 {
                return Err(format_error("bad Tc value"));
            }
            let th = self.tmp[0] & 0x0f;
            if th > MAX_TH || (self.baseline && th > 1) {
                return Err(format_error("bad Th value"));
            }
            let mut counts = [0i32; 16];
            let mut total = 0i32;
            for (i, count) in counts.iter_mut().enumerate() {
                *count = self.tmp[i + 1] as i32;
                total += *count;
            }
            if total == 0 {
                return Err(format_error("Huffman table has zero length"));
            }
            if total > 256 {
                return Err(format_error("Huffman table has excessive length"));
            }
            n -= total as i64 + 17;
            if n < 0 {
                return Err(format_error("DHT has wrong length"));
            }
            self.settle();
            let vals = self
                .data
                .get(self.at..self.at + total as usize)
                .ok_or(Error::Eof)?
                .to_vec();
            self.at += total as usize;
            let h = &mut self.huff[tc as usize][th as usize];
            h.n_codes = total;
            h.vals[..total as usize].copy_from_slice(&vals);
            h.lut = [0; 256];
            let (mut x, mut code) = (0usize, 0u32);
            for i in 0..8u32 {
                code <<= 1;
                for _ in 0..counts[i as usize] {
                    let base = ((code << (7 - i)) & 0xff) as usize;
                    let value = (h.vals[x] as u16) << 8 | (2 + i) as u16;
                    for k in 0..1usize << (7 - i) {
                        h.lut[base | k] = value;
                    }
                    code += 1;
                    x += 1;
                }
            }
            let (mut c, mut index) = (0i32, 0i32);
            for (i, &count) in counts.iter().enumerate() {
                if count == 0 {
                    h.min_codes[i] = -1;
                    h.max_codes[i] = -1;
                    h.vals_indices[i] = -1;
                } else {
                    h.min_codes[i] = c;
                    h.max_codes[i] = c + count - 1;
                    h.vals_indices[i] = index;
                    c += count;
                    index += count;
                }
                c <<= 1;
            }
        }
        Ok(())
    }

    fn process_dqt(&mut self, mut n: i64) -> Result<()> {
        while n > 0 {
            n -= 1;
            let x = self.read_byte()?;
            let tq = (x & 0x0f) as usize;
            if tq > MAX_TQ as usize {
                return Err(format_error("bad Tq value"));
            }
            match x >> 4 {
                0 => {
                    if n < BLOCK as i64 {
                        break;
                    }
                    n -= BLOCK as i64;
                    self.read_full(BLOCK)?;
                    for i in 0..BLOCK {
                        self.quant[tq][i] = self.tmp[i] as i32;
                    }
                }
                1 => {
                    if n < 2 * BLOCK as i64 {
                        break;
                    }
                    n -= 2 * BLOCK as i64;
                    self.read_full(2 * BLOCK)?;
                    for i in 0..BLOCK {
                        self.quant[tq][i] = (self.tmp[2 * i] as i32) << 8 | self.tmp[2 * i + 1] as i32;
                    }
                }
                _ => return Err(format_error("bad Pq value")),
            }
        }
        if n != 0 {
            return Err(format_error("DQT has wrong length"));
        }
        Ok(())
    }

    fn process_app0(&mut self, n: usize) -> Result<()> {
        if n < 5 {
            return self.ignore(n);
        }
        self.read_full(5)?;
        self.jfif = &self.tmp[..5] == b"JFIF\0";
        if n > 5 {
            return self.ignore(n - 5);
        }
        Ok(())
    }

    fn process_app14(&mut self, n: usize) -> Result<()> {
        if n < 12 {
            return self.ignore(n);
        }
        self.read_full(12)?;
        if &self.tmp[..5] == b"Adobe" {
            self.adobe_valid = true;
            self.adobe_transform = self.tmp[11];
        }
        if n > 12 {
            return self.ignore(n - 12);
        }
        Ok(())
    }

    fn decode(&mut self) -> Result<Picture> {
        self.read_full(2)?;
        if self.tmp[0] != 0xff || self.tmp[1] != SOI {
            return Err(format_error("missing SOI marker"));
        }
        loop {
            self.read_full(2)?;
            // extraneous data before a marker is skipped, as libjpeg skips it
            while self.tmp[0] != 0xff {
                self.tmp[0] = self.tmp[1];
                self.tmp[1] = self.read_byte()?;
            }
            let mut marker = self.tmp[1];
            if marker == 0 {
                continue; // "\xff\x00" is extraneous data too
            }
            while marker == 0xff {
                marker = self.read_byte()?; // fill bytes before a marker
            }
            if marker == EOI {
                break;
            }
            if (RST0..=RST7).contains(&marker) {
                continue; // a stray restart marker after the last segment
            }
            self.read_full(2)?;
            let n = ((self.tmp[0] as i64) << 8 | self.tmp[1] as i64) - 2;
            if n < 0 {
                return Err(format_error("short segment length"));
            }
            match marker {
                SOF0 | SOF1 | SOF2 => {
                    self.baseline = marker == SOF0;
                    self.progressive = marker == SOF2;
                    self.process_sof(n as usize)?;
                }
                DHT => self.process_dht(n)?,
                DQT => self.process_dqt(n)?,
                SOS => self.process_sos(n as usize)?,
                DRI => {
                    if n != 2 {
                        return Err(format_error("DRI has wrong length"));
                    }
                    self.read_full(2)?;
                    self.ri = (self.tmp[0] as usize) << 8 | self.tmp[1] as usize;
                }
                APP0 => self.process_app0(n as usize)?,
                APP14 => self.process_app14(n as usize)?,
                _ if (APP0..=APP15).contains(&marker) || marker == COM => self.ignore(n as usize)?,
                _ if marker < 0xc0 => return Err(format_error("unknown marker")),
                _ => return Err(unsupported("unknown marker")),
            }
        }
        if self.progressive {
            self.reconstruct_progressive()?;
        }
        if let Some((pix, stride)) = &self.gray {
            let mut picture = Picture::new(0, 0, self.width, self.height, [0, 0, 0]);
            for y in 0..self.height {
                for x in 0..self.width {
                    let v = pix[y * stride + x];
                    picture.set(x, y, [v, v, v]);
                }
            }
            return Ok(picture);
        }
        if self.planes.is_some() {
            return if !self.black.is_empty() {
                self.apply_black()
            } else if self.is_rgb() {
                Ok(self.convert_rgb())
            } else {
                Ok(self.convert_ycbcr())
            };
        }
        Err(format_error("missing SOS marker"))
    }

    // -- the scans ----------------------------------------------------------------------------

    fn make_image(&mut self, mxx: usize, myy: usize) {
        if self.n_comp == 1 {
            self.gray = Some((vec![0; 8 * mxx * 8 * myy], 8 * mxx));
            return;
        }
        let (h0, v0) = (self.comp[0].h, self.comp[0].v);
        let ratio = match (h0 / self.comp[1].h, v0 / self.comp[1].v) {
            (1, 1) => Ratio::R444,
            (1, 2) => Ratio::R440,
            (2, 1) => Ratio::R422,
            (2, 2) => Ratio::R420,
            (4, 1) => Ratio::R411,
            _ => Ratio::R410,
        };
        let (w, h) = (8 * h0 * mxx, 8 * v0 * myy);
        // image.NewYCbCr's plane sizes, for a rectangle at the origin
        let (cw, ch) = match ratio {
            Ratio::R422 => (w.div_ceil(2), h),
            Ratio::R420 => (w.div_ceil(2), h.div_ceil(2)),
            Ratio::R440 => (w, h.div_ceil(2)),
            Ratio::R411 => (w.div_ceil(4), h),
            Ratio::R410 => (w.div_ceil(4), h.div_ceil(2)),
            Ratio::R444 => (w, h),
        };
        self.planes = Some(Planes {
            y: vec![0; w * h],
            cb: vec![0; cw * ch],
            cr: vec![0; cw * ch],
            y_stride: w,
            c_stride: cw,
            ratio,
        });
        if self.n_comp == 4 {
            let (h3, v3) = (self.comp[3].h, self.comp[3].v);
            self.black = vec![0; 8 * h3 * mxx * 8 * v3 * myy];
            self.black_stride = 8 * h3 * mxx;
        }
    }

    fn process_sos(&mut self, n: usize) -> Result<()> {
        if self.n_comp == 0 {
            return Err(format_error("missing SOF marker"));
        }
        if n < 6 || 4 + 2 * self.n_comp < n || n % 2 != 0 {
            return Err(format_error("SOS has wrong length"));
        }
        self.read_full(n)?;
        let scan_comps = self.tmp[0] as usize;
        if n != 4 + 2 * scan_comps {
            return Err(format_error("SOS length inconsistent with number of components"));
        }
        // (component index, DC table, AC table) of each component in the scan
        let mut scan = [(0usize, 0usize, 0usize); 4];
        let mut total_hv = 0;
        for i in 0..scan_comps {
            let selector = self.tmp[1 + 2 * i];
            let index = (0..self.n_comp)
                .rev()
                .find(|&j| self.comp[j].c == selector)
                .ok_or_else(|| format_error("unknown component selector"))?;
            if scan[..i].iter().any(|s| s.0 == index) {
                return Err(format_error("repeated component selector"));
            }
            total_hv += self.comp[index].h * self.comp[index].v;
            let td = self.tmp[2 + 2 * i] >> 4;
            if td > MAX_TH || (self.baseline && td > 1) {
                return Err(format_error("bad Td value"));
            }
            let ta = self.tmp[2 + 2 * i] & 0x0f;
            if ta > MAX_TH || (self.baseline && ta > 1) {
                return Err(format_error("bad Ta value"));
            }
            scan[i] = (index, td as usize, ta as usize);
        }
        if self.n_comp > 1 && total_hv > 10 {
            return Err(format_error("total sampling factors too large"));
        }
        let (mut zig_start, mut zig_end, mut ah, mut al) = (0i32, BLOCK as i32 - 1, 0u32, 0u32);
        if self.progressive {
            zig_start = self.tmp[1 + 2 * scan_comps] as i32;
            zig_end = self.tmp[2 + 2 * scan_comps] as i32;
            ah = (self.tmp[3 + 2 * scan_comps] >> 4) as u32;
            al = (self.tmp[3 + 2 * scan_comps] & 0x0f) as u32;
            if (zig_start == 0 && zig_end != 0) || zig_start > zig_end || BLOCK as i32 <= zig_end {
                return Err(format_error("bad spectral selection bounds"));
            }
            if zig_start != 0 && scan_comps != 1 {
                return Err(format_error("progressive AC coefficients for more than one component"));
            }
            if ah != 0 && ah != al + 1 {
                return Err(format_error("bad successive approximation values"));
            }
        }
        let (h0, v0) = (self.comp[0].h, self.comp[0].v);
        let mxx = self.width.div_ceil(8 * h0);
        let myy = self.height.div_ceil(8 * v0);
        if self.gray.is_none() && self.planes.is_none() {
            self.make_image(mxx, myy);
        }
        if self.progressive {
            for &(index, _, _) in &scan[..scan_comps] {
                if self.prog_coeffs[index].is_empty() {
                    self.prog_coeffs[index] = vec![[0; BLOCK]; mxx * myy * self.comp[index].h * self.comp[index].v];
                }
            }
        }
        self.bits = Bits::default();
        let (mut mcu, mut expected_rst) = (0usize, RST0);
        let mut dc = [0i32; 4];
        let mut block_count = 0usize;
        for my in 0..myy {
            for mx in 0..mxx {
                for &(index, td, ta) in &scan[..scan_comps] {
                    let (hi, vi) = (self.comp[index].h, self.comp[index].v);
                    for j in 0..hi * vi {
                        let (bx, by);
                        if scan_comps != 1 {
                            bx = hi * mx + j % hi;
                            by = vi * my + j / hi;
                        } else {
                            // a non-interleaved scan visits only the blocks inside the picture
                            let q = mxx * hi;
                            bx = block_count % q;
                            by = block_count / q;
                            block_count += 1;
                            if bx * 8 >= self.width || by * 8 >= self.height {
                                continue;
                            }
                        }
                        let slot = by * mxx * hi + bx;
                        let mut b: Block = if self.progressive {
                            *self.prog_coeffs[index]
                                .get(slot)
                                .ok_or_else(|| format_error("block out of range"))?
                        } else {
                            [0; BLOCK]
                        };
                        if ah != 0 {
                            self.refine(&mut b, ta, zig_start, zig_end, 1 << al)?;
                        } else {
                            let mut zig = zig_start;
                            if zig == 0 {
                                zig += 1;
                                let value = self.decode_huffman(0, td)?;
                                if value > 16 {
                                    return Err(unsupported("excessive DC component"));
                                }
                                let delta = self.receive_extend(value)?;
                                dc[index] = dc[index].wrapping_add(delta);
                                b[0] = dc[index].wrapping_shl(al);
                            }
                            if zig <= zig_end && self.eob_run > 0 {
                                self.eob_run -= 1;
                            } else {
                                while zig <= zig_end {
                                    let value = self.decode_huffman(1, ta)?;
                                    let (val0, val1) = (value >> 4, value & 0x0f);
                                    if val1 != 0 {
                                        zig += val0 as i32;
                                        if zig > zig_end {
                                            break;
                                        }
                                        let ac = self.receive_extend(val1)?;
                                        b[UNZIG[zig as usize]] = ac.wrapping_shl(al);
                                    } else {
                                        if val0 != 0x0f {
                                            self.eob_run = 1u16 << val0;
                                            if val0 != 0 {
                                                let bits = self.decode_bits(val0 as i32)?;
                                                self.eob_run |= bits as u16;
                                            }
                                            self.eob_run -= 1;
                                            break;
                                        }
                                        zig += 0x0f;
                                    }
                                    zig += 1;
                                }
                            }
                        }
                        if self.progressive {
                            self.prog_coeffs[index][slot] = b;
                            continue;
                        }
                        self.reconstruct_block(&mut b, bx, by, index)?;
                    }
                }
                mcu += 1;
                if self.ri > 0 && mcu % self.ri == 0 && mcu < mxx * myy {
                    self.read_full(2)?;
                    if self.tmp[0] != 0xff || self.tmp[1] != expected_rst {
                        self.find_rst(expected_rst)?;
                    }
                    expected_rst = if expected_rst == RST7 { RST0 } else { expected_rst + 1 };
                    self.bits = Bits::default();
                    dc = [0; 4];
                    self.eob_run = 0;
                }
            }
        }
        Ok(())
    }

    /// A successive-approximation refinement of one block.
    fn refine(&mut self, b: &mut Block, ta: usize, zig_start: i32, zig_end: i32, delta: i32) -> Result<()> {
        if zig_start == 0 {
            if self.decode_bit()? {
                b[0] |= delta;
            }
            return Ok(());
        }
        let mut zig = zig_start;
        if self.eob_run == 0 {
            while zig <= zig_end {
                let mut z = 0i32;
                let value = self.decode_huffman(1, ta)?;
                let (val0, val1) = (value >> 4, value & 0x0f);
                match val1 {
                    0 => {
                        if val0 != 0x0f {
                            self.eob_run = 1u16 << val0;
                            if val0 != 0 {
                                let bits = self.decode_bits(val0 as i32)?;
                                self.eob_run |= bits as u16;
                            }
                            break;
                        }
                    }
                    1 => {
                        z = delta;
                        if !self.decode_bit()? {
                            z = -z;
                        }
                    }
                    _ => return Err(format_error("unexpected Huffman code")),
                }
                zig = self.refine_non_zeroes(b, zig, zig_end, val0 as i32, delta)?;
                if zig > zig_end {
                    return Err(format_error("too many coefficients"));
                }
                if z != 0 {
                    b[UNZIG[zig as usize]] = z;
                }
                zig += 1;
            }
        }
        if self.eob_run > 0 {
            self.eob_run -= 1;
            self.refine_non_zeroes(b, zig, zig_end, -1, delta)?;
        }
        Ok(())
    }

    /// Refines the non-zero entries of `b` in zig-zag order, skipping the first
    /// `nz` zero entries when `nz >= 0`.
    fn refine_non_zeroes(&mut self, b: &mut Block, mut zig: i32, zig_end: i32, mut nz: i32, delta: i32) -> Result<i32> {
        while zig <= zig_end {
            let u = UNZIG[zig as usize];
            if b[u] == 0 {
                if nz == 0 {
                    break;
                }
                nz -= 1;
                zig += 1;
                continue;
            }
            if self.decode_bit()? {
                b[u] = if b[u] >= 0 {
                    b[u].wrapping_add(delta)
                } else {
                    b[u].wrapping_sub(delta)
                };
            }
            zig += 1;
        }
        Ok(zig)
    }

    fn reconstruct_progressive(&mut self) -> Result<()> {
        let h0 = self.comp[0].h;
        let mxx = self.width.div_ceil(8 * h0);
        for i in 0..self.n_comp {
            if self.prog_coeffs[i].is_empty() {
                continue;
            }
            let v = 8 * self.comp[0].v / self.comp[i].v;
            let h = 8 * self.comp[0].h / self.comp[i].h;
            let stride = mxx * self.comp[i].h;
            let mut by = 0;
            while by * v < self.height {
                let mut bx = 0;
                while bx * h < self.width {
                    let mut b = *self.prog_coeffs[i]
                        .get(by * stride + bx)
                        .ok_or_else(|| format_error("block out of range"))?;
                    self.reconstruct_block(&mut b, bx, by, i)?;
                    bx += 1;
                }
                by += 1;
            }
        }
        Ok(())
    }

    /// Dequantises, inverse-transforms and stores one block.
    fn reconstruct_block(&mut self, b: &mut Block, bx: usize, by: usize, index: usize) -> Result<()> {
        let qt = &self.quant[self.comp[index].tq as usize];
        for zig in 0..BLOCK {
            b[UNZIG[zig]] = b[UNZIG[zig]].wrapping_mul(qt[zig]);
        }
        idct(b);
        let (dst, stride): (&mut Vec<u8>, usize) = if self.n_comp == 1 {
            let (pix, stride) = self.gray.as_mut().ok_or_else(|| format_error("missing SOF marker"))?;
            (pix, *stride)
        } else {
            let planes = self.planes.as_mut().ok_or_else(|| format_error("missing SOF marker"))?;
            match index {
                0 => (&mut planes.y, planes.y_stride),
                1 => (&mut planes.cb, planes.c_stride),
                2 => (&mut planes.cr, planes.c_stride),
                3 => (&mut self.black, self.black_stride),
                _ => return Err(unsupported("too many components")),
            }
        };
        let origin = 8 * (by * stride + bx);
        for y in 0..8 {
            for x in 0..8 {
                let c = b[y * 8 + x];
                let value = if c < -128 {
                    0
                } else if c > 127 {
                    255
                } else {
                    (c + 128) as u8
                };
                // a block past the planes is a corrupt file, and Go would panic on it
                let at = origin + y * stride + x;
                *dst.get_mut(at).ok_or_else(|| format_error("block out of range"))? = value;
            }
        }
        Ok(())
    }

    /// Advances past the restart marker `expected`.
    fn find_rst(&mut self, expected: u8) -> Result<()> {
        loop {
            let mut i = 0;
            if self.tmp[0] == 0xff {
                if self.tmp[1] == expected {
                    return Ok(());
                } else if self.tmp[1] == 0xff {
                    i = 1;
                } else if self.tmp[1] != 0x00 {
                    return Err(format_error("bad RST marker"));
                }
            } else if self.tmp[1] == 0xff {
                self.tmp[0] = 0xff;
                i = 1;
            }
            self.settle();
            for k in i..2 {
                self.tmp[k] = *self.data.get(self.at).ok_or(Error::Eof)?;
                self.at += 1;
            }
        }
    }

    // -- the colours --------------------------------------------------------------------------

    fn is_rgb(&self) -> bool {
        if self.jfif {
            return false;
        }
        if self.adobe_valid && self.adobe_transform == 0 {
            return true; // Adobe's "unknown", which in practice is RGB
        }
        self.comp[0].c == b'R' && self.comp[1].c == b'G' && self.comp[2].c == b'B'
    }

    fn convert_ycbcr(&self) -> Picture {
        let p = self.planes.as_ref().expect("a colour image");
        let mut picture = Picture::new(0, 0, self.width, self.height, [0, 0, 0]);
        for y in 0..self.height {
            for x in 0..self.width {
                let c = p.c_offset(x, y);
                picture.set(x, y, ycbcr_to_rgb(p.y[y * p.y_stride + x], p.cb[c], p.cr[c]));
            }
        }
        picture
    }

    fn convert_rgb(&self) -> Picture {
        let p = self.planes.as_ref().expect("a colour image");
        let scale = self.comp[0].h / self.comp[1].h;
        let mut picture = Picture::new(0, 0, self.width, self.height, [0, 0, 0]);
        for y in 0..self.height {
            let co = p.c_offset(0, y);
            for x in 0..self.width {
                picture.set(
                    x,
                    y,
                    [p.y[y * p.y_stride + x], p.cb[co + x / scale], p.cr[co + x / scale]],
                );
            }
        }
        picture
    }

    /// CMYK (Adobe transform 0) or YCbCrK (any other) into what
    /// `image.CMYK.At(x, y).RGBA() >> 8` reads.
    fn apply_black(&self) -> Result<Picture> {
        if !self.adobe_valid {
            return Err(unsupported(
                "unknown color model: 4-component JPEG doesn't have Adobe APP14 metadata",
            ));
        }
        let p = self.planes.as_ref().expect("a colour image");
        let mut picture = Picture::new(0, 0, self.width, self.height, [0, 0, 0]);
        for y in 0..self.height {
            for x in 0..self.width {
                let cmyk = if self.adobe_transform != 0 {
                    // YCbCrK: the YCbCr part to RGB, which then stands as C, M and Y
                    let c = p.c_offset(x, y);
                    let [r, g, b] = ycbcr_to_rgb(p.y[y * p.y_stride + x], p.cb[c], p.cr[c]);
                    [r, g, b, 255 - self.black[y * self.black_stride + x]]
                } else {
                    let mut ink = [0u8; 4];
                    let planes: [(&Vec<u8>, usize); 4] = [
                        (&p.y, p.y_stride),
                        (&p.cb, p.c_stride),
                        (&p.cr, p.c_stride),
                        (&self.black, self.black_stride),
                    ];
                    for (t, (src, stride)) in planes.iter().enumerate() {
                        let sub = self.comp[t].h != self.comp[0].h || self.comp[t].v != self.comp[0].v;
                        let (sx, sy) = if sub { (x / 2, y / 2) } else { (x, y) };
                        ink[t] = 255 - src[sy * stride + sx];
                    }
                    ink
                };
                picture.set(x, y, cmyk_to_rgb(cmyk));
            }
        }
        Ok(picture)
    }
}

/// `color.YCbCrToRGB`: the JFIF conversion in 16.16 fixed point, which is also
/// `YCbCr.RGBA() >> 8`.
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8) -> [u8; 3] {
    let yy1 = y as i32 * 0x10101;
    let cb1 = cb as i32 - 128;
    let cr1 = cr as i32 - 128;
    let clamp = |v: i32| -> u8 {
        if (v as u32) & 0xff00_0000 == 0 {
            (v >> 16) as u8
        } else {
            (!(v >> 31)) as u8
        }
    };
    [
        clamp(yy1 + 91881 * cr1),
        clamp(yy1 - 22554 * cb1 - 46802 * cr1),
        clamp(yy1 + 116130 * cb1),
    ]
}

/// `color.CMYK.RGBA() >> 8`.
fn cmyk_to_rgb([c, m, y, k]: [u8; 4]) -> [u8; 3] {
    let w = 0xffff - k as u32 * 0x101;
    let channel = |v: u8| (((0xffff - v as u32 * 0x101) * w / 0xffff) >> 8) as u8;
    [channel(c), channel(m), channel(y)]
}

const W1: i64 = 2841;
const W2: i64 = 2676;
const W3: i64 = 2408;
const W5: i64 = 1609;
const W6: i64 = 1108;
const W7: i64 = 565;
const W1PW7: i64 = W1 + W7;
const W1MW7: i64 = W1 - W7;
const W2PW6: i64 = W2 + W6;
const W2MW6: i64 = W2 - W6;
const W3PW5: i64 = W3 + W5;
const W3MW5: i64 = W3 - W5;
const R2: i64 = 181;

/// A value as Go's `int32` holds it: wrapped.
#[inline]
fn w(x: i64) -> i64 {
    x as i32 as i64
}

/// Go's integer IDCT (`image/jpeg/idct.go`, after the MPEG reference), in
/// 8.8 fixed point.
fn idct(src: &mut Block) {
    for y in 0..8 {
        let s = &mut src[y * 8..y * 8 + 8];
        if s[1..].iter().all(|&v| v == 0) {
            let dc = w((s[0] as i64) << 3) as i32;
            s.fill(dc);
            continue;
        }
        let mut x0 = (s[0] as i64) * 2048 + 128;
        let mut x1 = (s[4] as i64) * 2048;
        let mut x2 = s[6] as i64;
        let mut x3 = s[2] as i64;
        let mut x4 = s[1] as i64;
        let mut x5 = s[7] as i64;
        let mut x6 = s[5] as i64;
        let mut x7 = s[3] as i64;

        let mut x8 = W7 * (x4 + x5);
        x4 = x8 + W1MW7 * x4;
        x5 = x8 - W1PW7 * x5;
        x8 = W3 * (x6 + x7);
        x6 = x8 - W3MW5 * x6;
        x7 = x8 - W3PW5 * x7;

        x8 = x0 + x1;
        x0 -= x1;
        x1 = W6 * (x3 + x2);
        x2 = x1 - W2PW6 * x2;
        x3 = x1 + W2MW6 * x3;
        x1 = x4 + x6;
        x4 -= x6;
        x6 = x5 + x7;
        x5 -= x7;

        x7 = x8 + x3;
        x8 -= x3;
        x3 = x0 + x2;
        x0 -= x2;
        x2 = w(R2 * (x4 + x5) + 128) >> 8;
        x4 = w(R2 * (x4 - x5) + 128) >> 8;

        s[0] = (w(x7 + x1) >> 8) as i32;
        s[1] = (w(x3 + x2) >> 8) as i32;
        s[2] = (w(x0 + x4) >> 8) as i32;
        s[3] = (w(x8 + x6) >> 8) as i32;
        s[4] = (w(x8 - x6) >> 8) as i32;
        s[5] = (w(x0 - x4) >> 8) as i32;
        s[6] = (w(x3 - x2) >> 8) as i32;
        s[7] = (w(x7 - x1) >> 8) as i32;
    }
    for x in 0..8 {
        let at = |row: usize| x + 8 * row;
        let mut y0 = (src[at(0)] as i64) * 256 + 8192;
        let mut y1 = (src[at(4)] as i64) * 256;
        let mut y2 = src[at(6)] as i64;
        let mut y3 = src[at(2)] as i64;
        let mut y4 = src[at(1)] as i64;
        let mut y5 = src[at(7)] as i64;
        let mut y6 = src[at(5)] as i64;
        let mut y7 = src[at(3)] as i64;

        let mut y8 = W7 * (y4 + y5) + 4;
        y4 = w(y8 + W1MW7 * y4) >> 3;
        y5 = w(y8 - W1PW7 * y5) >> 3;
        y8 = W3 * (y6 + y7) + 4;
        y6 = w(y8 - W3MW5 * y6) >> 3;
        y7 = w(y8 - W3PW5 * y7) >> 3;

        y8 = y0 + y1;
        y0 -= y1;
        y1 = W6 * (y3 + y2) + 4;
        y2 = w(y1 - W2PW6 * y2) >> 3;
        y3 = w(y1 + W2MW6 * y3) >> 3;
        y1 = y4 + y6;
        y4 -= y6;
        y6 = y5 + y7;
        y5 -= y7;

        y7 = y8 + y3;
        y8 -= y3;
        y3 = y0 + y2;
        y0 -= y2;
        y2 = w(R2 * (y4 + y5) + 128) >> 8;
        y4 = w(R2 * (y4 - y5) + 128) >> 8;

        src[at(0)] = (w(y7 + y1) >> 14) as i32;
        src[at(1)] = (w(y3 + y2) >> 14) as i32;
        src[at(2)] = (w(y0 + y4) >> 14) as i32;
        src[at(3)] = (w(y8 + y6) >> 14) as i32;
        src[at(4)] = (w(y8 - y6) >> 14) as i32;
        src[at(5)] = (w(y0 - y4) >> 14) as i32;
        src[at(6)] = (w(y3 - y2) >> 14) as i32;
        src[at(7)] = (w(y7 - y1) >> 14) as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_colour_conversions_are_gos() {
        // what Go's color.YCbCrToRGB and color.CMYK.RGBA() >> 8 answer for the same inputs
        assert_eq!(ycbcr_to_rgb(0x7f, 0x80, 0x80), [0x7f, 0x7f, 0x7f]);
        assert_eq!(ycbcr_to_rgb(0, 0, 255), [178, 0, 0]);
        assert_eq!(ycbcr_to_rgb(255, 255, 0), [76, 255, 255]);
        assert_eq!(ycbcr_to_rgb(30, 200, 90), [0, 32, 157]);
        assert_eq!(cmyk_to_rgb([0, 0, 0, 0]), [255, 255, 255]);
        assert_eq!(cmyk_to_rgb([255, 128, 0, 64]), [0, 95, 191]);
        assert_eq!(cmyk_to_rgb([10, 20, 30, 40]), [207, 198, 190]);
    }

    #[test]
    fn a_flat_block_inverts_to_a_flat_block() {
        // a DC coefficient of 80 is a flat block of 80 / 8 before the level shift
        let mut b: Block = [0; BLOCK];
        b[0] = 80;
        idct(&mut b);
        assert!(b.iter().all(|&v| v == 10), "{b:?}");
        let mut wild: Block = [i32::MAX; BLOCK];
        idct(&mut wild); // wraps as Go's int32 does, and does not panic
    }

    #[test]
    fn rubbish_is_refused() {
        assert!(decode(b"\xff\xd8").is_err());
        assert!(decode(b"not a jpeg").is_err());
        assert!(decode(b"\xff\xd8\xff\xd9").unwrap_err().contains("missing SOS"));
    }
}
