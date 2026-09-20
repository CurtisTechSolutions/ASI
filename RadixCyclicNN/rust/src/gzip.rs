//! The gzip container, written out.
//!
//! `model.json.gz` is an ordinary model file (`GraphModel.save` gzips whenever
//! the path ends with `.gz`), so a port that cannot read one cannot read half
//! the files the other two write.  Python has `gzip`, Go has `compress/gzip`,
//! and the standard library here has neither - so this is RFC 1951 inflate and
//! an RFC 1952 wrapper, in the same spirit as the BLAKE2b and the Mersenne
//! Twister the Go port had to write out.
//!
//! **Reading** is complete: stored, fixed-Huffman and dynamic-Huffman blocks.
//! **Writing** emits stored (uncompressed) deflate blocks - a valid gzip stream
//! that Python and Go read back, but not a smaller one.  A compressor is real
//! work for a container detail; what the format costs is the reading.

/// The CRC-32 of a byte slice (the polynomial gzip uses).
pub fn crc32(data: &[u8]) -> u32 {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xff) as usize] ^ (crc >> 8);
    }
    crc ^ 0xffff_ffff
}

/// Whether a buffer starts with the gzip magic.
pub fn is_gzip(data: &[u8]) -> bool {
    data.len() >= 2 && data[0] == 0x1f && data[1] == 0x8b
}

/// Wraps `data` as a gzip stream of stored deflate blocks.
pub fn compress(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len() + data.len() / 65535 * 5 + 18);
    out.extend_from_slice(&[0x1f, 0x8b, 8, 0]); // magic, deflate, no flags
    out.extend_from_slice(&[0, 0, 0, 0]); // mtime 0: the file is the model, not the moment
    out.extend_from_slice(&[0, 3]); // no extra flags, Unix
    let mut chunks = data.chunks(0xffff).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xff, 0xff]); // one final, empty stored block
    }
    while let Some(chunk) = chunks.next() {
        let last = chunks.peek().is_none();
        out.push(if last { 1 } else { 0 }); // BFINAL, BTYPE = 00 (stored), then byte-aligned
        let len = chunk.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&crc32(data).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out
}

/// Unwraps a gzip stream, checking the CRC and the length.
pub fn decompress(data: &[u8]) -> Result<Vec<u8>, String> {
    if !is_gzip(data) || data.len() < 18 {
        return Err("not a gzip stream".to_string());
    }
    if data[2] != 8 {
        return Err(format!("unsupported compression method {}", data[2]));
    }
    let flags = data[3];
    let mut at = 10;
    if flags & 0b100 != 0 {
        // FEXTRA
        let len = u16::from_le_bytes([
            *data.get(at).ok_or("truncated header")?,
            *data.get(at + 1).ok_or("truncated header")?,
        ]) as usize;
        at += 2 + len;
    }
    for bit in [0b1000u8, 0b1_0000] {
        // FNAME, FCOMMENT: NUL-terminated
        if flags & bit != 0 {
            while *data.get(at).ok_or("truncated header")? != 0 {
                at += 1;
            }
            at += 1;
        }
    }
    if flags & 0b10 != 0 {
        at += 2; // FHCRC
    }
    if at + 8 > data.len() {
        return Err("truncated gzip stream".to_string());
    }
    let body = &data[at..data.len() - 8];
    let out = inflate(body)?;
    let tail = &data[data.len() - 8..];
    let want_crc = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let want_len = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
    if crc32(&out) != want_crc {
        return Err("gzip checksum mismatch".to_string());
    }
    if out.len() as u32 != want_len {
        return Err("gzip length mismatch".to_string());
    }
    Ok(out)
}

// -- inflate (RFC 1951) ------------------------------------------------------

const MAX_BITS: usize = 15;

/// A canonical Huffman decoder: how many codes of each length, and the symbols
/// in code order (the shape zlib's `puff` uses - small, and obviously right).
struct Huffman {
    count: [u16; MAX_BITS + 1],
    symbol: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Huffman {
        let mut count = [0u16; MAX_BITS + 1];
        for &l in lengths {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut offs = [0u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offs[len + 1] = offs[len] + count[len];
        }
        let mut symbol = vec![0u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                symbol[offs[l as usize] as usize] = sym as u16;
                offs[l as usize] += 1;
            }
        }
        Huffman { count, symbol }
    }
}

struct Bits<'a> {
    data: &'a [u8],
    at: usize,
    bit: u32,
    held: u32,
}

impl<'a> Bits<'a> {
    fn new(data: &'a [u8]) -> Bits<'a> {
        Bits {
            data,
            at: 0,
            bit: 0,
            held: 0,
        }
    }

    fn take(&mut self, need: u32) -> Result<u32, String> {
        if need == 0 {
            return Ok(0);
        }
        while self.bit < need {
            let byte = *self.data.get(self.at).ok_or("out of input")?;
            self.held |= (byte as u32) << self.bit;
            self.at += 1;
            self.bit += 8;
        }
        let value = self.held & ((1u32 << need) - 1);
        self.held >>= need;
        self.bit -= need;
        Ok(value)
    }

    fn align(&mut self) {
        self.held = 0;
        self.bit = 0;
    }

    fn decode(&mut self, huff: &Huffman) -> Result<u16, String> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=MAX_BITS {
            code |= self.take(1)? as i32;
            let count = huff.count[len] as i32;
            if code - count < first {
                return Ok(huff.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err("invalid Huffman code".to_string())
    }
}

const LENGTH_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258,
];
const LENGTH_EXTRA: [u32; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145,
    8193, 12289, 16385, 24577,
];
const DIST_EXTRA: [u32; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13,
];
const CODE_LENGTH_ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

/// Decompresses a raw deflate stream.
pub fn inflate(data: &[u8]) -> Result<Vec<u8>, String> {
    let mut bits = Bits::new(data);
    let mut out: Vec<u8> = Vec::with_capacity(data.len() * 4);
    loop {
        let last = bits.take(1)? == 1;
        match bits.take(2)? {
            0 => {
                bits.align();
                let at = bits.at;
                if at + 4 > bits.data.len() {
                    return Err("truncated stored block".to_string());
                }
                let len = u16::from_le_bytes([bits.data[at], bits.data[at + 1]]) as usize;
                let nlen = u16::from_le_bytes([bits.data[at + 2], bits.data[at + 3]]);
                if nlen != !(len as u16) {
                    return Err("stored block length is not its own complement".to_string());
                }
                let from = at + 4;
                let body = bits.data.get(from..from + len).ok_or("truncated stored block")?;
                out.extend_from_slice(body);
                bits.at = from + len;
            }
            1 => {
                let mut lengths = [0u8; 288];
                for (i, l) in lengths.iter_mut().enumerate() {
                    *l = match i {
                        0..=143 => 8,
                        144..=255 => 9,
                        256..=279 => 7,
                        _ => 8,
                    };
                }
                let lit = Huffman::new(&lengths);
                let dist = Huffman::new(&[5u8; 30]);
                inflate_block(&mut bits, &mut out, &lit, &dist)?;
            }
            2 => {
                let hlit = bits.take(5)? as usize + 257;
                let hdist = bits.take(5)? as usize + 1;
                let hclen = bits.take(4)? as usize + 4;
                let mut code_lengths = [0u8; 19];
                for &slot in CODE_LENGTH_ORDER.iter().take(hclen) {
                    code_lengths[slot] = bits.take(3)? as u8;
                }
                let code_huff = Huffman::new(&code_lengths);
                let mut lengths = vec![0u8; hlit + hdist];
                let mut i = 0;
                while i < lengths.len() {
                    let symbol = bits.decode(&code_huff)?;
                    match symbol {
                        0..=15 => {
                            lengths[i] = symbol as u8;
                            i += 1;
                        }
                        16 => {
                            if i == 0 {
                                return Err("no length to repeat".to_string());
                            }
                            let prev = lengths[i - 1];
                            let times = 3 + bits.take(2)? as usize;
                            for _ in 0..times {
                                if i < lengths.len() {
                                    lengths[i] = prev;
                                    i += 1;
                                }
                            }
                        }
                        17 => i += 3 + bits.take(3)? as usize,
                        18 => i += 11 + bits.take(7)? as usize,
                        _ => return Err("invalid code-length symbol".to_string()),
                    }
                }
                if i > lengths.len() {
                    return Err("code lengths overrun their table".to_string());
                }
                let lit = Huffman::new(&lengths[..hlit]);
                let dist = Huffman::new(&lengths[hlit..]);
                inflate_block(&mut bits, &mut out, &lit, &dist)?;
            }
            _ => return Err("reserved deflate block type".to_string()),
        }
        if last {
            return Ok(out);
        }
    }
}

fn inflate_block(bits: &mut Bits<'_>, out: &mut Vec<u8>, lit: &Huffman, dist: &Huffman) -> Result<(), String> {
    loop {
        let symbol = bits.decode(lit)?;
        match symbol {
            0..=255 => out.push(symbol as u8),
            256 => return Ok(()),
            257..=285 => {
                let i = symbol as usize - 257;
                let length = LENGTH_BASE[i] as usize + bits.take(LENGTH_EXTRA[i])? as usize;
                let d = bits.decode(dist)? as usize;
                if d >= DIST_BASE.len() {
                    return Err("invalid distance symbol".to_string());
                }
                let distance = DIST_BASE[d] as usize + bits.take(DIST_EXTRA[d])? as usize;
                if distance > out.len() {
                    return Err("distance reaches before the start of the output".to_string());
                }
                let from = out.len() - distance;
                for j in 0..length {
                    out.push(out[from + j]); // byte by byte: the run may overlap itself
                }
            }
            _ => return Err("invalid literal/length symbol".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn what_it_writes_it_reads_back() {
        for text in ["", "x", &"the cat sat on the mat. ".repeat(5000)] {
            let packed = compress(text.as_bytes());
            assert!(is_gzip(&packed));
            assert_eq!(decompress(&packed).unwrap(), text.as_bytes());
        }
    }

    #[test]
    fn it_reads_what_python_wrote() {
        // gzip.compress(b"the cat sat on the mat\n" * 4, mtime=0) - dynamic Huffman,
        // which is what Python actually emits for a model file
        const PYTHON_GZ: &[u8] = &[
            0x1f, 0x8b, 0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x03, 0x2b, 0xc9, 0x48, 0x55, 0x48, 0x4e, 0x2c, 0x51,
            0x28, 0x06, 0xe2, 0xfc, 0x3c, 0x85, 0x12, 0x20, 0x37, 0x37, 0xb1, 0x84, 0xab, 0x84, 0x1a, 0xc2, 0x00, 0x02,
            0xce, 0x25, 0x21, 0x5c, 0x00, 0x00, 0x00,
        ];
        let out = decompress(PYTHON_GZ).unwrap();
        assert_eq!(String::from_utf8(out).unwrap(), "the cat sat on the mat\n".repeat(4));
    }

    #[test]
    fn a_damaged_stream_is_refused() {
        let mut packed = compress(b"hello");
        let last = packed.len() - 1;
        packed[last] ^= 0xff;
        assert!(decompress(&packed).is_err());
        assert!(decompress(b"not gzip at all...").is_err());
    }
}
