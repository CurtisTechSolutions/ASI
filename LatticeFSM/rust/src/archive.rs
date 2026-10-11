//! Archives as streams: a deflate decoder that never holds more than its 32 KiB window, gzip (every member of a
//! multi-member file, its CRC checked), zip (ZIP64 too: archives and members of any size; stored or deflated) and
//! tar (ustar, GNU long names, pax paths and sizes, base-256 sizes) - what Python's `gzip`, `zipfile` and `tarfile`
//! read for `latticefsm/corpus.py`, member for member.  The standard library only.

use std::io::{self, Read, Seek, SeekFrom};

const WINDOW: usize = 1 << 15;
const MAX_BITS: usize = 15;

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, msg.into())
}

// ---- CRC-32 ----------------------------------------------------------------------------------------------------------

const CRC_TABLE: [u32; 256] = {
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
    table
};

/// A running CRC-32 (gzip's and zip's).
#[derive(Clone, Copy, Debug)]
pub struct Crc32(u32);

impl Default for Crc32 {
    fn default() -> Self {
        Crc32(0xffff_ffff)
    }
}

impl Crc32 {
    pub fn update(&mut self, data: &[u8]) {
        let mut c = self.0;
        for &b in data {
            c = CRC_TABLE[((c ^ b as u32) & 0xff) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }

    pub fn value(&self) -> u32 {
        self.0 ^ 0xffff_ffff
    }
}

// ---- inflate (RFC 1951), streamed ------------------------------------------------------------------------------------

/// A canonical Huffman decoder (zlib's `puff` shape): how many codes of each length, and the symbols in code order.
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

/// Bytes from a reader, buffered, and the bits of them deflate reads least significant first.
struct Source<R: Read> {
    src: R,
    buf: Vec<u8>,
    pos: usize,
    len: usize,
    held: u32,
    bits: u32,
}

impl<R: Read> Source<R> {
    fn new(src: R) -> Source<R> {
        Source {
            src,
            buf: vec![0; 1 << 16],
            pos: 0,
            len: 0,
            held: 0,
            bits: 0,
        }
    }

    /// The next byte, or `None` at the end of the input.
    fn byte(&mut self) -> io::Result<Option<u8>> {
        if self.pos == self.len {
            self.len = loop {
                match self.src.read(&mut self.buf) {
                    Ok(n) => break n,
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e),
                }
            };
            self.pos = 0;
            if self.len == 0 {
                return Ok(None);
            }
        }
        self.pos += 1;
        Ok(Some(self.buf[self.pos - 1]))
    }

    fn need(&mut self) -> io::Result<u8> {
        self.byte()?.ok_or_else(|| bad("the compressed stream ends early"))
    }

    fn take(&mut self, need: u32) -> io::Result<u32> {
        while self.bits < need {
            self.held |= (self.need()? as u32) << self.bits;
            self.bits += 8;
        }
        let v = if need == 0 { 0 } else { self.held & ((1u32 << need) - 1) };
        self.held = if need >= 32 { 0 } else { self.held >> need };
        self.bits -= need;
        Ok(v)
    }

    fn align(&mut self) {
        self.held = 0;
        self.bits = 0;
    }

    fn decode(&mut self, h: &Huffman) -> io::Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..=MAX_BITS {
            code |= self.take(1)? as i32;
            let count = h.count[len] as i32;
            if code - count < first {
                return Ok(h.symbol[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        Err(bad("an invalid Huffman code"))
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
const ORDER: [usize; 19] = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

enum Block {
    Header,
    Stored(usize),
    Codes(Box<(Huffman, Huffman)>),
    Done,
}

/// A raw deflate stream decoded as it is read: the window of the last 32 KiB is all it keeps.
pub struct Inflate<R: Read> {
    src: Source<R>,
    window: Vec<u8>,
    written: usize,
    out: Vec<u8>,
    opos: usize,
    block: Block,
    last: bool,
}

impl<R: Read> Inflate<R> {
    pub fn new(src: R) -> Inflate<R> {
        Inflate::over(Source::new(src))
    }

    fn over(src: Source<R>) -> Inflate<R> {
        Inflate {
            src,
            window: vec![0; WINDOW],
            written: 0,
            out: Vec::new(),
            opos: 0,
            block: Block::Header,
            last: false,
        }
    }

    fn put(&mut self, b: u8) {
        self.out.push(b);
        self.window[self.written & (WINDOW - 1)] = b;
        self.written += 1;
    }

    fn end_block(&mut self) {
        self.block = if self.last { Block::Done } else { Block::Header };
    }

    /// Decode until a good amount of output is ready or the stream ends.
    fn fill(&mut self) -> io::Result<()> {
        while self.out.len() < WINDOW {
            match &mut self.block {
                Block::Done => return Ok(()),
                Block::Header => {
                    self.last = self.src.take(1)? == 1;
                    self.block = match self.src.take(2)? {
                        0 => {
                            self.src.align();
                            let mut h = [0u8; 4];
                            for b in h.iter_mut() {
                                *b = self.src.need()?;
                            }
                            let len = u16::from_le_bytes([h[0], h[1]]);
                            if u16::from_le_bytes([h[2], h[3]]) != !len {
                                return Err(bad("a stored block's length is not its own complement"));
                            }
                            Block::Stored(len as usize)
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
                            Block::Codes(Box::new((Huffman::new(&lengths), Huffman::new(&[5u8; 30]))))
                        }
                        2 => Block::Codes(Box::new(self.dynamic()?)),
                        _ => return Err(bad("a reserved deflate block type")),
                    };
                }
                Block::Stored(n) => {
                    if *n == 0 {
                        self.end_block();
                        continue;
                    }
                    *n -= 1;
                    let b = self.src.need()?;
                    self.put(b);
                }
                Block::Codes(_) => {
                    let Block::Codes(codes) = std::mem::replace(&mut self.block, Block::Header) else {
                        unreachable!()
                    };
                    let ended = self.codes(&codes)?;
                    if ended {
                        self.end_block();
                    } else {
                        self.block = Block::Codes(codes);
                    }
                }
            }
        }
        Ok(())
    }

    fn dynamic(&mut self) -> io::Result<(Huffman, Huffman)> {
        let hlit = self.src.take(5)? as usize + 257;
        let hdist = self.src.take(5)? as usize + 1;
        let hclen = self.src.take(4)? as usize + 4;
        let mut cl = [0u8; 19];
        for &slot in ORDER.iter().take(hclen) {
            cl[slot] = self.src.take(3)? as u8;
        }
        let code = Huffman::new(&cl);
        let mut lengths = vec![0u8; hlit + hdist];
        let mut i = 0;
        while i < lengths.len() {
            let sym = self.src.decode(&code)?;
            let (value, times) = match sym {
                0..=15 => (sym as u8, 1),
                16 => {
                    if i == 0 {
                        return Err(bad("no code length to repeat"));
                    }
                    (lengths[i - 1], 3 + self.src.take(2)? as usize)
                }
                17 => (0, 3 + self.src.take(3)? as usize),
                18 => (0, 11 + self.src.take(7)? as usize),
                _ => return Err(bad("an invalid code-length symbol")),
            };
            if i + times > lengths.len() {
                return Err(bad("code lengths overrun their table"));
            }
            for _ in 0..times {
                lengths[i] = value;
                i += 1;
            }
        }
        Ok((Huffman::new(&lengths[..hlit]), Huffman::new(&lengths[hlit..])))
    }

    /// Symbols of a Huffman block until enough output is ready (`false`) or the block ends (`true`).
    fn codes(&mut self, codes: &(Huffman, Huffman)) -> io::Result<bool> {
        let (lit, dist) = codes;
        while self.out.len() < WINDOW {
            let sym = self.src.decode(lit)?;
            match sym {
                0..=255 => self.put(sym as u8),
                256 => return Ok(true),
                257..=285 => {
                    let i = sym as usize - 257;
                    let length = LENGTH_BASE[i] as usize + self.src.take(LENGTH_EXTRA[i])? as usize;
                    let d = self.src.decode(dist)? as usize;
                    if d >= DIST_BASE.len() {
                        return Err(bad("an invalid distance symbol"));
                    }
                    let distance = DIST_BASE[d] as usize + self.src.take(DIST_EXTRA[d])? as usize;
                    if distance > self.written.min(WINDOW) {
                        return Err(bad("a distance reaching before the output"));
                    }
                    let from = self.written - distance;
                    for j in 0..length {
                        let b = self.window[(from + j) & (WINDOW - 1)];
                        self.put(b);
                    }
                }
                _ => return Err(bad("an invalid literal/length symbol")),
            }
        }
        Ok(false)
    }

    fn done(&self) -> bool {
        matches!(self.block, Block::Done) && self.opos == self.out.len()
    }
}

impl<R: Read> Read for Inflate<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.opos == self.out.len() {
            self.out.clear();
            self.opos = 0;
            self.fill()?;
        }
        let n = buf.len().min(self.out.len() - self.opos);
        buf[..n].copy_from_slice(&self.out[self.opos..self.opos + n]);
        self.opos += n;
        Ok(n)
    }
}

// ---- gzip (RFC 1952) -------------------------------------------------------------------------------------------------

/// Every member of a gzip file, decompressed as it is read; each member's CRC and length checked at its end.
pub struct Gzip<R: Read> {
    inflate: Option<Inflate<R>>,
    crc: Crc32,
    size: u32,
}

impl<R: Read> Gzip<R> {
    pub fn new(src: R) -> io::Result<Gzip<R>> {
        let mut source = Source::new(src);
        if !Self::header(&mut source, true)? {
            return Err(bad("not a gzipped file"));
        }
        Ok(Gzip {
            inflate: Some(Inflate::over(source)),
            crc: Crc32::default(),
            size: 0,
        })
    }

    /// A member's header; `false` at the end of the file (after zero padding).
    fn header(s: &mut Source<R>, first: bool) -> io::Result<bool> {
        let mut b0 = s.byte()?;
        if !first {
            while b0 == Some(0) {
                b0 = s.byte()?;
            }
        }
        let Some(b0) = b0 else { return Ok(false) };
        if b0 != 0x1f || s.need()? != 0x8b {
            return Err(bad("not a gzipped file"));
        }
        if s.need()? != 8 {
            return Err(bad("unknown gzip compression method"));
        }
        let flags = s.need()?;
        for _ in 0..6 {
            s.need()?;
        }
        if flags & 4 != 0 {
            let len = s.need()? as usize | (s.need()? as usize) << 8;
            for _ in 0..len {
                s.need()?;
            }
        }
        for bit in [8u8, 16] {
            if flags & bit != 0 {
                while s.need()? != 0 {}
            }
        }
        if flags & 2 != 0 {
            s.need()?;
            s.need()?;
        }
        Ok(true)
    }
}

impl<R: Read> Read for Gzip<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        loop {
            let Some(inf) = self.inflate.as_mut() else { return Ok(0) };
            let n = inf.read(buf)?;
            if n > 0 {
                self.crc.update(&buf[..n]);
                self.size = self.size.wrapping_add(n as u32);
                return Ok(n);
            }
            if !inf.done() {
                continue;
            }
            // the member's trailer, then the next member or the end
            let mut inf = self.inflate.take().expect("there");
            inf.src.align();
            let mut t = [0u8; 8];
            for b in t.iter_mut() {
                *b = inf.src.need()?;
            }
            if u32::from_le_bytes([t[0], t[1], t[2], t[3]]) != self.crc.value() {
                return Err(bad("CRC check failed in a gzip member"));
            }
            if u32::from_le_bytes([t[4], t[5], t[6], t[7]]) != self.size {
                return Err(bad("incorrect length in a gzip member"));
            }
            let mut src = inf.src;
            if !Self::header(&mut src, false)? {
                return Ok(0);
            }
            self.inflate = Some(Inflate::over(src));
            self.crc = Crc32::default();
            self.size = 0;
        }
    }
}

// ---- zip -------------------------------------------------------------------------------------------------------------

const CP437_HIGH: &str = "ÇüéâäàåçêëèïîìÄÅÉæÆôöòûùÿÖÜ¢£¥₧ƒáíóúñÑªº¿⌐¬½¼¡«»░▒▓│┤╡╢╖╕╣║╗╝╜╛┐└┴┬├─┼╞╟╚╔╩╦╠═╬╧╨╤╥╙╘╒╓╫╪┘┌█▄▌▐▀αßΓπΣσµτΦΘΩδ∞φε∩≡±≥≤⌠⌡÷≈°∙·√ⁿ²■\u{a0}";

fn cp437(bytes: &[u8]) -> String {
    let high: Vec<char> = CP437_HIGH.chars().collect();
    bytes
        .iter()
        .map(|&b| if b < 128 { b as char } else { high[b as usize - 128] })
        .collect()
}

/// A member of a zip, as its central directory describes it.
#[derive(Clone, Debug)]
pub struct ZipEntry {
    pub name: String,
    pub flags: u16,
    pub method: u16,
    pub crc: u32,
    pub compressed: u64,
    pub size: u64,
    pub offset: u64,
}

impl ZipEntry {
    pub fn is_dir(&self) -> bool {
        self.name.ends_with('/')
    }
}

fn le16(b: &[u8], at: usize) -> u64 {
    u16::from_le_bytes([b[at], b[at + 1]]) as u64
}

fn le32(b: &[u8], at: usize) -> u64 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as u64
}

fn le64(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
}

fn rfind(hay: &[u8], needle: &[u8]) -> Option<usize> {
    (0..=hay.len().checked_sub(needle.len())?)
        .rev()
        .find(|&i| &hay[i..i + needle.len()] == needle)
}

/// Every member of a zip, in its central directory's order (ZIP64 end records and extra fields read; data
/// prepended to the archive allowed for, as Python's `zipfile` does).
pub fn zip_entries<R: Read + Seek>(r: &mut R) -> io::Result<Vec<ZipEntry>> {
    let size = r.seek(SeekFrom::End(0))?;
    let tail_len = size.min(22 + 65535);
    r.seek(SeekFrom::Start(size - tail_len))?;
    let mut tail = vec![0u8; tail_len as usize];
    r.read_exact(&mut tail)?;
    let at = rfind(&tail, b"PK\x05\x06")
        .filter(|&i| i + 22 <= tail.len())
        .ok_or_else(|| bad("not a zip file"))?;
    let eocd = &tail[at..];
    let eocd_pos = size - tail_len + at as u64;
    let (mut entries, mut cd_size, mut cd_offset, mut location) =
        (le16(eocd, 10), le32(eocd, 12), le32(eocd, 16), eocd_pos);
    if eocd_pos >= 20 {
        let mut loc = [0u8; 20];
        r.seek(SeekFrom::Start(eocd_pos - 20))?;
        r.read_exact(&mut loc)?;
        if &loc[..4] == b"PK\x06\x07" && eocd_pos >= 20 + 56 {
            let mut rec = [0u8; 56];
            r.seek(SeekFrom::Start(eocd_pos - 20 - 56))?;
            r.read_exact(&mut rec)?;
            if &rec[..4] == b"PK\x06\x06" {
                entries = le64(&rec, 32);
                cd_size = le64(&rec, 40);
                cd_offset = le64(&rec, 48);
                location = eocd_pos - 20 - 56;
            }
        }
    }
    let concat = location
        .checked_sub(cd_size + cd_offset)
        .ok_or_else(|| bad("a bad zip central directory"))?;
    r.seek(SeekFrom::Start(cd_offset + concat))?;
    let mut cd = io::BufReader::new(r.take(cd_size));
    let mut out = Vec::with_capacity(entries.min(1 << 20) as usize);
    let mut head = [0u8; 46];
    loop {
        match cd.read_exact(&mut head) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e),
        }
        if &head[..4] != b"PK\x01\x02" {
            return Err(bad("a bad zip central directory entry"));
        }
        let (nlen, elen, clen) = (
            le16(&head, 28) as usize,
            le16(&head, 30) as usize,
            le16(&head, 32) as usize,
        );
        let mut rest = vec![0u8; nlen + elen + clen];
        cd.read_exact(&mut rest)?;
        let flags = le16(&head, 8) as u16;
        let name = if flags & 0x800 != 0 {
            String::from_utf8_lossy(&rest[..nlen]).into_owned()
        } else {
            cp437(&rest[..nlen])
        };
        let mut e = ZipEntry {
            name,
            flags,
            method: le16(&head, 10) as u16,
            crc: le32(&head, 16) as u32,
            compressed: le32(&head, 20),
            size: le32(&head, 24),
            offset: le32(&head, 42),
        };
        let extra = &rest[nlen..nlen + elen];
        let mut i = 0;
        while i + 4 <= extra.len() {
            let (id, len) = (le16(extra, i), le16(extra, i + 2) as usize);
            let data = &extra[i + 4..(i + 4 + len).min(extra.len())];
            if id == 1 {
                let mut j = 0;
                for field in [&mut e.size, &mut e.compressed, &mut e.offset] {
                    if *field == 0xffff_ffff && j + 8 <= data.len() {
                        *field = le64(data, j);
                        j += 8;
                    }
                }
            }
            i += 4 + len;
        }
        e.offset += concat;
        out.push(e);
    }
    Ok(out)
}

/// A reader that checks a CRC-32 and a length when it reaches its end.
pub struct Checked<R: Read> {
    inner: R,
    crc: Crc32,
    want_crc: u32,
    want_len: u64,
    got: u64,
}

impl<R: Read> Read for Checked<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        if n == 0 && !buf.is_empty() {
            if self.got != self.want_len {
                return Err(bad("a zip member of the wrong length"));
            }
            if self.crc.value() != self.want_crc {
                return Err(bad("bad CRC-32 for a zip member"));
            }
            return Ok(0);
        }
        self.crc.update(&buf[..n]);
        self.got += n as u64;
        Ok(n)
    }
}

/// A member's bytes, decompressed as they are read (stored or deflated members only).
pub fn zip_open<'a, R: Read + Seek>(r: &'a mut R, e: &ZipEntry) -> io::Result<Box<dyn Read + 'a>> {
    r.seek(SeekFrom::Start(e.offset))?;
    let mut head = [0u8; 30];
    r.read_exact(&mut head)?;
    if &head[..4] != b"PK\x03\x04" {
        return Err(bad("a bad zip local header"));
    }
    let skip = le16(&head, 26) + le16(&head, 28);
    r.seek(SeekFrom::Current(skip as i64))?;
    let data = r.take(e.compressed);
    let check = |inner: Box<dyn Read + 'a>| Checked {
        inner,
        crc: Crc32::default(),
        want_crc: e.crc,
        want_len: e.size,
        got: 0,
    };
    Ok(match e.method {
        0 => Box::new(check(Box::new(data))),
        8 => Box::new(check(Box::new(Inflate::new(data)))),
        m => return Err(bad(format!("zip compression method {m} is not read"))),
    })
}

// ---- tar -------------------------------------------------------------------------------------------------------------

fn cstr(b: &[u8]) -> &[u8] {
    &b[..b.iter().position(|&c| c == 0).unwrap_or(b.len())]
}

/// A tar number: octal digits, or base-256 when the first byte's high bit is set.
fn tar_number(b: &[u8]) -> u64 {
    if b[0] == 0o200 || b[0] == 0o377 {
        let mut n: u64 = 0;
        for &c in &b[1..] {
            n = n.wrapping_shl(8) | c as u64;
        }
        return if b[0] == 0o377 { 0 } else { n };
    }
    let s = String::from_utf8_lossy(cstr(b));
    u64::from_str_radix(s.trim(), 8).unwrap_or(0)
}

fn pax_records(data: &[u8]) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        let Some(sp) = data[i..].iter().position(|&c| c == b' ') else {
            break;
        };
        let Ok(len) = String::from_utf8_lossy(&data[i..i + sp]).parse::<usize>() else {
            break;
        };
        if len == 0 || i + len > data.len() {
            break;
        }
        let rec = &data[i + sp + 1..i + len];
        let rec = rec.strip_suffix(b"\n").unwrap_or(rec);
        if let Some(eq) = rec.iter().position(|&c| c == b'=') {
            out.push((
                String::from_utf8_lossy(&rec[..eq]).into_owned(),
                String::from_utf8_lossy(&rec[eq + 1..]).into_owned(),
            ));
        }
        i += len;
    }
    out
}

fn skip(r: &mut impl Read, n: u64) -> io::Result<()> {
    let copied = io::copy(&mut r.take(n), &mut io::sink())?;
    if copied < n {
        return Err(bad("a tar archive ends early"));
    }
    Ok(())
}

/// Every regular member of a tar, in order, handed to `f` with a reader of its bytes (read it or not: the rest is
/// skipped).  It stops at the first zero block or a block whose checksum is wrong.
pub fn tar_each<R: Read>(
    mut src: R,
    f: &mut dyn FnMut(&str, &mut dyn Read) -> Result<(), String>,
) -> Result<(), String> {
    let err = |e: io::Error| e.to_string();
    let (mut long_name, mut pax_path, mut pax_size): (Option<String>, Option<String>, Option<u64>) = (None, None, None);
    let mut block = [0u8; 512];
    loop {
        let mut got = 0;
        while got < 512 {
            let n = src.read(&mut block[got..]).map_err(err)?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got == 0 || block.iter().all(|&c| c == 0) {
            return Ok(());
        }
        if got < 512 {
            return Err("a tar archive ends inside a header".to_string());
        }
        let sum: u64 = block
            .iter()
            .enumerate()
            .map(|(i, &c)| if (148..156).contains(&i) { 32 } else { c as u64 })
            .sum();
        if tar_number(&block[148..156]) != sum {
            return Ok(());
        }
        let kind = block[156];
        let mut size = tar_number(&block[124..136]);
        let padded = |n: u64| n.div_ceil(512) * 512;
        match kind {
            b'L' | b'x' | b'g' | b'K' => {
                let mut data = Vec::new();
                (&mut src).take(size).read_to_end(&mut data).map_err(err)?;
                if (data.len() as u64) < size {
                    return Err("a tar archive ends early".to_string());
                }
                skip(&mut src, padded(size) - size).map_err(err)?;
                if kind == b'L' {
                    long_name = Some(String::from_utf8_lossy(cstr(&data)).into_owned());
                } else if kind == b'x' {
                    for (k, v) in pax_records(&data) {
                        match k.as_str() {
                            "path" => pax_path = Some(v),
                            "size" => pax_size = v.parse().ok(),
                            _ => {}
                        }
                    }
                }
                continue;
            }
            _ => {}
        }
        let mut name = String::from_utf8_lossy(cstr(&block[..100])).into_owned();
        let prefix = String::from_utf8_lossy(cstr(&block[345..500])).into_owned();
        if !prefix.is_empty() && !matches!(kind, b'L' | b'K' | b'S') && &block[257..262] == b"ustar" {
            name = format!("{prefix}/{name}");
        }
        if let Some(n) = long_name.take() {
            name = n;
        }
        if let Some(p) = pax_path.take() {
            name = p;
        }
        if let Some(s) = pax_size.take() {
            size = s;
        }
        let regular = matches!(kind, b'0' | 0 | b'7') && !(kind == 0 && name.ends_with('/'));
        let mut data = (&mut src).take(size);
        if regular {
            f(&name, &mut data)?;
        }
        let left = io::copy(&mut data, &mut io::sink()).map_err(err)?;
        let _ = left;
        if data.limit() > 0 {
            return Err("a tar archive ends early".to_string());
        }
        skip(&mut src, padded(size) - size).map_err(err)?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inflate_reads_what_gzip_wrote() {
        let text: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8 ^ (i / 1000) as u8).collect();
        let gz = crate::gzip::compress(&text);
        let mut out = Vec::new();
        Gzip::new(&gz[..]).unwrap().read_to_end(&mut out).unwrap();
        assert_eq!(out, text);
        // two members, zero padding after them
        let mut two = gz.clone();
        two.extend(crate::gzip::compress(b"more"));
        two.extend([0, 0, 0]);
        let mut out = Vec::new();
        Gzip::new(&two[..]).unwrap().read_to_end(&mut out).unwrap();
        assert_eq!(out.len(), text.len() + 4);
        let mut broken = gz.clone();
        let n = broken.len();
        broken[n - 6] ^= 1;
        assert!(Gzip::new(&broken[..]).unwrap().read_to_end(&mut Vec::new()).is_err());
    }

    #[test]
    fn crc_matches_the_whole_buffer_one() {
        let mut c = Crc32::default();
        c.update(b"hello ");
        c.update(b"world");
        assert_eq!(c.value(), crate::gzip::crc32(b"hello world"));
        assert_eq!(cp437(&[0x80, 0xff, b'a']), "Ç\u{a0}a");
    }
}
