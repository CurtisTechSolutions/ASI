//! Reading ZIP archives, written out: the central directory, the local
//! headers, and `gzip.rs`'s inflate for what is inside.
//!
//! A corpus arrives as a `.zip` more often than as anything else - a source
//! tree, a folder of books, an export - and the other two implementations read
//! one wherever they read a text file (`radixnet/archive.py`,
//! `go/radixnet/source.go`).  Python has `zipfile`, Go has `archive/zip`, and
//! the standard library here has neither, so this is the part of APPNOTE.TXT a
//! reader needs, in the same spirit as the gzip container beside it.
//!
//! **What it reads**: stored and deflated entries, ZIP64 archives (more than
//! 65 535 entries, or past 4 GiB), data descriptors (the sizes are taken from
//! the central directory, which always has them), archives with bytes in front
//! of them (a self-extractor's stub), UTF-8 and CP437 names.  Every entry is
//! checked against its CRC-32 and its declared size, as `zipfile` checks it.
//!
//! **What it does not**: encryption, and the compression methods Python only
//! reads through optional modules (bzip2, LZMA) - such an entry is reported as
//! unreadable rather than guessed at.  Nothing here writes an archive.
//!
//! The archive is read through `Read + Seek`: the central directory is read
//! once, and an entry's bytes are read only when that entry is asked for - so
//! an archive on disk is never held in memory whole, only the entry being
//! unpacked (decision D-034: an upload is kept whole and unpacked on demand).

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom};
use std::path::Path;

/// The signatures an archive can start with: a local file header, an empty
/// archive's end record, and a spanned archive's marker - the same three
/// Python's `is_zip` and Go's `IsZipMagic` accept.  The name never decides.
pub const ZIP_MAGIC: [[u8; 4]; 3] = [*b"PK\x03\x04", *b"PK\x05\x06", *b"PK\x07\x08"];

const LOCAL_HEADER: u32 = 0x0403_4b50;
const CENTRAL_HEADER: u32 = 0x0201_4b50;
const END_RECORD: u32 = 0x0605_4b50;
const ZIP64_END_RECORD: u32 = 0x0606_4b50;
const ZIP64_LOCATOR: u32 = 0x0706_4b50;
const ZIP64_EXTRA: u16 = 0x0001;

/// Bit 0 of the general purpose flags: the entry is encrypted.
pub const FLAG_ENCRYPTED: u16 = 0x0001;
/// Bit 11: the name is UTF-8 (otherwise it is code page 437).
const FLAG_UTF8: u16 = 0x0800;

/// Whether `data` starts with one of the [`ZIP_MAGIC`] signatures.
pub fn is_zip(data: &[u8]) -> bool {
    data.len() >= 4 && ZIP_MAGIC.iter().any(|magic| data[..4] == magic[..])
}

/// Whether the file at `path` starts with a ZIP signature (reads four bytes;
/// a file that cannot be read is not an archive).
pub fn is_zip_file(path: impl AsRef<Path>) -> bool {
    let Ok(mut file) = File::open(path) else { return false };
    let mut head = [0u8; 4];
    file.read_exact(&mut head).is_ok() && is_zip(&head)
}

/// One member of an archive, as its central directory describes it.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    /// The path inside the archive, `/`-separated as the format writes it.
    pub name: String,
    pub flags: u16,
    /// 0 stored, 8 deflated; anything else cannot be read here.
    pub method: u16,
    pub crc: u32,
    /// Bytes the entry takes in the archive.
    pub compressed: u64,
    /// Bytes it unpacks to.
    pub size: u64,
    /// Where its local header starts (already corrected for a prefix).
    offset: u64,
}

impl Entry {
    /// A directory: the format marks one with a trailing slash.
    pub fn is_dir(&self) -> bool {
        self.name.ends_with('/')
    }

    /// Encrypted entries are never read.
    pub fn encrypted(&self) -> bool {
        self.flags & FLAG_ENCRYPTED != 0
    }
}

/// An open archive: its central directory, and the reader its entries are
/// unpacked from on demand.
pub struct Archive<R> {
    reader: R,
    entries: Vec<Entry>,
}

impl Archive<BufReader<File>> {
    /// Opens the archive at `path`; only the central directory is read.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, String> {
        let file =
            File::open(path.as_ref()).map_err(|err| format!("cannot read {}: {err}", path.as_ref().display()))?;
        Archive::new(BufReader::new(file))
    }
}

impl Archive<Cursor<Vec<u8>>> {
    /// An archive held in memory.
    pub fn from_bytes(data: Vec<u8>) -> Result<Self, String> {
        Archive::new(Cursor::new(data))
    }
}

impl<R: Read + Seek> Archive<R> {
    /// Reads the central directory of the archive behind `reader`.
    pub fn new(mut reader: R) -> Result<Self, String> {
        let entries = read_directory(&mut reader).map_err(|err| format!("not a valid ZIP archive ({err})"))?;
        Ok(Archive { reader, entries })
    }

    /// Every member, in the order the central directory lists them.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// The bytes of entry `index`, unpacked and checked against its CRC-32 and
    /// its declared size.
    pub fn read(&mut self, index: usize) -> Result<Vec<u8>, String> {
        let packed = self.read_packed(index)?;
        unpack(&self.entries[index], packed)
    }

    /// The bytes entry `index` takes in the archive, still compressed - the
    /// cheap, sequential half of [`Archive::read`], so the expensive half
    /// ([`unpack`]) can run on another thread.
    pub fn read_packed(&mut self, index: usize) -> Result<Vec<u8>, String> {
        let entry = self
            .entries
            .get(index)
            .ok_or_else(|| format!("no entry {index} in the archive"))?;
        if entry.encrypted() {
            return Err("file is encrypted, password required".to_string());
        }
        let (offset, compressed) = (entry.offset, entry.compressed);
        let name = entry.name.clone();
        let r = &mut self.reader;
        r.seek(SeekFrom::Start(offset)).map_err(io)?;
        let head = take(r, 30)?;
        if le32(&head, 0) != LOCAL_HEADER {
            return Err(format!("bad magic number for file header of {name:?}"));
        }
        // the local header's own name and extra lengths: they may differ from
        // the central directory's, and the data starts after them
        let skip = le16(&head, 26) as i64 + le16(&head, 28) as i64;
        r.seek(SeekFrom::Current(skip)).map_err(io)?;
        take(r, compressed as usize)
    }
}

/// An entry's bytes from what [`Archive::read_packed`] read: inflated when
/// the entry is deflated, then checked against its declared size and CRC-32.
pub fn unpack(entry: &Entry, packed: Vec<u8>) -> Result<Vec<u8>, String> {
    let data = match entry.method {
        0 => packed,
        8 => crate::gzip::inflate(&packed).map_err(|err| format!("{}: {err}", entry.name))?,
        other => {
            return Err(format!(
                "compression method {other} ({}) is not supported",
                method_name(other)
            ))
        }
    };
    if data.len() as u64 != entry.size {
        return Err(format!(
            "{} unpacked to {} bytes, its header says {}",
            entry.name,
            data.len(),
            entry.size
        ));
    }
    if crate::gzip::crc32(&data) != entry.crc {
        return Err(format!("bad CRC-32 for file {:?}", entry.name));
    }
    Ok(data)
}

/// What a compression method is called, for the message that refuses it.
fn method_name(method: u16) -> &'static str {
    match method {
        1..=6 => "a legacy PKWARE method",
        9 => "deflate64",
        12 => "bzip2",
        14 => "lzma",
        93 => "zstandard",
        95 => "xz",
        _ => "unknown",
    }
}

fn io(err: std::io::Error) -> String {
    err.to_string()
}

fn take<R: Read>(r: &mut R, n: usize) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    r.by_ref().take(n as u64).read_to_end(&mut buf).map_err(io)?;
    if buf.len() != n {
        return Err("truncated archive".to_string());
    }
    Ok(buf)
}

fn le16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn le32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn le64(b: &[u8], at: usize) -> u64 {
    let mut word = [0u8; 8];
    word.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(word)
}

/// Finds the end record, follows it (and its ZIP64 twin when a field
/// overflowed) to the central directory, and reads every entry.
fn read_directory<R: Read + Seek>(r: &mut R) -> Result<Vec<Entry>, String> {
    let len = r.seek(SeekFrom::End(0)).map_err(io)?;
    if len < 22 {
        return Err("file is not a zip file".to_string());
    }
    // the end record is the last 22 bytes, or sits before a comment of up to 64 KiB
    let window = len.min(22 + 0xffff);
    r.seek(SeekFrom::Start(len - window)).map_err(io)?;
    let tail = take(r, window as usize)?;
    let at = (0..=tail.len() - 22)
        .rev()
        .find(|&i| le32(&tail, i) == END_RECORD)
        .ok_or("file is not a zip file")?;
    let end_pos = len - window + at as u64;
    let end = &tail[at..at + 22];
    let mut count = le16(end, 10) as u64;
    let mut cd_size = le32(end, 12) as u64;
    let mut cd_offset = le32(end, 16) as u64;
    // where the central directory really is: bytes in front of the archive shift
    // every offset in it by the same amount (zipfile calls it `concat`)
    let mut directory_end = end_pos;
    if end_pos >= 20 + 56 {
        r.seek(SeekFrom::Start(end_pos - 20)).map_err(io)?;
        let locator = take(r, 20)?;
        if le32(&locator, 0) == ZIP64_LOCATOR {
            let record_at = le64(&locator, 8);
            // the locator's offset is not corrected for a prefix either: the
            // record sits right in front of the locator, so read it from there
            let record_pos = end_pos - 20 - 56;
            r.seek(SeekFrom::Start(record_pos)).map_err(io)?;
            let record = take(r, 56)?;
            if le32(&record, 0) != ZIP64_END_RECORD {
                return Err(format!("corrupt ZIP64 end record (expected at {record_at})"));
            }
            count = le64(&record, 32);
            cd_size = le64(&record, 40);
            cd_offset = le64(&record, 48);
            directory_end = record_pos;
        }
    }
    let cd_start = directory_end
        .checked_sub(cd_size)
        .ok_or("the central directory is larger than the file")?;
    let prefix = cd_start
        .checked_sub(cd_offset)
        .ok_or("the central directory offset points past the file")?;
    r.seek(SeekFrom::Start(cd_start)).map_err(io)?;
    let directory = take(r, cd_size as usize)?;
    let mut entries = Vec::with_capacity(count.min(1 << 20) as usize);
    let mut at = 0usize;
    while at + 46 <= directory.len() {
        let h = &directory[at..];
        if le32(h, 0) != CENTRAL_HEADER {
            return Err("bad magic number for central directory".to_string());
        }
        let flags = le16(h, 8);
        let method = le16(h, 10);
        let crc = le32(h, 16);
        let mut compressed = le32(h, 20) as u64;
        let mut size = le32(h, 24) as u64;
        let name_len = le16(h, 28) as usize;
        let extra_len = le16(h, 30) as usize;
        let comment_len = le16(h, 32) as usize;
        let mut offset = le32(h, 42) as u64;
        let total = 46 + name_len + extra_len + comment_len;
        if at + total > directory.len() {
            return Err("truncated central directory".to_string());
        }
        let raw_name = &h[46..46 + name_len];
        let extra = &h[46 + name_len..46 + name_len + extra_len];
        // ZIP64: the fields that overflowed are in the extra block, in this order
        let mut e = 0usize;
        while e + 4 <= extra.len() {
            let id = le16(extra, e);
            let n = le16(extra, e + 2) as usize;
            let body = &extra[e + 4..(e + 4 + n).min(extra.len())];
            if id == ZIP64_EXTRA {
                let mut k = 0usize;
                for field in [&mut size, &mut compressed, &mut offset] {
                    if *field == 0xffff_ffff && k + 8 <= body.len() {
                        *field = le64(body, k);
                        k += 8;
                    }
                }
            }
            e += 4 + n;
        }
        entries.push(Entry {
            name: decode_name(raw_name, flags),
            flags,
            method,
            crc,
            compressed,
            size,
            offset: offset + prefix,
        });
        at += total;
    }
    Ok(entries)
}

/// A member's name as `zipfile` reads it: UTF-8 when the flag says so, code
/// page 437 otherwise, cut at a NUL, with backslashes turned into slashes.
fn decode_name(raw: &[u8], flags: u16) -> String {
    let name: String = if flags & FLAG_UTF8 != 0 {
        String::from_utf8_lossy(raw).into_owned()
    } else {
        raw.iter()
            .map(|&b| {
                if b < 0x80 {
                    b as char
                } else {
                    CP437_HIGH[(b - 0x80) as usize]
                }
            })
            .collect()
    };
    let name = match name.find('\0') {
        Some(at) => name[..at].to_string(),
        None => name,
    };
    name.replace('\\', "/")
}

/// Code page 437's upper half, the encoding a ZIP name has without the UTF-8 flag.
const CP437_HIGH: [char; 128] = [
    'Ç', 'ü', 'é', 'â', 'ä', 'à', 'å', 'ç', 'ê', 'ë', 'è', 'ï', 'î', 'ì', 'Ä', 'Å', 'É', 'æ', 'Æ', 'ô', 'ö', 'ò', 'û',
    'ù', 'ÿ', 'Ö', 'Ü', '¢', '£', '¥', '₧', 'ƒ', 'á', 'í', 'ó', 'ú', 'ñ', 'Ñ', 'ª', 'º', '¿', '⌐', '¬', '½', '¼', '¡',
    '«', '»', '░', '▒', '▓', '│', '┤', '╡', '╢', '╖', '╕', '╣', '║', '╗', '╝', '╜', '╛', '┐', '└', '┴', '┬', '├', '─',
    '┼', '╞', '╟', '╚', '╔', '╩', '╦', '╠', '═', '╬', '╧', '╨', '╤', '╥', '╙', '╘', '╒', '╓', '╫', '╪', '┘', '┌', '█',
    '▄', '▌', '▐', '▀', 'α', 'ß', 'Γ', 'π', 'Σ', 'σ', 'µ', 'τ', 'Φ', 'Θ', 'Ω', 'δ', '∞', 'φ', 'ε', '∩', '≡', '±', '≥',
    '≤', '⌠', '⌡', '÷', '≈', '°', '∙', '·', '√', 'ⁿ', '²', '■', '\u{a0}',
];

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A ZIP archive of `(name, bytes, deflate)` members, written the plain
    /// way: a deflated member is a deflate stream of stored blocks (what
    /// `gzip::compress` writes), which exercises the inflate path all the same.
    pub(crate) fn write_zip(members: &[(&str, &[u8], bool)]) -> Vec<u8> {
        build_zip(members, false)
    }

    /// [`write_zip`], with every size and offset moved into the ZIP64 extra
    /// block and the end record behind a ZIP64 one - what an archive of more
    /// than 65 535 members or past 4 GiB looks like, in miniature.
    fn build_zip(members: &[(&str, &[u8], bool)], zip64: bool) -> Vec<u8> {
        let mut out = Vec::new();
        let mut central = Vec::new();
        for (name, data, deflate) in members {
            let packed = if *deflate {
                let gz = crate::gzip::compress(data);
                gz[10..gz.len() - 8].to_vec()
            } else {
                data.to_vec()
            };
            let method: u16 = if *deflate { 8 } else { 0 };
            let crc = crate::gzip::crc32(data);
            let offset = out.len() as u32;
            let mut head = Vec::new();
            head.extend_from_slice(&LOCAL_HEADER.to_le_bytes());
            head.extend_from_slice(&[20, 0, 0, 0]);
            head.extend_from_slice(&method.to_le_bytes());
            head.extend_from_slice(&[0, 0, 0, 0]);
            head.extend_from_slice(&crc.to_le_bytes());
            head.extend_from_slice(&(packed.len() as u32).to_le_bytes());
            head.extend_from_slice(&(data.len() as u32).to_le_bytes());
            head.extend_from_slice(&(name.len() as u16).to_le_bytes());
            head.extend_from_slice(&[0, 0]);
            out.extend_from_slice(&head);
            out.extend_from_slice(name.as_bytes());
            out.extend_from_slice(&packed);
            let wide = |n: u32| if zip64 { 0xffff_ffffu32 } else { n };
            central.extend_from_slice(&CENTRAL_HEADER.to_le_bytes());
            central.extend_from_slice(&[20, 0, 45, 0]);
            central.extend_from_slice(&FLAG_UTF8.to_le_bytes());
            central.extend_from_slice(&method.to_le_bytes());
            central.extend_from_slice(&[0, 0, 0, 0]);
            central.extend_from_slice(&crc.to_le_bytes());
            central.extend_from_slice(&wide(packed.len() as u32).to_le_bytes());
            central.extend_from_slice(&wide(data.len() as u32).to_le_bytes());
            central.extend_from_slice(&(name.len() as u16).to_le_bytes());
            central.extend_from_slice(&(if zip64 { 28u16 } else { 0 }).to_le_bytes());
            central.extend_from_slice(&[0u8; 10]);
            central.extend_from_slice(&wide(offset).to_le_bytes());
            central.extend_from_slice(name.as_bytes());
            if zip64 {
                central.extend_from_slice(&ZIP64_EXTRA.to_le_bytes());
                central.extend_from_slice(&24u16.to_le_bytes());
                central.extend_from_slice(&(data.len() as u64).to_le_bytes());
                central.extend_from_slice(&(packed.len() as u64).to_le_bytes());
                central.extend_from_slice(&(offset as u64).to_le_bytes());
            }
        }
        let cd_offset = out.len() as u64;
        out.extend_from_slice(&central);
        let count = members.len() as u64;
        if zip64 {
            let record_at = out.len() as u64;
            out.extend_from_slice(&ZIP64_END_RECORD.to_le_bytes());
            out.extend_from_slice(&44u64.to_le_bytes());
            out.extend_from_slice(&[45, 0, 45, 0]);
            out.extend_from_slice(&[0u8; 8]);
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&count.to_le_bytes());
            out.extend_from_slice(&(central.len() as u64).to_le_bytes());
            out.extend_from_slice(&cd_offset.to_le_bytes());
            out.extend_from_slice(&ZIP64_LOCATOR.to_le_bytes());
            out.extend_from_slice(&[0u8; 4]);
            out.extend_from_slice(&record_at.to_le_bytes());
            out.extend_from_slice(&1u32.to_le_bytes());
        }
        let (short_count, size, offset) = if zip64 {
            (0xffffu16, 0xffff_ffffu32, 0xffff_ffffu32)
        } else {
            (count as u16, central.len() as u32, cd_offset as u32)
        };
        out.extend_from_slice(&END_RECORD.to_le_bytes());
        out.extend_from_slice(&[0, 0, 0, 0]);
        out.extend_from_slice(&short_count.to_le_bytes());
        out.extend_from_slice(&short_count.to_le_bytes());
        out.extend_from_slice(&size.to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
        out.extend_from_slice(&[0, 0]);
        out
    }

    /// `zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED)` with
    /// `a/one.txt` (real Huffman-coded deflate), the directory `a/`, and
    /// `two.txt` stored - bytes Python wrote, so the reader is held to them.
    const PYTHON_ZIP: &[u8] = &[
        80, 75, 3, 4, 20, 0, 0, 0, 8, 0, 0, 0, 33, 0, 117, 239, 130, 123, 37, 0, 0, 0, 47, 0, 0, 0, 9, 0, 0, 0, 97, 47,
        111, 110, 101, 46, 116, 120, 116, 43, 201, 72, 85, 72, 78, 44, 81, 40, 6, 226, 252, 60, 133, 18, 32, 55, 55,
        177, 132, 171, 4, 42, 92, 148, 8, 20, 203, 7, 11, 167, 228, 231, 23, 113, 1, 0, 80, 75, 3, 4, 20, 0, 0, 0, 0,
        0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 97, 47, 80, 75, 3, 4, 20, 0, 0, 0, 0, 0, 0, 0,
        33, 0, 205, 174, 8, 183, 23, 0, 0, 0, 23, 0, 0, 0, 7, 0, 0, 0, 116, 119, 111, 46, 116, 120, 116, 116, 104, 101,
        32, 100, 111, 103, 32, 115, 97, 116, 32, 111, 110, 32, 116, 104, 101, 32, 108, 111, 103, 10, 80, 75, 1, 2, 20,
        3, 20, 0, 0, 0, 8, 0, 0, 0, 33, 0, 117, 239, 130, 123, 37, 0, 0, 0, 47, 0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 128, 1, 0, 0, 0, 0, 97, 47, 111, 110, 101, 46, 116, 120, 116, 80, 75, 1, 2, 20, 3, 20, 0, 0, 0, 0, 0, 0,
        0, 33, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 237, 65, 76, 0, 0, 0, 97,
        47, 80, 75, 1, 2, 20, 3, 20, 0, 0, 0, 0, 0, 0, 0, 33, 0, 205, 174, 8, 183, 23, 0, 0, 0, 23, 0, 0, 0, 7, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 1, 108, 0, 0, 0, 116, 119, 111, 46, 116, 120, 116, 80, 75, 5, 6, 0, 0, 0, 0, 3,
        0, 3, 0, 156, 0, 0, 0, 168, 0, 0, 0, 0, 0,
    ];

    #[test]
    fn it_reads_what_python_writes() {
        let mut archive = Archive::from_bytes(PYTHON_ZIP.to_vec()).unwrap();
        let names: Vec<&str> = archive.entries().iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["a/one.txt", "a/", "two.txt"]);
        assert_eq!(archive.entries()[0].method, 8, "a real deflate stream");
        assert!(archive.entries()[1].is_dir());
        assert_eq!(
            archive.read(0).unwrap(),
            b"the cat sat on the mat\nthe cat ran to the door\n"
        );
        assert_eq!(archive.read(2).unwrap(), b"the dog sat on the log\n");
        assert!(is_zip(PYTHON_ZIP));
    }

    #[test]
    fn stored_and_deflated_members_round_trip() {
        let bytes = write_zip(&[
            ("x.txt", b"hello\nworld\n", false),
            ("dir/y.txt", b"deflated text", true),
        ]);
        let mut archive = Archive::from_bytes(bytes).unwrap();
        assert_eq!(archive.entries().len(), 2);
        assert_eq!(archive.read(0).unwrap(), b"hello\nworld\n");
        assert_eq!(archive.read(1).unwrap(), b"deflated text");
        assert!(archive.read(2).is_err());
    }

    #[test]
    fn a_zip64_archive_reads_like_any_other() {
        let members: &[(&str, &[u8], bool)] = &[("big/a.txt", b"first\n", true), ("b.txt", b"second\n", false)];
        let mut archive = Archive::from_bytes(build_zip(members, true)).unwrap();
        assert_eq!(archive.entries().len(), 2);
        assert_eq!(archive.entries()[0].size, 6);
        assert_eq!(archive.read(0).unwrap(), b"first\n");
        assert_eq!(archive.read(1).unwrap(), b"second\n");
        // and behind a prefix too
        let mut prefixed = b"stub".to_vec();
        prefixed.extend(build_zip(members, true));
        assert_eq!(Archive::from_bytes(prefixed).unwrap().read(1).unwrap(), b"second\n");
    }

    #[test]
    fn a_prefix_in_front_of_the_archive_is_allowed_for() {
        // a self-extractor's stub: every offset in the directory is off by its length
        let mut bytes = b"#!/bin/sh\nexit 0\n".to_vec();
        bytes.extend(write_zip(&[("a.txt", b"one line\n", true)]));
        let mut archive = Archive::from_bytes(bytes).unwrap();
        assert_eq!(archive.read(0).unwrap(), b"one line\n");
    }

    #[test]
    fn a_damaged_member_is_an_error_not_a_crash() {
        let mut bytes = write_zip(&[("a.txt", b"some text here", false)]);
        // flip one byte of the stored data: the CRC no longer matches
        let at = 30 + "a.txt".len();
        bytes[at] ^= 0xff;
        let mut archive = Archive::from_bytes(bytes).unwrap();
        assert!(archive.read(0).unwrap_err().contains("CRC"));
        assert!(Archive::from_bytes(b"PK\x03\x04 not really".to_vec()).is_err());
        assert!(Archive::from_bytes(Vec::new()).is_err());
    }

    #[test]
    fn names_are_decoded_as_zipfile_decodes_them() {
        assert_eq!(decode_name(b"caf\x82.txt", 0), "café.txt", "code page 437");
        assert_eq!(decode_name("café.txt".as_bytes(), FLAG_UTF8), "café.txt");
        assert_eq!(decode_name(b"a\\b.txt", 0), "a/b.txt");
        assert_eq!(decode_name(b"cut\0here", 0), "cut");
    }

    #[test]
    fn the_magic_decides_not_the_name() {
        assert!(is_zip(b"PK\x05\x06"));
        assert!(!is_zip(b"PK"));
        assert!(!is_zip(b"just text"));
        let dir = std::env::temp_dir().join(format!("radixnet-zip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let text = dir.join("really-text.zip");
        std::fs::write(&text, "not an archive").unwrap();
        assert!(!is_zip_file(&text));
        let zip = dir.join("named.txt");
        std::fs::write(&zip, write_zip(&[("a.txt", b"x", false)])).unwrap();
        assert!(is_zip_file(&zip));
        let mut opened = Archive::open(&zip).unwrap();
        assert_eq!(opened.read(0).unwrap(), b"x");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
