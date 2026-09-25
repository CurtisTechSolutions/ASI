//! Streaming corpora: texts read part by part, ZIP archives entry by entry
//! (`go/radixnet/source.go`, `go/radixnet/pipeline.go`, `radixnet/archive.py`).
//!
//! A corpus is a file of lines, or an archive of such files - and an archive
//! is **kept whole** (D-034): it is never unpacked to disk, and nothing holds
//! it in memory; its text entries are read one at a time, when something asks
//! for them.  Everything that reads a training file goes through here, so the
//! CLI's `--data`, the server's uploads and the upload listing all agree on
//! what an archive holds.
//!
//! **Which entries are text** is decided by rules shared with the other two
//! implementations, each with a reason the listing reports: directories,
//! `__MACOSX` metadata, `._*` / `.DS_Store` / `Thumbs.db`, nested archives and
//! encrypted entries by their name and flags ([`skip_reason`]); unreadable
//! entries, binary ones (a NUL in the first 8 KiB) and empty ones by their
//! content.  A text entry is UTF-8, its byte order mark dropped and any bytes
//! that are not UTF-8 replaced - the way Python decodes it.
//!
//! **Parallel parts.** Go streams every entry on its own goroutine; here an
//! archive is read in batches - the compressed bytes of a batch sequentially
//! (the disk is one device), then inflated on the worker pool (the part that
//! costs CPU) - and the texts come out in archive order all the same.

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::SystemTime;

use crate::json::Json;
use crate::parallel::parallel_fill;
use crate::report::split_texts;
use crate::zip::{Archive, Entry};

/// How much of an entry is sniffed for a NUL byte before it is called binary.
pub const SNIFF_BYTES: usize = 8192;

/// An archive entry with one of these endings is another archive, not text.
const ARCHIVE_SUFFIXES: &[&str] = &[".zip", ".gz", ".tgz", ".tar", ".bz2", ".xz", ".7z", ".rar", ".jar"];

/// How many compressed bytes one parallel batch reads before it inflates them.
const BATCH_BYTES: u64 = 32 << 20;
/// ... and how many entries, whichever comes first.
const BATCH_ENTRIES: usize = 256;

/// An archive entry that was not used, and why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Skipped {
    pub path: String,
    pub reason: String,
}

impl Skipped {
    /// `{"path", "reason"}`, as the upload listing reports it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("path", Json::str(self.path.clone())),
            ("reason", Json::str(self.reason.clone())),
        ])
    }
}

/// Why an entry is not read as text, judged by its name and flags alone:
/// `None` for an entry that may be one.
pub fn skip_reason(entry: &Entry) -> Option<&'static str> {
    let path = entry.name.replace('\\', "/");
    let base = path.trim_end_matches('/').rsplit('/').next().unwrap_or("");
    if entry.is_dir() || path.ends_with('/') {
        return Some("directory");
    }
    if path.starts_with("__MACOSX/") || path.contains("/__MACOSX/") {
        return Some("macOS metadata");
    }
    if base.starts_with("._") || base == ".DS_Store" || base == "Thumbs.db" {
        return Some("system file");
    }
    let lower = base.to_lowercase();
    if ARCHIVE_SUFFIXES.iter().any(|suffix| lower.ends_with(suffix)) {
        return Some("nested archive");
    }
    if entry.encrypted() {
        return Some("encrypted");
    }
    None
}

/// Bytes of a text file as text: UTF-8, the byte order mark dropped and
/// anything that is not UTF-8 replaced (Python's `decode("utf-8-sig",
/// errors="replace")`).
pub fn decode_text(bytes: &[u8]) -> String {
    let body = bytes.strip_prefix(&[0xef, 0xbb, 0xbf][..]).unwrap_or(bytes);
    String::from_utf8_lossy(body).into_owned()
}

/// What one entry turned out to be.
enum Outcome {
    Text(String),
    Skip(String),
}

/// An entry's bytes judged by their content: binary, empty, or text.
fn judge(payload: Result<Vec<u8>, String>) -> Outcome {
    let bytes = match payload {
        Ok(bytes) => bytes,
        Err(err) => return Outcome::Skip(format!("unreadable ({err})")),
    };
    if bytes[..bytes.len().min(SNIFF_BYTES)].contains(&0) {
        return Outcome::Skip("binary".to_string());
    }
    let text = decode_text(&bytes);
    if text.trim().is_empty() {
        return Outcome::Skip("empty".to_string());
    }
    Outcome::Text(text)
}

/// Calls `emit(path, text, bytes)` for every text entry of `archive` (`bytes`
/// is what the entry unpacked to), in archive order, and returns the entries
/// it passed over with their reasons (in archive order too).  `emit`
/// returning `Break` ends the walk early.
///
/// The entries are inflated on `workers` threads (0 = the machine, 1 = this
/// one), a batch at a time: only one batch is ever in memory.
pub fn walk_texts<R: Read + Seek>(
    archive: &mut Archive<R>,
    workers: usize,
    emit: &mut dyn FnMut(&str, String, usize) -> ControlFlow<()>,
) -> Vec<Skipped> {
    /// One entry of a batch: its compressed bytes going in, what it was coming out.
    struct Work {
        index: usize,
        packed: Result<Vec<u8>, String>,
        outcome: Option<Outcome>,
    }
    let mut skipped = Vec::new();
    let total = archive.entries().len();
    let mut at = 0usize;
    while at < total {
        // gather a batch: the entries skipped by name cost nothing, the rest
        // are read (still compressed) until the batch is full
        let mut work: Vec<Work> = Vec::new();
        let mut bytes = 0u64;
        let start = at;
        while at < total && work.len() < BATCH_ENTRIES && bytes < BATCH_BYTES {
            if skip_reason(&archive.entries()[at]).is_none() {
                bytes += archive.entries()[at].compressed;
                work.push(Work {
                    index: at,
                    packed: archive.read_packed(at),
                    outcome: None,
                });
            }
            at += 1;
        }
        {
            let entries = archive.entries();
            parallel_fill(&mut work, workers, |_, w| {
                let packed = std::mem::replace(&mut w.packed, Ok(Vec::new()));
                w.outcome = Some(judge(packed.and_then(|p| crate::zip::unpack(&entries[w.index], p))));
            });
        }
        // back in archive order: the entries skipped by name and the ones read interleave
        let mut read = work.into_iter();
        for index in start..at {
            let entry = &archive.entries()[index];
            let path = entry.name.clone();
            if let Some(reason) = skip_reason(entry) {
                skipped.push(Skipped {
                    path,
                    reason: reason.to_string(),
                });
                continue;
            }
            let Some(Work {
                outcome: Some(outcome), ..
            }) = read.next()
            else {
                continue;
            };
            match outcome {
                Outcome::Skip(reason) => skipped.push(Skipped { path, reason }),
                Outcome::Text(text) => {
                    if emit(&path, text, entry.size as usize).is_break() {
                        return skipped;
                    }
                }
            }
        }
    }
    skipped
}

/// `(path, text)` of every text entry of the ZIP archive at `path` - Python's
/// `upload_entries` - or `None` when the file is not an archive.
pub fn archive_entries(path: &Path, workers: usize) -> Result<Option<Vec<(String, String)>>, String> {
    if !crate::zip::is_zip_file(path) {
        return Ok(None);
    }
    let mut archive = Archive::open(path)?;
    let mut out = Vec::new();
    walk_texts(&mut archive, workers, &mut |entry, text, _| {
        out.push((entry.to_string(), text));
        ControlFlow::Continue(())
    });
    Ok(Some(out))
}

/// The training texts of an uploaded archive, as the server reads an upload:
/// every entry whole (when it is not blank), or one text per non-blank line.
/// `None` when the file is not an archive.
pub fn upload_texts(path: &Path, whole_file: bool, workers: usize) -> Result<Option<Vec<String>>, String> {
    let Some(entries) = archive_entries(path, workers)? else {
        return Ok(None);
    };
    let mut texts = Vec::new();
    for (_, text) in entries {
        if whole_file {
            if !text.trim().is_empty() {
                texts.push(text);
            }
        } else {
            texts.extend(split_texts(&text, "lines", 0));
        }
    }
    Ok(Some(texts))
}

// -- what an archive holds, for the upload listing -------------------------------------------------

/// One text entry of an archive, as the listing counts it.
#[derive(Clone, Debug, PartialEq)]
pub struct EntryInfo {
    pub path: String,
    pub bytes: usize,
    pub lines: usize,
}

/// What an archive holds: its text entries with their line counts, and the
/// entries it passed over (Python's `_archive_summary`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ArchiveSummary {
    pub entries: Vec<EntryInfo>,
    pub skipped: Vec<Skipped>,
    pub lines: usize,
    pub chars: usize,
    /// Set when the archive could not be read at all.
    pub error: Option<String>,
}

impl ArchiveSummary {
    /// How many text files the archive holds.
    pub fn files(&self) -> usize {
        self.entries.len()
    }
}

/// Reads every text entry of the archive at `path` and counts it.
pub fn inspect(path: &Path, workers: usize) -> ArchiveSummary {
    let mut archive = match Archive::open(path) {
        Ok(archive) => archive,
        Err(err) => {
            return ArchiveSummary {
                error: Some(err),
                ..Default::default()
            }
        }
    };
    let mut summary = ArchiveSummary::default();
    let skipped = walk_texts(&mut archive, workers, &mut |entry, text, bytes| {
        let lines = count_lines(&text);
        summary.lines += lines;
        summary.chars += text.chars().count();
        summary.entries.push(EntryInfo {
            path: entry.to_string(),
            bytes,
            lines,
        });
        ControlFlow::Continue(())
    });
    summary.skipped = skipped;
    summary
}

/// What [`summary`] remembers: an archive's summary by its path, stamped with
/// the file version (size and modification time) it describes.
type Cache = Mutex<HashMap<PathBuf, (Version, ArchiveSummary)>>;

/// A file's size and modification time: the version of it a summary describes.
type Version = (u64, Option<SystemTime>);

fn cache() -> &'static Cache {
    static CACHE: OnceLock<Cache> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

fn version(path: &Path) -> Version {
    std::fs::metadata(path)
        .map(|m| (m.len(), m.modified().ok()))
        .unwrap_or((0, None))
}

/// [`inspect`], remembered per file version (size and modification time), so
/// listing the uploads does not inflate every archive every time.
pub fn summary(path: &Path, workers: usize) -> ArchiveSummary {
    let key = version(path);
    if let Some((stamp, found)) = cache().lock().unwrap_or_else(|e| e.into_inner()).get(path) {
        if *stamp == key {
            return found.clone();
        }
    }
    let found = inspect(path, workers);
    remember(path, found.clone());
    found
}

/// Keeps `found` as what the archive at `path` holds, as the file is now: the
/// upload route hands over the pass it made while validating an archive, so
/// the listing's record of it does not make a second one.
pub fn remember(path: &Path, found: ArchiveSummary) {
    let key = version(path);
    cache()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(path.to_path_buf(), (key, found));
}

/// The listing record of an upload that is a ZIP archive - its text entries'
/// lines and characters rather than its own bytes read as text - or `None`
/// for any other file.
pub fn archive_record(path: &Path, workers: usize) -> Option<Json> {
    if !crate::zip::is_zip_file(path) {
        return None;
    }
    let meta = std::fs::metadata(path).ok()?;
    let info = summary(path, workers);
    let name = path.file_name()?.to_string_lossy().into_owned();
    let modified = meta
        .modified()
        .map(|t| crate::checkpoint::iso_time(t, 3))
        .unwrap_or_default();
    let mut pairs = vec![
        ("name".to_string(), Json::str(name)),
        ("bytes".to_string(), Json::Int(meta.len() as i64)),
        ("chars".to_string(), Json::Int(info.chars as i64)),
        ("lines".to_string(), Json::Int(info.lines as i64)),
        ("modified".to_string(), Json::str(modified)),
        ("archive".to_string(), Json::Bool(true)),
        ("files".to_string(), Json::Int(info.files() as i64)),
        ("skipped".to_string(), Json::Int(info.skipped.len() as i64)),
    ];
    if let Some(err) = info.error {
        pairs.push(("error".to_string(), Json::str(err)));
    }
    Some(Json::Obj(pairs))
}

/// Non-blank lines, split the way Python's `str.splitlines` splits them - on
/// every line boundary Unicode knows, not only `\n`.
pub fn count_lines(text: &str) -> usize {
    let is_break = |c: char| {
        matches!(
            c,
            '\n' | '\r' | '\u{0b}' | '\u{0c}' | '\u{1c}' | '\u{1d}' | '\u{1e}' | '\u{85}' | '\u{2028}' | '\u{2029}'
        )
    };
    text.split(is_break).filter(|line| !line.trim().is_empty()).count()
}

// -- streaming sources ---------------------------------------------------------------------------

/// A corpus that can be read from the start as often as a pass needs it,
/// without being held in memory: `emit` gets every text in order, and
/// returning `Break` from it ends the reading early.
pub trait TextSource: Send + Sync {
    /// Streams every text; `Break` when `emit` asked to stop.
    fn each(&self, emit: &mut dyn FnMut(String) -> ControlFlow<()>) -> Result<ControlFlow<()>, String>;
}

/// A corpus already in memory.
pub struct SliceSource(pub Vec<String>);

impl TextSource for SliceSource {
    fn each(&self, emit: &mut dyn FnMut(String) -> ControlFlow<()>) -> Result<ControlFlow<()>, String> {
        for text in &self.0 {
            if emit(text.clone()).is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// A text file (gunzipped when its name ends in `.gz`), cut into texts by
/// `unit` - `lines`, `paragraphs`, `pages` or `file` ([`split_texts`]).
pub struct FileSource {
    pub path: PathBuf,
    pub unit: String,
    pub page_lines: usize,
}

impl TextSource for FileSource {
    fn each(&self, emit: &mut dyn FnMut(String) -> ControlFlow<()>) -> Result<ControlFlow<()>, String> {
        let content = crate::cli::read_file_text(&self.path.to_string_lossy())?;
        for text in split_texts(&content, &self.unit, self.page_lines) {
            if emit(text).is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// A ZIP archive on disk: entry by entry, each text entry cut into texts by
/// `unit` (`file` makes every entry one text).  Only a batch of entries is
/// ever in memory, inflated on `workers` threads.
pub struct ZipSource {
    pub path: PathBuf,
    pub unit: String,
    pub page_lines: usize,
    pub workers: usize,
}

impl TextSource for ZipSource {
    fn each(&self, emit: &mut dyn FnMut(String) -> ControlFlow<()>) -> Result<ControlFlow<()>, String> {
        let mut archive = Archive::open(&self.path)?;
        let mut flow = ControlFlow::Continue(());
        walk_texts(&mut archive, self.workers, &mut |_, text, _| {
            for piece in split_texts(&text, &self.unit, self.page_lines) {
                if emit(piece).is_break() {
                    flow = ControlFlow::Break(());
                    return flow;
                }
            }
            ControlFlow::Continue(())
        });
        Ok(flow)
    }
}

/// Several sources, one after the other.
pub struct MultiSource(pub Vec<Box<dyn TextSource>>);

impl TextSource for MultiSource {
    fn each(&self, emit: &mut dyn FnMut(String) -> ControlFlow<()>) -> Result<ControlFlow<()>, String> {
        for source in &self.0 {
            if source.each(emit)?.is_break() {
                return Ok(ControlFlow::Break(()));
            }
        }
        Ok(ControlFlow::Continue(()))
    }
}

/// The source for a file: a [`ZipSource`] when it is an archive (by its magic
/// bytes - the name never decides), a [`FileSource`] otherwise.
pub fn source_for_file(path: &Path, unit: &str, page_lines: usize, workers: usize) -> Box<dyn TextSource> {
    if crate::zip::is_zip_file(path) {
        Box::new(ZipSource {
            path: path.to_path_buf(),
            unit: unit.to_string(),
            page_lines,
            workers,
        })
    } else {
        Box::new(FileSource {
            path: path.to_path_buf(),
            unit: unit.to_string(),
            page_lines,
        })
    }
}

/// Every text of a source, in order.
pub fn collect_texts(source: &dyn TextSource) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let _ = source.each(&mut |text| {
        out.push(text);
        ControlFlow::Continue(())
    })?;
    Ok(out)
}

/// How many texts a source holds, without keeping any of them.
pub fn count_texts(source: &dyn TextSource) -> Result<usize, String> {
    let mut n = 0usize;
    let _ = source.each(&mut |_| {
        n += 1;
        ControlFlow::Continue(())
    })?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::zip::tests::write_zip;

    fn corpus_zip() -> Vec<u8> {
        write_zip(&[
            ("part1.txt", b"the cat sat on the mat\nthe dog sat on the log\n", true),
            ("docs/", b"", false),
            ("more/part2.txt", b"\xef\xbb\xbfa bird flew\n\nover the hill\n", false),
            ("cover.png", b"\x89PNG\x00\x00", false),
            ("__MACOSX/._part1.txt", b"meta", false),
            ("nested.zip", b"PK\x03\x04", false),
            ("blank.txt", b"  \n\n", false),
            ("café.txt", b"caf\xe9 au lait\n", false),
        ])
    }

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("radixnet-source-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn the_rules_say_what_is_text_and_why_the_rest_is_not() {
        let mut archive = Archive::from_bytes(corpus_zip()).unwrap();
        let mut seen = Vec::new();
        let skipped = walk_texts(&mut archive, 2, &mut |path, text, _| {
            seen.push((path.to_string(), text));
            ControlFlow::Continue(())
        });
        let paths: Vec<&str> = seen.iter().map(|(p, _)| p.as_str()).collect();
        assert_eq!(paths, vec!["part1.txt", "more/part2.txt", "café.txt"]);
        assert_eq!(
            seen[1].1, "a bird flew\n\nover the hill\n",
            "the byte order mark is dropped"
        );
        assert_eq!(
            seen[2].1, "caf\u{fffd} au lait\n",
            "bytes that are not UTF-8 are replaced"
        );
        let reasons: Vec<(&str, &str)> = skipped.iter().map(|s| (s.path.as_str(), s.reason.as_str())).collect();
        assert_eq!(
            reasons,
            vec![
                ("docs/", "directory"),
                ("cover.png", "binary"),
                ("__MACOSX/._part1.txt", "macOS metadata"),
                ("nested.zip", "nested archive"),
                ("blank.txt", "empty"),
            ]
        );
    }

    #[test]
    fn a_walk_stops_when_asked() {
        let mut archive = Archive::from_bytes(corpus_zip()).unwrap();
        let mut seen = 0;
        walk_texts(&mut archive, 1, &mut |_, _, _| {
            seen += 1;
            ControlFlow::Break(())
        });
        assert_eq!(seen, 1);
    }

    #[test]
    fn an_archive_streams_in_the_order_it_was_written() {
        // more entries than one batch holds, so the batches have to line up
        let bodies: Vec<String> = (0..600)
            .map(|i| format!("text number {i}\nand its second line\n"))
            .collect();
        let names: Vec<String> = (0..600).map(|i| format!("f{i:03}.txt")).collect();
        let members: Vec<(&str, &[u8], bool)> = names
            .iter()
            .zip(&bodies)
            .map(|(n, b)| (n.as_str(), b.as_bytes(), true))
            .collect();
        let path = temp_file("many.zip", &write_zip(&members));
        let source = source_for_file(&path, "lines", 0, 4);
        let texts = collect_texts(source.as_ref()).unwrap();
        assert_eq!(texts.len(), 1200);
        assert_eq!(texts[0], "text number 0");
        assert_eq!(texts[1199], "and its second line");
        assert_eq!(texts[598], "text number 299");
        assert_eq!(count_texts(source.as_ref()).unwrap(), 1200);
        let whole = collect_texts(source_for_file(&path, "file", 0, 1).as_ref()).unwrap();
        assert_eq!(whole.len(), 600, "an entry is one file");
        assert_eq!(whole[0], "text number 0\nand its second line");
    }

    #[test]
    fn a_plain_file_and_a_slice_are_sources_too() {
        let path = temp_file("plain.zip", b"not really an archive\nsecond\n");
        let texts = collect_texts(source_for_file(&path, "lines", 0, 0).as_ref()).unwrap();
        assert_eq!(
            texts,
            vec!["not really an archive", "second"],
            "the magic decides, not the name"
        );
        let both = MultiSource(vec![
            Box::new(SliceSource(vec!["one".to_string()])),
            source_for_file(&path, "file", 0, 0),
        ]);
        assert_eq!(
            collect_texts(&both).unwrap(),
            vec!["one", "not really an archive\nsecond"]
        );
        let mut first = Vec::new();
        let flow = both
            .each(&mut |t| {
                first.push(t);
                ControlFlow::Break(())
            })
            .unwrap();
        assert!(flow.is_break());
        assert_eq!(first, vec!["one"]);
    }

    #[test]
    fn an_upload_listing_counts_the_entries_not_the_bytes() {
        let path = temp_file("corpus.zip", &corpus_zip());
        let info = inspect(&path, 0);
        assert_eq!(info.files(), 3);
        assert_eq!(info.lines, 2 + 2 + 1);
        assert_eq!(info.skipped.len(), 5);
        assert_eq!(summary(&path, 0), info, "the cached summary is the same");
        let record = archive_record(&path, 0).unwrap();
        assert_eq!(record.at("archive").as_bool(), Some(true));
        assert_eq!(record.at("files").as_i64(), Some(3));
        assert_eq!(record.at("lines").as_i64(), Some(5));
        assert_eq!(record.at("skipped").as_i64(), Some(5));
        assert!(record.at("modified").as_str().unwrap().ends_with("+00:00"));
        let plain = temp_file("plain.txt", b"a\nb\n");
        assert!(archive_record(&plain, 0).is_none());
        let broken = temp_file("broken.bin", b"PK\x03\x04 and then nothing");
        let record = archive_record(&broken, 0).unwrap();
        assert!(record.at("error").as_str().unwrap().contains("not a valid ZIP archive"));
    }

    #[test]
    fn uploads_read_whole_or_by_line() {
        let path = temp_file("up.zip", &corpus_zip());
        let lines = upload_texts(&path, false, 0).unwrap().unwrap();
        assert_eq!(
            lines,
            vec![
                "the cat sat on the mat",
                "the dog sat on the log",
                "a bird flew",
                "over the hill",
                "caf\u{fffd} au lait"
            ]
        );
        let whole = upload_texts(&path, true, 0).unwrap().unwrap();
        assert_eq!(whole.len(), 3);
        assert_eq!(whole[0], "the cat sat on the mat\nthe dog sat on the log\n");
        let plain = temp_file("up.txt", b"x\n");
        assert!(upload_texts(&plain, false, 0).unwrap().is_none());
    }

    #[test]
    fn lines_break_where_python_breaks_them() {
        assert_eq!(count_lines("a\nb\r\nc\rd\u{2028}e\u{0c}f\n\n  \n"), 6);
        assert_eq!(count_lines(""), 0);
    }
}
