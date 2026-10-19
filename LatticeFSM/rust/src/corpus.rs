//! Reading a corpus: every file under some paths - text, markup, documents and archives of any size - as lines, or
//! as bytes.  The same reader as `latticefsm/corpus.py`, member for member and line for line: text (`.txt .md
//! .text .csv .tsv .log .rst .tex .org .srt .vtt`) and `.jsonl` streamed a line at a time, `.json`, HTML / XML
//! (tags stripped), docx / odt / epub, and zip (ZIP64 too), tar, tar.gz / tgz and gz archives opened member by
//! member (`archive.rs`), so neither an archive's size nor a member's is bounded by memory.  For `base64` every file
//! is its bytes, archives giving their members' bytes instead.

use std::fs;
use std::io::{self, BufRead, BufReader, Cursor, Read};
use std::path::Path;

use crate::archive::{tar_each, zip_entries, zip_open, Gzip, ZipEntry};
use crate::json::{parse, Json};
use crate::ngram::split_text_lines;

pub const TEXT: [&str; 11] = [
    ".txt", ".md", ".text", ".csv", ".tsv", ".log", ".rst", ".tex", ".org", ".srt", ".vtt",
];
pub const MARKUP: [&str; 4] = [".html", ".htm", ".xhtml", ".xml"];
/// Bytes a line is read in at most: a longer one is cut there.
pub const MAX_LINE: usize = 1 << 20;
/// How deep archives inside archives are followed.
pub const MAX_DEPTH: usize = 4;

const BLOCK: [&str; 29] = [
    "p",
    "h",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "br",
    "li",
    "tr",
    "title",
    "section",
    "article",
    "blockquote",
    "pre",
    "table",
    "ul",
    "ol",
    "dt",
    "dd",
    "hr",
    "header",
    "footer",
    "nav",
    "aside",
    "caption",
    "figcaption",
];
const SPACE: [&str; 3] = ["td", "th", "tab"];
const ENTITIES: [(&str, &str); 14] = [
    ("amp", "&"),
    ("lt", "<"),
    ("gt", ">"),
    ("quot", "\""),
    ("apos", "'"),
    ("nbsp", " "),
    ("hellip", "\u{2026}"),
    ("mdash", "\u{2014}"),
    ("ndash", "\u{2013}"),
    ("lsquo", "\u{2018}"),
    ("rsquo", "\u{2019}"),
    ("ldquo", "\u{201c}"),
    ("rdquo", "\u{201d}"),
    ("copy", "\u{a9}"),
];

/// What a file is, by its name.
pub fn file_kind(name: &str) -> Option<&'static str> {
    let n = name.to_lowercase();
    if n.ends_with(".tar.gz") || n.ends_with(".tgz") {
        return Some("tgz");
    }
    for (ext, kind) in [
        (".gz", "gz"),
        (".zip", "zip"),
        (".tar", "tar"),
        (".jsonl", "jsonl"),
        (".json", "json"),
        (".docx", "docx"),
        (".odt", "odt"),
        (".epub", "epub"),
    ] {
        if n.ends_with(ext) {
            return Some(kind);
        }
    }
    if MARKUP.iter().any(|x| n.ends_with(x)) {
        return Some("markup");
    }
    if TEXT.iter().any(|x| n.ends_with(x)) {
        return Some("text");
    }
    None
}

fn hidden(member: &str) -> bool {
    member.split('/').any(|p| p.starts_with('.') || p == "__MACOSX")
}

fn walk(dir: &Path, everything: bool, out: &mut Vec<(String, bool)>) -> Result<(), String> {
    let (mut files, mut dirs) = (Vec::new(), Vec::new());
    for e in fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let p = e.map_err(|e| e.to_string())?.path();
        let name = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if name.starts_with('.') {
            continue;
        }
        if p.is_dir() {
            dirs.push((name, p));
        } else if name != "README.md" && (everything || file_kind(&name).is_some()) {
            files.push((name, p));
        }
    }
    files.sort();
    dirs.sort();
    out.extend(
        files
            .into_iter()
            .map(|(_, p)| (p.to_string_lossy().into_owned(), false)),
    );
    for (_, d) in dirs {
        walk(&d, everything, out)?;
    }
    Ok(())
}

/// Every file under `paths` as (path, named directly): directories walked in name order, hidden files and
/// directories and `README.md` left out, and unless `everything` the files of no known kind too.
pub fn files(paths: &[String], everything: bool) -> Result<Vec<(String, bool)>, String> {
    let mut out = Vec::new();
    for p in paths {
        let path = Path::new(p);
        if path.is_dir() {
            walk(path, everything, &mut out)?;
        } else if path.exists() {
            out.push((p.clone(), true));
        } else {
            return Err(format!("{p}: no such file or directory"));
        }
    }
    Ok(out)
}

// ---- archives --------------------------------------------------------------------------------------------------------

fn wanted(e: &ZipEntry) -> bool {
    !(e.is_dir() || hidden(&e.name) || e.flags & 1 != 0 || !matches!(e.method, 0 | 8))
}

/// What reads a text document: `(path, label, kind, reader)`.
pub type TextVisitor<'a> = &'a mut dyn FnMut(&str, &str, &str, &mut dyn Read) -> Result<(), String>;
/// What reads a file's bytes: `(path, label, reader)`.
pub type ByteVisitor<'a> = &'a mut dyn FnMut(&str, &str, &mut dyn Read) -> Result<(), String>;

type Leaf<'a> = &'a mut dyn FnMut(&str, Option<&'static str>, &mut dyn Read) -> Result<(), String>;

enum Src<'a> {
    File(&'a mut fs::File),
    Stream(&'a mut dyn Read),
}

fn is_archive(kind: Option<&str>) -> bool {
    matches!(kind, Some("zip" | "tar" | "tgz" | "gz"))
}

fn zip_each<R: Read + io::Seek>(
    r: &mut R,
    label: &str,
    depth: usize,
    keep: &dyn Fn(Option<&str>) -> bool,
    f: Leaf<'_>,
) -> Result<(), String> {
    let entries = zip_entries(r).map_err(|e| format!("{label}: {e}"))?;
    for e in entries.iter().filter(|e| wanted(e)) {
        let mut member = zip_open(r, e).map_err(|err| format!("{label}/{}: {err}", e.name))?;
        let sub = format!("{label}/{}", e.name);
        expand(
            Src::Stream(&mut member),
            &e.name,
            &sub,
            depth + 1,
            file_kind(&e.name),
            keep,
            &mut *f,
        )?;
    }
    Ok(())
}

/// A file (called `name`, known as `label`) with its archives opened; `keep(kind)` says whether a document of that
/// kind is wanted as it is.
fn expand(
    src: Src<'_>,
    name: &str,
    label: &str,
    depth: usize,
    kind: Option<&'static str>,
    keep: &dyn Fn(Option<&str>) -> bool,
    f: Leaf<'_>,
) -> Result<(), String> {
    if is_archive(kind) {
        if depth >= MAX_DEPTH {
            return Ok(());
        }
        let ctx = |e: io::Error| format!("{label}: {e}");
        match (kind, src) {
            (Some("zip"), Src::File(file)) => zip_each(file, label, depth, keep, f),
            (Some("zip"), Src::Stream(r)) => {
                let mut data = Vec::new();
                r.read_to_end(&mut data).map_err(ctx)?;
                zip_each(&mut Cursor::new(data), label, depth, keep, f)
            }
            (Some("gz"), src) => {
                let r: &mut dyn Read = match src {
                    Src::File(file) => file,
                    Src::Stream(r) => r,
                };
                let mut g = Gzip::new(r).map_err(ctx)?;
                let inner = &name[..name.len() - 3];
                expand(
                    Src::Stream(&mut g),
                    inner,
                    label,
                    depth + 1,
                    file_kind(inner),
                    keep,
                    &mut *f,
                )
            }
            (Some(k), src) => {
                let r: &mut dyn Read = match src {
                    Src::File(file) => file,
                    Src::Stream(r) => r,
                };
                let mut each = |member: &str, m: &mut dyn Read| {
                    if hidden(member) {
                        return Ok(());
                    }
                    let sub = format!("{label}/{member}");
                    expand(
                        Src::Stream(m),
                        member,
                        &sub,
                        depth + 1,
                        file_kind(member),
                        keep,
                        &mut *f,
                    )
                };
                if k == "tgz" {
                    let g = Gzip::new(r).map_err(ctx)?;
                    tar_each(g, &mut each)
                } else {
                    tar_each(r, &mut each)
                }
                .map_err(|e| {
                    if e.starts_with(label) {
                        e
                    } else {
                        format!("{label}: {e}")
                    }
                })
            }
            _ => Ok(()),
        }
    } else if keep(kind) {
        let r: &mut dyn Read = match src {
            Src::File(file) => file,
            Src::Stream(r) => r,
        };
        f(label, kind, r)
    } else {
        Ok(())
    }
}

/// Every text document under `paths`, handed to `f(path, label, kind, reader)`: `path` the file it came from,
/// `label` the file or, inside an archive, `archive/member`.
pub fn text_documents(paths: &[String], f: TextVisitor<'_>) -> Result<(), String> {
    for (path, named) in files(paths, false)? {
        let mut kind = file_kind(&path);
        if kind.is_none() && named {
            kind = Some("text");
        }
        let mut file = fs::File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let base = Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut leaf = |label: &str, k: Option<&'static str>, r: &mut dyn Read| f(&path, label, k.unwrap_or("text"), r);
        expand(Src::File(&mut file), &base, &path, 0, kind, &|k| k.is_some(), &mut leaf)?;
    }
    Ok(())
}

/// Every file under `paths` as its bytes, handed to `f(path, label, reader)`, archives opened into their members.
pub fn byte_documents(paths: &[String], f: ByteVisitor<'_>) -> Result<(), String> {
    for (path, _) in files(paths, true)? {
        let mut file = fs::File::open(&path).map_err(|e| format!("{path}: {e}"))?;
        let base = Path::new(&path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let mut leaf = |label: &str, _: Option<&'static str>, r: &mut dyn Read| f(&path, label, r);
        expand(
            Src::File(&mut file),
            &base,
            &path,
            0,
            file_kind(&path),
            &|_| true,
            &mut leaf,
        )?;
    }
    Ok(())
}

/// `r` `size` bytes at a time, the last chunk shorter.
pub fn read_chunks(
    r: &mut dyn Read,
    size: usize,
    f: &mut dyn FnMut(&[u8]) -> Result<(), String>,
) -> Result<(), String> {
    let mut buf = vec![0u8; size];
    loop {
        let mut got = 0;
        while got < size {
            let n = r.read(&mut buf[got..]).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got == 0 {
            return Ok(());
        }
        f(&buf[..got])?;
        if got < size {
            return Ok(());
        }
    }
}

// ---- text ------------------------------------------------------------------------------------------------------------

fn json_text(v: &Json) -> String {
    match v {
        Json::String(s) => s.clone(),
        Json::Object(pairs) => {
            for key in ["text", "content", "sentence", "body"] {
                if let Some(Json::String(s)) = v.get(key) {
                    return s.clone();
                }
            }
            pairs
                .iter()
                .filter_map(|(_, x)| x.as_str())
                .collect::<Vec<_>>()
                .join(" ")
        }
        Json::Array(items) => items.iter().map(json_text).collect::<Vec<_>>().join("\n"),
        _ => String::new(),
    }
}

/// A tag's local name, lower-cased (ASCII): `</w:p>` is `p`.
fn tag_name(tag: &str) -> String {
    let t = tag.trim_start_matches('/');
    let end = t.find([' ', '\t', '\n', '\r', '\x0c', '\x0b', '/']).unwrap_or(t.len());
    let name = t[..end].to_ascii_lowercase();
    match name.rfind(':') {
        Some(i) => name[i + 1..].to_string(),
        None => name,
    }
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let b = s.as_bytes();
    let mut i = 0;
    let mut start = 0;
    while let Some(off) = s[i..].find('&') {
        let amp = i + off;
        let mut j = amp + 1;
        let mut replaced: Option<String> = None;
        if j < b.len() && b[j] == b'#' {
            j += 1;
            let hex = j < b.len() && (b[j] == b'x' || b[j] == b'X');
            if hex {
                j += 1;
            }
            let digits = j;
            while j < b.len()
                && (if hex {
                    b[j].is_ascii_hexdigit()
                } else {
                    b[j].is_ascii_digit()
                })
            {
                j += 1;
            }
            if j > digits && j < b.len() && b[j] == b';' {
                let v = u32::from_str_radix(&s[digits..j], if hex { 16 } else { 10 }).ok();
                replaced = Some(match v.filter(|&v| v > 0).and_then(char::from_u32) {
                    Some(c) => c.to_string(),
                    None => s[amp..=j].to_string(),
                });
            }
        } else {
            let letters = j;
            while j < b.len() && b[j].is_ascii_alphabetic() {
                j += 1;
            }
            if j > letters && j < b.len() && b[j] == b';' {
                let name = &s[letters..j];
                replaced = Some(
                    ENTITIES
                        .iter()
                        .find(|(k, _)| *k == name)
                        .map(|(_, v)| v.to_string())
                        .unwrap_or_else(|| s[amp..=j].to_string()),
                );
            }
        }
        match replaced {
            Some(r) => {
                out.push_str(&s[start..amp]);
                out.push_str(&r);
                i = j + 1;
                start = i;
            }
            None => i = amp + 1,
        }
    }
    out.push_str(&s[start..]);
    out
}

/// The text of HTML or XML, as `corpus.strip_markup` reads it.
pub fn strip_markup(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let n = s.len();
    let mut i = 0;
    while i < n {
        let Some(off) = s[i..].find('<') else {
            out.push_str(&s[i..]);
            break;
        };
        let j = i + off;
        out.push_str(&s[i..j]);
        if s[j..].starts_with("<!--") {
            i = s[j + 4..].find("-->").map(|k| j + 4 + k + 3).unwrap_or(n);
            continue;
        }
        if s[j..].starts_with("<![CDATA[") {
            match s[j + 9..].find("]]>") {
                Some(k) => {
                    out.push_str(&s[j + 9..j + 9 + k]);
                    i = j + 9 + k + 3;
                }
                None => {
                    out.push_str(&s[j + 9..]);
                    i = n;
                }
            }
            continue;
        }
        let Some(k) = s[j + 1..].find('>').map(|k| j + 1 + k) else {
            out.push_str(&s[j..]);
            break;
        };
        let tag = &s[j + 1..k];
        i = k + 1;
        if tag.starts_with('!') || tag.starts_with('?') {
            continue;
        }
        let name = tag_name(tag);
        if (name == "script" || name == "style") && !tag.starts_with('/') && !tag.ends_with('/') {
            loop {
                let Some(c) = s[i..].find("</").map(|c| i + c) else {
                    i = n;
                    break;
                };
                let Some(e) = s[c + 2..].find('>').map(|e| c + 2 + e) else {
                    i = n;
                    break;
                };
                i = e + 1;
                if tag_name(&s[c + 1..e]) == name {
                    break;
                }
            }
            continue;
        }
        if BLOCK.contains(&name.as_str()) {
            out.push('\n');
        } else if SPACE.contains(&name.as_str()) {
            out.push(' ');
        }
    }
    decode_entities(&out)
}

fn markup_of_zip(data: Vec<u8>, kind: &str) -> Option<String> {
    let mut cur = Cursor::new(data);
    let entries = zip_entries(&mut cur).ok()?;
    let names: Vec<&ZipEntry> = entries
        .iter()
        .filter(|e| wanted(e))
        .filter(|e| match kind {
            "docx" => e.name == "word/document.xml",
            "odt" => e.name == "content.xml",
            _ => {
                let l = e.name.to_lowercase();
                l.ends_with(".xhtml") || l.ends_with(".html") || l.ends_with(".htm")
            }
        })
        .collect();
    let mut parts = Vec::new();
    for e in names {
        let mut bytes = Vec::new();
        zip_open(&mut cur, e).ok()?.read_to_end(&mut bytes).ok()?;
        parts.push(strip_markup(&String::from_utf8_lossy(&bytes)));
    }
    Some(parts.join("\n"))
}

/// One line of at most `limit` bytes (its `\n` kept), or `false` at the end.
fn read_line_limited(r: &mut impl BufRead, buf: &mut Vec<u8>, limit: usize) -> io::Result<bool> {
    buf.clear();
    while buf.len() < limit {
        let avail = r.fill_buf()?;
        if avail.is_empty() {
            break;
        }
        let room = (limit - buf.len()).min(avail.len());
        match avail[..room].iter().position(|&c| c == b'\n') {
            Some(p) => {
                buf.extend_from_slice(&avail[..=p]);
                r.consume(p + 1);
                return Ok(true);
            }
            None => {
                buf.extend_from_slice(&avail[..room]);
                r.consume(room);
            }
        }
    }
    Ok(!buf.is_empty())
}

fn clean(lines: Vec<String>, f: &mut dyn FnMut(String) -> Result<(), String>) -> Result<(), String> {
    for l in lines {
        let l = l.trim();
        if !l.is_empty() {
            f(l.to_string())?;
        }
    }
    Ok(())
}

/// The non-empty lines (trimmed) of a document of `kind`, handed to `f` one at a time.
pub fn text_lines(kind: &str, r: &mut dyn Read, f: &mut dyn FnMut(String) -> Result<(), String>) -> Result<(), String> {
    let err = |e: io::Error| e.to_string();
    if kind == "text" || kind == "jsonl" {
        let mut br = BufReader::with_capacity(1 << 16, r);
        let mut buf = Vec::new();
        while read_line_limited(&mut br, &mut buf, MAX_LINE).map_err(err)? {
            for line in split_text_lines(&String::from_utf8_lossy(&buf)) {
                let l = line.trim();
                if l.is_empty() {
                    continue;
                }
                if kind == "text" {
                    f(l.to_string())?;
                    continue;
                }
                match parse(l) {
                    Ok(j) => clean(split_text_lines(&json_text(&j)), f)?,
                    Err(_) => f(l.to_string())?,
                }
            }
        }
        return Ok(());
    }
    let mut data = Vec::new();
    r.read_to_end(&mut data).map_err(err)?;
    match kind {
        "json" => {
            let text = String::from_utf8_lossy(&data).into_owned();
            match parse(&text) {
                Ok(j) => clean(split_text_lines(&json_text(&j)), f),
                Err(_) => clean(split_text_lines(&text), f),
            }
        }
        "markup" => clean(split_text_lines(&strip_markup(&String::from_utf8_lossy(&data))), f),
        "docx" | "odt" | "epub" => match markup_of_zip(data, kind) {
            Some(text) => clean(split_text_lines(&text), f),
            None => Ok(()),
        },
        _ => Ok(()),
    }
}
