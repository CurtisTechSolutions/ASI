//! Text, audio, images and video as symbols a network of machines can read.
//!
//! The same as `latticefsm/media.py`, in the same exact integer arithmetic, so the two ports turn every file into the
//! same symbols: text as lower-cased characters; a WAV file as the phonetic tokenizer's 64 acoustic units
//! (`../../PhoneticTokenizer/rust`); an image (PNG 8-bit not interlaced, PPM / PGM, BMP 24/32-bit) as a `grid x grid`
//! of brightness levels `L0..`; two frames as their cell-by-cell difference `D-2..D+2`; a video (a directory of frames
//! and `audio.wav`, or a file ffmpeg splits) as `|`, a keyframe or a difference, and the audio units of each frame.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use phonetok::acoustic::AcousticTokenizer;

use crate::gzip;
use crate::rng::Rng;

/// What text is read as: 43 symbols.
pub const TEXT_SYMBOLS: [&str; 43] = [
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w",
    "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "_", ".", ",", "?", "!", "'", "-",
];
/// What a change in a cell is read as, darker to lighter.
pub const DIFFS: [&str; 5] = ["D-2", "D-1", "D0", "D+1", "D+2"];
pub const IMAGE_EXTENSIONS: [&str; 5] = [".png", ".ppm", ".pgm", ".pnm", ".bmp"];

/// `text` lower-cased, whitespace as `_`, every character not in [`TEXT_SYMBOLS`] dropped.
pub fn text_symbols(text: &str) -> Vec<String> {
    text.to_lowercase()
        .chars()
        .map(|c| if c.is_whitespace() { '_' } else { c })
        .map(|c| c.to_string())
        .filter(|c| TEXT_SYMBOLS.contains(&c.as_str()))
        .collect()
}

/// An image: its width, height and one brightness 0-255 per pixel, row by row.
pub struct Image {
    pub width: usize,
    pub height: usize,
    pub px: Vec<u32>,
}

fn luma(r: u32, g: u32, b: u32) -> u32 {
    (299 * r + 587 * g + 114 * b) / 1000
}

/// An image file's bytes: PNG, PPM / PGM or BMP.
pub fn read_image_bytes(data: &[u8]) -> Result<Image, String> {
    if data.starts_with(b"\x89PNG\r\n\x1a\n") {
        png(data)
    } else if data.len() > 2 && data[0] == b'P' && b"2356".contains(&data[1]) {
        pnm(data)
    } else if data.starts_with(b"BM") {
        bmp(data)
    } else {
        Err("not a PNG, PPM / PGM or BMP image".to_string())
    }
}

pub fn read_image(path: &str) -> Result<Image, String> {
    read_image_bytes(&fs::read(path).map_err(|e| format!("{path}: {e}"))?).map_err(|e| format!("{path}: {e}"))
}

fn be32(d: &[u8], at: usize) -> usize {
    u32::from_be_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]]) as usize
}

fn le32(d: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([d[at], d[at + 1], d[at + 2], d[at + 3]])
}

fn png(data: &[u8]) -> Result<Image, String> {
    let (mut pos, mut idat, mut palette) = (8usize, Vec::new(), Vec::new());
    let (mut width, mut height, mut ctype) = (0usize, 0usize, 0u8);
    while pos + 8 <= data.len() {
        let n = be32(data, pos);
        let tag = &data[pos + 4..pos + 8];
        let body = data.get(pos + 8..pos + 8 + n).ok_or("a PNG chunk runs past the end")?;
        pos += 12 + n;
        match tag {
            b"IHDR" => {
                width = be32(body, 0);
                height = be32(body, 4);
                ctype = body[9];
                if body[8] != 8 || body[12] != 0 {
                    return Err("only 8-bit, non-interlaced PNGs are read".to_string());
                }
            }
            b"PLTE" => palette = body.to_vec(),
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {}
        }
    }
    let ch = match ctype {
        0 | 3 => 1,
        2 => 3,
        4 => 2,
        6 => 4,
        other => return Err(format!("PNG colour type {other}")),
    };
    if idat.len() < 2 {
        return Err("a PNG without image data".to_string());
    }
    let raw = gzip::inflate(&idat[2..])?;
    let stride = width * ch;
    let mut px = Vec::with_capacity(width * height);
    let mut prev = vec![0u8; stride];
    let mut at = 0;
    for _ in 0..height {
        let ftype = *raw.get(at).ok_or("PNG data too short")?;
        let mut line = raw.get(at + 1..at + 1 + stride).ok_or("PNG data too short")?.to_vec();
        at += 1 + stride;
        for i in 0..stride {
            let a = if i >= ch { line[i - ch] as i32 } else { 0 };
            let b = prev[i] as i32;
            let c = if i >= ch { prev[i - ch] as i32 } else { 0 };
            let add = match ftype {
                1 => a,
                2 => b,
                3 => (a + b) / 2,
                4 => {
                    let p = a + b - c;
                    let (pa, pb, pc) = ((p - a).abs(), (p - b).abs(), (p - c).abs());
                    if pa <= pb && pa <= pc {
                        a
                    } else if pb <= pc {
                        b
                    } else {
                        c
                    }
                }
                _ => 0,
            };
            line[i] = ((line[i] as i32 + add) & 255) as u8;
        }
        for x in 0..width {
            let p = &line[x * ch..(x + 1) * ch];
            px.push(if ctype == 3 {
                let k = p[0] as usize * 3;
                let pal = palette.get(k..k + 3).ok_or("a PNG palette index out of range")?;
                luma(pal[0] as u32, pal[1] as u32, pal[2] as u32)
            } else if ch >= 3 {
                luma(p[0] as u32, p[1] as u32, p[2] as u32)
            } else {
                p[0] as u32
            });
        }
        prev = line;
    }
    Ok(Image { width, height, px })
}

fn pnm(data: &[u8]) -> Result<Image, String> {
    let kind = data[1];
    let mut fields = Vec::new();
    let mut pos = 2;
    while fields.len() < 3 {
        while pos < data.len() && data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        if data.get(pos) == Some(&b'#') {
            while pos < data.len() && data[pos] != b'\n' {
                pos += 1;
            }
            continue;
        }
        let start = pos;
        while pos < data.len() && !data[pos].is_ascii_whitespace() {
            pos += 1;
        }
        let t = std::str::from_utf8(&data[start..pos]).map_err(|_| "a bad PPM header")?;
        fields.push(t.parse::<usize>().map_err(|_| "a bad PPM header")?);
    }
    let (width, height, top) = (fields[0], fields[1], fields[2]);
    pos += 1;
    let ch = if kind == b'3' || kind == b'6' { 3 } else { 1 };
    let n = width * height * ch;
    let mut values: Vec<u32> = if kind == b'5' || kind == b'6' {
        if top > 255 {
            return Err("only 8-bit PPM / PGM files are read".to_string());
        }
        data.get(pos..pos + n)
            .ok_or("PPM data too short")?
            .iter()
            .map(|&b| b as u32)
            .collect()
    } else {
        std::str::from_utf8(data.get(pos..).unwrap_or(&[]))
            .map_err(|_| "bad PPM text")?
            .split_ascii_whitespace()
            .take(n)
            .map(|t| t.parse::<u32>().map_err(|_| "bad PPM text".to_string()))
            .collect::<Result<_, _>>()?
    };
    if top != 255 {
        values = values.iter().map(|&v| v * 255 / top as u32).collect();
    }
    let px = if ch == 1 {
        values
    } else {
        values.chunks(3).map(|c| luma(c[0], c[1], c[2])).collect()
    };
    Ok(Image { width, height, px })
}

fn bmp(data: &[u8]) -> Result<Image, String> {
    if data.len() < 54 {
        return Err("a BMP too short".to_string());
    }
    let offset = le32(data, 10) as usize;
    let width = le32(data, 18) as i32;
    let height = le32(data, 22) as i32;
    let bits = u16::from_le_bytes([data[28], data[29]]);
    let compression = le32(data, 30);
    if !(bits == 24 || bits == 32) || !(compression == 0 || compression == 3) {
        return Err("only 24- or 32-bit uncompressed BMPs are read".to_string());
    }
    let (w, h) = (width as usize, height.unsigned_abs() as usize);
    let bpp = bits as usize / 8;
    let stride = (w * bpp).div_ceil(4) * 4;
    let mut rows = Vec::with_capacity(h);
    for r in 0..h {
        let at = offset + r * stride;
        let row: Vec<u32> = (0..w)
            .map(|x| {
                let p = at + x * bpp;
                luma(data[p + 2] as u32, data[p + 1] as u32, data[p] as u32)
            })
            .collect();
        rows.push(row);
    }
    if height > 0 {
        rows.reverse();
    }
    Ok(Image {
        width: w,
        height: h,
        px: rows.concat(),
    })
}

/// The image averaged down to `grid x grid` cells, row by row: each the integer mean of its pixels.
pub fn cells(img: &Image, grid: usize) -> Vec<u32> {
    let (w, h) = (img.width, img.height);
    let mut out = Vec::with_capacity(grid * grid);
    for cy in 0..grid {
        let y0 = cy * h / grid;
        let y1 = (y0 + 1).max((cy + 1) * h / grid).min(h);
        for cx in 0..grid {
            let x0 = cx * w / grid;
            let x1 = (x0 + 1).max((cx + 1) * w / grid).min(w);
            let (mut total, mut count) = (0u64, 0u64);
            for y in y0..y1 {
                for x in x0..x1 {
                    total += img.px[y * w + x] as u64;
                    count += 1;
                }
            }
            out.push(total.checked_div(count).unwrap_or(0) as u32);
        }
    }
    out
}

/// The order cells are read in: `rows`, or `center-out` - ring by ring from the middle, each ring row by row.
pub fn order(grid: usize, how: &str) -> Result<Vec<usize>, String> {
    let mut v: Vec<usize> = (0..grid * grid).collect();
    match how {
        "rows" => Ok(v),
        "center-out" => {
            let c2 = grid as i64 - 1;
            v.sort_by_key(|&i| {
                let (r, c) = ((i / grid) as i64, (i % grid) as i64);
                ((2 * r - c2).abs().max((2 * c - c2).abs()), r, c)
            });
            Ok(v)
        }
        other => Err(format!("order is rows or center-out, got {other:?}")),
    }
}

pub fn level_symbols(levels: usize) -> Vec<String> {
    (0..levels).map(|i| format!("L{i}")).collect()
}

fn side(n: usize) -> usize {
    (n as f64).sqrt().round() as usize
}

pub fn image_symbols(grid_values: &[u32], levels: usize, how: &str) -> Result<Vec<String>, String> {
    let grid = side(grid_values.len());
    Ok(order(grid, how)?
        .into_iter()
        .map(|i| format!("L{}", grid_values[i] as usize * levels / 256))
        .collect())
}

pub fn diff_symbols(before: &[u32], after: &[u32], how: &str) -> Result<Vec<String>, String> {
    let grid = side(after.len());
    Ok(order(grid, how)?
        .into_iter()
        .map(|i| {
            let d = after[i] as i64 - before[i] as i64;
            let m = d.abs();
            let k = if m < 8 {
                0
            } else if m < 48 {
                1
            } else {
                2
            };
            DIFFS[(2 + if d > 0 { k } else { -k }) as usize].to_string()
        })
        .collect())
}

/// The acoustic units, `q0` .. `q63`.
pub fn unit_names() -> Vec<String> {
    AcousticTokenizer::bundled().map(|t| t.book.names()).unwrap_or_default()
}

/// A WAV file's bytes heard as acoustic units.
pub fn audio_units_bytes(wav: &[u8]) -> Result<Vec<String>, String> {
    Ok(AcousticTokenizer::bundled()?.listen_wav(wav)?.units)
}

pub fn audio_units(path: &str) -> Result<Vec<String>, String> {
    audio_units_bytes(&fs::read(path).map_err(|e| format!("{path}: {e}"))?)
}

/// A video's frame image paths, its audio WAV path, and a temporary directory to remove when done.
pub type Frames = (Vec<String>, Option<String>, Option<PathBuf>);

/// A video as (frame image paths, audio WAV path, a temporary directory to remove): a directory's images in name order
/// and its `audio.wav`; a video file split by ffmpeg into `fps` frames a second of 64 x 64 and a 16 kHz mono track.
pub fn video_frames(source: &str, fps: usize) -> Result<Frames, String> {
    let list = |dir: &Path, exts: &[&str]| -> Result<Vec<String>, String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                let name = p.to_string_lossy().to_lowercase();
                exts.iter().any(|x| name.ends_with(x))
            })
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        v.sort();
        Ok(v)
    };
    let path = Path::new(source);
    if path.is_dir() {
        let audio = path.join("audio.wav");
        let audio = audio.exists().then(|| audio.to_string_lossy().into_owned());
        return Ok((list(path, &IMAGE_EXTENSIONS)?, audio, None));
    }
    let tmp = std::env::temp_dir().join(format!("latticefsm-video-{}-{}", std::process::id(), fastid(source)));
    fs::create_dir_all(&tmp).map_err(|e| e.to_string())?;
    let ok = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-i",
            source,
            "-vf",
            &format!("fps={fps},scale=64:64"),
        ])
        .arg(tmp.join("frame%05d.ppm"))
        .status()
        .map_err(|_| format!("{source}: a video file needs ffmpeg; or give a directory of frames (and audio.wav)"))?;
    if !ok.success() {
        return Err(format!("{source}: ffmpeg could not read it"));
    }
    let audio = tmp.join("audio.wav");
    let got = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-i",
            source,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
        ])
        .arg(&audio)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    let audio = (got && audio.exists()).then(|| audio.to_string_lossy().into_owned());
    Ok((list(&tmp, &[".ppm"])?, audio, Some(tmp)))
}

fn fastid(s: &str) -> u64 {
    s.bytes()
        .fold(0xcbf29ce484222325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100000001b3))
}

/// The frames of a video averaged to grids, the first `frames` of them (all for 0), and its audio's units.
pub fn video_grids(
    source: &str,
    grid: usize,
    frames: usize,
    audio: bool,
    fps: usize,
) -> Result<(Vec<Vec<u32>>, Vec<String>), String> {
    let (mut paths, wav, tmp) = video_frames(source, fps)?;
    if frames > 0 {
        paths.truncate(frames);
    }
    let result = (|| {
        let grids = paths
            .iter()
            .map(|p| read_image(p).map(|i| cells(&i, grid)))
            .collect::<Result<Vec<_>, _>>()?;
        let units = match (&wav, audio) {
            (Some(w), true) => audio_units(w)?,
            _ => Vec::new(),
        };
        Ok((grids, units))
    })();
    if let Some(t) = tmp {
        let _ = fs::remove_dir_all(t);
    }
    result
}

/// `|` before each frame; the first (and every `keyframe`-th) as an image, the rest as differences; after each frame
/// the audio units that sound during it (the units split evenly over the frames).
#[allow(clippy::too_many_arguments)]
pub fn video_symbols(
    source: &str,
    grid: usize,
    levels: usize,
    how: &str,
    keyframe: usize,
    frames: usize,
    audio: bool,
    fps: usize,
) -> Result<Vec<String>, String> {
    let (grids, units) = video_grids(source, grid, frames, audio, fps)?;
    let n = grids.len();
    let mut out = Vec::new();
    for (t, g) in grids.iter().enumerate() {
        out.push("|".to_string());
        if t == 0 || (keyframe > 0 && t % keyframe == 0) {
            out.extend(image_symbols(g, levels, how)?);
        } else {
            out.extend(diff_symbols(&grids[t - 1], g, how)?);
        }
        out.extend(units[t * units.len() / n..(t + 1) * units.len() / n].iter().cloned());
    }
    Ok(out)
}

/// The differences between consecutive frames, each after a `|`.
pub fn diff_stream(source: &str, grid: usize, how: &str, frames: usize) -> Result<Vec<String>, String> {
    let (grids, _) = video_grids(source, grid, frames, false, 8)?;
    let mut out = Vec::new();
    for w in grids.windows(2) {
        out.push("|".to_string());
        out.extend(diff_symbols(&w[0], &w[1], how)?);
    }
    Ok(out)
}

/// A `input<TAB>label` file: each input (a path relative to the file's directory, when `resolve`) and its label.
pub fn read_labels(path: &str, resolve: bool) -> Result<Vec<(String, String)>, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let base = Path::new(path).parent().map(Path::to_path_buf).unwrap_or_default();
    let base = fs::canonicalize(if base.as_os_str().is_empty() {
        Path::new(".")
    } else {
        &base
    })
    .map_err(|e| format!("{path}: {e}"))?;
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() || line.starts_with('#') {
            continue;
        }
        let (x, y) = line
            .split_once('\t')
            .ok_or_else(|| format!("{path}: each line is input<TAB>label, got {line:?}"))?;
        let x = x.trim();
        let x = if resolve && !Path::new(x).is_absolute() {
            base.join(x).to_string_lossy().into_owned()
        } else {
            x.to_string()
        };
        out.push((x, y.trim().to_string()));
    }
    Ok(out)
}

/// A verdict label: 1/0, yes/no, true/false, accept/reject.
pub fn verdict(label: &str) -> Result<bool, String> {
    match label.trim().to_lowercase().as_str() {
        "1" | "yes" | "true" | "accept" | "y" | "t" => Ok(true),
        "0" | "no" | "false" | "reject" | "n" | "f" => Ok(false),
        other => Err(format!("a verdict is 1/0, yes/no or true/false, got {other:?}")),
    }
}

fn ppm(path: &Path, w: usize, h: usize, px: &[u8]) -> Result<(), String> {
    let mut bytes = format!("P5\n{w} {h}\n255\n").into_bytes();
    bytes.extend_from_slice(px);
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn speak(path: &Path, text: &str) -> Result<(), String> {
    use phonetok::synth::{wav_bytes, Synthesizer, VoiceSettings, RATE};
    let tokens = crate::phonetic::tokenizer(true).tokens(text);
    let pcm = Synthesizer::new(RATE, VoiceSettings::default()).speak(&tokens);
    fs::write(path, wav_bytes(&pcm, RATE)).map_err(|e| format!("{}: {e}", path.display()))
}

/// Write a small labelled data set of `kind` (text, image, audio, video) into `out` with its `labels.tsv`, drawn from
/// this port's generator (`media.py` has the same sets, drawn from Python's); returns the file's path.
pub fn make_dataset(kind: &str, out: &str, count: usize, seed: u64) -> Result<String, String> {
    let mut rng = Rng::new(seed);
    let dir = Path::new(out);
    fs::create_dir_all(dir).map_err(|e| format!("{out}: {e}"))?;
    let mut rows: Vec<(String, u8)> = Vec::new();
    match kind {
        "text" => {
            let words: Vec<&str> = "the a cat dog sat ran saw big small red old house tree by in on"
                .split(' ')
                .collect();
            for i in 0..count {
                let n = rng.randint(3, 6);
                let mut s: Vec<&str> = (0..n).map(|_| *rng.choice(&words)).collect();
                let neg = i % 2 == 1;
                if neg {
                    let at = rng.randint(1, s.len());
                    s.insert(at, *rng.choice(&["not", "no", "never"]));
                }
                rows.push((s.join(" "), neg as u8));
            }
        }
        "image" => {
            for i in 0..count {
                let vertical = i % 2 == 0;
                let period = rng.randint(4, 8);
                let phase = rng.randint(0, period - 1);
                let mut px = Vec::with_capacity(32 * 32);
                for y in 0..32 {
                    for x in 0..32 {
                        let on = ((if vertical { x } else { y }) + phase) % period < period / 2;
                        let v = if on { 200 } else { 40 } + rng.randint(0, 60) as i64 - 30;
                        px.push(v.clamp(0, 255) as u8);
                    }
                }
                let name = format!("img{i:03}.ppm");
                ppm(&dir.join(&name), 32, 32, &px)?;
                rows.push((name, vertical as u8));
            }
        }
        "audio" => {
            for (one, many) in crate::phonetic::plurals().into_iter().take((count / 2).max(1)) {
                for (word, label) in [(one.clone(), 0), (many.clone(), 1)] {
                    let name = format!("{word}.wav");
                    speak(&dir.join(&name), &word)?;
                    rows.push((name, label));
                }
            }
        }
        "video" => {
            let numbers = ["one", "two", "three", "four"];
            for i in 0..count {
                let right = i % 2 == 0;
                let clip = dir.join(format!("clip{i:03}"));
                fs::create_dir_all(&clip).map_err(|e| e.to_string())?;
                let mut x = if right { rng.randint(2, 6) } else { rng.randint(18, 22) } as i64;
                let y = rng.randint(4, 20) as i64;
                let step = rng.randint(2, 3) as i64;
                for t in 0..8 {
                    let mut px = vec![20u8; 32 * 32];
                    for yy in y..y + 8 {
                        for xx in x..x + 8 {
                            if (0..32).contains(&xx) && (0..32).contains(&yy) {
                                px[(yy * 32 + xx) as usize] = 230;
                            }
                        }
                    }
                    ppm(&clip.join(format!("frame{t:02}.ppm")), 32, 32, &px)?;
                    x += if right { step } else { -step };
                }
                speak(&clip.join("audio.wav"), rng.choice(&numbers))?;
                rows.push((format!("clip{i:03}"), right as u8));
            }
        }
        other => return Err(format!("kind is text, image, audio or video, got {other:?}")),
    }
    let path = dir.join("labels.tsv");
    let text: String = rows.iter().map(|(x, y)| format!("{x}\t{y}\n")).collect();
    fs::write(&path, text).map_err(|e| e.to_string())?;
    Ok(path.to_string_lossy().into_owned())
}

/// Symbols of files kept by path, size and modification time, so a file is decoded once however many epochs read it.
#[derive(Default)]
pub struct Cache {
    map: HashMap<(String, u64, u128), Vec<usize>>,
}

impl Cache {
    fn key(path: &str) -> Option<(String, u64, u128)> {
        let meta = fs::metadata(path).ok()?;
        let mtime = meta
            .modified()
            .ok()?
            .duration_since(std::time::UNIX_EPOCH)
            .ok()?
            .as_nanos();
        let abs = fs::canonicalize(path).ok()?.to_string_lossy().into_owned();
        Some((abs, meta.len(), mtime))
    }

    pub fn get(&self, path: &str) -> Option<Vec<usize>> {
        Cache::key(path).and_then(|k| self.map.get(&k).cloned())
    }

    pub fn put(&mut self, path: &str, symbols: &[usize]) {
        if let Some(k) = Cache::key(path) {
            self.map.insert(k, symbols.to_vec());
        }
    }
}
