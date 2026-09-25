//! Acoustic units (the port of acoustic.py): speech as the sounds a codebook
//! learned from it, with nothing written down.  Audio at 16 kHz is cut into
//! 25 ms frames every 10 ms, each frame is 40 log-mel energies with the
//! utterance's mean taken out, every frame goes to its nearest codebook entry
//! and a run of one entry is one unit - a token like `q17`.  The codebook is
//! learned by k-means from any recordings ([`learn`]), it is a data file that
//! carries its own analysis settings (the bundled one is `data/acoustic.tsv`),
//! and the [`Vocoder`] turns units back into a waveform frame by frame.
//!
//! Every operation here is the Python module's, in the same order, so a unit
//! is the same unit in every port.  The index loops are kept as they are in
//! Python on purpose.
#![allow(clippy::needless_range_loop)]

use crate::mt::{Mt, Rng};

/// The bundled codebook: the very file the Python package reads.
pub const ACOUSTIC_TEXT: &str = include_str!("../../phonetok/data/acoustic.tsv");

/// The sample rate the analysis runs at.
pub const ACOUSTIC_RATE: u32 = 16000;

/// Every unit token starts with this: q (for quantised) and the codebook index.
pub const UNIT_PREFIX: &str = "q";

/// Added to every band's energy before the log.
pub const LOG_FLOOR: f64 = 1e-10;

/// The vocoder's pulse train, in Hz.
pub const PITCH: f64 = 120.0;

const TWO_PI: f64 = 2.0 * std::f64::consts::PI;

/// How audio becomes frames: the settings a codebook carries.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Analysis {
    pub rate: u32,
    pub frame: usize,
    pub hop: usize,
    pub fft: usize,
    pub bands: usize,
    pub fmin: f64,
    pub fmax: f64,
    pub preemphasis: f64,
    pub normalize: bool,
}

impl Default for Analysis {
    /// 25 ms frames every 10 ms at 16 kHz, 40 bands between 20 Hz and 8 kHz, normalised.
    fn default() -> Analysis {
        Analysis {
            rate: ACOUSTIC_RATE,
            frame: 400,
            hop: 160,
            fft: 512,
            bands: 40,
            fmin: 20.0,
            fmax: 8000.0,
            preemphasis: 0.97,
            normalize: true,
        }
    }
}

impl Analysis {
    /// Refuses settings no analysis can run with.
    pub fn validate(&self) -> Result<(), String> {
        if self.rate < 1 || self.frame < 2 || self.hop < 1 || self.bands < 1 {
            return Err(format!("impossible analysis: {self:?}"));
        }
        if self.fft < self.frame || self.fft & (self.fft - 1) != 0 {
            return Err(format!(
                "fft must be a power of two of at least the frame ({}), got {}",
                self.frame, self.fft
            ));
        }
        if !(0.0 <= self.fmin && self.fmin < self.fmax && self.fmax <= self.rate as f64 / 2.0) {
            return Err(format!(
                "the bands must lie in 0 .. {} Hz, got {} .. {}",
                self.rate as f64 / 2.0,
                self.fmin,
                self.fmax
            ));
        }
        Ok(())
    }
}

// -- audio files --------------------------------------------------------------------------

/// The samples of a WAV file in [-1, 1], mono (channels averaged), and its rate.
/// PCM of 8, 16, 24 and 32 bits and IEEE float of 32 and 64 bits are read.
pub fn read_wav(data: &[u8]) -> Result<(Vec<f64>, u32), String> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a WAV file (no RIFF/WAVE header)".to_string());
    }
    let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]) as usize;
    let u32_at =
        |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
    let (mut tag, mut channels, mut rate, mut bits) = (0usize, 0usize, 0usize, 0usize);
    let mut have_fmt = false;
    let mut body: Option<&[u8]> = None;
    let mut pos = 12;
    while pos + 8 <= data.len() {
        let chunk = &data[pos..pos + 4];
        let size = u32_at(pos + 4);
        let end = (pos + 8 + size).min(data.len());
        let payload = &data[pos + 8..end];
        if chunk == b"fmt " && payload.len() >= 16 {
            tag = u16_at(pos + 8);
            channels = u16_at(pos + 10);
            rate = u32_at(pos + 12);
            bits = u16_at(pos + 22);
            if tag == 0xFFFE && payload.len() >= 26 {
                tag = u16_at(pos + 8 + 24);
            }
            have_fmt = true;
        } else if chunk == b"data" {
            body = Some(payload);
        }
        pos += 8 + size + (size & 1);
    }
    let body = match (have_fmt, body) {
        (true, Some(b)) => b,
        _ => return Err("not a WAV file (no fmt or data chunk)".to_string()),
    };
    if channels < 1 || rate < 1 {
        return Err(format!("WAV with {channels} channels at {rate} Hz"));
    }
    let values: Vec<f64> = match (tag, bits) {
        (1, 8) => body.iter().map(|&b| (b as f64 - 128.0) / 128.0).collect(),
        (1, 16) => body
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
            .collect(),
        (1, 24) => body
            .chunks_exact(3)
            .map(|c| {
                let mut v = c[0] as i32 | (c[1] as i32) << 8 | (c[2] as i32) << 16;
                if v & 0x800000 != 0 {
                    v -= 0x1000000;
                }
                v as f64 / 8388608.0
            })
            .collect(),
        (1, 32) => body
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64 / 2147483648.0)
            .collect(),
        (3, 32) => body
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
            .collect(),
        (3, 64) => body
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect(),
        (1, _) => return Err(format!("{bits}-bit PCM WAV is not supported")),
        (3, _) => return Err(format!("{bits}-bit float WAV is not supported")),
        _ => {
            return Err(format!(
                "WAV format tag {tag} is not supported (PCM and IEEE float are)"
            ))
        }
    };
    let values = if channels > 1 {
        values
            .chunks_exact(channels)
            .map(|frame| {
                let mut s = 0.0;
                for v in frame {
                    s += v;
                }
                s / channels as f64
            })
            .collect()
    } else {
        values
    };
    Ok((values, rate as u32))
}

/// Reads a WAV file.
pub fn load_wav(path: &str) -> Result<(Vec<f64>, u32), String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    read_wav(&data)
}

/// 16-bit mono PCM (what the synthesizer makes) as samples in [-1, 1].
pub fn pcm_samples(pcm: &[u8]) -> Vec<f64> {
    pcm.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]) as f64 / 32768.0)
        .collect()
}

/// Samples at `src` Hz brought to `dst` Hz by a windowed-sinc interpolation.
pub fn resample(samples: &[f64], src: u32, dst: u32) -> Result<Vec<f64>, String> {
    if src == dst {
        return Ok(samples.to_vec());
    }
    if src < 1 || dst < 1 {
        return Err(format!("cannot resample {src} Hz to {dst} Hz"));
    }
    const TAPS: i64 = 16;
    let n = samples.len() as i64;
    let count = samples.len() * dst as usize / src as usize;
    let ratio = src as f64 / dst as f64;
    let cutoff = (dst as f64 / src as f64).min(1.0) * 0.95;
    let mut out = Vec::with_capacity(count);
    for j in 0..count {
        let center = j as f64 * ratio;
        let base = center.floor() as i64;
        let mut acc = 0.0;
        let mut norm = 0.0;
        for i in (base - TAPS + 1)..=(base + TAPS) {
            if i < 0 || i >= n {
                continue;
            }
            let x = i as f64 - center;
            let w = if x == 0.0 {
                1.0
            } else {
                (std::f64::consts::PI * cutoff * x).sin() / (std::f64::consts::PI * cutoff * x)
                    * (0.5 + 0.5 * (std::f64::consts::PI * x / TAPS as f64).cos())
            };
            acc += samples[i as usize] * w;
            norm += w;
        }
        out.push(if norm != 0.0 { acc / norm } else { 0.0 });
    }
    Ok(out)
}

// -- the analysis -------------------------------------------------------------------------

/// The periodic Hann window of `n` samples.
pub fn hann(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (TWO_PI * i as f64 / n as f64).cos())
        .collect()
}

/// Transforms `re + i im` in place (a length that is a power of two); `inverse` transforms the other way
/// and divides by the length.  Radix-2, the twiddle advanced by multiplication, exactly as Python does it.
pub fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let sign = if inverse { 1.0 } else { -1.0 };
    let mut length = 2;
    while length <= n {
        let ang = sign * TWO_PI / length as f64;
        let wr = ang.cos();
        let wi = ang.sin();
        let half = length >> 1;
        let mut start = 0;
        while start < n {
            let mut cr = 1.0;
            let mut ci = 0.0;
            for k in start..start + half {
                let m = k + half;
                let tr = re[m] * cr - im[m] * ci;
                let ti = re[m] * ci + im[m] * cr;
                re[m] = re[k] - tr;
                im[m] = im[k] - ti;
                re[k] += tr;
                im[k] += ti;
                let ncr = cr * wr - ci * wi;
                let nci = cr * wi + ci * wr;
                cr = ncr;
                ci = nci;
            }
            start += length;
        }
        length <<= 1;
    }
    if inverse {
        let fnn = n as f64;
        for i in 0..n {
            re[i] /= fnn;
            im[i] /= fnn;
        }
    }
}

/// The HTK mel scale.
pub fn mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// Its inverse.
pub fn mel_to_hz(m: f64) -> f64 {
    700.0 * (10.0f64.powf(m / 2595.0) - 1.0)
}

/// One triangular filter: the first bin it touches and its weight on each bin from there on.
#[derive(Clone, Debug)]
pub struct Filter {
    pub first: usize,
    pub weights: Vec<f64>,
}

/// The filters of an analysis, and per band the summed weight of its filter.
#[derive(Clone, Debug)]
pub struct Filters {
    pub bands: Vec<Filter>,
    pub areas: Vec<f64>,
}

impl Filters {
    pub fn of(a: &Analysis) -> Filters {
        let (lo, hi) = (mel(a.fmin), mel(a.fmax));
        let edges: Vec<f64> = (0..a.bands + 2)
            .map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (a.bands + 1) as f64))
            .collect();
        let half = a.fft / 2;
        let mut bands = Vec::with_capacity(a.bands);
        for b in 0..a.bands {
            let (left, center, right) = (edges[b], edges[b + 1], edges[b + 2]);
            let mut first: Option<usize> = None;
            let mut weights = Vec::new();
            for k in 0..=half {
                let f = (k * a.rate as usize) as f64 / a.fft as f64;
                let w = if f <= left || f >= right {
                    0.0
                } else if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                };
                if w > 0.0 {
                    if first.is_none() {
                        first = Some(k);
                    }
                    weights.push(w);
                } else if first.is_some() {
                    break;
                }
            }
            let first = match first {
                Some(k) => k,
                None => {
                    weights = vec![1.0];
                    (center * a.fft as f64 / a.rate as f64 + 0.5) as usize
                }
            };
            bands.push(Filter { first, weights });
        }
        let areas = bands
            .iter()
            .map(|f| {
                let mut total = 0.0;
                for w in &f.weights {
                    total += w;
                }
                total
            })
            .collect();
        Filters { bands, areas }
    }
}

/// y[n] = x[n] - p x[n-1].
pub fn preemphasize(samples: &[f64], p: f64) -> Vec<f64> {
    let mut out = samples.to_vec();
    for i in (1..out.len()).rev() {
        out[i] -= p * out[i - 1];
    }
    out
}

/// The log-mel frames of samples (already at `a.rate`); at least one frame, zero-padded if need be.
/// `normalize` subtracts every band's mean over the utterance (the learner reads the raw frames first).
pub fn frames(samples: &[f64], a: &Analysis, normalize: bool) -> Result<Vec<Vec<f64>>, String> {
    a.validate()?;
    let mut y = if a.preemphasis != 0.0 {
        preemphasize(samples, a.preemphasis)
    } else {
        samples.to_vec()
    };
    while y.len() < a.frame {
        y.push(0.0);
    }
    let count = (y.len() - a.frame) / a.hop + 1;
    let window = hann(a.frame);
    let filters = Filters::of(a);
    let half = a.fft / 2;
    let mut out = Vec::with_capacity(count);
    let mut re = vec![0.0; a.fft];
    let mut im = vec![0.0; a.fft];
    let mut power = vec![0.0; half + 1];
    for t in 0..count {
        let start = t * a.hop;
        for i in 0..a.frame {
            re[i] = y[start + i] * window[i];
        }
        for i in a.frame..a.fft {
            re[i] = 0.0;
        }
        for v in im.iter_mut() {
            *v = 0.0;
        }
        fft(&mut re, &mut im, false);
        for k in 0..=half {
            power[k] = re[k] * re[k] + im[k] * im[k];
        }
        let mut frame = Vec::with_capacity(a.bands);
        for f in &filters.bands {
            let mut e = 0.0;
            for (i, w) in f.weights.iter().enumerate() {
                e += w * power[f.first + i];
            }
            frame.push((e + LOG_FLOOR).ln());
        }
        out.push(frame);
    }
    if normalize && !out.is_empty() {
        let means = band_means(&out);
        for frame in out.iter_mut() {
            for b in 0..a.bands {
                frame[b] -= means[b];
            }
        }
    }
    Ok(out)
}

/// [`frames`] with the analysis' own normalisation setting.
pub fn frames_of(samples: &[f64], a: &Analysis) -> Result<Vec<Vec<f64>>, String> {
    frames(samples, a, a.normalize)
}

/// The mean of every dimension over the frames.
pub fn band_means(frames: &[Vec<f64>]) -> Vec<f64> {
    if frames.is_empty() {
        return Vec::new();
    }
    let dims = frames[0].len();
    let mut sums = vec![0.0; dims];
    for frame in frames {
        for b in 0..dims {
            sums[b] += frame[b];
        }
    }
    sums.iter().map(|s| s / frames.len() as f64).collect()
}

/// The squared Euclidean distance, summed in order.
pub fn dist2(x: &[f64], y: &[f64]) -> f64 {
    let mut d = 0.0;
    for (a, b) in x.iter().zip(y) {
        let diff = a - b;
        d += diff * diff;
    }
    d
}

// -- the codebook -------------------------------------------------------------------------

/// What was learned: the centroids and what the training frames said about them.
#[derive(Clone, Debug)]
pub struct Codebook {
    pub analysis: Analysis,
    pub centroids: Vec<Vec<f64>>,
    /// Mean length in frames of a run of each unit in the training audio (decoding holds a unit that long).
    pub runs: Vec<f64>,
    /// Frames each unit claimed in the training audio.
    pub counts: Vec<usize>,
    /// The mean log-mel of the training frames before normalisation: decoding adds it back.
    pub mean: Vec<f64>,
    pub seed: i64,
    pub note: String,
    pub inertia: f64,
}

impl Codebook {
    /// The number of units.
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Refuses a codebook whose tables disagree.
    pub fn validate(&self) -> Result<(), String> {
        self.analysis.validate()?;
        if self.centroids.is_empty() {
            return Err("an empty codebook".to_string());
        }
        for c in &self.centroids {
            if c.len() != self.analysis.bands {
                return Err(format!(
                    "a centroid of {} values in a codebook of {} bands",
                    c.len(),
                    self.analysis.bands
                ));
            }
        }
        if self.runs.len() != self.k()
            || self.counts.len() != self.k()
            || self.mean.len() != self.analysis.bands
        {
            return Err("a codebook whose tables disagree about its size".to_string());
        }
        Ok(())
    }

    /// The token of a unit.
    pub fn name(&self, index: usize) -> String {
        format!("{UNIT_PREFIX}{index}")
    }

    /// Every unit's token, in order.
    pub fn names(&self) -> Vec<String> {
        (0..self.k()).map(|i| self.name(i)).collect()
    }

    /// The codebook index of a unit token, or `None` when the token is not one of this codebook's units.
    pub fn index(&self, token: &str) -> Option<usize> {
        let digits = token.strip_prefix(UNIT_PREFIX)?;
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        let i: usize = digits.parse().ok()?;
        if i >= self.k() || token != self.name(i) {
            return None;
        }
        Some(i)
    }

    /// The index of the centroid nearest the frame (the first of equals).
    pub fn nearest(&self, frame: &[f64]) -> usize {
        let mut best = 0;
        let mut best_d = dist2(frame, &self.centroids[0]);
        for i in 1..self.k() {
            let d = dist2(frame, &self.centroids[i]);
            if d < best_d {
                best = i;
                best_d = d;
            }
        }
        best
    }

    /// Every frame's nearest unit.
    pub fn codes(&self, frames: &[Vec<f64>]) -> Vec<usize> {
        frames.iter().map(|f| self.nearest(f)).collect()
    }

    /// The unit tokens of the frames, runs collapsed when asked.
    pub fn units(&self, frames: &[Vec<f64>], collapse: bool) -> Vec<String> {
        let codes = self.codes(frames);
        let codes = if collapse {
            collapse_runs(&codes)
        } else {
            codes
        };
        codes.iter().map(|&c| self.name(c)).collect()
    }

    /// The frames a unit is held for when decoded: its typical run, at least one.
    pub fn hold(&self, index: usize) -> usize {
        ((self.runs[index] + 0.5) as usize).max(1)
    }

    /// The absolute log-mel frame a unit stands for: its centroid plus the mean.
    pub fn frame_of(&self, index: usize) -> Vec<f64> {
        self.centroids[index]
            .iter()
            .zip(&self.mean)
            .map(|(c, m)| c + m)
            .collect()
    }

    /// The codebook as its TSV file.
    pub fn dumps(&self) -> String {
        let a = &self.analysis;
        let mut out = format!(
            "# phonetok acoustic codebook: {} units learned by k-means over log-mel frames\n",
            self.k()
        );
        if !self.note.is_empty() {
            for line in self.note.lines() {
                out.push_str(&format!("# {line}\n"));
            }
        }
        out.push_str(&format!(
            "rate\t{}\nframe\t{}\nhop\t{}\nfft\t{}\nbands\t{}\nfmin\t{:?}\nfmax\t{:?}\npreemphasis\t{:?}\n\
             normalize\t{}\nseed\t{}\ninertia\t{:?}\nunits\t{}\n",
            a.rate,
            a.frame,
            a.hop,
            a.fft,
            a.bands,
            a.fmin,
            a.fmax,
            a.preemphasis,
            if a.normalize { 1 } else { 0 },
            self.seed,
            self.inertia,
            self.k()
        ));
        out.push_str("mean");
        for v in &self.mean {
            out.push_str(&format!("\t{v:?}"));
        }
        out.push('\n');
        for (i, c) in self.centroids.iter().enumerate() {
            out.push_str(&format!(
                "{}\t{}\t{:?}",
                self.name(i),
                self.counts[i],
                self.runs[i]
            ));
            for v in c {
                out.push_str(&format!("\t{v:?}"));
            }
            out.push('\n');
        }
        out
    }

    /// Writes the codebook file.
    pub fn dump(&self, path: &str) -> Result<(), String> {
        std::fs::write(path, self.dumps()).map_err(|e| format!("{path}: {e}"))
    }

    /// Reads a codebook file's text.
    pub fn loads(text: &str) -> Result<Codebook, String> {
        let mut settings: Vec<(String, String)> = Vec::new();
        let mut note: Vec<String> = Vec::new();
        let mut mean: Vec<f64> = Vec::new();
        let mut centroids: Vec<Vec<f64>> = Vec::new();
        let mut counts: Vec<usize> = Vec::new();
        let mut runs: Vec<f64> = Vec::new();
        let floats = |parts: &[&str]| -> Result<Vec<f64>, String> {
            parts
                .iter()
                .map(|p| {
                    p.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("unreadable number {p:?} in the codebook"))
                })
                .collect()
        };
        for raw in text.lines() {
            let line = raw.trim_end_matches('\r');
            if line.trim().is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix('#') {
                if !line.starts_with("# phonetok acoustic codebook") {
                    note.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
                }
                continue;
            }
            let parts: Vec<&str> = line.split('\t').collect();
            let key = parts[0];
            let unit_row = key
                .strip_prefix(UNIT_PREFIX)
                .map(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit()))
                .unwrap_or(false);
            if key == "mean" {
                mean = floats(&parts[1..])?;
            } else if unit_row {
                if parts.len() < 4 {
                    return Err(format!("codebook row {key:?} is incomplete"));
                }
                let expected = format!("{UNIT_PREFIX}{}", centroids.len());
                if key != expected {
                    return Err(format!(
                        "codebook rows out of order: {key:?} where {expected:?} was expected"
                    ));
                }
                counts.push(
                    parts[1]
                        .parse()
                        .map_err(|_| format!("unreadable count in codebook row {key:?}"))?,
                );
                runs.push(
                    parts[2]
                        .parse()
                        .map_err(|_| format!("unreadable run in codebook row {key:?}"))?,
                );
                centroids.push(floats(&parts[3..])?);
            } else if parts.len() == 2 {
                settings.push((key.to_string(), parts[1].to_string()));
            } else {
                return Err(format!("unreadable codebook line: {line:?}"));
            }
        }
        let setting = |name: &str| {
            settings
                .iter()
                .rev()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.as_str())
        };
        let bad = |name: &str| format!("unreadable codebook setting {name}");
        let mut a = Analysis::default();
        if let Some(v) = setting("rate") {
            a.rate = v.parse().map_err(|_| bad("rate"))?;
        }
        if let Some(v) = setting("frame") {
            a.frame = v.parse().map_err(|_| bad("frame"))?;
        }
        if let Some(v) = setting("hop") {
            a.hop = v.parse().map_err(|_| bad("hop"))?;
        }
        if let Some(v) = setting("fft") {
            a.fft = v.parse().map_err(|_| bad("fft"))?;
        }
        if let Some(v) = setting("bands") {
            a.bands = v.parse().map_err(|_| bad("bands"))?;
        }
        if let Some(v) = setting("fmin") {
            a.fmin = v.parse().map_err(|_| bad("fmin"))?;
        }
        if let Some(v) = setting("fmax") {
            a.fmax = v.parse().map_err(|_| bad("fmax"))?;
        }
        if let Some(v) = setting("preemphasis") {
            a.preemphasis = v.parse().map_err(|_| bad("preemphasis"))?;
        }
        if let Some(v) = setting("normalize") {
            a.normalize = !matches!(v, "0" | "false" | "no");
        }
        let seed: i64 = match setting("seed") {
            Some(v) => v.parse().map_err(|_| bad("seed"))?,
            None => 0,
        };
        let inertia: f64 = match setting("inertia") {
            Some(v) => v.parse().map_err(|_| bad("inertia"))?,
            None => 0.0,
        };
        let book = Codebook {
            analysis: a,
            centroids,
            runs,
            counts,
            mean,
            seed,
            note: note.join("\n"),
            inertia,
        };
        book.validate()?;
        Ok(book)
    }

    /// Reads a codebook file.
    pub fn load(path: &str) -> Result<Codebook, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        Codebook::loads(&text)
    }

    /// The bundled codebook.
    pub fn bundled() -> Result<Codebook, String> {
        Codebook::loads(ACOUSTIC_TEXT)
    }
}

/// Folds every run of one code into one.
pub fn collapse_runs(codes: &[usize]) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    for &c in codes {
        if out.last() != Some(&c) {
            out.push(c);
        }
    }
    out
}

/// Per code, the frames it claimed and the runs it made.
pub fn run_lengths(codes: &[usize], k: usize) -> (Vec<usize>, Vec<usize>) {
    let mut frames = vec![0; k];
    let mut runs = vec![0; k];
    let mut prev = usize::MAX;
    for &c in codes {
        frames[c] += 1;
        if c != prev {
            runs[c] += 1;
        }
        prev = c;
    }
    (frames, runs)
}

// -- learning -----------------------------------------------------------------------------

/// What [`kmeans`] found.
#[derive(Clone, Debug)]
pub struct Learned {
    pub centroids: Vec<Vec<f64>>,
    pub assignments: Vec<usize>,
    pub inertia: f64,
    pub iterations: usize,
    pub converged: bool,
}

/// k-means++ seeding, then Lloyd's iterations until nothing moves (or `iterations` are up).  Every draw comes
/// from `rng` (CPython's Mersenne Twister), so the same seed learns the same codebook in every port.
pub fn kmeans(
    points: &[Vec<f64>],
    k: usize,
    rng: &mut dyn Rng,
    iterations: usize,
) -> Result<Learned, String> {
    let n = points.len();
    if k < 1 {
        return Err("k must be at least 1".to_string());
    }
    if n < k {
        return Err(format!("{n} frames cannot make {k} units"));
    }
    let first = rng.rand_below(n);
    let mut centroids: Vec<Vec<f64>> = vec![points[first].clone()];
    let mut d2: Vec<f64> = points.iter().map(|p| dist2(p, &centroids[0])).collect();
    while centroids.len() < k {
        let mut total = 0.0;
        for d in &d2 {
            total += d;
        }
        let mut chosen = n - 1;
        if total > 0.0 {
            let r = rng.float64() * total;
            let mut acc = 0.0;
            for (i, d) in d2.iter().enumerate() {
                acc += d;
                if acc >= r {
                    chosen = i;
                    break;
                }
            }
        } else {
            chosen = rng.rand_below(n);
        }
        let centre = points[chosen].clone();
        for (i, p) in points.iter().enumerate() {
            let d = dist2(p, &centre);
            if d < d2[i] {
                d2[i] = d;
            }
        }
        centroids.push(centre);
    }
    let dims = if n > 0 { points[0].len() } else { 0 };
    let mut assignments = vec![usize::MAX; n];
    let mut inertia = 0.0;
    let mut converged = false;
    let mut done = 0;
    for it in 0..iterations {
        done = it + 1;
        let mut changed = 0;
        inertia = 0.0;
        for (i, p) in points.iter().enumerate() {
            let mut best = 0;
            let mut best_d = dist2(p, &centroids[0]);
            for c in 1..k {
                let d = dist2(p, &centroids[c]);
                if d < best_d {
                    best = c;
                    best_d = d;
                }
            }
            if best != assignments[i] {
                changed += 1;
                assignments[i] = best;
            }
            d2[i] = best_d;
            inertia += best_d;
        }
        if changed == 0 {
            converged = true;
            break;
        }
        let mut sums = vec![vec![0.0; dims]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = assignments[i];
            for b in 0..dims {
                sums[c][b] += p[b];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                centroids[c] = sums[c].iter().map(|v| v / counts[c] as f64).collect();
            } else {
                let mut far = 0;
                let mut far_d = -1.0;
                for i in 0..n {
                    if d2[i] > far_d {
                        far = i;
                        far_d = d2[i];
                    }
                }
                centroids[c] = points[far].clone();
                assignments[far] = c;
                d2[far] = 0.0;
            }
        }
    }
    Ok(Learned {
        centroids,
        assignments,
        inertia,
        iterations: done,
        converged,
    })
}

/// A codebook of `k` units learned from recordings (each samples at `a.rate`).  Nothing but the audio is
/// looked at: the frames are analysed raw, their mean over all the recordings is kept for decoding, every
/// recording is normalised on its own, and k-means clusters the lot.
pub fn learn(
    recordings: &[Vec<f64>],
    k: usize,
    seed: i64,
    iterations: usize,
    a: &Analysis,
    note: &str,
) -> Result<Codebook, String> {
    a.validate()?;
    let mut utterances: Vec<Vec<Vec<f64>>> = Vec::new();
    let mut sums = vec![0.0; a.bands];
    let mut total = 0usize;
    for samples in recordings {
        let mut fr = frames(samples, a, false)?;
        for f in &fr {
            for b in 0..a.bands {
                sums[b] += f[b];
            }
        }
        total += fr.len();
        if a.normalize {
            let means = band_means(&fr);
            for f in fr.iter_mut() {
                for b in 0..a.bands {
                    f[b] -= means[b];
                }
            }
        }
        utterances.push(fr);
    }
    if total == 0 {
        return Err("no audio to learn from".to_string());
    }
    let mean: Vec<f64> = sums.iter().map(|s| s / total as f64).collect();
    let points: Vec<Vec<f64>> = utterances
        .iter()
        .flat_map(|fr| fr.iter().cloned())
        .collect();
    let mut rng = Mt::new(seed as u64);
    let found = kmeans(&points, k, &mut rng, iterations)?;
    let mut counts = vec![0usize; k];
    let mut runs = vec![0usize; k];
    let mut pos = 0;
    for fr in &utterances {
        let codes = &found.assignments[pos..pos + fr.len()];
        pos += fr.len();
        let (f, r) = run_lengths(codes, k);
        for c in 0..k {
            counts[c] += f[c];
            runs[c] += r[c];
        }
    }
    let mean_runs = (0..k)
        .map(|c| {
            if runs[c] > 0 {
                counts[c] as f64 / runs[c] as f64
            } else {
                1.0
            }
        })
        .collect();
    Ok(Codebook {
        analysis: *a,
        centroids: found.centroids,
        runs: mean_runs,
        counts,
        mean,
        seed,
        note: note.to_string(),
        inertia: found.inertia,
    })
}

// -- decoding -----------------------------------------------------------------------------

/// The linear magnitude spectrum (`fft/2 + 1` bins) a log-mel frame stands for: a band's energy was summed
/// over its bins, so it is spread back as that energy per unit of filter weight through the same triangles;
/// where two filters overlap their weights sum to one, so a bin under both takes their mean.
pub fn magnitudes_of(logmel: &[f64], a: &Analysis, filters: &Filters) -> Vec<f64> {
    let half = a.fft / 2;
    let mut power = vec![0.0; half + 1];
    for (b, f) in filters.bands.iter().enumerate() {
        let e = logmel[b].exp() / filters.areas[b];
        for (i, w) in f.weights.iter().enumerate() {
            power[f.first + i] += w * e;
        }
    }
    power.iter().map(|p| p.sqrt()).collect()
}

/// The centre frequency of every band, in Hz.
pub fn band_centres(a: &Analysis) -> Vec<f64> {
    let (lo, hi) = (mel(a.fmin), mel(a.fmax));
    (0..a.bands)
        .map(|b| mel_to_hz(lo + (hi - lo) * (b + 1) as f64 / (a.bands + 1) as f64))
        .collect()
}

/// The bands under 1 kHz and the bands over 3 kHz: what the voicing is read off.
pub fn voicing_bands(a: &Analysis) -> (Vec<usize>, Vec<usize>) {
    let centres = band_centres(a);
    let low = (0..a.bands).filter(|&b| centres[b] < 1000.0).collect();
    let high = (0..a.bands).filter(|&b| centres[b] > 3000.0).collect();
    (low, high)
}

/// How voiced a frame sounds, 0 to 1, read off its tilt: the low bands over the high.  Fully voiced from a
/// tilt of 4 nats, fully unvoiced from -2.
pub fn voicing_of(logmel: &[f64], bands: &(Vec<usize>, Vec<usize>)) -> f64 {
    let (low, high) = bands;
    if low.is_empty() || high.is_empty() {
        return 1.0;
    }
    let mut sl = 0.0;
    for &b in low {
        sl += logmel[b];
    }
    let mut sh = 0.0;
    for &b in high {
        sh += logmel[b];
    }
    let tilt = sl / low.len() as f64 - sh / high.len() as f64;
    ((tilt + 2.0) / 6.0).clamp(0.0, 1.0)
}

/// A deterministic noise source (xorshift32), the same numbers in every port.
struct Noise(u32);

impl Noise {
    fn new() -> Noise {
        Noise(0x9E37_79B9)
    }

    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as f64 / 2147483648.0 - 1.0
    }
}

/// What the vocoder takes its phases from: a pulse train at the voice's pitch, and noise, both running on
/// continuously so consecutive frames agree about the phase of every bin.
struct Excitation {
    a: Analysis,
    step: f64,
    phase: f64,
    noise: Noise,
    pulses: Vec<f64>,
    noises: Vec<f64>,
    offset: usize,
    window: Vec<f64>,
}

impl Excitation {
    fn new(a: &Analysis, pitch: f64) -> Result<Excitation, String> {
        if pitch <= 0.0 {
            return Err(format!("pitch must be positive, got {pitch}"));
        }
        Ok(Excitation {
            a: *a,
            step: pitch / a.rate as f64,
            phase: 0.0,
            noise: Noise::new(),
            pulses: Vec::new(),
            noises: Vec::new(),
            offset: 0,
            window: hann(a.frame),
        })
    }

    fn ensure(&mut self, upto: usize) {
        while self.offset + self.pulses.len() < upto {
            self.phase += self.step;
            if self.phase >= 1.0 {
                self.phase -= 1.0;
                self.pulses.push(1.0);
            } else {
                self.pulses.push(0.0);
            }
            self.noises.push(self.noise.next());
        }
    }

    /// The unit phasors (cos, sin per bin, `fft/2 + 1` of them) of the excitation frame at `start`.
    fn phases(&mut self, start: usize, voicing: f64) -> (Vec<f64>, Vec<f64>) {
        let a = self.a;
        self.ensure(start + a.frame);
        if start - self.offset > 4 * a.frame {
            let drop = start - self.offset - a.frame;
            self.pulses.drain(..drop);
            self.noises.drain(..drop);
            self.offset += drop;
        }
        let noise_gain = 0.25 * (1.0 - voicing) + 0.02;
        let base = start - self.offset;
        let mut re = vec![0.0; a.fft];
        let mut im = vec![0.0; a.fft];
        for i in 0..a.frame {
            re[i] = (voicing * self.pulses[base + i] + noise_gain * self.noises[base + i])
                * self.window[i];
        }
        fft(&mut re, &mut im, false);
        let half = a.fft / 2;
        let mut cos = vec![1.0; half + 1];
        let mut sin = vec![0.0; half + 1];
        for k in 0..=half {
            let m = (re[k] * re[k] + im[k] * im[k]).sqrt();
            if m > 0.0 {
                cos[k] = re[k] / m;
                sin[k] = im[k] / m;
            }
        }
        (cos, sin)
    }
}

/// The full (Hermitian) spectrum of a frame from its magnitudes and unit phasors.
fn spectrum(mags: &[f64], cos: &[f64], sin: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let half = n / 2;
    let mut re = vec![0.0; n];
    let mut im = vec![0.0; n];
    for k in 0..=half {
        re[k] = mags[k] * cos[k];
        im[k] = mags[k] * sin[k];
    }
    im[0] = 0.0;
    im[half] = 0.0;
    for k in 1..half {
        re[n - k] = re[k];
        im[n - k] = -im[k];
    }
    (re, im)
}

fn pack_pcm(samples: &[f64], gain: f64) -> Vec<u8> {
    let mut out = Vec::with_capacity(2 * samples.len());
    let scale = gain * 32767.0;
    for &x in samples {
        let s = (x * scale).clamp(-32767.0, 32767.0);
        out.extend_from_slice(&(s as i16).to_le_bytes());
    }
    out
}

/// Units to a waveform as they come: every unit is its centroid held for its typical run, every frame is
/// spread back over the linear spectrum with the excitation's phases, and the samples no later frame can
/// touch are handed back at once.  [`Vocoder::end`] flushes the tail.
pub struct Vocoder {
    pub book: Codebook,
    pub gain: f64,
    pub pitch: f64,
    pub total_samples: usize,
    window: Vec<f64>,
    filters: Filters,
    voicing: (Vec<usize>, Vec<usize>),
    excitation: Excitation,
    acc: Vec<f64>,
    norm: Vec<f64>,
    frames: usize,
    emitted: usize,
}

impl Vocoder {
    pub fn new(book: Codebook, gain: f64, pitch: f64) -> Result<Vocoder, String> {
        book.validate()?;
        let a = book.analysis;
        Ok(Vocoder {
            window: hann(a.frame),
            filters: Filters::of(&a),
            voicing: voicing_bands(&a),
            excitation: Excitation::new(&a, pitch)?,
            book,
            gain,
            pitch,
            total_samples: 0,
            acc: Vec::new(),
            norm: Vec::new(),
            frames: 0,
            emitted: 0,
        })
    }

    /// One unit; the PCM that no later unit can change comes back.
    pub fn feed(&mut self, unit: &str) -> Result<Vec<u8>, String> {
        let index = self
            .book
            .index(unit)
            .ok_or_else(|| format!("not a unit of this codebook: {unit:?}"))?;
        let frame = self.book.frame_of(index);
        let mut out = Vec::new();
        for _ in 0..self.book.hold(index) {
            out.extend(self.feed_frame(&frame));
        }
        Ok(out)
    }

    /// One absolute log-mel frame (the codebook's mean already added); the PCM that is ready comes back.
    pub fn feed_frame(&mut self, logmel: &[f64]) -> Vec<u8> {
        let a = self.book.analysis;
        let start = self.frames * a.hop;
        let mags = magnitudes_of(logmel, &a, &self.filters);
        let (cos, sin) = self
            .excitation
            .phases(start, voicing_of(logmel, &self.voicing));
        let (mut re, mut im) = spectrum(&mags, &cos, &sin, a.fft);
        fft(&mut re, &mut im, true);
        let end = start + a.frame;
        if self.acc.len() < end {
            self.acc.resize(end, 0.0);
            self.norm.resize(end, 0.0);
        }
        for i in 0..a.frame {
            let w = self.window[i];
            self.acc[start + i] += re[i] * w;
            self.norm[start + i] += w * w;
        }
        self.frames += 1;
        let ready = (self.frames * a.hop).min(self.acc.len());
        self.emit(ready)
    }

    fn emit(&mut self, upto: usize) -> Vec<u8> {
        if upto <= self.emitted {
            return Vec::new();
        }
        let chunk: Vec<f64> = (self.emitted..upto)
            .map(|i| {
                let norm = self.norm[i];
                if norm > 1e-3 {
                    self.acc[i] / norm
                } else {
                    self.acc[i] / 1e-3
                }
            })
            .collect();
        self.emitted = upto;
        self.total_samples += chunk.len();
        pack_pcm(&chunk, self.gain)
    }

    /// The tail: what the last frames left pending.  The vocoder is ready for the next utterance.
    pub fn end(&mut self) -> Vec<u8> {
        let out = self.emit(self.acc.len());
        self.excitation =
            Excitation::new(&self.book.analysis, self.pitch).expect("a pitch that was accepted");
        self.acc.clear();
        self.norm.clear();
        self.frames = 0;
        self.emitted = 0;
        out
    }
}

type Spectra = Vec<(Vec<f64>, Vec<f64>)>;

fn stft(samples: &[f64], a: &Analysis, count: usize) -> Spectra {
    let window = hann(a.frame);
    let mut out = Vec::with_capacity(count);
    for t in 0..count {
        let start = t * a.hop;
        let mut re = vec![0.0; a.fft];
        let mut im = vec![0.0; a.fft];
        for i in 0..a.frame {
            let s = if start + i < samples.len() {
                samples[start + i]
            } else {
                0.0
            };
            re[i] = s * window[i];
        }
        fft(&mut re, &mut im, false);
        out.push((re, im));
    }
    out
}

fn istft(spectra: &Spectra, a: &Analysis) -> Vec<f64> {
    let window = hann(a.frame);
    let length = if spectra.is_empty() {
        0
    } else {
        (spectra.len() - 1) * a.hop + a.frame
    };
    let mut acc = vec![0.0; length];
    let mut norm = vec![0.0; length];
    for (t, (re, im)) in spectra.iter().enumerate() {
        let mut r = re.clone();
        let mut i = im.clone();
        fft(&mut r, &mut i, true);
        let start = t * a.hop;
        for j in 0..a.frame {
            let w = window[j];
            acc[start + j] += r[j] * w;
            norm[start + j] += w * w;
        }
    }
    (0..length)
        .map(|i| {
            if norm[i] > 1e-3 {
                acc[i] / norm[i]
            } else {
                acc[i] / 1e-3
            }
        })
        .collect()
}

/// Samples for a run of absolute log-mel frames, polished: the first pass is exactly the [`Vocoder`]'s, and
/// each iteration transforms the samples back, keeps the phases that came out and restores the magnitudes.
pub fn griffin_lim(
    frames: &[Vec<f64>],
    a: &Analysis,
    iterations: usize,
    pitch: f64,
) -> Result<Vec<f64>, String> {
    let n = a.fft;
    let half = n / 2;
    let count = frames.len();
    if count == 0 {
        return Ok(Vec::new());
    }
    let filters = Filters::of(a);
    let bands = voicing_bands(a);
    let magnitudes: Vec<Vec<f64>> = frames
        .iter()
        .map(|f| magnitudes_of(f, a, &filters))
        .collect();
    let mut excitation = Excitation::new(a, pitch)?;
    let mut spectra: Spectra = Vec::with_capacity(count);
    for (t, f) in frames.iter().enumerate() {
        let (cos, sin) = excitation.phases(t * a.hop, voicing_of(f, &bands));
        spectra.push(spectrum(&magnitudes[t], &cos, &sin, n));
    }
    for _ in 0..iterations {
        let samples = istft(&spectra, a);
        let mut fresh = stft(&samples, a, count);
        for (t, (re, im)) in fresh.iter_mut().enumerate() {
            let mags = &magnitudes[t];
            for k in 0..=half {
                let m = (re[k] * re[k] + im[k] * im[k]).sqrt();
                if m > 0.0 {
                    re[k] = re[k] / m * mags[k];
                    im[k] = im[k] / m * mags[k];
                } else {
                    re[k] = mags[k];
                    im[k] = 0.0;
                }
            }
            im[0] = 0.0;
            im[half] = 0.0;
            for k in 1..half {
                re[n - k] = re[k];
                im[n - k] = -im[k];
            }
        }
        spectra = fresh;
    }
    Ok(istft(&spectra, a))
}

/// The absolute log-mel frames a run of units stands for: each centroid plus the mean, held for its run.
pub fn unit_frames(units: &[String], book: &Codebook) -> Result<Vec<Vec<f64>>, String> {
    let mut frames = Vec::new();
    for unit in units {
        let index = book
            .index(unit)
            .ok_or_else(|| format!("not a unit of this codebook: {unit:?}"))?;
        let frame = book.frame_of(index);
        for _ in 0..book.hold(index) {
            frames.push(frame.clone());
        }
    }
    Ok(frames)
}

/// 16-bit PCM at the codebook's rate for a run of units; `polish` is the number of Griffin-Lim iterations
/// over the whole utterance (0 is exactly what the streaming [`Vocoder`] makes).
pub fn synthesize(
    units: &[String],
    book: &Codebook,
    polish: usize,
    gain: f64,
    pitch: f64,
) -> Result<Vec<u8>, String> {
    if polish == 0 {
        let mut voc = Vocoder::new(book.clone(), gain, pitch)?;
        let mut out = Vec::new();
        for u in units {
            out.extend(voc.feed(u)?);
        }
        out.extend(voc.end());
        return Ok(out);
    }
    let frames = unit_frames(units, book)?;
    let samples = griffin_lim(&frames, &book.analysis, polish, pitch)?;
    Ok(pack_pcm(&samples, gain))
}

// -- the facade ---------------------------------------------------------------------------

/// What an utterance was heard as.
#[derive(Clone, Debug)]
pub struct Heard {
    pub units: Vec<String>,
    /// Every frame's code, before runs collapse.
    pub codes: Vec<usize>,
    pub seconds: f64,
    pub frames: usize,
}

impl Heard {
    pub fn text(&self) -> String {
        self.units.join(" ")
    }
}

/// Audio to units and back, over one codebook.
pub struct AcousticTokenizer {
    pub book: Codebook,
    /// Folds a run of one unit into one token; off, every frame is a token.
    pub collapse: bool,
}

impl AcousticTokenizer {
    pub fn new(book: Codebook, collapse: bool) -> Result<AcousticTokenizer, String> {
        book.validate()?;
        Ok(AcousticTokenizer { book, collapse })
    }

    /// Over the bundled codebook.
    pub fn bundled() -> Result<AcousticTokenizer, String> {
        AcousticTokenizer::new(Codebook::bundled()?, true)
    }

    /// A WAV file's bytes as samples at the codebook's rate.
    pub fn samples(&self, wav: &[u8]) -> Result<Vec<f64>, String> {
        let (values, rate) = read_wav(wav)?;
        resample(&values, rate, self.book.analysis.rate)
    }

    /// Everything an utterance (samples at `rate`, 0 for the codebook's) was heard as.
    pub fn listen(&self, samples: &[f64], rate: u32) -> Result<Heard, String> {
        let rate = if rate == 0 {
            self.book.analysis.rate
        } else {
            rate
        };
        let samples = resample(samples, rate, self.book.analysis.rate)?;
        let frames = frames_of(&samples, &self.book.analysis)?;
        let codes = self.book.codes(&frames);
        let kept = if self.collapse {
            collapse_runs(&codes)
        } else {
            codes.clone()
        };
        Ok(Heard {
            units: kept.iter().map(|&c| self.book.name(c)).collect(),
            codes,
            seconds: samples.len() as f64 / self.book.analysis.rate as f64,
            frames: frames.len(),
        })
    }

    /// [`AcousticTokenizer::listen`] over a WAV file's bytes.
    pub fn listen_wav(&self, wav: &[u8]) -> Result<Heard, String> {
        let (values, rate) = read_wav(wav)?;
        self.listen(&values, rate)
    }

    /// The units of an utterance.
    pub fn hear(&self, samples: &[f64], rate: u32) -> Result<Vec<String>, String> {
        Ok(self.listen(samples, rate)?.units)
    }

    pub fn is_unit(&self, token: &str) -> bool {
        self.book.index(token).is_some()
    }

    /// The unit tokens of a text of units, checked against the codebook (runs collapsed when the tokenizer does).
    pub fn units_of(&self, text: &str) -> Result<Vec<String>, String> {
        let mut units: Vec<String> = Vec::new();
        for token in text.split_whitespace() {
            if self.book.index(token).is_none() {
                return Err(format!("not a unit of this codebook: {token:?}"));
            }
            if self.collapse && units.last().map(|u| u == token).unwrap_or(false) {
                continue;
            }
            units.push(token.to_string());
        }
        Ok(units)
    }

    /// The text form of a text of units: one line, space-separated; idempotent.
    pub fn text(&self, text: &str) -> Result<String, String> {
        Ok(self.units_of(text)?.join(" "))
    }

    /// The units spoken back, as 16-bit PCM at the codebook's rate.
    pub fn synthesize(
        &self,
        units: &[String],
        polish: usize,
        gain: f64,
        pitch: f64,
    ) -> Result<Vec<u8>, String> {
        synthesize(units, &self.book, polish, gain, pitch)
    }

    /// A streaming vocoder over the codebook.
    pub fn vocoder(&self, gain: f64, pitch: f64) -> Result<Vocoder, String> {
        Vocoder::new(self.book.clone(), gain, pitch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth::{Synthesizer, VoiceSettings};

    const CLIPS: [&str; 3] = [
        "DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T",
        "DH AH0 # K W IH1 K # B R AW1 N # F AA1 K S # JH AH1 M P S # OW1 V ER0 # DH AH0 # L EY1 Z IY0 # D AO1 G",
        "P R AE1 K T AH0 S # M EY1 K S # P ER1 F IH0 K T",
    ];

    fn spoken(tokens: &str, pitch: f64) -> Vec<f64> {
        let tokens: Vec<String> = tokens.split_whitespace().map(String::from).collect();
        let voice = VoiceSettings {
            pitch,
            ..VoiceSettings::default()
        };
        pcm_samples(&Synthesizer::new(ACOUSTIC_RATE, voice).speak(&tokens))
    }

    fn clips() -> Vec<Vec<f64>> {
        CLIPS.iter().map(|c| spoken(c, 120.0)).collect()
    }

    fn small_book() -> Codebook {
        learn(&clips(), 8, 3, 30, &Analysis::default(), "test").unwrap()
    }

    #[test]
    fn the_fft_is_the_dft() {
        let mut state = 12345u32;
        let xs: Vec<f64> = (0..256)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                (state >> 8) as f64 / 8388608.0 - 1.0
            })
            .collect();
        let mut re = xs.clone();
        let mut im = vec![0.0; 256];
        fft(&mut re, &mut im, false);
        for &k in &[0usize, 1, 5, 100, 128, 200, 255] {
            let (mut wr, mut wi) = (0.0, 0.0);
            for (n, x) in xs.iter().enumerate() {
                let ang = -TWO_PI * k as f64 * n as f64 / 256.0;
                wr += x * ang.cos();
                wi += x * ang.sin();
            }
            assert!(
                (wr - re[k]).abs() < 1e-9 && (wi - im[k]).abs() < 1e-9,
                "bin {k}"
            );
        }
        fft(&mut re, &mut im, true);
        for (a, b) in re.iter().zip(&xs) {
            assert!((a - b).abs() < 1e-12);
        }
        assert!((hann(400).iter().sum::<f64>() / 400.0 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn wav_files_are_read() {
        let pcm: Vec<u8> = [0i16, 16384, -16384]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let (samples, rate) = read_wav(&crate::synth::wav_bytes(&pcm, 16000)).unwrap();
        assert_eq!(rate, 16000);
        assert_eq!(samples, vec![0.0, 0.5, -0.5]);
        assert!(read_wav(b"not a wav at all").is_err());
        assert_eq!(pcm_samples(&pcm), vec![0.0, 0.5, -0.5]);
        let src: Vec<f64> = (0..8000)
            .map(|i| (TWO_PI * 440.0 * i as f64 / 8000.0).sin())
            .collect();
        let out = resample(&src, 8000, 16000).unwrap();
        assert_eq!(out.len(), 16000);
        let crossings = (1001..15001)
            .filter(|&i| (out[i - 1] < 0.0) != (out[i] < 0.0))
            .count();
        assert!(
            (crossings as f64 / (14000.0 / 16000.0) - 880.0).abs() < 6.0,
            "{crossings}"
        );
    }

    #[test]
    fn frames_and_filters() {
        let a = Analysis::default();
        let filters = Filters::of(&a);
        assert_eq!(filters.bands.len(), a.bands);
        let mut covered = vec![0.0; a.fft / 2 + 1];
        for f in &filters.bands {
            assert!(!f.weights.is_empty());
            for (i, w) in f.weights.iter().enumerate() {
                covered[f.first + i] += w;
            }
        }
        assert!(covered[3..254].iter().all(|&c| c > 0.0));
        let samples = spoken(CLIPS[0], 120.0);
        let fr = frames_of(&samples, &a).unwrap();
        assert_eq!(fr.len(), (samples.len() - a.frame) / a.hop + 1);
        for b in 0..a.bands {
            let mean: f64 = fr.iter().map(|f| f[b]).sum::<f64>() / fr.len() as f64;
            assert!(mean.abs() < 1e-9, "band {b} mean {mean}");
        }
        assert_eq!(frames_of(&[0.0, 0.0], &a).unwrap().len(), 1);
        assert!(Analysis {
            fft: 500,
            ..Analysis::default()
        }
        .validate()
        .is_err());
        let bands = voicing_bands(&a);
        let vowel = frames(&spoken("AE1 AE1 AE1", 120.0), &a, false).unwrap();
        let hiss = frames(&spoken("S S S", 120.0), &a, false).unwrap();
        assert_eq!(voicing_of(&vowel[vowel.len() / 2], &bands), 1.0);
        assert_eq!(voicing_of(&hiss[hiss.len() / 2], &bands), 0.0);
    }

    #[test]
    fn kmeans_finds_separated_blobs() {
        let mut rng = Mt::new(5);
        let mut points = Vec::new();
        for c in [(0.0, 0.0), (10.0, 0.0), (0.0, 10.0)] {
            for _ in 0..40 {
                points.push(vec![c.0 + rng.float64() - 0.5, c.1 + rng.float64() - 0.5]);
            }
        }
        let mut seed = Mt::new(1);
        let found = kmeans(&points, 3, &mut seed, 50).unwrap();
        assert!(found.converged);
        let mut centres: Vec<(i64, i64)> = found
            .centroids
            .iter()
            .map(|c| (c[0].round() as i64, c[1].round() as i64))
            .collect();
        centres.sort();
        assert_eq!(centres, vec![(0, 0), (0, 10), (10, 0)]);
        let mut seed = Mt::new(1);
        assert_eq!(
            kmeans(&points, 3, &mut seed, 50).unwrap().centroids,
            found.centroids
        );
        assert!(kmeans(&points[..2], 3, &mut Mt::new(1), 5).is_err());
    }

    #[test]
    fn learning_a_codebook_from_speech() {
        let book = small_book();
        assert_eq!((book.k(), book.mean.len()), (8, 40));
        assert!(book.counts.iter().all(|&c| c > 0));
        assert!(book.runs.iter().all(|&r| r >= 1.0));
        let back = Codebook::loads(&book.dumps()).unwrap();
        assert_eq!(back.centroids, book.centroids);
        assert_eq!(
            (
                back.runs.clone(),
                back.counts.clone(),
                back.mean.clone(),
                back.seed
            ),
            (book.runs.clone(), book.counts.clone(), book.mean.clone(), 3)
        );
        assert_eq!(back.analysis, book.analysis);
        assert_eq!(back.note, "test");
        assert_eq!(back.dumps(), book.dumps());
        assert_eq!(book.index("q0"), Some(0));
        assert_eq!(book.index("q7"), Some(7));
        for bad in ["q8", "q07", "q", "7", "Q1", "DH", "#"] {
            assert_eq!(book.index(bad), None, "{bad}");
        }
        assert_eq!(collapse_runs(&[1, 1, 2, 2, 2, 1, 3, 3]), vec![1, 2, 1, 3]);
        assert_eq!(
            run_lengths(&[1, 1, 2, 2, 2, 1], 3),
            (vec![0, 3, 3], vec![0, 2, 1])
        );
        assert!(learn(&[], 4, 1, 5, &Analysis::default(), "").is_err());
    }

    #[test]
    fn hearing_and_speaking_back() {
        let book = small_book();
        let tok = AcousticTokenizer::new(book.clone(), true).unwrap();
        let clips = clips();
        let heard = tok.listen(&clips[0], 0).unwrap();
        assert_eq!(heard.codes.len(), heard.frames);
        assert!(!heard.units.is_empty() && heard.units.len() < heard.frames);
        assert!(
            heard.units.windows(2).all(|w| w[0] != w[1]),
            "a run survived"
        );
        let pcm: Vec<u8> = clips[0]
            .iter()
            .flat_map(|s| ((s * 32768.0).round() as i16).to_le_bytes())
            .collect();
        assert_eq!(
            tok.listen_wav(&crate::synth::wav_bytes(&pcm, 16000))
                .unwrap()
                .units,
            heard.units
        );
        assert_eq!(tok.text("q1 q1 q2   q3\nq3").unwrap(), "q1 q2 q3");
        assert!(tok.text("q1 hello").is_err());
        // frame for frame, the sounds heard again are mostly the same units
        let mut voc = tok.vocoder(1.0, PITCH).unwrap();
        let mut out = Vec::new();
        for &code in &heard.codes {
            out.extend(voc.feed_frame(&book.frame_of(code)));
        }
        out.extend(voc.end());
        assert_eq!(out.len() / 2, (heard.frames - 1) * 160 + 400);
        let again = tok.listen(&pcm_samples(&out), 0).unwrap();
        let agree = heard
            .codes
            .iter()
            .zip(&again.codes)
            .filter(|(a, b)| a == b)
            .count();
        assert!(
            agree as f64 / heard.frames as f64 > 0.6,
            "{agree} of {}",
            heard.frames
        );
        let peak = out
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]).abs())
            .max()
            .unwrap();
        assert!(peak > 3000 && peak < 32767, "peak {peak}");
        let whole = tok.synthesize(&heard.units, 0, 1.0, PITCH).unwrap();
        let mut voc = tok.vocoder(1.0, PITCH).unwrap();
        let mut streamed = Vec::new();
        for u in &heard.units {
            streamed.extend(voc.feed(u).unwrap());
        }
        streamed.extend(voc.end());
        assert_eq!(streamed, whole);
        let held: usize = heard
            .units
            .iter()
            .map(|u| book.hold(book.index(u).unwrap()))
            .sum();
        assert_eq!(whole.len() / 2, (held - 1) * 160 + 400);
        assert_eq!(
            tok.synthesize(&heard.units, 2, 0.5, 200.0).unwrap().len(),
            whole.len()
        );
        assert!(tok
            .synthesize(&["q1".to_string(), "nope".to_string()], 0, 1.0, PITCH)
            .is_err());
        assert!(Vocoder::new(book.clone(), 1.0, 0.0).is_err());
        assert_eq!(
            synthesize(&[], &book, 0, 1.0, PITCH).unwrap(),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn the_bundled_codebook() {
        let tok = AcousticTokenizer::bundled().unwrap();
        assert_eq!(tok.book.k(), 64);
        assert_eq!(tok.book.analysis, Analysis::default());
        assert!(tok.book.note.contains("synthesizer"));
        let heard = tok.listen(&spoken(CLIPS[1], 120.0), 0).unwrap();
        let distinct: std::collections::HashSet<&String> = heard.units.iter().collect();
        assert!(distinct.len() > 10 && heard.units.len() < heard.frames);
        assert!(heard.units.iter().all(|u| tok.is_unit(u)));
        let other = tok.hear(&spoken(CLIPS[1], 175.0), 0).unwrap();
        let seen: std::collections::HashSet<&String> = other.iter().collect();
        let shared = distinct.iter().filter(|u| seen.contains(*u)).count();
        assert!(
            shared as f64 / distinct.len() as f64 > 0.5,
            "{shared} of {}",
            distinct.len()
        );
    }
}
