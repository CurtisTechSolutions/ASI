//! The neural vocoder for the acoustic units (`phonetok/neural.py`, ported): the units heard back through a
//! filter learned from recordings, over the vocoder's own excitation.
//!
//! A pulse train at the voice's pitch and the vocoder's deterministic noise run on under the utterance, and a
//! small network - an embedding of the unit codes, residual blocks of dilated convolutions at the frame rate,
//! and a head - writes for every frame and every bin of the transform how loud each of the two should be
//! there.  The frames are overlap-added exactly as the vocoder's own are, so this port and the others hear the
//! same samples from the same weights.  Inference only: training is the Python module's (numpy), and it writes
//! the weights file this reads (`<codebook stem>.vocoder.json`, tensors as base64 little-endian float32).

use crate::acoustic::{hann, istft, pack_pcm, stft, Analysis, Codebook, Noise, Spectra};
use crate::json::{self, Json};

/// The vocoder trained for the bundled codebook.
pub const VOCODER_TEXT: &str = include_str!("../../phonetok/data/acoustic.vocoder.json");
pub const VOCODER_SUFFIX: &str = ".vocoder.json";
/// The slope of the leaky rectifier below zero.
pub const SLOPE: f64 = 0.1;
/// The range a predicted log gain is held to.
pub const CLIP: (f64, f64) = (-24.0, 8.0);
/// What `synthesize` scales the waveform by at gain 1: three decibels of headroom for the peaks the dispersed
/// excitation still has.
pub const HEADROOM: f64 = 0.7;

/// The architecture: what the file says.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    pub units: usize,
    pub bins: usize,
    pub channels: usize,
    pub kernel: usize,
    pub dilations: Vec<usize>,
}

impl Spec {
    pub fn validate(&self) -> Result<(), String> {
        if self.units < 1 || self.bins < 2 || self.channels < 1 {
            return Err(format!("impossible vocoder: {self:?}"));
        }
        if self.kernel < 1 || self.kernel % 2 == 0 {
            return Err(format!("the kernel must be odd, got {}", self.kernel));
        }
        if self.dilations.is_empty() || self.dilations.iter().any(|&d| d < 1) {
            return Err(format!(
                "the dilations must be at least 1, got {:?}",
                self.dilations
            ));
        }
        Ok(())
    }

    /// Frames on either side that can reach a frame's output: the receptive field's radius.
    pub fn context(&self) -> usize {
        self.dilations
            .iter()
            .map(|d| (self.kernel - 1) / 2 * d)
            .sum()
    }

    pub fn parameters(&self) -> usize {
        let (c, k) = (self.channels, self.kernel);
        self.units * c
            + self.dilations.len() * (c * c * k + c + c * c + c)
            + 2 * self.bins * c
            + 2 * self.bins
            + self.units * 2 * self.bins
    }
}

#[derive(Clone, Debug)]
struct Block {
    /// The convolution's weights arranged per tap: `[tap][out][in]`.
    taps: Vec<f64>,
    b1: Vec<f64>,
    /// `[out][in]`.
    w2: Vec<f64>,
    b2: Vec<f64>,
}

/// Codes to samples through the learned filter over the vocoder's excitation.
#[derive(Clone, Debug)]
pub struct UnitVocoder {
    pub analysis: Analysis,
    pub spec: Spec,
    /// `[units][channels]`.
    embed: Vec<f64>,
    blocks: Vec<Block>,
    /// `[2 * bins][channels]`: the pulse train's log gains, then the noise's.
    head_w: Vec<f64>,
    head_b: Vec<f64>,
    /// `[units][2 * bins]`: every unit's own log gains, added to the head's output.
    template: Vec<f64>,
    pub note: String,
    pub checksum: f64,
    pub trained: Json,
}

fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let mut buf: u32 = 0;
    let mut bits = 0;
    for c in text.bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' => 62,
            b'/' => 63,
            b'=' => break,
            b'\n' | b'\r' | b' ' => continue,
            _ => return Err(format!("not base64: {:?}", c as char)),
        };
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push(((buf >> bits) & 0xFF) as u8);
        }
    }
    Ok(out)
}

fn tensor(t: &Json, shape: &[usize]) -> Result<Vec<f64>, String> {
    let got: Vec<usize> = t.get("shape").ints().iter().map(|&v| v as usize).collect();
    if got != shape {
        return Err(format!(
            "a tensor of shape {got:?} where {shape:?} was expected"
        ));
    }
    let count: usize = shape.iter().product();
    let text = t
        .get("f32")
        .as_str()
        .ok_or("a tensor without its f32 text")?;
    let bytes = base64_decode(text)?;
    if bytes.len() != 4 * count {
        return Err(format!(
            "a tensor of {count} values packed as {} bytes",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
        .collect())
}

impl UnitVocoder {
    pub fn loads(text: &str) -> Result<UnitVocoder, String> {
        let doc = json::parse(text)?;
        if doc.get("phonetok").as_str() != Some("acoustic vocoder") {
            return Err("not a phonetok acoustic vocoder file".to_string());
        }
        let model = doc.get("model").as_str().unwrap_or("source-filter");
        let version = doc.get("version").as_f64().unwrap_or(1.0);
        if model != "source-filter" || version != 1.0 {
            return Err(format!(
                "a vocoder of a kind this version does not know: {model} v{version}"
            ));
        }
        let num = |j: &Json, key: &str| -> Result<f64, String> {
            j.get(key)
                .as_f64()
                .ok_or_else(|| format!("the vocoder file has no {key}"))
        };
        let an = doc.get("analysis");
        let analysis = Analysis {
            rate: num(an, "rate")? as u32,
            frame: num(an, "frame")? as usize,
            hop: num(an, "hop")? as usize,
            fft: num(an, "fft")? as usize,
            ..Analysis::default()
        };
        let spec = Spec {
            units: num(&doc, "units")? as usize,
            bins: analysis.fft / 2 + 1,
            channels: num(&doc, "channels")? as usize,
            kernel: num(&doc, "kernel")? as usize,
            dilations: doc
                .get("dilations")
                .ints()
                .iter()
                .map(|&d| d.max(0) as usize)
                .collect(),
        };
        spec.validate()?;
        let (c, k) = (spec.channels, spec.kernel);
        let w = doc.get("weights");
        let embed = tensor(w.get("embed"), &[spec.units, c])?;
        let blocks_json = w.get("blocks").as_arr();
        if blocks_json.len() != spec.dilations.len() {
            return Err(format!(
                "{} blocks of weights for {} dilations",
                blocks_json.len(),
                spec.dilations.len()
            ));
        }
        let mut blocks = Vec::with_capacity(blocks_json.len());
        for b in blocks_json {
            let w1 = tensor(b.get("w1"), &[c, c, k])?; // [out][in][tap]
            let mut taps = vec![0.0; k * c * c];
            for o in 0..c {
                for i in 0..c {
                    for j in 0..k {
                        taps[(j * c + o) * c + i] = w1[(o * c + i) * k + j];
                    }
                }
            }
            blocks.push(Block {
                taps,
                b1: tensor(b.get("b1"), &[c])?,
                w2: tensor(b.get("w2"), &[c, c])?,
                b2: tensor(b.get("b2"), &[c])?,
            });
        }
        let head = w.get("head");
        Ok(UnitVocoder {
            analysis,
            head_w: tensor(head.get("w"), &[2 * spec.bins, c])?,
            head_b: tensor(head.get("b"), &[2 * spec.bins])?,
            template: tensor(w.get("template"), &[spec.units, 2 * spec.bins])?,
            spec,
            embed,
            blocks,
            note: doc.get("note").as_str().unwrap_or("").to_string(),
            checksum: doc.get("checksum").as_f64().unwrap_or(0.0),
            trained: doc.get("trained").clone(),
        })
    }

    pub fn load(path: &str) -> Result<UnitVocoder, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        UnitVocoder::loads(&text)
    }

    /// The vocoder trained for the bundled codebook.
    pub fn bundled() -> Result<UnitVocoder, String> {
        UnitVocoder::loads(VOCODER_TEXT)
    }

    /// Whether this vocoder was trained for `book`: the same units, analysis and fingerprint.
    pub fn matches(&self, book: &Codebook) -> bool {
        let (a, b) = (&self.analysis, &book.analysis);
        self.spec.units == book.k()
            && (a.rate, a.frame, a.hop, a.fft) == (b.rate, b.frame, b.hop, b.fft)
            && (self.checksum - codebook_checksum(book)).abs() < 1e-3
    }

    pub fn describe(&self) -> Json {
        let s = &self.spec;
        let a = &self.analysis;
        Json::obj(vec![
            ("units", Json::Num(s.units as f64)),
            ("channels", Json::Num(s.channels as f64)),
            ("kernel", Json::Num(s.kernel as f64)),
            (
                "dilations",
                Json::nums(&s.dilations.iter().map(|&d| d as f64).collect::<Vec<_>>()),
            ),
            ("bins", Json::Num(s.bins as f64)),
            ("parameters", Json::Num(s.parameters() as f64)),
            ("context_frames", Json::Num(s.context() as f64)),
            (
                "context_seconds",
                Json::Num(s.context() as f64 * a.hop as f64 / a.rate as f64),
            ),
            (
                "analysis",
                Json::obj(vec![
                    ("rate", Json::Num(a.rate as f64)),
                    ("frame", Json::Num(a.frame as f64)),
                    ("hop", Json::Num(a.hop as f64)),
                    ("fft", Json::Num(a.fft as f64)),
                ]),
            ),
            ("checksum", Json::Num(self.checksum)),
            ("note", Json::str(&self.note)),
            ("trained", self.trained.clone()),
        ])
    }

    /// The log gains of every frame: the pulse train's per bin, and the noise's (each `[frames][bins]`).
    #[allow(clippy::type_complexity)]
    pub fn filters(&self, codes: &[usize]) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), String> {
        let s = &self.spec;
        let (c, k) = (s.channels, s.kernel);
        let centre = ((k - 1) / 2) as isize;
        let count = codes.len();
        for &code in codes {
            if code >= s.units {
                return Err(format!(
                    "code {code} is not one of this vocoder's {} units",
                    s.units
                ));
            }
        }
        let mut x: Vec<f64> = Vec::with_capacity(count * c);
        for &code in codes {
            x.extend_from_slice(&self.embed[code * c..(code + 1) * c]);
        }
        for (block, &d) in self.blocks.iter().zip(&s.dilations) {
            let mut pre = vec![0.0; count * c];
            for t in 0..count {
                pre[t * c..(t + 1) * c].copy_from_slice(&block.b1);
            }
            for j in 0..k {
                let shift = (j as isize - centre) * d as isize;
                let tap = &block.taps[j * c * c..(j + 1) * c * c];
                for t in 0..count {
                    let src = t as isize + shift;
                    if src < 0 || src >= count as isize {
                        continue;
                    }
                    let xs = &x[src as usize * c..(src as usize + 1) * c];
                    let acc = &mut pre[t * c..(t + 1) * c];
                    for o in 0..c {
                        let row = &tap[o * c..(o + 1) * c];
                        let mut total = 0.0;
                        for i in 0..c {
                            total += row[i] * xs[i];
                        }
                        acc[o] += total;
                    }
                }
            }
            for t in 0..count {
                let h: Vec<f64> = pre[t * c..(t + 1) * c]
                    .iter()
                    .map(|&v| if v > 0.0 { v } else { SLOPE * v })
                    .collect();
                let xt = &mut x[t * c..(t + 1) * c];
                for (o, out) in xt.iter_mut().enumerate() {
                    let row = &block.w2[o * c..(o + 1) * c];
                    let mut total = block.b2[o];
                    for i in 0..c {
                        total += row[i] * h[i];
                    }
                    *out += total;
                }
            }
        }
        let bins = s.bins;
        let (lo, hi) = CLIP;
        let mut lp = Vec::with_capacity(count);
        let mut ln = Vec::with_capacity(count);
        for t in 0..count {
            let xt = &x[t * c..(t + 1) * c];
            let template = &self.template[codes[t] * 2 * bins..(codes[t] + 1) * 2 * bins];
            let mut z = vec![0.0; 2 * bins];
            for (o, out) in z.iter_mut().enumerate() {
                let row = &self.head_w[o * c..(o + 1) * c];
                let mut total = template[o] + self.head_b[o];
                for i in 0..c {
                    total += row[i] * xt[i];
                }
                *out = total.clamp(lo, hi);
            }
            ln.push(z.split_off(bins));
            lp.push(z);
        }
        Ok((lp, ln))
    }

    /// The full (Hermitian) spectrum of every frame, as the inverse transform takes it.
    pub fn spectra(&self, codes: &[usize], pitch: f64) -> Result<Spectra, String> {
        let (lp, ln) = self.filters(codes)?;
        let (pulses, noises) = excitation_spectra(codes.len(), &self.analysis, pitch)?;
        let n = self.analysis.fft;
        let half = n / 2;
        let mut out = Vec::with_capacity(codes.len());
        for t in 0..codes.len() {
            let mut re = vec![0.0; n];
            let mut im = vec![0.0; n];
            let (pre, pim) = &pulses[t];
            let (nre, nim) = &noises[t];
            for k in 0..=half {
                let gp = lp[t][k].exp();
                let gn = ln[t][k].exp();
                re[k] = pre[k] * gp + nre[k] * gn;
                im[k] = pim[k] * gp + nim[k] * gn;
            }
            im[0] = 0.0;
            im[half] = 0.0;
            for k in 1..half {
                re[n - k] = re[k];
                im[n - k] = -im[k];
            }
            out.push((re, im));
        }
        Ok(out)
    }

    /// The waveform of a run of codes, in [-1, 1].
    pub fn samples(&self, codes: &[usize], pitch: f64) -> Result<Vec<f64>, String> {
        Ok(istft(&self.spectra(codes, pitch)?, &self.analysis))
    }

    /// 16-bit PCM at the analysis' rate: the waveform at `gain` times [`HEADROOM`].
    pub fn synthesize(&self, codes: &[usize], gain: f64, pitch: f64) -> Result<Vec<u8>, String> {
        Ok(pack_pcm(&self.samples(codes, pitch)?, gain * HEADROOM))
    }
}

/// A cheap fingerprint of a codebook, kept in its vocoder so the two are not mixed up.
pub fn codebook_checksum(book: &Codebook) -> f64 {
    let mut total = 0.0;
    for c in &book.centroids {
        for v in c {
            total += v;
        }
    }
    for v in &book.mean {
        total += v;
    }
    ((total + book.k() as f64) * 1e6).round() / 1e6
}

/// The code of every frame a run of units stands for: each unit held for its typical run.
pub fn codes_of(units: &[String], book: &Codebook) -> Result<Vec<usize>, String> {
    let mut codes = Vec::new();
    for unit in units {
        let index = book
            .index(unit)
            .ok_or_else(|| format!("not a unit of this codebook: {unit:?}"))?;
        codes.extend(std::iter::repeat_n(index, book.hold(index)));
    }
    Ok(codes)
}

/// Where a codebook's vocoder lives: `codebook.tsv` -> `codebook.vocoder.json`.
pub fn vocoder_path_for(codebook_path: &str) -> String {
    let stem = match codebook_path.rsplit_once('.') {
        Some((stem, ext)) if ext.eq_ignore_ascii_case("tsv") && !stem.is_empty() => stem,
        _ => codebook_path,
    };
    format!("{stem}{VOCODER_SUFFIX}")
}

/// The vocoder that belongs to a codebook, when there is one: `path` names the file outright; else
/// `$PHONETOK_VOCODER`; else the file beside the codebook's own (`source`), or the bundled vocoder for a codebook
/// with no `source`, when it matches.  A file that claims to belong to the codebook but does not match it is an
/// error; none at all is `None`.
pub fn find_vocoder(
    book: &Codebook,
    source: Option<&str>,
    path: Option<&str>,
) -> Result<Option<UnitVocoder>, String> {
    let env = std::env::var("PHONETOK_VOCODER")
        .ok()
        .filter(|p| !p.is_empty());
    let named = path.map(String::from).or(env);
    let candidate = |vocoder: UnitVocoder, name: &str| -> Result<Option<UnitVocoder>, String> {
        if !vocoder.matches(book) {
            return Err(format!("{name} was not trained for this codebook"));
        }
        Ok(Some(vocoder))
    };
    if let Some(named) = named {
        if !std::path::Path::new(&named).is_file() {
            return Err(format!("vocoder file not found: {named}"));
        }
        return candidate(UnitVocoder::load(&named)?, &named);
    }
    match source {
        Some(src) => {
            let beside = vocoder_path_for(src);
            if std::path::Path::new(&beside).is_file() {
                candidate(UnitVocoder::load(&beside)?, &beside)
            } else {
                Ok(None)
            }
        }
        // a codebook that came from nowhere may be the bundled one: the bundled vocoder is tried, and
        // merely not the codebook's when it does not match
        None => Ok(UnitVocoder::bundled().ok().filter(|v| v.matches(book))),
    }
}

// -- the excitation: the source every port makes the same ---------------------------------

/// How many samples each pulse is spread over: a Hann-windowed linear chirp of unit energy sweeping from
/// zero to the Nyquist frequency, so the waveform peaks where the recordings' do instead of three times
/// higher (an impulse's harmonics are all in phase at the pulse).
pub const CHIRP: usize = 64;

/// The chirp every pulse becomes, of unit energy.
pub fn chirp_kernel() -> Vec<f64> {
    let window = hann(CHIRP);
    let kernel: Vec<f64> = (0..CHIRP)
        .map(|t| ((std::f64::consts::PI * (t * t) as f64) / (2.0 * CHIRP as f64)).cos() * window[t])
        .collect();
    let norm = kernel.iter().map(|v| v * v).sum::<f64>().sqrt();
    kernel.iter().map(|v| v / norm).collect()
}

/// The pulse train with every pulse spread into the chirp (pulses taken in order, so every port sums alike).
pub fn disperse(pulses: &[f64]) -> Vec<f64> {
    let kernel = chirp_kernel();
    let mut out = vec![0.0; pulses.len()];
    for (p, &v) in pulses.iter().enumerate() {
        if v == 0.0 {
            continue;
        }
        for (t, k) in kernel.iter().enumerate() {
            if p + t >= out.len() {
                break;
            }
            out[p + t] += v * k;
        }
    }
    out
}

/// Unit impulses at `pitch` from phase zero: exactly the vocoder's pulse train (see [`disperse`]).
pub fn pulse_train(length: usize, pitch: f64, rate: u32) -> Result<Vec<f64>, String> {
    if pitch <= 0.0 {
        return Err(format!("pitch must be positive, got {pitch}"));
    }
    let step = pitch / rate as f64;
    let mut phase = 0.0;
    let mut out = vec![0.0; length];
    for v in out.iter_mut() {
        phase += step;
        if phase >= 1.0 {
            phase -= 1.0;
            *v = 1.0;
        }
    }
    Ok(out)
}

/// The first `length` values of the vocoder's noise (xorshift32 from its fixed seed), in [-1, 1).
pub fn noise_train(length: usize) -> Vec<f64> {
    let mut noise = Noise::new();
    (0..length).map(|_| noise.next()).collect()
}

pub fn excitation_length(count: usize, a: &Analysis) -> usize {
    if count == 0 {
        0
    } else {
        (count - 1) * a.hop + a.frame
    }
}

/// The spectra of the dispersed pulse train's frames and of the noise's, `count` of each (full transforms;
/// the first `fft/2 + 1` bins are the half the network's gains apply to).
pub fn excitation_spectra(
    count: usize,
    a: &Analysis,
    pitch: f64,
) -> Result<(Spectra, Spectra), String> {
    let length = excitation_length(count, a);
    let pulses = stft(&disperse(&pulse_train(length, pitch, a.rate)?), a, count);
    let noises = stft(&noise_train(length), a, count);
    Ok((pulses, noises))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_pulse_train_and_the_noise() {
        let pulses = pulse_train(16000, 100.0, 16000).unwrap();
        // the phase accumulates 0.00625 a sample, which is not exact in binary: the first pulse lands on sample
        // 160 and ninety-nine fit in a second, in every port alike
        assert_eq!(pulses.iter().filter(|&&v| v == 1.0).count(), 99);
        assert_eq!(pulses.iter().position(|&v| v == 1.0), Some(160));
        assert!(pulse_train(10, 0.0, 16000).is_err());
        let noise = noise_train(5);
        assert_eq!(noise, noise_train(8)[..5]);
        assert!(noise.iter().all(|v| (-1.0..1.0).contains(v)));
        let kernel = chirp_kernel();
        assert_eq!(kernel.len(), CHIRP);
        assert!((kernel.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        let dispersed = disperse(&pulse_train(1600, 100.0, 16000).unwrap());
        assert!((dispersed.iter().map(|v| v * v).sum::<f64>() - 9.0).abs() < 1e-9);
        assert_eq!(dispersed[100], 0.0);
        assert_eq!(dispersed[160], 0.0); // the window starts at zero
        assert_ne!(dispersed[161], 0.0);
    }

    #[test]
    fn base64_and_tensors() {
        assert_eq!(base64_decode("AAAAAA==").unwrap(), vec![0, 0, 0, 0]);
        assert_eq!(base64_decode("AACAPw==").unwrap(), vec![0, 0, 128, 63]); // 1.0f32
        let t = json::parse(r#"{"shape":[2],"f32":"AACAPwAAAEA="}"#).unwrap();
        assert_eq!(tensor(&t, &[2]).unwrap(), vec![1.0, 2.0]);
        assert!(tensor(&t, &[3]).is_err());
    }

    #[test]
    fn the_bundled_vocoder_speaks_its_codebook() {
        let book = Codebook::bundled().unwrap();
        let vocoder = UnitVocoder::bundled().unwrap();
        assert!(vocoder.matches(&book));
        assert_eq!(vocoder.spec.units, book.k());
        let units: Vec<String> = ["q1", "q2", "q3"].iter().map(|s| s.to_string()).collect();
        let codes = codes_of(&units, &book).unwrap();
        let held: usize = (1..=3).map(|i| book.hold(i)).sum();
        assert_eq!(codes.len(), held);
        let pcm = vocoder.synthesize(&codes, 1.0, 120.0).unwrap();
        assert_eq!(pcm.len() / 2, excitation_length(held, &book.analysis));
        assert!(vocoder.filters(&[book.k()]).is_err());
        assert!(codes_of(&["nope".to_string()], &book).is_err());
        let found = find_vocoder(&book, None, None).unwrap().unwrap();
        assert_eq!(found.checksum, vocoder.checksum);
        assert_eq!(vocoder_path_for("mine.tsv"), "mine.vocoder.json");
        assert_eq!(vocoder_path_for("mine"), "mine.vocoder.json");
        let tok = crate::acoustic::AcousticTokenizer::bundled().unwrap();
        assert_eq!(tok.vocoder_name(None).unwrap(), "neural");
        assert_eq!(tok.synthesize(&units, 0, 1.0, 120.0).unwrap(), pcm);
        assert_ne!(
            tok.synthesize_with(&units, 0, 1.0, 120.0, Some(false))
                .unwrap(),
            pcm
        );
    }
}
