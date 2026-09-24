//! A voice for the sounds (the port of synth.py): a source-filter formant
//! synthesizer that speaks phones as they arrive, over the same voice table.

use crate::phones::{base, is_vowel, stress_of, BOS, EOS, PAUSE_FULL, PAUSE_QUESTION, PAUSE_SHORT};
use crate::tokenizer::parse_token;
use std::collections::HashMap;
use std::sync::OnceLock;

/// The voice table: the very file the Python package reads.
pub const VOICE_TEXT: &str = include_str!("../../phonetok/data/voice.tsv");

/// The default sample rate.
pub const RATE: u32 = 16000;

/// One row of the voice table: what a phone is made of.
#[derive(Clone, Debug)]
pub struct Sound {
    pub phone: String,
    pub kind: String,
    pub f: [f64; 3],
    pub bw: [f64; 3],
    pub ms: f64,
    pub voiced: bool,
    pub noise_f: f64,
    pub noise_bw: f64,
    pub noise_amp: f64,
    pub amp: f64,
    pub f2: Option<[f64; 3]>,
}

/// Phone -> Sound, read once.
pub fn voice() -> &'static HashMap<String, Sound> {
    static TABLE: OnceLock<HashMap<String, Sound>> = OnceLock::new();
    TABLE.get_or_init(|| {
        let mut table = HashMap::new();
        for line in VOICE_TEXT.lines() {
            if line.trim().is_empty() || line.starts_with('#') {
                continue;
            }
            let cols: Vec<&str> = line.split_whitespace().collect();
            let n: Vec<f64> = cols[2..].iter().map(|c| c.parse().unwrap_or(0.0)).collect();
            let kind = cols[1].to_string();
            let f2 = if kind == "diph" {
                Some([n[12], n[13], n[14]])
            } else {
                None
            };
            table.insert(
                cols[0].to_string(),
                Sound {
                    phone: cols[0].to_string(),
                    kind,
                    f: [n[0], n[1], n[2]],
                    bw: [n[3], n[4], n[5]],
                    ms: n[6],
                    voiced: n[7] >= 1.0,
                    noise_f: n[8],
                    noise_bw: n[9],
                    noise_amp: n[10],
                    amp: n[11],
                    f2,
                },
            );
        }
        table
    })
}

/// How the voice is set: its pitch, its pace, its loudness, the pauses it takes.
#[derive(Clone, Copy, Debug)]
pub struct VoiceSettings {
    pub pitch: f64,
    pub tempo: f64,
    pub gain: f64,
    pub short_pause: f64,
    pub full_pause: f64,
    pub lead: f64,
}

impl Default for VoiceSettings {
    fn default() -> Self {
        VoiceSettings {
            pitch: 120.0,
            tempo: 1.0,
            gain: 0.5,
            short_pause: 0.15,
            full_pause: 0.40,
            lead: 0.05,
        }
    }
}

struct Noise(u32);

impl Noise {
    fn next(&mut self) -> f64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x as f64 / 2147483648.0 - 1.0
    }
}

fn resonator(f: f64, bw: f64, rate: f64) -> (f64, f64, f64) {
    let c = -(-2.0 * std::f64::consts::PI * bw / rate).exp();
    let b = 2.0
        * (-std::f64::consts::PI * bw / rate).exp()
        * (2.0 * std::f64::consts::PI * f / rate).cos();
    (1.0 - b - c, b, c)
}

#[derive(Clone, Debug)]
struct Segment {
    f_from: [f64; 3],
    f_to: [f64; 3],
    bw: [f64; 3],
    seconds: f64,
    voiced: bool,
    amp: f64,
    noise_f: f64,
    noise_bw: f64,
    noise_amp: f64,
    pitch_from: f64,
    pitch_to: f64,
    aspiration: bool,
}

/// Speaks phones as they arrive: feed it tokens, take the PCM; `end` closes the utterance.
pub struct Synthesizer {
    pub rate: u32,
    pub voice: VoiceSettings,
    noise: Noise,
    pending: Vec<String>,
    last_f: [f64; 3],
    y: [[f64; 2]; 4],
    phase: f64,
    src_prev: f64,
    started: bool,
    elapsed: f64,
    pub total_samples: usize,
}

impl Synthesizer {
    pub fn new(rate: u32, voice: VoiceSettings) -> Synthesizer {
        Synthesizer {
            rate,
            voice,
            noise: Noise(0x9E3779B9),
            pending: Vec::new(),
            last_f: [500.0, 1500.0, 2500.0],
            y: [[0.0; 2]; 4],
            phase: 1.0,
            src_prev: 0.0,
            started: false,
            elapsed: 0.0,
            total_samples: 0,
        }
    }

    /// One token in; the PCM that can be committed comes back (often nothing).
    pub fn feed(&mut self, token: &str) -> Vec<u8> {
        if token == EOS {
            return self.end();
        }
        if token == BOS {
            return Vec::new();
        }
        let Some((kind, phones)) = parse_token(token) else {
            return Vec::new();
        };
        match kind {
            "boundary" => self.flush(None),
            "pause" => {
                let mut out = self.flush(Some(token));
                let pause = if token == PAUSE_SHORT {
                    self.voice.short_pause
                } else {
                    self.voice.full_pause
                };
                out.extend(self.silence(pause));
                out
            }
            "special" => Vec::new(),
            _ => {
                self.pending.extend(phones);
                Vec::new()
            }
        }
    }

    /// The final sentinel: what is pending is spoken with the closing intonation, then the utterance ends.
    pub fn end(&mut self) -> Vec<u8> {
        let mut out = self.flush(Some(PAUSE_FULL));
        if self.started {
            out.extend(self.silence(0.12));
        }
        self.started = false;
        self.elapsed = 0.0;
        self.last_f = [500.0, 1500.0, 2500.0];
        out
    }

    /// The whole of a token stream, ended.
    pub fn speak(&mut self, tokens: &[String]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            out.extend(self.feed(t));
        }
        out.extend(self.end());
        out
    }

    fn flush(&mut self, final_: Option<&str>) -> Vec<u8> {
        let phones = std::mem::take(&mut self.pending);
        if phones.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        if !self.started {
            out.extend(self.silence(self.voice.lead));
            self.started = true;
        }
        for seg in self.word_segments(&phones, final_) {
            out.extend(self.render(&seg));
        }
        out
    }

    fn pitch_at(
        &self,
        seconds: f64,
        stress: Option<u8>,
        final_: Option<&str>,
        last_syllable: bool,
    ) -> f64 {
        let mut f0 = self.voice.pitch * (1.0 - 0.12 * (seconds / 3.0).min(1.0));
        match stress {
            Some(1) => f0 *= 1.18,
            Some(2) => f0 *= 1.08,
            _ => {}
        }
        if last_syllable {
            match final_ {
                Some(PAUSE_QUESTION) => f0 *= 1.35,
                Some(PAUSE_FULL) => f0 *= 0.82,
                Some(PAUSE_SHORT) => f0 *= 1.06,
                _ => {}
            }
        }
        f0
    }

    fn word_segments(&mut self, phones: &[String], final_: Option<&str>) -> Vec<Segment> {
        let last_vowel = phones
            .iter()
            .rposition(|p| is_vowel(p))
            .unwrap_or(phones.len().saturating_sub(1));
        let mut segs = Vec::new();
        let mut t = self.elapsed;
        let tempo = self.voice.tempo;
        for (i, phone) in phones.iter().enumerate() {
            let b = base(phone);
            let Some(sound) = voice().get(b).cloned() else {
                continue;
            };
            let stress = stress_of(phone);
            let last_syllable = i >= last_vowel;
            let mut seconds = sound.ms / 1000.0 / tempo;
            if is_vowel(b) {
                match stress {
                    Some(1) => seconds *= 1.25,
                    Some(0) => seconds *= 0.75,
                    _ => {}
                }
            }
            if last_syllable && final_.is_some() {
                seconds *= if is_vowel(b) { 1.35 } else { 1.15 };
            }
            let mut amp = sound.amp;
            if is_vowel(b) && stress == Some(0) {
                amp *= 0.7;
            }
            let pitch_from = self.pitch_at(t, stress, final_, last_syllable);
            let pitch_to = self.pitch_at(t + seconds, stress, final_, last_syllable);
            match sound.kind.as_str() {
                "stop" | "affr" => {
                    let closure = seconds * if sound.kind == "stop" { 0.55 } else { 0.4 };
                    segs.push(Segment {
                        f_from: self.last_f,
                        f_to: sound.f,
                        bw: sound.bw,
                        seconds: closure,
                        voiced: sound.voiced,
                        amp: sound.amp * 0.4,
                        noise_f: 0.0,
                        noise_bw: 0.0,
                        noise_amp: 0.0,
                        pitch_from,
                        pitch_to: pitch_from,
                        aspiration: false,
                    });
                    let burst = if sound.kind == "stop" {
                        0.012
                    } else {
                        seconds - closure
                    };
                    segs.push(Segment {
                        f_from: sound.f,
                        f_to: sound.f,
                        bw: sound.bw,
                        seconds: burst,
                        voiced: false,
                        amp: 0.0,
                        noise_f: sound.noise_f,
                        noise_bw: sound.noise_bw,
                        noise_amp: sound.noise_amp,
                        pitch_from,
                        pitch_to: pitch_from,
                        aspiration: false,
                    });
                    if !sound.voiced && sound.kind == "stop" {
                        segs.push(Segment {
                            f_from: sound.f,
                            f_to: sound.f,
                            bw: sound.bw,
                            seconds: 0.035,
                            voiced: false,
                            amp: 0.0,
                            noise_f: 0.0,
                            noise_bw: 0.0,
                            noise_amp: 0.18,
                            pitch_from,
                            pitch_to: pitch_from,
                            aspiration: true,
                        });
                    }
                    self.last_f = sound.f;
                }
                "asp" => {
                    segs.push(Segment {
                        f_from: self.last_f,
                        f_to: sound.f,
                        bw: sound.bw,
                        seconds,
                        voiced: false,
                        amp: 0.0,
                        noise_f: 0.0,
                        noise_bw: 0.0,
                        noise_amp: sound.noise_amp,
                        pitch_from,
                        pitch_to,
                        aspiration: true,
                    });
                    self.last_f = sound.f;
                }
                _ => {
                    if let (true, Some(f2)) = (sound.kind == "diph", sound.f2) {
                        let half = seconds / 2.0;
                        let mid = (pitch_from + pitch_to) / 2.0;
                        segs.push(Segment {
                            f_from: self.last_f,
                            f_to: sound.f,
                            bw: sound.bw,
                            seconds: half,
                            voiced: true,
                            amp,
                            noise_f: 0.0,
                            noise_bw: 0.0,
                            noise_amp: 0.0,
                            pitch_from,
                            pitch_to: mid,
                            aspiration: false,
                        });
                        segs.push(Segment {
                            f_from: sound.f,
                            f_to: f2,
                            bw: sound.bw,
                            seconds: half,
                            voiced: true,
                            amp,
                            noise_f: 0.0,
                            noise_bw: 0.0,
                            noise_amp: 0.0,
                            pitch_from: mid,
                            pitch_to,
                            aspiration: false,
                        });
                        self.last_f = f2;
                    } else {
                        segs.push(Segment {
                            f_from: self.last_f,
                            f_to: sound.f,
                            bw: sound.bw,
                            seconds,
                            voiced: sound.voiced,
                            amp,
                            noise_f: sound.noise_f,
                            noise_bw: sound.noise_bw,
                            noise_amp: sound.noise_amp,
                            pitch_from,
                            pitch_to,
                            aspiration: false,
                        });
                        self.last_f = sound.f;
                    }
                }
            }
            t += seconds;
        }
        self.elapsed = t;
        segs
    }

    fn silence(&mut self, seconds: f64) -> Vec<u8> {
        let n = (seconds * self.rate as f64).round() as usize;
        self.total_samples += n;
        self.elapsed += seconds;
        vec![0u8; 2 * n]
    }

    fn render(&mut self, seg: &Segment) -> Vec<u8> {
        let rate = self.rate as f64;
        let total = ((seg.seconds * rate).round() as usize).max(1);
        let frame = ((rate * 5.0 / 1000.0) as usize).max(1);
        let mut out = Vec::with_capacity(2 * total);
        let gain = self.voice.gain * 32767.0;
        let nyquist = rate * 0.45;
        let mut done = 0;
        while done < total {
            let n = frame.min(total - done);
            let mid = (done as f64 + n as f64 / 2.0) / total as f64;
            let g = (mid / 0.4).min(1.0);
            let mut coefs = [(0.0, 0.0, 0.0); 3];
            for (k, coef) in coefs.iter_mut().enumerate() {
                let f = seg.f_from[k] + (seg.f_to[k] - seg.f_from[k]) * g;
                *coef = resonator(f.min(nyquist), seg.bw[k], rate);
            }
            let noise_coefs = if seg.noise_amp > 0.0 && !seg.aspiration {
                Some(resonator(
                    seg.noise_f.min(nyquist),
                    seg.noise_bw.max(50.0),
                    rate,
                ))
            } else {
                None
            };
            let pitch = seg.pitch_from + (seg.pitch_to - seg.pitch_from) * mid;
            let period = rate / pitch.max(40.0);
            for _ in 0..n {
                let mut src = 0.0;
                if seg.voiced && seg.amp > 0.0 {
                    self.phase += 1.0 / period;
                    if self.phase >= 1.0 {
                        self.phase -= 1.0;
                    }
                    let ph = self.phase;
                    let flow = if ph < 0.4 {
                        0.5 * (1.0 - (std::f64::consts::PI * ph / 0.4).cos())
                    } else if ph < 0.6 {
                        (std::f64::consts::PI * (ph - 0.4) / 0.4).cos()
                    } else {
                        0.0
                    };
                    src = (flow - self.src_prev) * 4.0 * seg.amp;
                    self.src_prev = flow;
                }
                if seg.aspiration && seg.noise_amp > 0.0 {
                    src += self.noise.next() * seg.noise_amp * 0.6;
                }
                let mut x = src;
                for (k, &(a, b, c)) in coefs.iter().enumerate() {
                    let m = &mut self.y[k];
                    let v = a * x + b * m[0] + c * m[1];
                    m[1] = m[0];
                    m[0] = v;
                    x = v;
                }
                if let Some((a, b, c)) = noise_coefs {
                    let m = &mut self.y[3];
                    let v = a * self.noise.next() * seg.noise_amp + b * m[0] + c * m[1];
                    m[1] = m[0];
                    m[0] = v;
                    x += v * 3.0;
                }
                let s = (x * gain * 0.09).clamp(-32767.0, 32767.0);
                out.extend_from_slice(&(s as i16).to_le_bytes());
            }
            done += n;
        }
        self.total_samples += total;
        out
    }
}

/// The 44-byte header of a 16-bit mono WAV; `samples` unknown gives the streaming form.
pub fn wav_header(rate: u32, samples: Option<usize>) -> Vec<u8> {
    let size: u32 = samples.map(|n| (n * 2) as u32).unwrap_or(0xFFFF_FFFF);
    let riff: u32 = samples.map(|_| 36 + size).unwrap_or(0xFFFF_FFFF);
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&riff.to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes());
    h.extend_from_slice(&rate.to_le_bytes());
    h.extend_from_slice(&(rate * 2).to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&size.to_le_bytes());
    h
}

/// A WAV file holding `pcm`.
pub fn wav_bytes(pcm: &[u8], rate: u32) -> Vec<u8> {
    let mut out = wav_header(rate, Some(pcm.len() / 2));
    out.extend_from_slice(pcm);
    out
}
