//! Compress the matrix into its central node, from the outside in; rebuild it from the middle outward.
//!
//! The code the central node holds ([`Core`]) is the whole matrix with the least loss there is, laid out in
//! [`geometry::center_out`] order: one bit per cell (whether anything was ever written to that edge - an untouched
//! edge is the prototype's and comes back exactly), then the touched edges' six integers and fifteen numbers,
//! packed little-endian, and a small lossless record of the rest of the machine. `Precision::Exact` packs the
//! numbers as float64, so rebuilding gives back every field of every edge bit for bit; `Float32` and `Float16`
//! pack them smaller, and [`fidelity`] measures what that costs. Because the code runs from the centre out,
//! rebuilding can stop at any shell, and the machine can be walked straight from the code. The same code, byte
//! for byte, as `../latticefsm/compress.py`; `DESIGN.md` §11.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};

use crate::edge::{Edge, Weighting};
use crate::geometry::{center, center_out, shell_sizes, shells};
use crate::gzip;
use crate::json::{parse, Json};
use crate::lattice::Lattice;
use crate::machine::{argmax, softmax, Machine};

/// An edge's integers, in the order the code packs them.
pub const INTS: [&str; 6] = [
    "seen",
    "first_seen",
    "last_seen",
    "last_rewarded",
    "last_punished",
    "width_stamp",
];
/// An edge's numbers, in the order the code packs them.
pub const FLOATS: [&str; 15] = [
    "recent", "rewarded", "punished", "width", "w_bias", "w_seen", "w_recent", "w_net", "w_age", "w_width", "f_seen",
    "f_recent", "f_net", "f_age", "f_width",
];
/// The largest KL divergence (nats) a row may move by for [`fidelity`] to call the behaviour preserved.
pub const MAX_KL: f64 = 1e-3;
/// Two log-weights this close are a tie: a rebuilt greedy choice among the original's tied best is not a change.
pub const TIE: f64 = 1e-6;

const HALF_MAX: f64 = 65504.0;
const FORMAT: &str = "latticefsm-core";

/// How the fifteen numbers are packed, least lossy first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    Exact,
    Float32,
    Float16,
}

impl Precision {
    pub const ALL: [Precision; 3] = [Precision::Exact, Precision::Float32, Precision::Float16];

    pub fn name(self) -> &'static str {
        match self {
            Precision::Exact => "exact",
            Precision::Float32 => "float32",
            Precision::Float16 => "float16",
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Precision::Exact => 8,
            Precision::Float32 => 4,
            Precision::Float16 => 2,
        }
    }

    pub fn parse(name: &str) -> Result<Precision, String> {
        Precision::ALL
            .into_iter()
            .find(|p| p.name() == name)
            .ok_or_else(|| format!("precision must be exact, float32 or float16, got {name:?}"))
    }

    fn pack(self, x: f64, out: &mut Vec<u8>) {
        match self {
            Precision::Exact => out.extend_from_slice(&x.to_le_bytes()),
            Precision::Float32 => {
                out.extend_from_slice(&((x.clamp(-(f32::MAX as f64), f32::MAX as f64)) as f32).to_le_bytes())
            }
            Precision::Float16 => out.extend_from_slice(&f64_to_f16(x.clamp(-HALF_MAX, HALF_MAX)).to_le_bytes()),
        }
    }

    fn unpack(self, b: &[u8]) -> f64 {
        match self {
            Precision::Exact => f64::from_le_bytes(b[..8].try_into().expect("8 bytes")),
            Precision::Float32 => f32::from_le_bytes(b[..4].try_into().expect("4 bytes")) as f64,
            Precision::Float16 => f16_to_f64(u16::from_le_bytes(b[..2].try_into().expect("2 bytes"))),
        }
    }
}

/// An IEEE half from a double, rounding to nearest with ties to even - what Python's `struct.pack("e")` does.
pub fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a.is_nan() {
        return sign | 0x7e00;
    }
    if a == 0.0 {
        return sign;
    }
    if a < 6.103515625e-05 {
        // subnormal: units of 2^-24; a carry to 1024 is the smallest normal, whose bits are the same number
        return sign | (a * 16_777_216.0).round_ties_even() as u16;
    }
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let mut m = (a * 2f64.powi(10 - e)).round_ties_even() as u32 - 1024;
    let mut e = e;
    if m == 1024 {
        m = 0;
        e += 1;
    }
    if e > 15 {
        return sign | 0x7c00;
    }
    sign | (((e + 15) as u16) << 10) | m as u16
}

/// A double from an IEEE half, exactly.
pub fn f16_to_f64(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f64;
    sign * match e {
        0 => m * 2f64.powi(-24),
        31 => {
            if m == 0.0 {
                f64::INFINITY
            } else {
                f64::NAN
            }
        }
        _ => (1.0 + m / 1024.0) * 2f64.powi(e - 15),
    }
}

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
pub fn base64_encode(data: &[u8]) -> String {
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let n =
            (chunk[0] as u32) << 16 | (*chunk.get(1).unwrap_or(&0) as u32) << 8 | *chunk.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= chunk.len() {
                out.push(B64[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                out.push('=');
            }
        }
    }
    out
}

pub fn base64_decode(text: &str) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(text.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    for c in text.bytes() {
        if c == b'=' {
            break;
        }
        let v = B64
            .iter()
            .position(|&b| b == c)
            .ok_or_else(|| format!("bad base64 character {:?}", c as char))?;
        acc = acc << 6 | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits & 0xff) as u8);
        }
    }
    Ok(out)
}

fn bits_to_hex(bits: &[bool]) -> String {
    bits.chunks(4)
        .map(|c| {
            let nib = c.iter().enumerate().fold(0u8, |n, (j, &b)| n | (b as u8) << (3 - j));
            char::from(b"0123456789abcdef"[nib as usize])
        })
        .collect()
}

fn hex_to_bits(text: &str, n: usize) -> Result<Vec<bool>, String> {
    let mut bits = Vec::with_capacity(text.len() * 4);
    for c in text.chars() {
        let nib = c.to_digit(16).ok_or("the touched bitmap is not hex")?;
        bits.extend((0..4).map(|j| nib >> (3 - j) & 1 == 1));
    }
    if bits.len() < n {
        return Err("the touched bitmap is too short".to_string());
    }
    bits.truncate(n);
    Ok(bits)
}

fn edge_ints(e: &Edge) -> [i64; 6] {
    [
        e.seen as i64,
        e.first_seen,
        e.last_seen,
        e.last_rewarded,
        e.last_punished,
        e.width_stamp,
    ]
}

fn edge_floats(e: &Edge) -> [f64; 15] {
    let w = &e.weighting;
    let f = e.features;
    [
        e.recent, e.rewarded, e.punished, e.width, w.bias, w.seen, w.recent, w.net, w.age, w.width, f[0], f[1], f[2],
        f[3], f[4],
    ]
}

/// The whole matrix, compressed into its central node.
#[derive(Clone, Debug)]
pub struct Core {
    pub shape: (usize, usize, usize),
    pub alphabet: Vec<String>,
    pub precision: Precision,
    /// 4 or 8: the integers are packed as int32 when every one fits, else int64.
    pub int_width: usize,
    /// One bit per cell, in centre-out order.
    pub touched: Vec<bool>,
    pub ints: Vec<u8>,
    pub floats: Vec<u8>,
    /// Everything that is not an edge: settings, clock, stimulation, states, counters, prototype.
    pub machine: Json,
    order: Vec<usize>,
    positions: HashMap<usize, usize>,
}

/// The machine's record: what [`Machine::from_json`] needs besides the lattice, the generator left out.
fn machine_record(m: &Machine) -> Json {
    Json::object()
        .with("settings", m.settings_json())
        .with("clock", m.clock.into())
        .with("state", m.state.into())
        .with("runs", (m.runs as f64).into())
        .with("credits", (m.credits as f64).into())
        .with("compressions", (m.compressions as f64).into())
        .with("since_compression", (m.since_compression as f64).into())
        .with("last_compressed", m.last_compressed.into())
        .with("stimulation", m.to_json_stimulation())
        .with(
            "states",
            Json::Array(m.lattice.states.iter().map(|s| s.to_json()).collect()),
        )
        .with("lattice_prototype", m.lattice.prototype.to_json())
}

/// The record's keys a rebuilt machine takes back.
const RECORD_KEYS: [&str; 9] = [
    "settings",
    "clock",
    "state",
    "runs",
    "credits",
    "compressions",
    "since_compression",
    "last_compressed",
    "stimulation",
];

impl Core {
    #[allow(clippy::too_many_arguments)]
    fn assemble(
        shape: (usize, usize, usize),
        alphabet: Vec<String>,
        precision: Precision,
        int_width: usize,
        touched: Vec<bool>,
        ints: Vec<u8>,
        floats: Vec<u8>,
        machine: Json,
    ) -> Core {
        let order = center_out(shape);
        let mut positions = HashMap::new();
        let mut k = 0;
        for (&cell, &hit) in order.iter().zip(touched.iter()) {
            if hit {
                positions.insert(cell, k);
                k += 1;
            }
        }
        Core {
            shape,
            alphabet,
            precision,
            int_width,
            touched,
            ints,
            floats,
            machine,
            order,
            positions,
        }
    }

    pub fn center(&self) -> (usize, usize, usize) {
        center(self.shape)
    }

    pub fn n_touched(&self) -> usize {
        self.positions.len()
    }

    /// What the edges cost: the bitmap, the integers and the numbers. The machine record is besides.
    pub fn bytes(&self) -> usize {
        self.touched.len().div_ceil(8) + self.ints.len() + self.floats.len()
    }

    fn prototype(&self) -> Result<Weighting, String> {
        Weighting::from_json(
            self.machine
                .get("lattice_prototype")
                .ok_or("the core has no prototype")?,
        )
    }

    fn decode(&self, cell: usize, k: usize, prototype: &Weighting) -> Edge {
        let (s_n, a_n, _) = self.shape;
        let (s, rest) = (cell / (a_n * s_n), cell % (a_n * s_n));
        let (a, t) = (rest / s_n, rest % s_n);
        let iw = self.int_width;
        let ib = &self.ints[k * INTS.len() * iw..];
        let iv: Vec<i64> = (0..INTS.len())
            .map(|i| {
                let b = &ib[i * iw..];
                if iw == 4 {
                    i32::from_le_bytes(b[..4].try_into().expect("4 bytes")) as i64
                } else {
                    i64::from_le_bytes(b[..8].try_into().expect("8 bytes"))
                }
            })
            .collect();
        let fb = self.precision.bytes();
        let fbytes = &self.floats[k * FLOATS.len() * fb..];
        let fv: Vec<f64> = (0..FLOATS.len())
            .map(|i| self.precision.unpack(&fbytes[i * fb..]))
            .collect();
        let w = Weighting {
            bias: fv[4],
            seen: fv[5],
            recent: fv[6],
            net: fv[7],
            age: fv[8],
            width: fv[9],
            rate: prototype.rate,
        };
        let mut e = Edge::new(s, a, t, w);
        e.seen = iv[0] as u64;
        e.first_seen = iv[1];
        e.last_seen = iv[2];
        e.last_rewarded = iv[3];
        e.last_punished = iv[4];
        e.width_stamp = iv[5];
        e.recent = fv[0];
        e.rewarded = fv[1];
        e.punished = fv[2];
        e.width = fv[3];
        e.features = [fv[10], fv[11], fv[12], fv[13], fv[14]];
        e
    }

    /// One edge, decoded on its own; an untouched cell is the prototype's edge.
    pub fn edge(&self, s: usize, a: usize, t: usize) -> Result<Edge, String> {
        let (s_n, a_n, _) = self.shape;
        let p = self.prototype()?;
        let cell = (s * a_n + a) * s_n + t;
        Ok(match self.positions.get(&cell) {
            Some(&k) => self.decode(cell, k, &p),
            None => Edge::new(s, a, t, p),
        })
    }

    /// The machine rebuilt from the central node outward; with `shells = Some(k)`, only the first `k` shells (the
    /// central node is shell 0) - every cell beyond them comes back as the prototype's edge.
    pub fn decompress(&self, shells: Option<usize>) -> Result<Machine, String> {
        let p = self.prototype()?;
        let mut lattice = Lattice::new(self.shape.0, &self.alphabet, p.clone())?;
        let limit = match shells {
            Some(k) => shell_sizes(self.shape).iter().take(k).sum(),
            None => self.order.len(),
        };
        let mut k = 0;
        for (i, (&cell, &hit)) in self.order.iter().zip(self.touched.iter()).enumerate() {
            if i >= limit {
                break;
            }
            if hit {
                lattice.edges[cell] = self.decode(cell, k, &p);
                k += 1;
            }
        }
        let states = self.machine.get("states").cloned().unwrap_or(Json::Array(vec![]));
        let mut lat_json = lattice.to_json();
        lat_json.set("state_records", states);
        let mut doc = Json::object()
            .with("format", "latticefsm-machine".into())
            .with("version", 1.0.into());
        for key in RECORD_KEYS {
            if let Some(v) = self.machine.get(key) {
                doc.set(key, v.clone());
            }
        }
        doc.set("lattice", lat_json);
        Machine::from_json(&doc)
    }

    /// Put the code's edges back into `lattice` (of the same shape), every cell: a touched one decoded, an untouched
    /// one the prototype's.  The machine around the lattice is left as it is.
    pub fn rebuild_into(&self, lattice: &mut Lattice) -> Result<(), String> {
        if lattice.shape() != self.shape {
            return Err(format!(
                "the code is {:?}, the matrix {:?}",
                self.shape,
                lattice.shape()
            ));
        }
        let p = self.prototype()?;
        let (s_n, a_n, _) = self.shape;
        for (&cell, &hit) in self.order.iter().zip(self.touched.iter()) {
            lattice.edges[cell] = if hit {
                self.decode(cell, self.positions[&cell], &p)
            } else {
                let (s, rest) = (cell / (a_n * s_n), cell % (a_n * s_n));
                Edge::new(s, rest / s_n, rest % s_n, p.clone())
            };
        }
        Ok(())
    }

    fn setting(&self, key: &str) -> f64 {
        self.machine.get("settings").map(|s| s.num(key, 0.0)).unwrap_or(0.0)
    }

    /// The stimulation now, as the machine had it.
    pub fn stimulation(&self) -> f64 {
        let st = self.machine.get("stimulation").and_then(Json::as_array);
        let (value, stamp) = match st {
            Some(v) if v.len() == 2 => (v[0].as_f64().unwrap_or(1.0), v[1].as_f64().unwrap_or(0.0)),
            _ => (self.setting("baseline"), 0.0),
        };
        let baseline = self.setting("baseline");
        let excess = value - baseline;
        if excess == 0.0 {
            baseline
        } else {
            let clock = self.machine.num("clock", 0.0);
            baseline + excess * 0.5f64.powf((clock - stamp) / self.setting("calm"))
        }
    }

    /// A row's distribution, read straight from the code: only that row's cells are decoded.
    pub fn probabilities(
        &self,
        s: usize,
        a: usize,
        stimulation: Option<f64>,
        temperature: Option<f64>,
    ) -> Result<Vec<f64>, String> {
        let clock = self.machine.num("clock", 0.0) as i64;
        let life = self.setting("life");
        let stim = stimulation.unwrap_or_else(|| self.stimulation());
        let logits: Vec<f64> = (0..self.shape.2)
            .map(|t| self.edge(s, a, t).map(|e| e.log_weight(clock, life, stim)))
            .collect::<Result<_, _>>()?;
        Ok(softmax(
            &logits,
            temperature.unwrap_or_else(|| self.setting("temperature")),
        ))
    }

    /// The greedy walk of `text`, read straight from the code, from the start state or (`from_middle`) the middle
    /// state: the states it passes through, and whether the last accepts.
    pub fn run(&self, text: &str, from_middle: bool) -> Result<(Vec<usize>, bool), String> {
        let mut state = if from_middle {
            self.center().0
        } else {
            self.setting("start") as usize
        };
        let mut states = vec![state];
        for ch in text.chars().filter(|c| !c.is_whitespace()) {
            let a = self
                .alphabet
                .iter()
                .position(|s| *s == ch.to_string())
                .ok_or_else(|| format!("{ch:?} is not in the alphabet"))?;
            state = argmax(&self.probabilities(state, a, None, Some(0.0))?);
            states.push(state);
        }
        let accepting = self
            .machine
            .get("states")
            .and_then(Json::as_array)
            .map(|v| {
                v.iter().any(|s| {
                    s.as_array().is_some_and(|r| {
                        r.len() == 4 && r[0].as_f64() == Some(state as f64) && r[1].as_bool() == Some(true)
                    })
                })
            })
            .unwrap_or(false);
        Ok((states, accepting))
    }

    /// Per shell, from the central node out: cells, touched edges, and the bytes the code spends there.
    pub fn shell_table(&self) -> Json {
        let mut i = 0;
        let per_edge = INTS.len() * self.int_width + FLOATS.len() * self.precision.bytes();
        Json::Array(
            shell_sizes(self.shape)
                .into_iter()
                .enumerate()
                .map(|(k, n)| {
                    let hits = self.touched[i..i + n].iter().filter(|&&b| b).count();
                    i += n;
                    Json::object()
                        .with("shell", k.into())
                        .with("cells", n.into())
                        .with("touched", hits.into())
                        .with("bytes", (n.div_ceil(8) + hits * per_edge).into())
                })
                .collect(),
        )
    }

    pub fn summary(&self) -> Json {
        let (s, a, t) = self.shape;
        let cells = s * a * t;
        let dense = cells * (INTS.len() * 8 + FLOATS.len() * 8);
        let c = self.center();
        Json::object()
            .with("shape", Json::numbers(&[s as f64, a as f64, t as f64]))
            .with("center", Json::numbers(&[c.0 as f64, c.1 as f64, c.2 as f64]))
            .with(
                "center_label",
                Json::Array(vec![c.0.into(), self.alphabet[c.1].as_str().into(), c.2.into()]),
            )
            .with("shells", shells(self.shape).into())
            .with("precision", self.precision.name().into())
            .with("touched", self.n_touched().into())
            .with("cells", cells.into())
            .with("bytes", self.bytes().into())
            .with("dense_bytes", dense.into())
            .with("ratio", (dense as f64 / self.bytes().max(1) as f64).into())
            .with("record_bytes", self.machine.dump().len().into())
            .with("shell_table", self.shell_table())
    }

    pub fn to_json(&self) -> Json {
        let c = self.center();
        Json::object()
            .with("format", FORMAT.into())
            .with("version", 1.0.into())
            .with(
                "shape",
                Json::numbers(&[self.shape.0 as f64, self.shape.1 as f64, self.shape.2 as f64]),
            )
            .with("alphabet", Json::strings(&self.alphabet))
            .with("center", Json::numbers(&[c.0 as f64, c.1 as f64, c.2 as f64]))
            .with("order", "center-out".into())
            .with("ints_fields", Json::Array(INTS.iter().map(|&s| s.into()).collect()))
            .with("floats_fields", Json::Array(FLOATS.iter().map(|&s| s.into()).collect()))
            .with("precision", self.precision.name().into())
            .with("int_width", self.int_width.into())
            .with("touched", bits_to_hex(&self.touched).into())
            .with("ints", base64_encode(&self.ints).into())
            .with("floats", base64_encode(&self.floats).into())
            .with("machine", self.machine.clone())
    }

    pub fn from_json(v: &Json) -> Result<Core, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let names = |key: &str| -> Vec<String> {
            v.get(key)
                .and_then(Json::as_array)
                .map(|a| a.iter().filter_map(|s| s.as_str().map(str::to_string)).collect())
                .unwrap_or_default()
        };
        if names("ints_fields") != INTS || names("floats_fields") != FLOATS {
            return Err("the core's fields are not this version's".to_string());
        }
        let shape = v.get("shape").and_then(Json::as_array).ok_or("core.shape")?;
        if shape.len() != 3 {
            return Err("core.shape is three numbers".to_string());
        }
        let n = |i: usize| {
            shape[i]
                .as_f64()
                .map(|x| x as usize)
                .ok_or("core.shape is three numbers")
        };
        let shape = (n(0)?, n(1)?, n(2)?);
        let alphabet = names("alphabet");
        if alphabet.len() != shape.1 {
            return Err("the core's alphabet does not match its shape".to_string());
        }
        let precision = Precision::parse(v.str_or("precision", ""))?;
        let int_width = v.num("int_width", 8.0) as usize;
        if int_width != 4 && int_width != 8 {
            return Err("int_width is 4 or 8".to_string());
        }
        let touched = hex_to_bits(v.str_or("touched", ""), shape.0 * shape.1 * shape.2)?;
        let ints = base64_decode(v.str_or("ints", ""))?;
        let floats = base64_decode(v.str_or("floats", ""))?;
        let hits = touched.iter().filter(|&&b| b).count();
        if ints.len() != hits * INTS.len() * int_width || floats.len() != hits * FLOATS.len() * precision.bytes() {
            return Err("the core's packed fields do not match its bitmap".to_string());
        }
        let machine = v.get("machine").cloned().ok_or("core.machine")?;
        Ok(Core::assemble(
            shape, alphabet, precision, int_width, touched, ints, floats, machine,
        ))
    }

    /// Write the code to `path` (`.json`, or `.json.gz` to gzip it).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        fs::File::create(path)
            .and_then(|mut f| f.write_all(&bytes))
            .map_err(|e| format!("{path}: {e}"))
    }
}

/// Read a code from `path` (`.json` or `.json.gz`).
pub fn load_core(path: &str) -> Result<Core, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    Core::from_json(&parse(&text)?)
}

fn encode(machine: &Machine, precision: Precision) -> Core {
    let lat = &machine.lattice;
    let order = center_out(lat.shape());
    let touched: Vec<bool> = order.iter().map(|&c| lat.edges[c].touched()).collect();
    let edges: Vec<&Edge> = order
        .iter()
        .zip(touched.iter())
        .filter(|(_, &h)| h)
        .map(|(&c, _)| &lat.edges[c])
        .collect();
    let all_ints: Vec<i64> = edges.iter().flat_map(|e| edge_ints(e)).collect();
    let int_width = if all_ints.iter().all(|&x| x >= i32::MIN as i64 && x <= i32::MAX as i64) {
        4
    } else {
        8
    };
    let mut ints = Vec::with_capacity(all_ints.len() * int_width);
    for x in all_ints {
        if int_width == 4 {
            ints.extend_from_slice(&(x as i32).to_le_bytes());
        } else {
            ints.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut floats = Vec::with_capacity(edges.len() * FLOATS.len() * precision.bytes());
    for e in &edges {
        for x in edge_floats(e) {
            precision.pack(x, &mut floats);
        }
    }
    Core::assemble(
        lat.shape(),
        lat.symbols.clone(),
        precision,
        int_width,
        touched,
        ints,
        floats,
        machine_record(machine),
    )
}

/// The matrix, folded into its central node with the least loss. `precision` packs the numbers (default exact:
/// no loss at all); with a `budget` in bytes and no precision, the least lossy precision whose code fits, else
/// float16.
pub fn compress(machine: &Machine, precision: Option<Precision>, budget: Option<usize>) -> Core {
    if let Some(p) = precision {
        return encode(machine, p);
    }
    let Some(budget) = budget else {
        return encode(machine, Precision::Exact);
    };
    let mut core = encode(machine, Precision::Exact);
    for p in Precision::ALL {
        core = encode(machine, p);
        if core.bytes() <= budget {
            break;
        }
    }
    core
}

/// How far `restored` is from `original`: in behaviour, per row, the KL divergence of its distribution (at the
/// original's stimulation; temperature 1 when the original's is 0) and whether the greedy choice is still one of the
/// original's tied best; in state, the largest relative error over every number of every edge, and how many edges
/// differ at all.
pub fn fidelity(original: &Machine, restored: &Machine) -> Json {
    let (s_n, a_n) = (original.n_states(), original.alphabet().len());
    let stim = original.stimulation();
    let temp = if original.temperature > 0.0 {
        original.temperature
    } else {
        1.0
    };
    let (mut kls, mut agree) = (Vec::with_capacity(s_n * a_n), 0usize);
    for s in 0..s_n {
        for a in 0..a_n {
            let lo = original.log_weights(s, a, Some(stim));
            let lr = restored.log_weights(s, a, Some(stim));
            let (p, q) = (softmax(&lo, temp), softmax(&lr, temp));
            let kl: f64 = p
                .iter()
                .zip(q.iter())
                .filter(|(&pi, _)| pi > 0.0)
                .map(|(&pi, &qi)| pi * (pi.ln() - qi.max(1e-300).ln()))
                .sum();
            kls.push(kl.max(0.0));
            let best = lo.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            if lo[argmax(&lr)] >= best - TIE * (1.0 + best.abs()) {
                agree += 1;
            }
        }
    }
    let rows = s_n * a_n;
    let (mut worst, mut differ) = (0.0f64, 0usize);
    for (eo, er) in original.lattice.edges.iter().zip(restored.lattice.edges.iter()) {
        let (fo, fr) = (edge_floats(eo), edge_floats(er));
        if edge_ints(eo) != edge_ints(er) || fo != fr {
            differ += 1;
        }
        for (x, y) in fo.iter().zip(fr.iter()) {
            if x != y {
                worst = worst.max((x - y).abs() / x.abs().max(1e-12));
            }
        }
    }
    let mean_kl = kls.iter().sum::<f64>() / rows as f64;
    let max_kl = kls.iter().cloned().fold(0.0, f64::max);
    Json::object()
        .with("rows", rows.into())
        .with("mean_kl", mean_kl.into())
        .with("max_kl", max_kl.into())
        .with("greedy_agreement", (agree as f64 / rows as f64).into())
        .with("greedy_changed", (rows - agree).into())
        .with("preserved", (agree == rows && max_kl <= MAX_KL).into())
        .with("edges_differing", differ.into())
        .with("max_relative_error", worst.into())
        .with("lossless", (differ == 0).into())
}

/// Rebuild `core` one more shell at a time from the central node outward, and measure each step against `original`.
pub fn expansion(original: &Machine, core: &Core) -> Result<Json, String> {
    let sizes = shell_sizes(core.shape);
    let table = core.shell_table();
    let rows = table.as_array().cloned().unwrap_or_default();
    let mut out = Vec::new();
    for k in 1..=sizes.len() {
        let part = core.decompress(Some(k))?;
        let f = fidelity(original, &part);
        out.push(
            Json::object()
                .with("shells", k.into())
                .with("cells", sizes[..k].iter().sum::<usize>().into())
                .with(
                    "touched",
                    rows[..k]
                        .iter()
                        .map(|r| r.num("touched", 0.0) as usize)
                        .sum::<usize>()
                        .into(),
                )
                .with("greedy_changed", f.get("greedy_changed").cloned().unwrap_or(Json::Null))
                .with("max_kl", f.get("max_kl").cloned().unwrap_or(Json::Null)),
        );
    }
    Ok(Json::Array(out))
}
