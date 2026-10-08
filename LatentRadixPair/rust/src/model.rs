//! A trained tokenizer with a primed radix pair over its codes: reading, judging, predicting, files.

use crate::files;
use crate::nn::Rng;
use crate::pair::{self, Pair, PathResult, Score, Settings};
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::time::{SystemTime, UNIX_EPOCH};

pub const FORMAT: &str = "latentpair";

pub struct Model {
    pub tok: Tokenizer,
    pub pair: Pair,
    pub seed: i64,
    pub created: String,
    pub history: Vec<Value>,
    rng: Rng,
    draws: usize,
}

impl Model {
    pub fn prime(tok: Tokenizer, settings: Settings, seed: i64) -> Result<Model, String> {
        let pair = Pair::new(&tok.radices(), settings)?;
        Ok(Model { tok, pair, seed, created: now_rfc3339(), history: Vec::new(), rng: Rng::new(seed as u64), draws: 0 })
    }

    fn record(&mut self, r: Value) -> Value {
        self.history.push(r.clone());
        r
    }

    /// Reads texts into the count tree.
    pub fn train_bytes(&mut self, texts: &[Vec<u8>]) -> Value {
        let mut units = 0;
        for text in texts {
            let codes = self.tok.encode_all(text);
            self.pair.count.observe(&codes, &pair::outcomes(text));
            units += text.len() + 1;
        }
        self.record(json!({"call": "train", "texts": texts.len(), "units": units}))
    }
    pub fn train(&mut self, texts: &[String]) -> Value {
        let b: Vec<Vec<u8>> = texts.iter().map(|t| t.as_bytes().to_vec()).collect();
        self.train_bytes(&b)
    }

    /// Judges texts: amount = sign * strength * weight / outcomes each, from the position after the prefix.
    #[allow(clippy::too_many_arguments)]
    fn credit(&mut self, texts: &[Vec<u8>], sign: f64, strength: f64, weights: Option<&[f64]>, read: bool, call: &str, prefix: &[u8], outcomes: usize) -> Result<Value, String> {
        let strength = if strength <= 0.0 { self.pair.settings.strength } else { strength };
        let outcomes = if outcomes == 0 { self.pair.settings.outcomes } else { outcomes };
        let unit = 1.0 / outcomes as f64;
        if let Some(w) = weights {
            if w.len() != texts.len() {
                return Err(format!("{} weights for {} texts", w.len(), texts.len()));
            }
        }
        let mut cells = 0;
        for (i, text) in texts.iter().enumerate() {
            let w = weights.map_or(1.0, |w| w[i]);
            if w < 0.0 {
                return Err(format!("weights must be >= 0, got {w}"));
            }
            let mut full = prefix.to_vec();
            full.extend_from_slice(text);
            let codes = self.tok.encode_all(&full);
            let steps = pair::outcomes(&full);
            if read {
                self.pair.count.observe(&codes, &steps);
            }
            cells += self.pair.reward.credit(&codes, &steps, sign * strength * w * unit, &self.pair.settings.rungs.clone(), prefix.len())?;
        }
        let mut r = json!({"call": call, "texts": texts.len(), "strength": strength, "outcomes": outcomes, "amount": strength * unit, "cells": cells, "read": read});
        if !prefix.is_empty() {
            r["prefix"] = json!(String::from_utf8_lossy(prefix));
        }
        if call.is_empty() {
            return Ok(r);
        }
        Ok(self.record(r))
    }

    pub fn reward(&mut self, texts: &[String], strength: f64, weights: Option<&[f64]>, read: bool, prefix: &str, outcomes: usize) -> Result<Value, String> {
        self.credit(&as_bytes(texts), 1.0, strength, weights, read, "reward", prefix.as_bytes(), outcomes)
    }
    pub fn punish(&mut self, texts: &[String], strength: f64, weights: Option<&[f64]>, prefix: &str, outcomes: usize) -> Result<Value, String> {
        self.credit(&as_bytes(texts), -1.0, strength, weights, false, "punish", prefix.as_bytes(), outcomes)
    }
    pub fn two_nrl(&mut self, bad: &[String], good: &[String], strength: f64, prefix: &str, outcomes: usize) -> Result<Value, String> {
        let a = self.credit(&as_bytes(bad), -1.0, strength, None, false, "", prefix.as_bytes(), outcomes)?;
        let b = self.credit(&as_bytes(good), 1.0, strength, None, true, "", prefix.as_bytes(), outcomes)?;
        let mut r = json!({"call": "two_nrl", "punished": a["cells"], "rewarded": b["cells"], "bad": bad.len(), "good": good.len(),
            "strength": a["strength"], "outcomes": a["outcomes"], "amount": a["amount"]});
        if !prefix.is_empty() {
            r["prefix"] = json!(prefix);
        }
        Ok(self.record(r))
    }
    /// A positive mark rewards with that weight, a negative one punishes with its size, zero is ignored.
    pub fn feedback(&mut self, texts: &[String], marks: &[f64], prefix: &str, outcomes: usize) -> Result<Value, String> {
        if marks.len() != texts.len() {
            return Err(format!("{} marks for {} texts", marks.len(), texts.len()));
        }
        let (mut good, mut bad, mut gw, mut bw) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (i, &mk) in marks.iter().enumerate() {
            if mk > 0.0 {
                good.push(texts[i].clone());
                gw.push(mk);
            } else if mk < 0.0 {
                bad.push(texts[i].clone());
                bw.push(-mk);
            }
        }
        let outcomes = if outcomes == 0 { self.pair.settings.outcomes } else { outcomes };
        if !good.is_empty() {
            self.credit(&as_bytes(&good), 1.0, 0.0, Some(&gw), true, "", prefix.as_bytes(), outcomes)?;
        }
        if !bad.is_empty() {
            self.credit(&as_bytes(&bad), -1.0, 0.0, Some(&bw), false, "", prefix.as_bytes(), outcomes)?;
        }
        let mut r = json!({"call": "feedback", "rewarded": good.len(), "punished": bad.len(), "ignored": texts.len() - good.len() - bad.len(), "outcomes": outcomes});
        if !prefix.is_empty() {
            r["prefix"] = json!(prefix);
        }
        Ok(self.record(r))
    }

    pub fn code(&self, prefix: &str) -> Vec<i32> {
        self.tok.encode(prefix.as_bytes())
    }
    pub fn fold(&mut self, prefix: &str, traversal: &str, backoff: &str) -> Result<Vec<f64>, String> {
        let traversal = if traversal.is_empty() { "reward" } else { traversal };
        let code = self.code(prefix);
        self.pair.fold(&code, traversal, backoff)
    }

    /// Continues a prefix: the fold's most probable outcome each step ("greedy") or a sample ("sample").
    #[allow(clippy::too_many_arguments)]
    pub fn predict(&mut self, prefix: &str, length: usize, mode: &str, traversal: &str, temperature: f64, to_end: bool, backoff: &str, max_units: usize) -> Result<PathResult, String> {
        let traversal = if traversal.is_empty() { "reward" } else { traversal };
        let tok = self.tok.clone();
        let encode = move |ctx: &[u8]| tok.encode(ctx);
        match mode {
            "" | "greedy" => self.pair.walk(prefix.as_bytes(), &encode, length, traversal, None, 0.0, to_end, backoff, max_units),
            "sample" => {
                let mut rng = self.rng.clone();
                let res = self.pair.walk(prefix.as_bytes(), &encode, length, traversal, Some(&mut rng), temperature, to_end, backoff, max_units)?;
                self.draws += res.units.len(); // one draw per step, so a saved model replays its random state
                self.rng = rng;
                Ok(res)
            }
            _ => Err(format!("mode must be greedy or sample, got {mode:?}")),
        }
    }

    pub fn score(&mut self, text: &str, traversal: &str, backoff: &str) -> Result<Score, String> {
        let traversal = if traversal.is_empty() { "reward" } else { traversal };
        let b = text.as_bytes();
        let codes = self.tok.encode_all(b);
        self.pair.score_text(&codes, &pair::outcomes(b), traversal, backoff)
    }
    pub fn decode(&self, prefix: &str, known: usize) -> Result<String, String> {
        self.tok.decode(&self.code(prefix), known)
    }
    pub fn coverage(&mut self, text: &str) -> f64 {
        let codes = self.tok.encode_all(text.as_bytes());
        self.pair.count.coverage(&codes)
    }

    pub fn info(&self) -> Value {
        let (mut nz, mut nr) = (0, 0);
        self.pair.count.nonzero(|_, _| nz += 1);
        self.pair.reward.nonzero(|_, _, _| nr += 1);
        let used = self.pair.count.ctx.iter().filter(|&&c| c > 0).count();
        json!({
            "format": FORMAT, "seed": self.seed, "created": self.created, "tokenizer": self.tok.info(),
            "radices": self.pair.addr.radices, "depth": self.pair.addr.d(), "nodes": self.pair.addr.n,
            "cells": self.pair.addr.cells(), "memory_bytes": self.pair.memory_bytes(),
            "read": {"texts": self.pair.count.texts, "units": self.pair.count.units, "contexts_used": used},
            "judged": {"texts": self.pair.reward.judged, "rewards_total": self.pair.reward.rewards_total, "penalties_total": self.pair.reward.penalties_total},
            "nonzero_counts": nz, "nonzero_rewards": nr, "settings": self.pair.settings, "history": self.history.len(),
        })
    }

    pub fn to_json(&self) -> Result<Vec<u8>, String> {
        let tok: Value = serde_json::from_slice(&self.tok.to_json()?).map_err(|e| e.to_string())?;
        let mut f = FileJson {
            format: FORMAT.into(),
            version: 1,
            tokenizer: tok,
            seed: self.seed,
            draws: self.draws,
            settings: self.pair.settings.clone(),
            created: self.created.clone(),
            read: Read { texts: self.pair.count.texts, units: self.pair.count.units },
            judged: Judged { texts: self.pair.reward.judged, rewards_total: self.pair.reward.rewards_total, penalties_total: self.pair.reward.penalties_total },
            history: self.history.clone(),
            counts: Counts::default(),
            rewards: Rewards::default(),
        };
        self.pair.count.nonzero(|cell, c| {
            f.counts.cells.push(cell);
            f.counts.values.push(c);
        });
        self.pair.reward.nonzero(|cell, plus, minus| {
            f.rewards.cells.push(cell);
            f.rewards.plus.push(plus);
            f.rewards.minus.push(minus);
        });
        serde_json::to_vec(&f).map_err(|e| e.to_string())
    }

    pub fn from_json(data: &[u8]) -> Result<Model, String> {
        let mut f: FileJson = serde_json::from_slice(data).map_err(|e| e.to_string())?;
        if f.format != FORMAT {
            return Err(format!("not a {FORMAT} file (format {:?})", f.format));
        }
        let tok = Tokenizer::from_json(&serde_json::to_vec(&f.tokenizer).map_err(|e| e.to_string())?).map_err(|e| format!("tokenizer: {e}"))?;
        if f.settings.outcomes == 0 {
            f.settings.outcomes = pair::ENGLISH_PHONES; // files written before verdicts had units
        }
        let mut m = Model::prime(tok, f.settings, f.seed)?;
        m.created = f.created;
        m.history = f.history;
        for _ in 0..f.draws {
            m.rng.next_u64();
        }
        m.draws = f.draws;
        if f.counts.cells.len() != f.counts.values.len() || f.rewards.cells.len() != f.rewards.plus.len() || f.rewards.cells.len() != f.rewards.minus.len() {
            return Err("cell lists of unequal length".into());
        }
        let out = m.pair.addr.out;
        for (i, &cell) in f.counts.cells.iter().enumerate() {
            if cell >= m.pair.count.cnt.len() {
                return Err(format!("count cell {cell} outside the address space"));
            }
            m.pair.count.cnt[cell] = f.counts.values[i];
            m.pair.count.ctx[cell / out] += f.counts.values[i];
        }
        for (i, &cell) in f.rewards.cells.iter().enumerate() {
            if cell >= m.pair.reward.plus.len() {
                return Err(format!("reward cell {cell} outside the address space"));
            }
            m.pair.reward.plus[cell] = f.rewards.plus[i];
            m.pair.reward.minus[cell] = f.rewards.minus[i];
        }
        m.pair.count.texts = f.read.texts;
        m.pair.count.units = f.read.units;
        m.pair.reward.judged = f.judged.texts;
        m.pair.reward.rewards_total = f.judged.rewards_total;
        m.pair.reward.penalties_total = f.judged.penalties_total;
        Ok(m)
    }

    pub fn save(&self, path: &str) -> Result<(), String> {
        files::write_file(path, &self.to_json()?).map_err(|e| e.to_string())
    }
    pub fn load(path: &str) -> Result<Model, String> {
        let data = files::read_file(path).map_err(|e| format!("{path}: {e}"))?;
        Model::from_json(&data)
    }
}

fn as_bytes(texts: &[String]) -> Vec<Vec<u8>> {
    texts.iter().map(|t| t.as_bytes().to_vec()).collect()
}

#[derive(Serialize, Deserialize, Default)]
struct Read {
    texts: usize,
    units: usize,
}
#[derive(Serialize, Deserialize, Default)]
struct Judged {
    texts: usize,
    rewards_total: f64,
    penalties_total: f64,
}
#[derive(Serialize, Deserialize, Default)]
struct Counts {
    cells: Vec<usize>,
    values: Vec<i64>,
}
#[derive(Serialize, Deserialize, Default)]
struct Rewards {
    cells: Vec<usize>,
    plus: Vec<f64>,
    minus: Vec<f64>,
}
#[derive(Serialize, Deserialize)]
struct FileJson {
    format: String,
    version: u32,
    tokenizer: Value,
    seed: i64,
    #[serde(default)]
    draws: usize,
    settings: Settings,
    #[serde(default)]
    created: String,
    #[serde(default)]
    read: Read,
    #[serde(default)]
    judged: Judged,
    #[serde(default)]
    history: Vec<Value>,
    #[serde(default)]
    counts: Counts,
    #[serde(default)]
    rewards: Rewards,
}

/// The current time as RFC 3339 in UTC, without a calendar crate.
pub fn now_rfc3339() -> String {
    let secs = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    // civil-from-days (Howard Hinnant)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", rem / 3600, (rem % 3600) / 60, rem % 60)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{Config, Options, Tokenizer};

    const TEXTS: [&str; 8] = ["the cat sat on the mat", "the cat sat on the log", "the dog ate the bone", "a cat and a dog",
        "the bird sang on the wire", "a dog and a cat sat", "the mat was on the floor", "the log was on the fire"];

    fn tiny_model() -> Model {
        let cfg = Config { window: 8, embed: 8, enc_hidden: 32, dec_hidden: 32, out_embed: 8, levels: vec![vec![4, 4], vec![4], vec![2, 2]],
            noise: 0.02, recency: 0.6, predict: 1.0, start_share: 0.1 };
        let mut tok = Tokenizer::new(cfg, 4).unwrap();
        let corpus: Vec<Vec<u8>> = TEXTS.iter().map(|t| t.as_bytes().to_vec()).collect();
        tok.train(&corpus, &Options { steps: 200, batch: 64, eval_every: 100, eval_windows: 64, lr: 5e-3, ..Options::default() }, |_| {}).unwrap();
        let mut m = Model::prime(tok, Settings::default(), 1).unwrap();
        m.train(&TEXTS.iter().map(|t| t.to_string()).collect::<Vec<_>>());
        m
    }

    #[test]
    fn verdicts_are_in_outcome_units() {
        let mut m = tiny_model();
        assert_eq!(m.pair.settings.outcomes, pair::ENGLISH_PHONES);
        let cell = |m: &Model, prefix: &str, x: u8| {
            let code = m.code(prefix);
            m.pair.addr.cell(m.pair.addr.of(&code, m.pair.addr.d()), x as usize)
        };
        let r = m.reward(&["x".into()], 1.0, None, false, "game ", 2).unwrap();
        assert!((m.pair.reward.plus[cell(&m, "game ", b'x')] - 0.5).abs() < 1e-12);
        assert_eq!(r["outcomes"], 2);
        m.reward(&["y".into()], 1.0, None, false, "english ", 0).unwrap();
        assert!((m.pair.reward.plus[cell(&m, "english ", b'y')] - 1.0 / 69.0).abs() < 1e-12);
        m.punish(&["z".into()], 3.0, None, "raw ", 1).unwrap();
        assert!((m.pair.reward.minus[cell(&m, "raw ", b'z')] - 3.0).abs() < 1e-12);
        let r = m.two_nrl(&["b".into()], &["g".into()], 1.0, "p ", 4).unwrap();
        assert_eq!(r["outcomes"], 4);
        assert!((m.pair.reward.minus[cell(&m, "p ", b'b')] - 0.25).abs() < 1e-12);
        let r = m.feedback(&["f".into(), "ignored".into()], &[0.5, 0.0], "q ", 10).unwrap();
        assert_eq!(r["ignored"], 1);
        assert!((m.pair.reward.plus[cell(&m, "q ", b'f')] - 0.05).abs() < 1e-12);
        assert!(m.feedback(&["x".into()], &[1.0, 2.0], "", 0).is_err());
        assert_eq!(m.history.len(), 6);
        // a reward lifts, a punishment sinks, and a punishment is not bought back
        let before = m.fold("a cat sat on the ", "reward", "").unwrap()[b'm' as usize];
        m.reward(&["mat".into()], 5.0, None, true, "a cat sat on the ", 2).unwrap();
        let after = m.fold("a cat sat on the ", "reward", "").unwrap()[b'm' as usize];
        assert!(after > before);
        let pen = m.fold("a cat sat on the ", "punishment", "").unwrap()[b'm' as usize];
        m.punish(&["mat".into()], 20.0, None, "a cat sat on the ", 2).unwrap();
        let pen2 = m.fold("a cat sat on the ", "punishment", "").unwrap()[b'm' as usize];
        assert!(pen2 < pen);
        m.reward(&["mat".into()], 100.0, None, false, "a cat sat on the ", 2).unwrap();
        let pen3 = m.fold("a cat sat on the ", "punishment", "").unwrap()[b'm' as usize];
        assert!(pen3 <= pen2 + 1e-12);
    }

    #[test]
    fn walks_and_files() {
        let mut m = tiny_model();
        for _ in 0..10 {
            m.train(&["the cat sat on the mat".to_string()]);
        }
        let a = m.predict("the cat sat on the ", 6, "greedy", "reward", 0.0, false, "", 0).unwrap();
        let b = m.predict("the cat sat on the ", 6, "greedy", "reward", 0.0, false, "", 0).unwrap();
        assert_eq!(a.text, b.text);
        assert!(a.units.len() <= 6 && a.step_costs.len() == a.units.len() && a.peak > 0.0);
        let e = m.predict("the cat sat on the mat", 0, "greedy", "reward", 0.0, true, "", 50).unwrap();
        assert!(e.reached_end && e.text.is_empty(), "{e:?}");
        let c = m.predict("the cat sat on the ma", 1, "sample", "reward", 0.01, false, "", 0).unwrap();
        assert_eq!(c.text, "t");
        assert!(m.predict("x", 1, "dijkstra", "reward", 0.0, false, "", 0).is_err());
        let read = m.score("the cat sat on the mat", "reward", "").unwrap();
        let unread = m.score("zq xk vj qq", "reward", "").unwrap();
        assert!(read.bits < unread.bits && read.units == 22 && read.per_unit[22].unit == "</s>");
        assert!(m.coverage("the cat sat on the mat") > m.coverage("zq xk vj qq"));
        let s1 = m.predict("the ", 4, "sample", "reward", 1.0, false, "", 0).unwrap();
        let dir = std::env::temp_dir().join(format!("latentpair-model-{}", std::process::id()));
        let path = dir.join("model.json.gz");
        m.save(path.to_str().unwrap()).unwrap();
        let mut back = Model::load(path.to_str().unwrap()).unwrap();
        assert_eq!(back.pair.count.cnt, m.pair.count.cnt);
        assert_eq!(back.pair.count.ctx, m.pair.count.ctx);
        assert_eq!(back.pair.reward.plus, m.pair.reward.plus);
        assert_eq!(back.history.len(), m.history.len());
        let s2 = m.predict("the ", 4, "sample", "reward", 1.0, false, "", 0).unwrap();
        let s3 = back.predict("the ", 4, "sample", "reward", 1.0, false, "", 0).unwrap();
        assert_eq!(s2.text, s3.text, "random state not restored (first draw {:?})", s1.text);
        assert!(now_rfc3339().ends_with('Z') && now_rfc3339().len() == 20);
        std::fs::remove_dir_all(dir).ok();
    }
}
