//! How a training run walks its texts: the order, the curriculum, replay and
//! early stopping (`../../SPEC-SearchAndTraining.md` sections 3-6).
//!
//! The Rust twin of `radixnet/training.py` and `go/radixnet/training.go`,
//! kept number for number: the shuffle and the replay buffer order their
//! texts by SplitMix64 keys of the model's seed, so no random number is drawn
//! and the model's own generator stays where it was; every rounding and every
//! tie is the one the spec names.
//!
//! A [`TrainingPlan`] is built once per `train` call.  It says, epoch by
//! epoch, which of the run's texts to walk and in what order, which texts of
//! the model's [`ReplayBuffer`] to rehearse after them, and whether the run
//! should stop.  Everything is off by default, and a plan that is off walks
//! every text in corpus order every epoch - the pass the model always had.

use crate::json::{py_repr, Json};

/// The orders a run can walk its texts in; `"corpus"` is the one it always had.
pub const ORDERS: [&str; 4] = ["corpus", "shortest-first", "longest-first", "shuffle"];

const SHUFFLE_SALT: u64 = 0x0053_4855_4646_4C45; // "SHUFFLE"
const REPLAY_SALT: u64 = 0x0000_5245_504C_4159; // "REPLAY"

/// The SplitMix64 finaliser (wrapping arithmetic).
pub fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Where text `i` goes in the shuffled order of epoch number `epoch`
/// (smallest first).  A negative seed is taken in two's complement.
pub fn shuffle_key(seed: i64, epoch: i64, i: usize) -> u64 {
    splitmix64(splitmix64(splitmix64(seed as u64 ^ SHUFFLE_SALT) ^ epoch as u64) ^ i as u64)
}

/// The priority of the `g`-th text ever offered to a replay buffer (the
/// smallest are kept).
pub fn replay_key(seed: i64, g: i64) -> u64 {
    splitmix64(splitmix64(seed as u64 ^ REPLAY_SALT) ^ g as u64)
}

/// The settings of how a training run walks its texts; the default is off
/// in every one of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    /// One of [`ORDERS`].
    pub order: String,
    /// The share of the ordered texts the first epoch walks, growing to all
    /// of them by the last (1 = off).
    pub curriculum: f64,
    /// How many texts of the replay buffer each epoch rehearses, as a share
    /// of the run's texts (0 = off).
    pub replay: f64,
    /// The replay buffer's capacity from this run on (`None` leaves the
    /// model's alone, 0 drops it).
    pub replay_size: Option<usize>,
    /// Stop after this many full epochs without the loss improving by
    /// `min_delta` (0 = off).
    pub patience: usize,
    pub min_delta: f64,
    /// Read every text backwards, in the encoding's units ([`read`], the
    /// spec's section 9), before anything else sees it - so the order, the
    /// buffer and the counters all see the reversed texts.
    pub reverse: bool,
}

impl Default for Plan {
    fn default() -> Plan {
        Plan {
            order: "corpus".to_string(),
            curriculum: 1.0,
            replay: 0.0,
            replay_size: None,
            patience: 0,
            min_delta: 0.0,
            reverse: false,
        }
    }
}

/// The texts of one training call as `plan` reads them: each backwards, unit
/// by unit, with `reverse` ([`crate::encoding::Encoding::reverse`]), and
/// borrowed as they are without.  Every kind reads its texts through this
/// first, exactly once per run.
pub fn read<'a>(enc: &crate::encoding::Encoding, texts: &'a [String], plan: &Plan) -> std::borrow::Cow<'a, [String]> {
    if plan.reverse {
        std::borrow::Cow::Owned(texts.iter().map(|t| enc.reverse(t)).collect())
    } else {
        std::borrow::Cow::Borrowed(texts)
    }
}

impl Plan {
    /// Whether the run needs the whole list before the first epoch - an
    /// order, a curriculum, a rehearsal or a buffer to offer the texts to -
    /// and so reads a streaming source into memory.
    pub fn needs_texts(&self) -> bool {
        self.order != "corpus" || self.curriculum < 1.0 || self.replay > 0.0 || self.replay_size.is_some()
    }

    /// Whether anything about the run differs from the plain pass.
    pub fn active(&self) -> bool {
        self.needs_texts() || self.patience > 0
    }

    /// The error for a setting out of range (the spec's section 7).
    pub fn check(&self) -> Result<(), String> {
        if !ORDERS.contains(&self.order.as_str()) {
            return Err(format!(
                "unknown order {}; expected one of: {}",
                crate::negative::python_repr(&self.order),
                ORDERS.join(", ")
            ));
        }
        if !(self.curriculum > 0.0 && self.curriculum <= 1.0) {
            return Err(format!(
                "curriculum must lie in (0, 1], got {}",
                py_repr(self.curriculum)
            ));
        }
        if !self.replay.is_finite() || self.replay < 0.0 {
            return Err(format!(
                "replay must be a finite number >= 0, got {}",
                py_repr(self.replay)
            ));
        }
        if !self.min_delta.is_finite() || self.min_delta < 0.0 {
            return Err(format!(
                "min_delta must be a finite number >= 0, got {}",
                py_repr(self.min_delta)
            ));
        }
        Ok(())
    }

    /// The seven settings as a training config writes them, after its own -
    /// Python's `TrainConfig` field order.
    pub fn json_pairs(&self) -> Vec<(String, Json)> {
        vec![
            ("order".to_string(), Json::str(self.order.clone())),
            ("curriculum".to_string(), Json::Num(self.curriculum)),
            ("replay".to_string(), Json::Num(self.replay)),
            (
                "replay_size".to_string(),
                self.replay_size.map(|n| Json::Int(n as i64)).unwrap_or(Json::Null),
            ),
            ("patience".to_string(), Json::Int(self.patience as i64)),
            ("min_delta".to_string(), Json::Num(self.min_delta)),
            ("reverse".to_string(), Json::Bool(self.reverse)),
        ]
    }
}

/// How many of the list epoch `j` (from 0) of `epochs` walks: the first
/// `curriculum` of it, growing to all of it by the last epoch.
pub fn paced(n: usize, curriculum: f64, j: usize, epochs: usize) -> usize {
    if n == 0 {
        return 0;
    }
    if curriculum >= 1.0 || epochs <= 1 {
        return n;
    }
    let frac = curriculum + ((1.0 - curriculum) * j as f64) / (epochs - 1) as f64;
    let m = (frac * n as f64).ceil();
    if m >= n as f64 {
        n
    } else if m < 1.0 {
        1
    } else {
        m as usize
    }
}

/// How many buffered texts an epoch rehearses: `replay` of the run's `n`
/// texts, rounded half up, at most the pool.
pub fn replay_count(replay: f64, n: usize, pool: usize) -> usize {
    if replay <= 0.0 || pool == 0 {
        return 0;
    }
    let r = (replay * n as f64 + 0.5).floor();
    if r >= pool as f64 {
        pool
    } else {
        r as usize
    }
}

/// A bounded, uniform sample of every text a model was ever trained on.
///
/// The `g`-th text ever offered gets the priority [`replay_key`] of the
/// model's seed and `g`; the buffer keeps the `size` entries with the smallest
/// `(priority, g)`.  That is bottom-k sampling: whatever order the corpus came
/// in, every text ever offered is equally likely to be here, and no random
/// number is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct ReplayBuffer {
    pub seed: i64,
    pub size: usize,
    /// How many texts were ever offered.
    pub seen: i64,
    /// `(priority, g, text)`, smallest first.
    items: Vec<(u64, i64, String)>,
}

impl ReplayBuffer {
    /// A buffer holding `entries` (`(g, text)`), priorities recomputed from
    /// `seed` and only the `size` smallest kept.
    pub fn new(size: usize, seed: i64, seen: i64, entries: Vec<(i64, String)>) -> ReplayBuffer {
        let mut items: Vec<(u64, i64, String)> =
            entries.into_iter().map(|(g, t)| (replay_key(seed, g), g, t)).collect();
        items.sort();
        items.truncate(size);
        ReplayBuffer {
            seed,
            size,
            seen,
            items,
        }
    }

    /// How many texts the buffer holds.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The kept texts, in priority order - the order an epoch rehearses them in.
    pub fn texts(&self) -> Vec<String> {
        self.items.iter().map(|(_, _, t)| t.clone()).collect()
    }

    /// Offers `texts` in order; each is kept while it is among the `size`
    /// smallest priorities.
    pub fn offer(&mut self, texts: &[String]) {
        for text in texts {
            let g = self.seen;
            self.seen += 1;
            if self.size == 0 {
                continue;
            }
            let key = (replay_key(self.seed, g), g);
            if self.items.len() >= self.size {
                let last = &self.items[self.items.len() - 1];
                if key >= (last.0, last.1) {
                    continue;
                }
                self.items.pop();
            }
            let at = self.items.partition_point(|(k, i, _)| (*k, *i) < key);
            self.items.insert(at, (key.0, key.1, text.clone()));
        }
    }

    /// A new capacity; a smaller one drops the entries with the largest priorities.
    pub fn resize(&mut self, size: usize) {
        self.size = size;
        self.items.truncate(size);
    }

    /// The file's `replay` block: `{size, seen, index, texts}`, entries in
    /// priority order.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("size", Json::Int(self.size as i64)),
            ("seen", Json::Int(self.seen)),
            ("index", Json::ints(self.items.iter().map(|(_, g, _)| *g))),
            ("texts", Json::strs(self.items.iter().map(|(_, _, t)| t.clone()))),
        ])
    }

    /// Reads a `replay` block; the priorities are recomputed from `seed`,
    /// not trusted.
    pub fn from_json(doc: &Json, seed: i64) -> Result<ReplayBuffer, String> {
        let index = doc.at("index").to_i64s();
        let texts = doc.at("texts").to_strings();
        if index.len() != texts.len() {
            return Err(format!(
                "replay block has {} indices for {} texts",
                index.len(),
                texts.len()
            ));
        }
        let size = doc.at("size").as_i64().unwrap_or(texts.len() as i64);
        if size < 0 {
            return Err(format!("replay_size must be >= 0, got {size}"));
        }
        let seen = doc.at("seen").as_i64().unwrap_or(texts.len() as i64);
        Ok(ReplayBuffer::new(
            size as usize,
            seed,
            seen,
            index.into_iter().zip(texts).collect(),
        ))
    }
}

/// Stops a run that has stopped improving: `patience` epochs without a loss
/// `min_delta` below the best.
#[derive(Clone, Debug)]
pub struct EarlyStop {
    pub patience: usize,
    pub min_delta: f64,
    best: f64,
    stale: usize,
}

impl EarlyStop {
    pub fn new(patience: usize, min_delta: f64) -> EarlyStop {
        EarlyStop {
            patience,
            min_delta,
            best: f64::INFINITY,
            stale: 0,
        }
    }

    /// Takes one epoch's loss; `true` when the run should stop after it.
    pub fn update(&mut self, loss: f64) -> bool {
        if loss < self.best - self.min_delta {
            self.best = loss;
            self.stale = 0;
        } else {
            self.stale += 1;
        }
        self.patience > 0 && self.stale >= self.patience
    }
}

/// Which texts each epoch of one `train` call walks, what it rehearses, and
/// when it stops.
///
/// `texts` are the run's usable texts (the ones too short for a gram already
/// dropped).  The model's buffer is read as the run begins - a copy, never
/// written: [`TrainingPlan::finish`] hands back the buffer the model keeps
/// afterwards, so a run that fails leaves the model's as it was.
#[derive(Clone, Debug)]
pub struct TrainingPlan {
    pub texts: Vec<String>,
    seed: i64,
    epochs: usize,
    order: String,
    curriculum: f64,
    replay_size: Option<usize>,
    buffer: Option<ReplayBuffer>,
    pool: Vec<String>,
    rehearse: usize,
    stopper: EarlyStop,
    base: Vec<usize>,
}

impl TrainingPlan {
    /// The plan of one run over `texts` (`lengths` in the encoding's units)
    /// for a model of `seed` whose buffer is `buffer`.
    pub fn new(
        texts: Vec<String>,
        lengths: &[usize],
        seed: i64,
        epochs: usize,
        plan: &Plan,
        buffer: Option<ReplayBuffer>,
    ) -> Result<TrainingPlan, String> {
        plan.check()?;
        let pool = buffer.as_ref().map(ReplayBuffer::texts).unwrap_or_default();
        let rehearse = replay_count(plan.replay, texts.len(), pool.len());
        let mut base: Vec<usize> = (0..texts.len()).collect();
        match plan.order.as_str() {
            "shortest-first" => base.sort_by(|&a, &b| lengths[a].cmp(&lengths[b]).then(a.cmp(&b))),
            "longest-first" => base.sort_by(|&a, &b| lengths[b].cmp(&lengths[a]).then(a.cmp(&b))),
            _ => {}
        }
        Ok(TrainingPlan {
            texts,
            seed,
            epochs,
            order: plan.order.clone(),
            curriculum: plan.curriculum,
            replay_size: plan.replay_size,
            buffer,
            pool,
            rehearse,
            stopper: EarlyStop::new(plan.patience, plan.min_delta),
            base,
        })
    }

    /// Whether every epoch walks every text in corpus order and rehearses
    /// nothing - the pass it always was.
    pub fn plain(&self) -> bool {
        self.order == "corpus" && self.curriculum >= 1.0 && self.rehearse == 0
    }

    /// The buffered texts the structure pass observes after the run's own:
    /// the whole pool, when any is rehearsed.  They were trained before, so
    /// observing them is a no-op on a graph that still holds them; observing
    /// all of them, in priority order, keeps the structure pass one list long
    /// in every port.
    pub fn replayed(&self) -> Vec<String> {
        if self.rehearse == 0 {
            Vec::new()
        } else {
            self.pool.clone()
        }
    }

    /// How many of the run's texts epoch `j` walks.
    pub fn walked(&self, j: usize) -> usize {
        paced(self.texts.len(), self.curriculum, j, self.epochs)
    }

    /// Epoch `j` (from 0), whose record will carry epoch `number`: the
    /// indices of the run's texts it walks, in order, and the rehearsed texts
    /// that follow them.
    pub fn epoch(&self, j: usize, number: i64) -> (Vec<usize>, Vec<String>) {
        let n = self.texts.len();
        let mut chosen = if self.order == "shuffle" {
            let keys: Vec<u64> = (0..n).map(|i| shuffle_key(self.seed, number, i)).collect();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| keys[a].cmp(&keys[b]).then(a.cmp(&b)));
            order
        } else {
            self.base.clone()
        };
        chosen.truncate(self.walked(j));
        if self.rehearse == 0 {
            return (chosen, Vec::new());
        }
        let size = self.pool.len();
        let start = j * self.rehearse;
        let again = (0..self.rehearse)
            .map(|i| self.pool[(start + i) % size].clone())
            .collect();
        (chosen, again)
    }

    /// The texts epoch `j` walks: the chosen texts of the run, then the
    /// rehearsed ones.
    pub fn epoch_texts(&self, j: usize, number: i64) -> Vec<String> {
        let (chosen, again) = self.epoch(j, number);
        let mut out: Vec<String> = chosen.into_iter().map(|i| self.texts[i].clone()).collect();
        out.extend(again);
        out
    }

    /// Whether the run stops after epoch `j`, given its loss; an epoch inside
    /// the curriculum does not count.
    pub fn stop(&mut self, j: usize, loss: f64) -> bool {
        if self.walked(j) < self.texts.len() {
            return false;
        }
        self.stopper.update(loss)
    }

    /// The replay buffer the model keeps after this run: resized if asked,
    /// with the run's texts offered.
    pub fn finish(&mut self) -> Option<ReplayBuffer> {
        let mut buffer = self.buffer.take();
        if let Some(size) = self.replay_size {
            if size == 0 {
                return None;
            }
            match buffer.as_mut() {
                Some(b) => b.resize(size),
                None => buffer = Some(ReplayBuffer::new(size, self.seed, 0, Vec::new())),
            }
        }
        if let Some(b) = buffer.as_mut() {
            b.offer(&self.texts);
        }
        buffer
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_matches_the_reference_values() {
        // the first outputs of the reference SplitMix64 generator seeded with 0
        // are splitmix64(k * golden) for k = 0, 1, ...: the finaliser of 0 is
        // the published first value
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
    }

    #[test]
    fn the_keys_are_pinned() {
        // the Python and Go ports assert the same numbers
        assert_eq!(shuffle_key(7, 1, 0), 2915934175157713820);
        assert_eq!(shuffle_key(-3, 2, 5), 14722852399570101988);
        assert_eq!(replay_key(7, 0), 5594249859331236493);
        assert_eq!(replay_key(-1, 9), 6800963518696358292);
    }

    #[test]
    fn planned_training_keeps_a_buffer_stops_and_ignores_feedback() {
        use crate::model::{Model, TrainOptions};
        let texts: Vec<String> = (0..30).map(|i| format!("the cat number {i} sat on the mat")).collect();
        let mut m = Model::new(2, crate::GraphOptions::default()).unwrap();
        let opts = TrainOptions {
            epochs: 6,
            plan: Plan {
                replay_size: Some(6),
                patience: 1,
                min_delta: 100.0,
                ..Plan::default()
            },
            ..TrainOptions::default()
        };
        let records = m.train(&texts, &opts).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[1].extra_value("early_stop"), Some(&Json::Bool(true)));
        assert_eq!(records[0].extra_value("early_stop"), None);
        let before = m.replay.clone().unwrap();
        assert_eq!((before.len(), before.seen), (6, 30));
        // a thumbs up, or any pass stamped with a phase, leaves the buffer alone
        m.reward(&texts[..3], 1, 1.0).unwrap();
        let stamped = TrainOptions {
            epochs: 1,
            phase: Some("negative".into()),
            ..TrainOptions::default()
        };
        m.train(&texts[3..6], &stamped).unwrap();
        assert_eq!(m.replay.as_ref(), Some(&before));
        // saved last, read back
        let doc = m.to_doc();
        let Json::Obj(pairs) = &doc else { panic!("a document") };
        assert_eq!(pairs.last().map(|(k, _)| k.as_str()), Some("replay"));
        let back = Model::from_doc(&doc).unwrap();
        assert_eq!(back.replay, m.replay);
        // 0 drops it, and the file no longer mentions it
        let drop = TrainOptions {
            epochs: 1,
            plan: Plan {
                replay_size: Some(0),
                ..Plan::default()
            },
            ..TrainOptions::default()
        };
        m.train(&texts[..2], &drop).unwrap();
        assert!(m.replay.is_none());
        assert!(m.to_doc().get("replay").is_none());
    }

    #[test]
    fn reverse_reads_the_units_backwards() {
        let chars = crate::encoding::Encoding::default();
        let words = crate::encoding::parse_encoding("word:2:1").unwrap();
        assert_eq!(chars.reverse("héllo wörld"), "dlröw olléh");
        assert_eq!(chars.reverse("the cat\n"), "\ntac eht");
        assert_eq!(chars.reverse(""), "");
        assert_eq!(words.reverse("  the  cat\tsat\n"), "sat cat the");
        assert_eq!(words.reverse("one"), "one");
        assert_eq!(words.reverse(" \t "), "");
        // a combining accent stays a code point of its own, so twice is the text again
        let text = "cafe\u{301} au lait";
        assert_eq!(chars.reverse(&chars.reverse(text)), text);
        // read borrows the texts when the plan reads them forwards
        let texts = vec!["abc".to_string()];
        assert!(matches!(
            read(&chars, &texts, &Plan::default()),
            std::borrow::Cow::Borrowed(_)
        ));
    }

    /// A model's document with the clock taken out: what two runs that read
    /// the same texts must agree on byte for byte.
    fn timeless(doc: &Json) -> Json {
        match doc {
            Json::Obj(pairs) => Json::Obj(
                pairs
                    .iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "seconds" | "saved_at" | "created"))
                    .map(|(k, v)| (k.clone(), timeless(v)))
                    .collect(),
            ),
            Json::Arr(items) => Json::Arr(items.iter().map(timeless).collect()),
            other => other.clone(),
        }
    }

    #[test]
    fn a_reversed_run_is_a_run_over_the_reversed_texts() {
        use crate::kinds::{new_model, train, TrainSettings};
        let texts: Vec<String> = (0..20).map(|i| format!("the cat number {i} sat on the mat")).collect();
        for kind in ["count", "radix", "resonant", "negative"] {
            for spec in ["char:3:1", "word:2:1"] {
                let enc = crate::encoding::parse_encoding(spec).unwrap();
                let reversed: Vec<String> = texts.iter().map(|t| enc.reverse(t)).collect();
                // every kind trains through kinds::train, as the servers and the CLI do
                let run = |texts: &[String], reverse: bool| {
                    let mut m = new_model(kind, 3, enc, &[]).unwrap();
                    let mut s = TrainSettings::default();
                    s.config.epochs = 2;
                    s.config.plan.reverse = reverse;
                    train(&mut m, texts, &s, &mut |_| true).unwrap();
                    timeless(&m.to_doc()).render(0)
                };
                let want = run(&reversed, false);
                assert_ne!(want, run(&texts, false), "{kind} {spec}: backwards is not forwards");
                assert_eq!(run(&texts, true), want, "{kind} {spec}");
            }
        }
        // and the library's own entry point reads them backwards exactly once
        use crate::model::{Model, TrainOptions};
        let reversed: Vec<String> = texts
            .iter()
            .map(|t| crate::encoding::Encoding::default().reverse(t))
            .collect();
        let run = |texts: &[String], reverse: bool| {
            let mut m = Model::new(3, crate::GraphOptions::default()).unwrap();
            let opts = TrainOptions {
                epochs: 2,
                plan: Plan {
                    reverse,
                    ..Plan::default()
                },
                ..TrainOptions::default()
            };
            m.train(texts, &opts).unwrap();
            timeless(&m.to_doc()).render(0)
        };
        assert_eq!(run(&texts, true), run(&reversed, false));
    }

    #[test]
    fn a_negative_seed_is_two_s_complement() {
        assert_eq!(shuffle_key(-1, 3, 4), shuffle_key(-1, 3, 4));
        assert_eq!(replay_key(-1, 0), splitmix64(splitmix64(u64::MAX ^ REPLAY_SALT)));
    }

    #[test]
    fn the_curriculum_grows_to_the_whole_list() {
        assert_eq!(paced(10, 0.3, 0, 4), 3);
        assert_eq!(paced(10, 0.3, 3, 4), 10);
        assert_eq!(paced(10, 0.3, 1, 4), 6); // 0.3 + 0.7/3 = 0.5333.. -> ceil(5.33) = 6
        assert_eq!(paced(10, 0.01, 0, 4), 1);
        assert_eq!(paced(10, 1.0, 0, 4), 10);
        assert_eq!(paced(10, 0.5, 0, 1), 10);
        assert_eq!(paced(0, 0.5, 0, 3), 0);
    }

    #[test]
    fn replay_rounds_half_up_and_stops_at_the_pool() {
        assert_eq!(replay_count(0.25, 10, 100), 3); // 2.5 -> 3
        assert_eq!(replay_count(0.24, 10, 100), 2);
        assert_eq!(replay_count(5.0, 10, 7), 7);
        assert_eq!(replay_count(1e300, 10, 7), 7);
        assert_eq!(replay_count(0.0, 10, 7), 0);
        assert_eq!(replay_count(1.0, 10, 0), 0);
    }

    #[test]
    fn the_buffer_keeps_the_smallest_priorities_whatever_the_order() {
        let texts: Vec<String> = (0..50).map(|i| format!("text {i}")).collect();
        let mut one = ReplayBuffer::new(8, 7, 0, Vec::new());
        one.offer(&texts);
        let mut two = ReplayBuffer::new(8, 7, 0, Vec::new());
        for t in texts.chunks(3) {
            two.offer(t);
        }
        assert_eq!(one, two);
        assert_eq!(one.len(), 8);
        assert_eq!(one.seen, 50);
        let mut keys: Vec<(u64, i64)> = (0..50).map(|g| (replay_key(7, g), g)).collect();
        keys.sort();
        let want: Vec<i64> = keys[..8].iter().map(|&(_, g)| g).collect();
        assert_eq!(one.to_json().at("index").to_i64s(), want);
        let back = ReplayBuffer::from_json(&one.to_json(), 7).unwrap();
        assert_eq!(back, one);
    }

    #[test]
    fn a_smaller_buffer_drops_the_largest_priorities() {
        let texts: Vec<String> = (0..20).map(|i| format!("t{i}")).collect();
        let mut b = ReplayBuffer::new(10, 3, 0, Vec::new());
        b.offer(&texts);
        let first: Vec<String> = b.texts()[..4].to_vec();
        b.resize(4);
        assert_eq!(b.texts(), first);
    }

    #[test]
    fn early_stopping_waits_for_patience() {
        let mut s = EarlyStop::new(2, 0.1);
        assert!(!s.update(5.0));
        assert!(!s.update(4.95)); // not 0.1 better
        assert!(s.update(4.99));
        let mut off = EarlyStop::new(0, 0.0);
        for _ in 0..10 {
            assert!(!off.update(1.0));
        }
    }

    #[test]
    fn a_plan_orders_paces_and_rehearses() {
        let texts: Vec<String> = ["ccc", "a", "bb", "dddd"].iter().map(|s| s.to_string()).collect();
        let lengths = [3, 1, 2, 4];
        let plan = Plan {
            order: "shortest-first".to_string(),
            curriculum: 0.5,
            ..Plan::default()
        };
        let p = TrainingPlan::new(texts.clone(), &lengths, 1, 3, &plan, None).unwrap();
        assert!(!p.plain());
        assert_eq!(p.epoch(0, 1).0, vec![1, 2]);
        assert_eq!(p.epoch(1, 2).0, vec![1, 2, 0]);
        assert_eq!(p.epoch(2, 3).0, vec![1, 2, 0, 3]);
        let longest = Plan {
            order: "longest-first".to_string(),
            ..Plan::default()
        };
        let p = TrainingPlan::new(texts.clone(), &lengths, 1, 3, &longest, None).unwrap();
        assert_eq!(p.epoch(0, 1).0, vec![3, 0, 2, 1]);
        let mut buffer = ReplayBuffer::new(3, 1, 0, Vec::new());
        buffer.offer(&["x".to_string(), "y".to_string(), "z".to_string()]);
        let rehearsing = Plan {
            replay: 0.5,
            ..Plan::default()
        };
        let p = TrainingPlan::new(texts, &lengths, 1, 3, &rehearsing, Some(buffer.clone())).unwrap();
        let pool = buffer.texts();
        assert_eq!(p.replayed(), pool);
        assert_eq!(p.epoch(0, 1).1, vec![pool[0].clone(), pool[1].clone()]);
        assert_eq!(p.epoch(1, 2).1, vec![pool[2].clone(), pool[0].clone()]);
    }

    #[test]
    fn a_shuffle_is_a_permutation_that_moves_with_the_epoch() {
        let texts: Vec<String> = (0..12).map(|i| format!("{i}")).collect();
        let lengths = vec![1; 12];
        let plan = Plan {
            order: "shuffle".to_string(),
            ..Plan::default()
        };
        let p = TrainingPlan::new(texts, &lengths, 9, 2, &plan, None).unwrap();
        let (a, _) = p.epoch(0, 1);
        let (b, _) = p.epoch(1, 2);
        let mut sorted = a.clone();
        sorted.sort();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
        assert_ne!(a, b);
    }

    #[test]
    fn settings_out_of_range_are_refused() {
        let bad = |p: Plan| p.check().unwrap_err();
        assert!(bad(Plan {
            order: "random".into(),
            ..Plan::default()
        })
        .contains("unknown order 'random'"));
        assert!(bad(Plan {
            curriculum: 0.0,
            ..Plan::default()
        })
        .contains("curriculum"));
        assert!(bad(Plan {
            replay: -1.0,
            ..Plan::default()
        })
        .contains("replay"));
        assert!(bad(Plan {
            min_delta: f64::NAN,
            ..Plan::default()
        })
        .contains("min_delta"));
    }

    #[test]
    fn finishing_offers_the_run_and_honours_the_size() {
        let texts: Vec<String> = (0..5).map(|i| format!("t{i}")).collect();
        let lengths = vec![2; 5];
        let sized = Plan {
            replay_size: Some(3),
            ..Plan::default()
        };
        let mut p = TrainingPlan::new(texts.clone(), &lengths, 4, 1, &sized, None).unwrap();
        let b = p.finish().unwrap();
        assert_eq!((b.size, b.seen, b.len()), (3, 5, 3));
        let dropped = Plan {
            replay_size: Some(0),
            ..Plan::default()
        };
        let mut p = TrainingPlan::new(texts.clone(), &lengths, 4, 1, &dropped, Some(b.clone())).unwrap();
        assert!(p.finish().is_none());
        let mut p = TrainingPlan::new(texts, &lengths, 4, 1, &Plan::default(), Some(b)).unwrap();
        let kept = p.finish().unwrap();
        assert_eq!(kept.seen, 10);
        let mut p = TrainingPlan::new(Vec::new(), &[], 4, 1, &Plan::default(), None).unwrap();
        assert!(p.finish().is_none());
    }
}
