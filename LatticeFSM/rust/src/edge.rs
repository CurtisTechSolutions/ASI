//! One cell of the 3D matrix: a dense record of everything the machine knows about one transition.
//!
//! An edge is `(source, symbol, target)`: *in state* `source`, *reading* `symbol`,
//! *the machine may move to* `target`.  The matrix holds one for every
//! combination, so an edge is never created or deleted - only written to.
//! It holds a history (`seen`, `first_seen`, `last_seen`, the fading `recent`
//! trace), a verdict (`rewarded`, `punished` and when), a `width` that use and
//! reward widen, punishment narrows and disuse relaxes, its own adaptive
//! [`Weighting`], and the `features` as they were at its last traversal - what
//! the adaptation credits.  `DESIGN.md` §2-§4.

use crate::json::Json;

/// The features an edge's weighting reads, in the order [`Edge::features_at`] returns them.
pub const FEATURES: [&str; 5] = ["seen", "recent", "net", "age", "width"];

/// A channel's width at rest.
pub const WIDTH_REST: f64 = 1.0;
/// The narrowest a channel can be.
pub const WIDTH_MIN: f64 = 0.05;
/// The widest a channel can be.
pub const WIDTH_MAX: f64 = 20.0;
/// Every coefficient of a weighting is clipped to `[-limit, limit]` after adapting, so no edge can become certain.
pub const COEFFICIENT_LIMIT: f64 = 8.0;

/// The coefficients of one edge's weighting function:
/// `log_weight = bias + seen·f_seen + recent·f_recent + net·f_net + age·f_age + width·f_width`.
/// The stimulation's own width term is the machine's, not the edge's.  `rate` is this edge's learning rate.
#[derive(Clone, Debug, PartialEq)]
pub struct Weighting {
    pub bias: f64,
    pub seen: f64,
    pub recent: f64,
    pub net: f64,
    pub age: f64,
    pub width: f64,
    pub rate: f64,
}

impl Default for Weighting {
    /// The prototype a fresh machine gives every edge: nothing but `net`, so an unadapted edge weighs its own
    /// credit per traversal and nothing else.
    fn default() -> Weighting {
        Weighting {
            bias: 0.0,
            seen: 0.0,
            recent: 0.0,
            net: 1.0,
            age: 0.0,
            width: 0.0,
            rate: 0.1,
        }
    }
}

fn clip(v: f64) -> f64 {
    v.clamp(-COEFFICIENT_LIMIT, COEFFICIENT_LIMIT)
}

impl Weighting {
    /// This edge's opinion of itself: its coefficients applied to its features.
    pub fn log_weight(&self, f: &[f64; 5]) -> f64 {
        self.bias + self.seen * f[0] + self.recent * f[1] + self.net * f[2] + self.age * f[3] + self.width * f[4]
    }

    /// Move the coefficients with `credit` (reward positive, punishment negative): the bias by `rate · credit`,
    /// each coefficient by `rate · credit · feature` - the feature as it was when the edge was traversed.
    pub fn adapt(&mut self, credit: f64, f: &[f64; 5]) {
        let step = self.rate * credit;
        self.bias = clip(self.bias + step);
        self.seen = clip(self.seen + step * f[0]);
        self.recent = clip(self.recent + step * f[1]);
        self.net = clip(self.net + step * f[2]);
        self.age = clip(self.age + step * f[3]);
        self.width = clip(self.width + step * f[4]);
    }

    pub fn to_json(&self) -> Json {
        Json::numbers(&[
            self.bias,
            self.seen,
            self.recent,
            self.net,
            self.age,
            self.width,
            self.rate,
        ])
    }

    pub fn from_json(v: &Json) -> Result<Weighting, String> {
        let list = v.as_array().ok_or("a weighting is a list of seven numbers")?;
        let n = |i: usize| {
            list.get(i)
                .and_then(Json::as_f64)
                .ok_or(format!("weighting[{i}] is not a number"))
        };
        Ok(Weighting {
            bias: n(0)?,
            seen: n(1)?,
            recent: n(2)?,
            net: n(3)?,
            age: n(4)?,
            width: n(5)?,
            rate: n(6)?,
        })
    }
}

/// One transition of the machine and everything it has lived through.
#[derive(Clone, Debug, PartialEq)]
pub struct Edge {
    pub source: usize,
    pub symbol: usize,
    pub target: usize,
    /// Traversals, ever.
    pub seen: u64,
    /// The clock reading of the first traversal; `-1` never.
    pub first_seen: i64,
    /// The clock reading of the latest traversal; `-1` never.
    pub last_seen: i64,
    /// The traversal trace as written at `last_seen`; [`Edge::recent_at`] reads it through the clock.
    pub recent: f64,
    /// Reward credited, ever.
    pub rewarded: f64,
    /// Punishment credited, ever (a positive number).
    pub punished: f64,
    pub last_rewarded: i64,
    pub last_punished: i64,
    /// The width as written at `width_stamp`; [`Edge::width_at`] reads it through the clock.
    pub width: f64,
    pub width_stamp: i64,
    /// This edge's own weighting function.
    pub weighting: Weighting,
    /// The features at the latest traversal: what [`Edge::credit`] adapts the weighting with.
    pub features: [f64; 5],
}

impl Edge {
    pub fn new(source: usize, symbol: usize, target: usize, weighting: Weighting) -> Edge {
        Edge {
            source,
            symbol,
            target,
            seen: 0,
            first_seen: -1,
            last_seen: -1,
            recent: 0.0,
            rewarded: 0.0,
            punished: 0.0,
            last_rewarded: -1,
            last_punished: -1,
            width: WIDTH_REST,
            width_stamp: 0,
            weighting,
            features: [0.0; 5],
        }
    }

    fn half(&self, clock: i64, life: f64) -> f64 {
        0.5f64.powf((clock - self.last_seen) as f64 / life)
    }

    /// The traversal trace, faded by a half every `life` ticks since the last traversal.
    pub fn recent_at(&self, clock: i64, life: f64) -> f64 {
        if self.last_seen < 0 {
            0.0
        } else {
            self.recent * self.half(clock, life)
        }
    }

    /// How recently the edge was traversed: one at the moment, a half a life later, zero if never.
    pub fn age_at(&self, clock: i64, life: f64) -> f64 {
        if self.last_seen < 0 {
            0.0
        } else {
            self.half(clock, life)
        }
    }

    /// The width, relaxed toward [`WIDTH_REST`] by a half every `life` ticks since it was last written.
    pub fn width_at(&self, clock: i64, life: f64) -> f64 {
        if self.width == WIDTH_REST {
            WIDTH_REST
        } else {
            WIDTH_REST + (self.width - WIDTH_REST) * 0.5f64.powf((clock - self.width_stamp) as f64 / life)
        }
    }

    /// Credit per traversal: `(rewarded − punished) / (seen + 1)`.
    pub fn net(&self) -> f64 {
        (self.rewarded - self.punished) / (self.seen as f64 + 1.0)
    }

    /// The five features of [`FEATURES`], as they are at `clock`.
    pub fn features_at(&self, clock: i64, life: f64) -> [f64; 5] {
        [
            (self.seen as f64).ln_1p(),
            self.recent_at(clock, life),
            self.net(),
            self.age_at(clock, life),
            self.width_at(clock, life).ln(),
        ]
    }

    /// What the walk selects by: the edge's own function of its features, plus `stimulation · log(width)`.
    pub fn log_weight(&self, clock: i64, life: f64, stimulation: f64) -> f64 {
        let f = self.features_at(clock, life);
        self.weighting.log_weight(&f) + stimulation * f[4]
    }

    /// Record a traversal at `clock`: features noted, `seen` adds one, the trace is set, the channel widens.
    pub fn traverse(&mut self, clock: i64, life: f64, trace: f64, widen: f64) {
        self.features = self.features_at(clock, life);
        self.recent = self.recent_at(clock, life) + trace;
        if self.first_seen < 0 {
            self.first_seen = clock;
        }
        self.last_seen = clock;
        self.seen += 1;
        if widen != 0.0 {
            let w = self.width_at(clock, life) * (1.0 + widen);
            self.set_width(w, clock);
        }
    }

    /// Receive `amount` of credit: reward if positive, punishment if negative.  The verdict is written down, the
    /// channel widens by `widen · amount` (or narrows by `narrow · |amount|`), and the weighting adapts with the
    /// features of the traversal being credited.
    pub fn credit(&mut self, amount: f64, clock: i64, life: f64, widen: f64, narrow: f64) {
        if amount > 0.0 {
            self.rewarded += amount;
            self.last_rewarded = clock;
            if widen != 0.0 {
                let w = self.width_at(clock, life) * (1.0 + widen * amount);
                self.set_width(w, clock);
            }
        } else if amount < 0.0 {
            self.punished -= amount;
            self.last_punished = clock;
            if narrow != 0.0 {
                let w = self.width_at(clock, life) / (1.0 + narrow * -amount);
                self.set_width(w, clock);
            }
        } else {
            return;
        }
        let f = self.features;
        self.weighting.adapt(amount, &f);
    }

    /// Write the width outright (clipped), stamped at `clock`.
    pub fn set_width(&mut self, width: f64, clock: i64) {
        self.width = width.clamp(WIDTH_MIN, WIDTH_MAX);
        self.width_stamp = clock;
    }

    /// Whether anything was ever written to this edge.
    pub fn touched(&self) -> bool {
        self.seen > 0 || self.rewarded != 0.0 || self.punished != 0.0 || self.width != WIDTH_REST
    }

    /// The edge as the flat list the Python port writes: `[source, symbol, target, seen, first_seen, last_seen,
    /// recent, rewarded, punished, last_rewarded, last_punished, width, width_stamp, [weighting], [features]]`.
    pub fn to_json(&self) -> Json {
        Json::Array(vec![
            self.source.into(),
            self.symbol.into(),
            self.target.into(),
            (self.seen as f64).into(),
            self.first_seen.into(),
            self.last_seen.into(),
            self.recent.into(),
            self.rewarded.into(),
            self.punished.into(),
            self.last_rewarded.into(),
            self.last_punished.into(),
            self.width.into(),
            self.width_stamp.into(),
            self.weighting.to_json(),
            Json::numbers(&self.features),
        ])
    }

    pub fn from_json(v: &Json) -> Result<Edge, String> {
        let list = v.as_array().ok_or("an edge is a list")?;
        if list.len() != 15 {
            return Err(format!("an edge is a list of 15 values, got {}", list.len()));
        }
        let n = |i: usize| list[i].as_f64().ok_or(format!("edge[{i}] is not a number"));
        let f = list[14].as_array().ok_or("edge features are a list")?;
        if f.len() != 5 {
            return Err("an edge has five features".to_string());
        }
        let mut features = [0.0; 5];
        for (i, x) in f.iter().enumerate() {
            features[i] = x.as_f64().ok_or("a feature is a number")?;
        }
        Ok(Edge {
            source: n(0)? as usize,
            symbol: n(1)? as usize,
            target: n(2)? as usize,
            seen: n(3)? as u64,
            first_seen: n(4)? as i64,
            last_seen: n(5)? as i64,
            recent: n(6)?,
            rewarded: n(7)?,
            punished: n(8)?,
            last_rewarded: n(9)? as i64,
            last_punished: n(10)? as i64,
            width: n(11)?,
            width_stamp: n(12)? as i64,
            weighting: Weighting::from_json(&list[13])?,
            features,
        })
    }

    /// The edge as an object, for the API and `stats`: every field by name, with the fading ones read at `clock`.
    pub fn describe(&self, clock: i64, life: f64, stimulation: f64) -> Json {
        Json::object()
            .with("source", self.source.into())
            .with("symbol", self.symbol.into())
            .with("target", self.target.into())
            .with("seen", (self.seen as f64).into())
            .with("first_seen", self.first_seen.into())
            .with("last_seen", self.last_seen.into())
            .with("recent", self.recent_at(clock, life).into())
            .with("age", self.age_at(clock, life).into())
            .with("rewarded", self.rewarded.into())
            .with("punished", self.punished.into())
            .with("net", self.net().into())
            .with("last_rewarded", self.last_rewarded.into())
            .with("last_punished", self.last_punished.into())
            .with("width", self.width_at(clock, life).into())
            .with("weighting", self.weighting.to_json())
            .with("features", Json::numbers(&self.features))
            .with("log_weight", self.log_weight(clock, life, stimulation).into())
    }
}
