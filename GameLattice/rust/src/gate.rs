//! The thalamic gate (`DESIGN.md` §10.1): a reserve per modality, surges on onsets, and the meta-player's hand
//! on it - capped in duration and logged, so a sense is never starved to bankruptcy.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct Gate {
    pub reserves: BTreeMap<String, f64>,
    /// `(modality, reserve set, steps left)`: a suppression or attention the meta-player asked for.
    pub holds: Vec<(String, f64, u64)>,
    pub default_reserve: f64,
    /// The longest a hold may last.
    pub cap: u64,
    pub log: Vec<String>,
}

impl Default for Gate {
    fn default() -> Gate {
        Gate {
            reserves: BTreeMap::new(),
            holds: Vec::new(),
            default_reserve: 0.01,
            cap: 64,
            log: Vec::new(),
        }
    }
}

impl Gate {
    pub fn reserve(&self, modality: &str) -> f64 {
        self.reserves.get(modality).copied().unwrap_or(self.default_reserve)
    }

    /// Raise a modality's reserve for at most `cap` steps (SUPPRESS: looking away) or lower it (ATTEND).
    pub fn hold(&mut self, modality: &str, reserve: f64, steps: u64) {
        let steps = steps.min(self.cap);
        self.holds.push((modality.to_string(), reserve, steps));
        self.reserves.insert(modality.to_string(), reserve);
        self.log
            .push(format!("{modality}: reserve {reserve} for {steps} steps"));
    }

    /// One step passed: holds run down, and a hold that ran out gives the modality its default back.
    pub fn step(&mut self) {
        let mut released = Vec::new();
        for h in self.holds.iter_mut() {
            h.2 = h.2.saturating_sub(1);
            if h.2 == 0 {
                released.push(h.0.clone());
            }
        }
        self.holds.retain(|h| h.2 > 0);
        for m in released {
            self.reserves.remove(&m);
            self.log.push(format!("{m}: hold released"));
        }
    }
}

/// The surge reflex table (`DESIGN.md` §17 E10): what an event adds to a lattice's stimulation, with a soft
/// ceiling.  Per lattice; the gain is the one learned number and starts at 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Surge {
    pub comparator_miss: f64,
    pub jolt: f64,
    pub punishment_per_unit: f64,
    pub guard_reject: f64,
    pub bankruptcy: f64,
    pub reward: f64,
    pub ceiling: f64,
    pub gain: f64,
}

impl Default for Surge {
    fn default() -> Surge {
        Surge {
            comparator_miss: 0.5,
            jolt: 1.0,
            punishment_per_unit: 0.5,
            guard_reject: 0.5,
            bankruptcy: 2.0,
            reward: -0.25,
            ceiling: 4.0,
            gain: 1.0,
        }
    }
}

impl Surge {
    /// The amount to add for an event, given the current level: the ceiling is soft (what is added shrinks as
    /// the level nears it).
    pub fn amount(&self, event: f64, level: f64) -> f64 {
        let room = (self.ceiling - level).max(0.0) / self.ceiling;
        let a = self.gain * event;
        if a > 0.0 {
            a * room
        } else {
            a
        }
    }
}
