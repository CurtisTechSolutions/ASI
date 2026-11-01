//! The efference copy (`DESIGN.md` §8): every emission is copied into the alphabet of the sense that will read
//! it back, and what returns is one of three things - reafferent (it matched the copy: credited nothing),
//! an error (it differed, or the copy expired: a lesson on the forward model), or exafferent (no copy: the
//! world spoke, and only this may carry the world's credit).

#[derive(Clone, Debug, PartialEq)]
pub struct Copy {
    /// The emitted symbol as the reading sense will see it.
    pub symbol: String,
    pub modality: String,
    /// The clock when it was made.
    pub clock: i64,
    /// How long it may wait for its sensation: the loop's latency, never a life.
    pub ttl: i64,
    /// What the forward model predicted the next state would be, if it predicted.
    pub predicted: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Return {
    Reafferent,
    Error,
    Exafferent,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Efference {
    live: Vec<Copy>,
    pub reafferent: u64,
    pub errors: u64,
    pub exafferent: u64,
    pub expired: u64,
}

impl Efference {
    pub fn copy(&mut self, symbol: &str, modality: &str, clock: i64, ttl: i64, predicted: Option<usize>) {
        self.live.push(Copy {
            symbol: symbol.to_string(),
            modality: modality.to_string(),
            clock,
            ttl: ttl.max(1),
            predicted,
        });
    }

    /// Drop copies whose sensation can no longer arrive; each is an error (the act had no echo).
    pub fn expire(&mut self, clock: i64) -> usize {
        let before = self.live.len();
        self.live.retain(|c| clock - c.clock <= c.ttl);
        let gone = before - self.live.len();
        self.expired += gone as u64;
        self.errors += gone as u64;
        gone
    }

    /// Classify a returning symbol of `modality`: the oldest live copy of that modality is consumed.
    /// `matches` says whether the symbol is the copy's within the sense's tolerance.
    pub fn observe(
        &mut self,
        symbol: &str,
        modality: &str,
        matches: impl Fn(&str, &str) -> bool,
    ) -> (Return, Option<Copy>) {
        let pos = self.live.iter().position(|c| c.modality == modality);
        match pos {
            None => {
                self.exafferent += 1;
                (Return::Exafferent, None)
            }
            Some(i) => {
                let c = self.live.remove(i);
                if matches(&c.symbol, symbol) {
                    self.reafferent += 1;
                    (Return::Reafferent, Some(c))
                } else {
                    self.errors += 1;
                    (Return::Error, Some(c))
                }
            }
        }
    }

    pub fn live(&self) -> usize {
        self.live.len()
    }

    /// The reafferent share of everything observed: a meta-feature.
    pub fn reafferent_share(&self) -> f64 {
        let n = self.reafferent + self.errors + self.exafferent;
        if n == 0 {
            0.0
        } else {
            self.reafferent as f64 / n as f64
        }
    }
}
