//! Phonotactics (the port of phonotactics.py): which sounds go well together,
//! and how to build a word out of them.

use std::collections::HashMap;

use crate::lexicon::{Entry, Lexicon};
use crate::mt::Rng;
use crate::phones::{base, is_vowel, strip_stress, with_stress, BOUNDARY};
use crate::syllables::{is_legal_coda, is_legal_onset, syllabify, Syllable};

/// Sound-to-sound habits counted from pronunciations, plus the law of the syllable.
#[derive(Clone, Debug)]
pub struct Phonotactics {
    pub stress: bool,
    pub smoothing: f64,
    pub bigrams: HashMap<(String, String), usize>,
    pub unigrams: HashMap<String, usize>,
    followers: HashMap<String, HashMap<String, usize>>,
    follower_order: HashMap<String, Vec<String>>,
    pub onsets_initial: HashMap<String, usize>,
    pub onsets_medial: HashMap<String, usize>,
    pub nuclei: HashMap<String, usize>,
    pub codas_medial: HashMap<String, usize>,
    pub codas_final: HashMap<String, usize>,
    pub syllable_counts: HashMap<usize, usize>,
    pub words: usize,
}

fn cluster_parts(s: &str) -> Vec<String> {
    if s.is_empty() {
        Vec::new()
    } else {
        s.split(' ').map(|p| p.to_string()).collect()
    }
}

/// Python compares tuples of strings element by element, a shorter prefix first.
fn cluster_less(a: &str, b: &str) -> std::cmp::Ordering {
    cluster_parts(a).cmp(&cluster_parts(b))
}

fn sample(items: &[(String, f64)], rng: &mut dyn Rng) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let total: f64 = items.iter().map(|(_, w)| w).sum();
    let r = rng.float64() * total;
    let mut acc = 0.0;
    for (k, w) in items {
        acc += w;
        if r < acc {
            return Some(k.clone());
        }
    }
    Some(items[items.len() - 1].0.clone())
}

impl Phonotactics {
    pub fn new(stress: bool) -> Phonotactics {
        Phonotactics {
            stress,
            smoothing: 0.5,
            bigrams: HashMap::new(),
            unigrams: HashMap::new(),
            followers: HashMap::new(),
            follower_order: HashMap::new(),
            onsets_initial: HashMap::new(),
            onsets_medial: HashMap::new(),
            nuclei: HashMap::new(),
            codas_medial: HashMap::new(),
            codas_final: HashMap::new(),
            syllable_counts: HashMap::new(),
            words: 0,
        }
    }

    fn key(&self, phone: &str) -> String {
        if self.stress {
            phone.to_string()
        } else {
            base(phone).to_string()
        }
    }

    /// Counts the transitions and the syllable parts of every pronunciation.
    pub fn fit(&mut self, pronunciations: &[Vec<String>]) -> &mut Phonotactics {
        for phones in pronunciations {
            if phones.is_empty() {
                continue;
            }
            let mut keys = vec![BOUNDARY.to_string()];
            keys.extend(phones.iter().map(|p| self.key(p)));
            keys.push(BOUNDARY.to_string());
            for w in keys.windows(2) {
                let (a, b) = (w[0].clone(), w[1].clone());
                *self.bigrams.entry((a.clone(), b.clone())).or_insert(0) += 1;
                *self.unigrams.entry(a.clone()).or_insert(0) += 1;
                let table = self.followers.entry(a.clone()).or_default();
                if !table.contains_key(&b) {
                    self.follower_order.entry(a).or_default().push(b.clone());
                }
                *table.entry(b).or_insert(0) += 1;
            }
            *self.unigrams.entry(BOUNDARY.to_string()).or_insert(0) += 1;
            let syllables = syllabify(phones, true);
            *self.syllable_counts.entry(syllables.len()).or_insert(0) += 1;
            let last = syllables.len() - 1;
            for (i, s) in syllables.iter().enumerate() {
                let onset = s.onset.join(" ");
                if i == 0 {
                    *self.onsets_initial.entry(onset).or_insert(0) += 1;
                } else {
                    *self.onsets_medial.entry(onset).or_insert(0) += 1;
                }
                let nucleus = if !self.stress && is_vowel(&s.nucleus) {
                    with_stress(&s.nucleus, 0)
                } else {
                    s.nucleus.clone()
                };
                *self.nuclei.entry(nucleus).or_insert(0) += 1;
                let coda = s.coda.join(" ");
                if i == last {
                    *self.codas_final.entry(coda).or_insert(0) += 1;
                } else {
                    *self.codas_medial.entry(coda).or_insert(0) += 1;
                }
            }
            self.words += 1;
        }
        self
    }

    /// Fits on every pronunciation of a lexicon.
    pub fn from_lexicon(lexicon: &Lexicon, stress: bool) -> Phonotactics {
        let prons: Vec<Vec<String>> = lexicon.items().into_iter().map(|(_, p)| p).collect();
        let mut p = Phonotactics::new(stress);
        p.fit(&prons);
        p
    }

    pub fn alphabet_size(&self) -> usize {
        self.unigrams.len().max(1)
    }

    /// `P(b | a)`, add-k smoothed over the alphabet seen.
    pub fn prob(&self, a: &str, b: &str) -> f64 {
        let (a, b) = (self.key(a), self.key(b));
        let k = self.smoothing;
        let bigram = *self.bigrams.get(&(a.clone(), b)).unwrap_or(&0) as f64;
        let unigram = *self.unigrams.get(&a).unwrap_or(&0) as f64;
        (bigram + k) / (unigram + k * self.alphabet_size() as f64)
    }

    pub fn log_prob(&self, a: &str, b: &str) -> f64 {
        self.prob(a, b).ln()
    }

    /// How well `b` follows `a`, as pointwise mutual information in bits (0 = chance).
    pub fn affinity(&self, a: &str, b: &str) -> f64 {
        let (a, b) = (self.key(a), self.key(b));
        let total = self.unigrams.values().sum::<usize>().max(1) as f64;
        let k = self.smoothing;
        let size = self.alphabet_size() as f64;
        let bigram = *self.bigrams.get(&(a.clone(), b.clone())).unwrap_or(&0) as f64;
        let p_ab = (bigram + k) / (total + k * size * size);
        let p_a = (*self.unigrams.get(&a).unwrap_or(&0) as f64 + k) / (total + k * size);
        let p_b = (*self.unigrams.get(&b).unwrap_or(&0) as f64 + k) / (total + k * size);
        (p_ab / (p_a * p_b)).log2()
    }

    /// Every transition of a word with its log-probability, the boundaries included.
    pub fn transitions(&self, phones: &[String]) -> Vec<(String, String, f64)> {
        let mut keys = vec![BOUNDARY.to_string()];
        keys.extend(phones.iter().map(|p| self.key(p)));
        keys.push(BOUNDARY.to_string());
        keys.windows(2)
            .map(|w| (w[0].clone(), w[1].clone(), self.log_prob(&w[0], &w[1])))
            .collect()
    }

    /// The mean log-probability per transition.
    pub fn score(&self, phones: &[String]) -> f64 {
        let steps = self.transitions(phones);
        if steps.is_empty() {
            return 0.0;
        }
        steps.iter().map(|s| s.2).sum::<f64>() / steps.len() as f64
    }

    /// The index and log-probability of the least likely step.
    pub fn weakest(&self, phones: &[String]) -> (usize, f64) {
        let steps = self.transitions(phones);
        let mut best = 0;
        for (i, s) in steps.iter().enumerate() {
            if s.2 < steps[best].2 {
                best = i;
            }
        }
        (best, steps[best].2)
    }

    /// The `k` likeliest sounds after `phone` ("#" for the start of a word), with `P`.
    pub fn next(&self, phone: &str, k: usize) -> Vec<(String, f64)> {
        let a = self.key(phone);
        let Some(table) = self.followers.get(&a) else {
            return Vec::new();
        };
        let mut order: Vec<String> = self.follower_order.get(&a).cloned().unwrap_or_default();
        order.sort_by(|x, y| table[y].cmp(&table[x])); // stable: ties keep first-seen order
        let total: usize = table.values().sum();
        order
            .into_iter()
            .take(k)
            .map(|b| (b.clone(), table[&b] as f64 / total as f64))
            .collect()
    }

    fn draw(
        &self,
        table: &HashMap<String, usize>,
        rng: &mut dyn Rng,
        prev: &str,
    ) -> Option<Vec<String>> {
        if table.is_empty() {
            return Some(Vec::new());
        }
        let mut keys: Vec<&String> = table.keys().collect();
        keys.sort_by(|a, b| cluster_less(a, b));
        let size = self.alphabet_size() as f64;
        let items: Vec<(String, f64)> = keys
            .iter()
            .map(|k| {
                let count = table[*k] as f64;
                let parts = cluster_parts(k);
                let weight = match parts.first() {
                    Some(first) => count * (self.prob(prev, first) * size),
                    None => count,
                };
                ((*k).clone(), weight)
            })
            .collect();
        sample(&items, rng).map(|k| cluster_parts(&k))
    }

    fn draw_nucleus(&self, rng: &mut dyn Rng, prev: &str, stress: u8) -> String {
        let mut keys: Vec<&String> = self.nuclei.keys().filter(|k| is_vowel(k)).collect();
        keys.sort();
        let size = self.alphabet_size() as f64;
        let items: Vec<(String, f64)> = keys
            .iter()
            .map(|k| {
                (
                    (*k).clone(),
                    self.nuclei[*k] as f64 * self.prob(prev, k) * size,
                )
            })
            .collect();
        let chosen = sample(&items, rng).unwrap_or_else(|| "AH0".to_string());
        with_stress(&chosen, stress)
    }

    fn sample_syllables(&self, rng: &mut dyn Rng, lo: usize, hi: usize) -> usize {
        let mut ns: Vec<usize> = self
            .syllable_counts
            .keys()
            .copied()
            .filter(|&n| lo <= n && n <= hi)
            .collect();
        ns.sort();
        let items: Vec<(String, f64)> = ns
            .iter()
            .map(|n| (n.to_string(), self.syllable_counts[n] as f64))
            .collect();
        sample(&items, rng)
            .and_then(|s| s.parse().ok())
            .unwrap_or(lo)
    }

    /// Coins a pronounceable word from the habits (see phonotactics.py's build); `syllables` 0 means
    /// "as many as words usually have, between `min` and `max`"; `avoid` lists pronunciations not to return.
    pub fn build(
        &self,
        rng: &mut dyn Rng,
        syllables: usize,
        min: usize,
        max: usize,
        avoid: &[Entry],
        tries: usize,
    ) -> Vec<String> {
        assert!(
            self.words > 0,
            "fit the phonotactics on some pronunciations first"
        );
        let forbidden: std::collections::HashSet<String> = avoid
            .iter()
            .map(|(_, p)| strip_stress(p).join(" "))
            .collect();
        for _ in 0..tries {
            let n = if syllables > 0 {
                syllables
            } else {
                self.sample_syllables(rng, min, max)
            };
            let stressed = if n == 1 || rng.float64() < 0.6 {
                0
            } else {
                rng.rand_below(n)
            };
            let mut phones: Vec<String> = Vec::new();
            let mut ok = true;
            for i in 0..n {
                let prev = phones
                    .last()
                    .cloned()
                    .unwrap_or_else(|| BOUNDARY.to_string());
                let table = if i == 0 {
                    &self.onsets_initial
                } else {
                    &self.onsets_medial
                };
                let Some(onset) = self.draw(table, rng, &prev) else {
                    ok = false;
                    break;
                };
                phones.extend(onset);
                let prev = phones
                    .last()
                    .cloned()
                    .unwrap_or_else(|| BOUNDARY.to_string());
                let nucleus = self.draw_nucleus(rng, &prev, if i == stressed { 1 } else { 0 });
                phones.push(nucleus.clone());
                let table = if i == n - 1 {
                    &self.codas_final
                } else {
                    &self.codas_medial
                };
                let Some(coda) = self.draw(table, rng, &nucleus) else {
                    ok = false;
                    break;
                };
                phones.extend(coda);
            }
            if !ok || !well_formed(&phones) {
                continue;
            }
            if forbidden.contains(&strip_stress(&phones).join(" ")) {
                continue;
            }
            return phones;
        }
        Vec::new()
    }

    /// The best portmanteau of two words: the phones, the syllables kept of `a` and dropped of `b`.
    pub fn blend(&self, a: &[String], b: &[String]) -> Result<(Vec<String>, usize, usize), String> {
        if !a.iter().any(|p| is_vowel(p)) || !b.iter().any(|p| is_vowel(p)) {
            return Err("both words need a vowel to be blended".to_string());
        }
        let (sa, sb) = (syllabify(a, true), syllabify(b, true));
        let flat = |s: &[Syllable]| -> Vec<String> { s.iter().flat_map(|x| x.phones()).collect() };
        let mut candidates: Vec<(Vec<String>, usize, usize)> = Vec::new();
        for i in 1..=sa.len() {
            for j in 0..sb.len() {
                let mut head = flat(&sa[..i]);
                head.extend(flat(&sb[j..]));
                candidates.push((head, i, j));
                let mut head_open = flat(&sa[..i - 1]);
                head_open.extend(sa[i - 1].onset.iter().cloned());
                head_open.extend(sb[j].rime());
                head_open.extend(flat(&sb[j + 1..]));
                candidates.push((head_open, i, j));
            }
        }
        let longest = a.len().max(b.len()) as f64;
        let mut best: Option<(f64, usize)> = None;
        for (idx, (phones, i, j)) in candidates.iter().enumerate() {
            if !well_formed(phones) || phones.len() < 2 {
                continue;
            }
            let joint = self.score(phones);
            let balance = -((*i as f64) / (sa.len() as f64) - 0.5).abs()
                - (((sb.len() - j) as f64) / (sb.len() as f64) - 0.5).abs();
            let length = phones.len() as f64 / longest;
            let total = joint + 0.5 * balance + 0.2 * length;
            if best.map(|(t, _)| total > t).unwrap_or(true) {
                best = Some((total, idx));
            }
        }
        let Some((_, idx)) = best else {
            let mut out = a.to_vec();
            out.extend(b.iter().cloned());
            return Ok((out, sa.len(), 0));
        };
        let (mut phones, i, j) = candidates[idx].clone();
        let primaries = phones.iter().filter(|p| p.ends_with('1')).count();
        if primaries > 1 {
            let mut seen = false;
            for p in phones.iter_mut() {
                if p.ends_with('1') {
                    if seen {
                        *p = with_stress(p, 2);
                    }
                    seen = true;
                }
            }
        }
        if primaries == 0 {
            if let Some(p) = phones.iter_mut().find(|p| is_vowel(p)) {
                *p = with_stress(p, 1);
            }
        }
        Ok((phones, i, j))
    }

    pub fn describe(&self) -> String {
        format!(
            "Phonotactics(words={}, pairs={}, stress={})",
            self.words,
            self.bigrams.len(),
            self.stress
        )
    }
}

/// Could an English speaker say this: every onset legal, every coda falling, a vowel somewhere?
pub fn well_formed(phones: &[String]) -> bool {
    if phones.is_empty() || !phones.iter().any(|p| is_vowel(p)) {
        return false;
    }
    syllabify(phones, true)
        .iter()
        .all(|s| is_legal_onset(&s.onset) && is_legal_coda(&s.coda))
}

/// What is wrong with a sequence, in words; empty when nothing is.
pub fn violations(phones: &[String]) -> Vec<String> {
    if phones.is_empty() {
        return vec!["empty".to_string()];
    }
    let mut out = Vec::new();
    if !phones.iter().any(|p| is_vowel(p)) {
        out.push("no vowel".to_string());
    }
    for s in syllabify(phones, true) {
        if !is_legal_onset(&s.onset) {
            out.push(format!("{} cannot start a syllable", s.onset.join(" ")));
        }
        if !is_legal_coda(&s.coda) {
            out.push(format!("{} cannot end a syllable", s.coda.join(" ")));
        }
    }
    out
}
