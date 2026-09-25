//! Shortest-path prediction: what a walk may take, how a walk is ranked, and
//! the stochastic walk.

use crate::fsum::fsum;
use crate::graph::{is_origin, Graph, BACK, END, FIRST, THINK};
use crate::mt19937::Mt19937;
use crate::penalty::PenaltyCosts;
use crate::weights::ChildCost;

/// How a walk chooses its way through the graph.
///
/// The model has always searched by what went *right*: an edge's weight is a
/// function of how often the corpus traversed it and of the reward it
/// collected, and the cheapest path is the likeliest one.
/// [`Traversal::LeastPunished`] searches by what went *wrong* instead - it ranks
/// a step by the blame on it and lets the ordinary cost decide only between
/// steps nothing is held against.  See `../../SPEC-LeastPunished.md`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Traversal {
    /// The original search: `-log` softmax of the dual frequency weight.
    #[default]
    Reward,
    /// Punishment first (the edge's penalty plus what the walks through its
    /// context got wrong), cost only to break ties; a whole path is ranked by
    /// its *worst* step.
    LeastPunished,
}

impl Traversal {
    /// The traversal's name as the CLI and the API spell it.
    pub fn name(self) -> &'static str {
        match self {
            Traversal::Reward => "reward",
            Traversal::LeastPunished => "least-punished",
        }
    }
}

impl std::fmt::Display for Traversal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// Reads a traversal name: `""`, `"reward"` or `"rewards"` for the original
/// search, `"least-punished"` (also `"least_punished"`, `"punished"`,
/// `"punish"`, `"blame"`) for the one that follows the blame.
/// Reads a traversal name as a *ranking*.
///
/// `"punishment"` is the other punishment traversal ([`crate::penalty`]), which
/// prices a step rather than ordering the walks: the beams rank it exactly as
/// they rank a rewarded one, so it reads as [`Traversal::Reward`] here and is
/// never an alias of [`Traversal::LeastPunished`] - the two are different
/// currencies.
pub fn parse_traversal(name: &str) -> Result<Traversal, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "reward" | "rewards" | "cost" | "punishment" | "penalty" => Ok(Traversal::Reward),
        "least-punished" | "least_punished" | "leastpunished" | "blame" => Ok(Traversal::LeastPunished),
        other => Err(format!(
            "unknown traversal {other:?}; expected 'reward', 'punishment' or 'least-punished'"
        )),
    }
}

/// How close two punishments have to be to count as equal: the same penalty
/// applied in a different order can land a bit or two apart, and a walk is not
/// "more punished" for that.
pub const PUNISH_TOLERANCE: f64 = 1e-12;

/// Cuts `costs` down to the children a walk may actually take, given what the
/// model has learned about going round.
///
/// `BACK` is not a continuation - it emits nothing and no text passes through
/// it - so it never appears in a path.  But it *competes* with the real
/// children for probability, and when it is the cheapest of them the model's
/// most likely next step at this node is to stop rather than carry on: the walk
/// hands over, which here means the branch offers nothing and the search goes
/// on with its others.
///
/// `THINK` is not a continuation either, and the walk never takes it: stopping
/// to think is not stopping, so it is dropped from the options and the real
/// children stay on offer ([`crate::thinking`]).
pub fn onward(costs: &mut Vec<ChildCost>) {
    let mut back = f64::INFINITY;
    let mut has_back = false;
    let mut has_think = false;
    for it in costs.iter() {
        if it.child == BACK && (!has_back || it.cost < back) {
            back = it.cost;
            has_back = true;
        }
        if it.child == THINK {
            has_think = true;
        }
    }
    if !has_back {
        if has_think {
            costs.retain(|it| it.child != THINK);
        }
        return;
    }
    let hand_over = !costs
        .iter()
        .any(|it| it.child != BACK && it.child != THINK && it.cost < back);
    if hand_over {
        costs.clear();
        return;
    }
    costs.retain(|it| it.child != BACK && it.child != THINK);
}

/// Cuts `costs` down to the children the model has the least against - the
/// whole of the change, in one function.
///
/// Where nothing at this node was ever punished (every step of an untutored
/// graph) every child ties at zero and the list comes back untouched, so the
/// walk costs exactly what it always did; where something *was* punished, the
/// steps that carry more blame than the cleanest one here are not options any
/// more, however well rewarded they are.
pub fn least_punished(costs: &mut Vec<ChildCost>) {
    if costs.len() < 2 {
        return;
    }
    let first = costs[0].punish;
    let mut low = first;
    let mut same = true;
    for it in &costs[1..] {
        if it.punish != first {
            same = false;
        }
        if it.punish < low {
            low = it.punish;
        }
    }
    if same {
        return;
    }
    let keep = low + PUNISH_TOLERANCE;
    costs.retain(|it| it.punish <= keep);
}

/// The outcome of a prediction / generation walk.
#[derive(Clone, Debug, Default)]
pub struct PathResult {
    /// The emitted continuation.
    pub text: String,
    /// The path's labels, sentinels included.
    pub labels: Vec<String>,
    pub node_ids: Vec<usize>,
    /// The summed edge cost.
    pub cost: f64,
    pub step_costs: Vec<f64>,
    /// The search effort.
    pub expanded: usize,
    pub reached_end: bool,
    /// Prefix + text.
    pub full_text: String,
    /// The walk's worst step under the least-punished traversal (0 everywhere
    /// nothing was ever punished).
    pub punish: f64,
}

impl PathResult {
    /// `exp(-cost)` (0 for an infinite cost).
    pub fn probability(&self) -> f64 {
        if self.cost.is_infinite() || self.cost.is_nan() {
            return 0.0;
        }
        (-self.cost).exp()
    }
}

/// The number of characters the start node emits (its remainder after the
/// matched trigram); the sentinels emit nothing.
pub(crate) fn start_emission(g: &Graph, start_node: usize, start_offset: usize) -> Result<usize, String> {
    if start_node >= g.num_node_ids() || !g.is_alive(start_node) {
        return Err(format!("start node {start_node} is not alive"));
    }
    if start_node < FIRST {
        return Ok(0);
    }
    let len = g.label_len(start_node);
    if start_offset + g.enc.n > len {
        return Err(format!(
            "start_offset {start_offset} out of range for label {:?}",
            g.label(start_node)
        ));
    }
    Ok(len - (start_offset + g.enc.n))
}

/// Decodes a node path into a [`PathResult`].  `include_context` `None`
/// defaults to "true from `START`, false otherwise"; `max_chars` `None` means
/// no cap.
pub(crate) fn build_result(
    g: &Graph,
    node_ids: Vec<usize>,
    step_costs: Vec<f64>,
    start_offset: usize,
    max_chars: Option<usize>,
    expanded: usize,
    include_context: Option<bool>,
) -> PathResult {
    let labels: Vec<String> = node_ids.iter().map(|&n| g.label(n).to_string()).collect();
    let start_node = node_ids[0];
    // from START (or THINK) there is no matched context to strip: the first node in full
    let ctx = include_context.unwrap_or(is_origin(start_node));
    let offset = if start_node < FIRST { 0 } else { start_offset };
    // sentinels are stripped by id: a real node may carry the label "<s>" or "</s>"
    let real: Vec<&str> = node_ids
        .iter()
        .enumerate()
        .filter(|(_, &n)| n >= FIRST)
        .map(|(i, _)| labels[i].as_str())
        .collect();
    let mut text = g.enc.decode_path(&real, offset, ctx);
    if let Some(cap) = max_chars {
        text = g.enc.truncate(&text, cap);
    }
    let cost = fsum(&step_costs);
    let reached_end = *node_ids.last().unwrap() == END;
    PathResult {
        text,
        labels,
        node_ids,
        cost,
        step_costs,
        expanded,
        reached_end,
        full_text: String::new(),
        punish: 0.0,
    }
}

impl Graph {
    /// A stochastic walk sampling each child from `softmax(-cost / temperature)`;
    /// temperature 0 is greedy.  `max_chars` `None` means no limit, `rng` `None`
    /// uses the graph's own generator.
    ///
    /// Under [`Traversal::LeastPunished`] only the least punished children are
    /// on offer at every node and the sampling then runs exactly as before:
    /// punishment decides *what* may be walked, the cost decides which of those
    /// it is.
    #[allow(clippy::too_many_arguments)] // the walk's knobs, one per knob
    pub fn sample_walk(
        &mut self,
        start_node: usize,
        start_offset: usize,
        max_chars: Option<usize>,
        temperature: f64,
        rng: Option<&mut Mt19937>,
        include_context: Option<bool>,
        costs: Option<&PenaltyCosts>,
        traversal: Traversal,
    ) -> Result<PathResult, String> {
        self.sample_walk_filtered(
            start_node,
            start_offset,
            max_chars,
            temperature,
            rng,
            include_context,
            costs,
            traversal,
            SamplingFilter::default(),
        )
    }

    /// [`Graph::sample_walk`] with a [`SamplingFilter`]: every step draws from
    /// what the filter keeps, with the one random number it always drew - off,
    /// the walk is draw for draw the one it always was.
    #[allow(clippy::too_many_arguments)] // the walk's knobs, one per knob
    pub fn sample_walk_filtered(
        &mut self,
        start_node: usize,
        start_offset: usize,
        max_chars: Option<usize>,
        temperature: f64,
        rng: Option<&mut Mt19937>,
        include_context: Option<bool>,
        costs: Option<&PenaltyCosts>,
        traversal: Traversal,
        filter: SamplingFilter,
    ) -> Result<PathResult, String> {
        filter.check()?;
        self.prepare();
        match rng {
            Some(rng) => self.walk(
                start_node,
                start_offset,
                max_chars,
                temperature,
                rng,
                include_context,
                costs,
                traversal,
                filter,
            ),
            None => {
                // the graph's own generator, lent to the walk and put back
                let mut own = std::mem::replace(&mut self.rng, Mt19937::new(0));
                let walk = self.walk(
                    start_node,
                    start_offset,
                    max_chars,
                    temperature,
                    &mut own,
                    include_context,
                    costs,
                    traversal,
                    filter,
                );
                self.rng = own;
                walk
            }
        }
    }

    /// One stochastic walk over a prepared graph.
    #[allow(clippy::too_many_arguments)]
    fn walk(
        &self,
        start_node: usize,
        start_offset: usize,
        max_chars: Option<usize>,
        temperature: f64,
        rng: &mut Mt19937,
        include_context: Option<bool>,
        walk_costs: Option<&PenaltyCosts>,
        traversal: Traversal,
        filter: SamplingFilter,
    ) -> Result<PathResult, String> {
        if temperature < 0.0 {
            return Err("temperature must be >= 0".to_string());
        }
        let mut chars = start_emission(self, start_node, start_offset)?;
        let mut node = start_node;
        let mut node_ids = vec![node];
        let mut step_costs: Vec<f64> = Vec::new();
        let mut punish = 0.0f64;
        let mut steps = 0;
        let mut came_from = if is_origin(start_node) { Some(start_node) } else { None };
        let mut costs: Vec<ChildCost> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        loop {
            if node == END || max_chars.is_some_and(|cap| chars >= cap) {
                break;
            }
            // a node the model expects to go round offers nothing
            self.step_costs_into(node, came_from, walk_costs, &mut costs);
            onward(&mut costs);
            if traversal == Traversal::LeastPunished {
                least_punished(&mut costs);
            }
            if costs.is_empty() {
                break;
            }
            let pick = if temperature == 0.0 || costs.len() == 1 {
                let mut pick = costs[0];
                for &item in &costs[1..] {
                    if item.cost < pick.cost {
                        pick = item;
                    }
                }
                pick
            } else {
                // the draw below still runs, however few options are left: one random number per step
                filter.filter(&mut costs, temperature);
                let inv_t = 1.0 / temperature;
                let lowest = costs.iter().map(|c| c.cost).fold(f64::INFINITY, f64::min);
                weights.clear();
                weights.extend(costs.iter().map(|item| (-(item.cost - lowest) * inv_t).exp()));
                let r = rng.next_f64() * fsum(&weights);
                let mut pick = costs[costs.len() - 1];
                let mut acc = 0.0;
                for (i, &item) in costs.iter().enumerate() {
                    acc += weights[i];
                    if r < acc {
                        pick = item;
                        break;
                    }
                }
                pick
            };
            step_costs.push(pick.cost);
            if pick.punish > punish {
                punish = pick.punish; // a walk is as punished as its worst step
            }
            node_ids.push(pick.child);
            if pick.child != END {
                chars += self.label_len(pick.child) - self.enc.overlap();
            }
            came_from = Some(node);
            node = pick.child;
            steps += 1;
        }
        let mut walk = build_result(
            self,
            node_ids,
            step_costs,
            start_offset,
            max_chars,
            steps,
            include_context,
        );
        walk.punish = punish;
        Ok(walk)
    }
}

/// What every step of a stochastic walk draws from (`../../SPEC-SearchAndTraining.md`
/// section 1).  The default is off: `top_k` 0 keeps every option, `top_p` 1
/// the whole mass, `min_p` 0 every weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingFilter {
    /// The `top_k` cheapest options (0 = off).
    pub top_k: usize,
    /// Nucleus: the smallest set of cheapest options holding `top_p` of the mass (1 = off).
    pub top_p: f64,
    /// The options at least `min_p` times as likely as the best (0 = off).
    pub min_p: f64,
}

impl Default for SamplingFilter {
    fn default() -> SamplingFilter {
        SamplingFilter {
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
        }
    }
}

impl SamplingFilter {
    /// Whether it keeps anything less than every option.
    pub fn active(&self) -> bool {
        self.top_k > 0 || self.top_p < 1.0 || self.min_p > 0.0
    }

    /// The error for a filter out of range: `top_p` outside `(0, 1]`, `min_p` outside `[0, 1)`.
    pub fn check(&self) -> Result<(), String> {
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(format!("top_p must lie in (0, 1], got {}", self.top_p));
        }
        if !(self.min_p >= 0.0 && self.min_p < 1.0) {
            return Err(format!("min_p must lie in [0, 1), got {}", self.min_p));
        }
        Ok(())
    }

    /// Narrows `options` in place to what a stochastic step may draw from, in
    /// the order the node offered them.  Options rank by `(cost, position)`;
    /// `top_k`, then `min_p`, then `top_p`; the cheapest always survives, so
    /// the weights the draw then uses are the numbers it would have used
    /// unfiltered.
    pub fn filter(&self, options: &mut Vec<ChildCost>, temperature: f64) {
        self.filter_by(options, temperature, |o| o.cost);
    }

    /// [`SamplingFilter::filter`] over any list of options, `cost` reading an
    /// option's cost - the phase walk's steps carry a phase beside theirs.
    pub fn filter_by<T>(&self, options: &mut Vec<T>, temperature: f64, cost: impl Fn(&T) -> f64) {
        let n = options.len();
        if n <= 1 || temperature == 0.0 || !self.active() {
            return;
        }
        let costs: Vec<f64> = options.iter().map(cost).collect();
        let mut ranked: Vec<usize> = (0..n).collect();
        // IEEE order, not `total_cmp`: -0 and 0 are one cost here, as they are in Python and Go
        ranked.sort_by(|&a, &b| {
            costs[a]
                .partial_cmp(&costs[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if self.top_k > 0 && self.top_k < n {
            ranked.truncate(self.top_k);
        }
        let lowest = costs[ranked[0]];
        let inv_t = 1.0 / temperature;
        let mut weight = vec![0.0f64; n];
        for &i in &ranked {
            weight[i] = (-(costs[i] - lowest) * inv_t).exp();
        }
        if self.min_p > 0.0 {
            ranked.retain(|&i| weight[i] >= self.min_p);
        }
        if self.top_p < 1.0 {
            let ws: Vec<f64> = ranked.iter().map(|&i| weight[i]).collect();
            let total = fsum(&ws);
            let mut acc = 0.0;
            let mut kept = 0;
            for &i in &ranked {
                kept += 1;
                acc += weight[i];
                if acc >= self.top_p * total {
                    break;
                }
            }
            ranked.truncate(kept);
        }
        if ranked.len() == n {
            return;
        }
        let mut keep = vec![false; n];
        for &i in &ranked {
            keep[i] = true;
        }
        let mut at = 0;
        options.retain(|_| {
            let kept = keep[at];
            at += 1;
            kept
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `(child, edge, cost)` options, child = position, as a node offers them.
    fn options(costs: &[f64]) -> Vec<ChildCost> {
        costs
            .iter()
            .enumerate()
            .map(|(i, &cost)| ChildCost {
                child: i,
                edge: 100 + i,
                cost,
                punish: 0.0,
            })
            .collect()
    }

    fn kept(costs: &[f64], temperature: f64, top_k: usize, top_p: f64, min_p: f64) -> Vec<usize> {
        let mut o = options(costs);
        SamplingFilter { top_k, top_p, min_p }.filter(&mut o, temperature);
        o.iter().map(|c| c.child).collect()
    }

    #[test]
    fn off_or_greedy_keeps_every_option() {
        assert_eq!(kept(&[0.5, 0.1, 2.0], 1.0, 0, 1.0, 0.0), vec![0, 1, 2]);
        assert_eq!(kept(&[0.5, 0.1, 2.0], 0.0, 1, 1.0, 0.0), vec![0, 1, 2]);
    }

    #[test]
    fn the_filters_keep_what_python_keeps() {
        // the cases of tests/test_search_training.py, number for number
        assert_eq!(kept(&[0.5, 0.1, 2.0, 0.3], 1.0, 2, 1.0, 0.0), vec![1, 3]);
        assert_eq!(kept(&[0.2, 0.2, 0.2], 1.0, 2, 1.0, 0.0), vec![0, 1]);
        assert_eq!(kept(&[0.1, 0.5, 2.0], 1.0, 0, 1.0, 0.5), vec![0, 1]);
        assert_eq!(kept(&[0.1, 0.5, 2.0], 1.0, 0, 1.0, 0.9), vec![0]);
        assert_eq!(kept(&[0.1, 0.5, 2.0], 10.0, 0, 1.0, 0.5), vec![0, 1, 2]);
        assert_eq!(kept(&[0.0; 4], 1.0, 0, 0.5, 0.0).len(), 2);
        assert_eq!(kept(&[0.0; 4], 1.0, 0, 0.51, 0.0).len(), 3);
        assert_eq!(kept(&[0.0, 0.1, 0.2, 5.0, 6.0], 1.0, 4, 0.4, 0.5), vec![0, 1]);
        for (top_k, top_p, min_p) in [(1, 1.0, 0.0), (0, 1e-9, 0.0), (0, 1.0, 0.999)] {
            assert_eq!(kept(&[3.0, 0.7, 1.0], 1.0, top_k, top_p, min_p), vec![1]);
        }
    }

    #[test]
    fn out_of_range_is_refused() {
        let bad = |top_p: f64, min_p: f64| SamplingFilter { top_k: 0, top_p, min_p }.check().is_err();
        assert!(bad(0.0, 0.0) && bad(1.5, 0.0) && bad(f64::NAN, 0.0) && bad(1.0, 1.0) && bad(1.0, -0.1));
        assert!(!bad(0.5, 0.99));
    }
}
