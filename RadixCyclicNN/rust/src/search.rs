//! Shortest-path prediction: what a walk may take, how a walk is ranked, and
//! the stochastic walk.

use crate::fsum::fsum;
use crate::graph::{Graph, BACK, END, START};
use crate::mt19937::Mt19937;
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
pub fn parse_traversal(name: &str) -> Result<Traversal, String> {
    match name.trim().to_ascii_lowercase().as_str() {
        "" | "reward" | "rewards" | "cost" => Ok(Traversal::Reward),
        "least-punished" | "least_punished" | "leastpunished" | "punished" | "punish" | "blame" => {
            Ok(Traversal::LeastPunished)
        }
        other => Err(format!(
            "unknown traversal {other:?}; expected 'reward' or 'least-punished'"
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
pub fn onward(costs: &mut Vec<ChildCost>) {
    let mut back = f64::INFINITY;
    let mut has_back = false;
    for it in costs.iter() {
        if it.child == BACK && (!has_back || it.cost < back) {
            back = it.cost;
            has_back = true;
        }
    }
    if !has_back {
        return;
    }
    let hand_over = !costs.iter().any(|it| it.child != BACK && it.cost < back);
    if hand_over {
        costs.clear();
        return;
    }
    costs.retain(|it| it.child != BACK);
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
/// matched trigram); `START` and `END` emit nothing.
pub(crate) fn start_emission(g: &Graph, start_node: usize, start_offset: usize) -> Result<usize, String> {
    if start_node >= g.num_node_ids() || !g.is_alive(start_node) {
        return Err(format!("start node {start_node} is not alive"));
    }
    if start_node == START || start_node == END {
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
    let ctx = include_context.unwrap_or(start_node == START);
    let offset = if start_node == START || start_node == END {
        0
    } else {
        start_offset
    };
    let real: Vec<&str> = node_ids
        .iter()
        .enumerate()
        .filter(|(_, &n)| n != START && n != END)
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
        traversal: Traversal,
    ) -> Result<PathResult, String> {
        self.prepare();
        match rng {
            Some(rng) => self.walk(
                start_node,
                start_offset,
                max_chars,
                temperature,
                rng,
                include_context,
                traversal,
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
                    traversal,
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
        traversal: Traversal,
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
        let mut came_from = if start_node == START { Some(START) } else { None };
        let mut costs: Vec<ChildCost> = Vec::new();
        let mut weights: Vec<f64> = Vec::new();
        loop {
            if node == END || max_chars.is_some_and(|cap| chars >= cap) {
                break;
            }
            // a node the model expects to go round offers nothing
            self.child_costs_into(node, came_from, &mut costs);
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
