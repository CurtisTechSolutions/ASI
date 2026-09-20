//! The model file: `radixnet-count`, the format Python and Go read and write.
//!
//! This is the contract between the three implementations, so it is a port of
//! `CountRewardNet.to_dict` / `from_dict` and of the graph document under it
//! (`radixnet/countnet.py`, `radixnet/graph.py`; `go/radixnet/json.go` is the
//! same thing in Go) rather than anything this port invented:
//!
//! * dead nodes and edges are compacted away and the ids remapped, so two
//!   models that were trained the same way produce the same file whatever
//!   splitting and merging they did on the way;
//! * the edges come out in node order and, within a node, in the order that
//!   node's children were first seen - which is the order the softmax adds them
//!   up in, and therefore part of the numbers;
//! * the Mersenne Twister's state rides along, so training continues on the
//!   other side exactly where it stopped;
//! * every counter is written as its odometer reading plus, only once something
//!   has actually wrapped, its reset count.
//!
//! A file is gzipped when the path ends with `.gz` (see [`crate::gzip`]), and
//! written through a temporary file and a rename, so an interrupted save never
//! leaves half a model behind.

use std::sync::atomic::Ordering;

use crate::clock::utc_now;
use crate::counter::Counter;
use crate::encoding::{Encoding, Unit, BACK_LABEL, END_LABEL, START_LABEL};
use crate::graph::{Graph, GraphOptions, Loc, BACK, END, FIRST, START};
use crate::json::{parse, Json};
use crate::model::{EpochRecord, Model};
use crate::weights::SMOOTHING;

/// The model file format shared with the Python and Go implementations.
pub const MODEL_FORMAT: &str = "radixnet-count";
/// The model document version this port writes.
pub const MODEL_FORMAT_VERSION: i64 = 1;
/// The encoding a graph document carries.
///
/// A document without an `encoding` block is the character trigram of stride 1,
/// which is what every file was before the encoding became a choice, so an old
/// file reads exactly as it always did.
fn read_encoding(doc: &Json) -> Result<Encoding, String> {
    let Some(block) = doc.get("encoding") else {
        return Ok(Encoding::default());
    };
    let name = block.at("unit").as_str().unwrap_or("char");
    let unit = Unit::parse(name).ok_or_else(|| format!("unknown graph unit {name:?}"))?;
    let n = block.at("n").as_i64().unwrap_or(crate::encoding::WINDOW as i64);
    let stride = block.at("stride").as_i64().unwrap_or(1);
    if n < 1 || stride < 1 {
        return Err(format!("graph encoding {name}:{n}:{stride} is not usable"));
    }
    Encoding::new(unit, n as usize, stride as usize)
}

/// The graph document format.
pub const GRAPH_FORMAT: &str = "radixnet-graph";
/// 3 added the `BACK` sentinel; older documents gain an unvisited one on load.
pub const GRAPH_FORMAT_VERSION: i64 = 3;

/// The sine model's defaults, written so its reader finds what it expects: the
/// count model's activation is the constant 1 (`a = 0`, `k = 1`), which is why
/// none of these numbers affect anything here.
const DEFAULT_B: f64 = 1.0 / 3.0;
const DEFAULT_H: f64 = 0.0;
/// The `BACK` sentinel's state, fixed at the far edge of the range the sine
/// model draws from rather than drawn from it.
const BACK_Z: f64 = 4.5;

impl Graph {
    /// The graph as a `radixnet-graph` document.
    pub fn to_doc(&mut self) -> Json {
        self.prepare();
        self.carry_counters(false); // a saved file always holds a wrapped reading

        let order: Vec<usize> = (0..self.labels.len()).filter(|&i| self.alive[i]).collect();
        let mut remap = vec![usize::MAX; self.labels.len()];
        for (new, &old) in order.iter().enumerate() {
            remap[old] = new;
        }

        let mut labels = Vec::with_capacity(order.len());
        let mut z = Vec::with_capacity(order.len());
        let mut count = Vec::with_capacity(order.len());
        let mut count_resets = Vec::with_capacity(order.len());
        for &old in &order {
            labels.push(self.labels[old].clone());
            z.push(if old == BACK { BACK_Z } else { 0.0 });
            count.push(self.count[old].load(Ordering::Relaxed));
            count_resets.push(self.count_resets.get(&old).copied().unwrap_or(0));
        }
        let n = order.len();
        let mut nodes = vec![
            ("labels".to_string(), Json::strs(labels)),
            ("z".to_string(), Json::nums(z)),
            ("a".to_string(), Json::nums(vec![0.0; n])),
            ("b".to_string(), Json::nums(vec![DEFAULT_B; n])),
            ("h".to_string(), Json::nums(vec![DEFAULT_H; n])),
            ("k".to_string(), Json::nums(vec![1.0; n])),
            ("count".to_string(), Json::ints(count)),
        ];
        // the reset counts ride along only once something has actually wrapped
        if count_resets.iter().any(|&r| r != 0) {
            nodes.push(("count_resets".to_string(), Json::ints(count_resets)));
        }

        let mut src = Vec::new();
        let mut dst = Vec::new();
        let mut w = Vec::new();
        let mut edge_count = Vec::new();
        let mut edge_resets = Vec::new();
        let mut reward = Vec::new();
        let mut edge_index = vec![usize::MAX; self.edge_w.len()];
        for &old in &order {
            let adj = &self.children[old];
            for (i, &c) in adj.order.iter().enumerate() {
                let e = adj.edges[i];
                edge_index[e] = src.len();
                src.push(remap[old] as i64);
                dst.push(remap[c] as i64);
                w.push(self.edge_w[e]);
                edge_count.push(self.edge_count[e].load(Ordering::Relaxed));
                edge_resets.push(self.edge_count_resets.get(&e).copied().unwrap_or(0));
                reward.push(self.edge_reward[e]);
            }
        }
        let mut edges = vec![
            ("src".to_string(), Json::ints(src)),
            ("dst".to_string(), Json::ints(dst)),
            ("w".to_string(), Json::nums(w)),
            ("count".to_string(), Json::ints(edge_count)),
        ];
        if edge_resets.iter().any(|&r| r != 0) {
            edges.push(("count_resets".to_string(), Json::ints(edge_resets)));
        }
        edges.push(("reward".to_string(), Json::nums(reward)));

        let (words, index) = self.rng.state();
        let mut state: Vec<Json> = words.iter().map(|&x| Json::Int(x as i64)).collect();
        state.push(Json::Int(index as i64));
        let rng_state = Json::Arr(vec![Json::Int(3), Json::Arr(state), Json::Null]);

        let events: Vec<i64> = self.window[self.window_head..]
            .iter()
            .filter_map(|&e| match edge_index.get(e) {
                Some(&at) if at != usize::MAX => Some(at as i64),
                _ => None,
            })
            .collect();
        let weights = Json::obj([
            ("function", Json::str("dual-frequency")),
            ("count_scale", Json::Num(self.count_scale)),
            ("global_scale", Json::Num(self.global_scale)),
            ("window_scale", Json::Num(self.window_scale)),
            ("reward_scale", Json::Num(self.reward_scale)),
            ("path_scale", Json::Num(self.path_scale)),
            ("window", Json::Int(self.window_size as i64)),
            ("smoothing", Json::Num(SMOOTHING)),
            ("kind", Json::str("count-reward")),
            ("total_traversals", Json::Int(self.total_traversals().value)),
            ("total_traversals_resets", Json::Int(self.total_traversals().resets)),
            ("window_events", Json::ints(events)),
        ]);

        // the three implementations keep their tables in different orders; the file has one
        let mut rows: Vec<(i64, i64, i64, i64, i64)> = self
            .paths
            .iter()
            .filter_map(|(key, row)| {
                let at = *edge_index.get(key.edge)?;
                let prev = *remap.get(key.prev)?;
                if at == usize::MAX || prev == usize::MAX {
                    return None;
                }
                Some((prev as i64, at as i64, row.seen, row.correct, row.incorrect))
            })
            .collect();
        rows.sort_unstable();
        let paths = Json::obj([
            ("prev", Json::ints(rows.iter().map(|r| r.0))),
            ("edge", Json::ints(rows.iter().map(|r| r.1))),
            ("seen", Json::ints(rows.iter().map(|r| r.2))),
            ("correct", Json::ints(rows.iter().map(|r| r.3))),
            ("incorrect", Json::ints(rows.iter().map(|r| r.4))),
        ]);

        let graph = Json::obj([
            ("format", Json::str(GRAPH_FORMAT)),
            ("format_version", Json::Int(GRAPH_FORMAT_VERSION)),
            ("seed", Json::Int(self.seed)),
            ("inverted", Json::Bool(self.inverted)),
            ("version", Json::Int(self.version.value)),
            ("version_resets", Json::Int(self.version.resets)),
            ("structure_version", Json::Int(self.structure_version.value)),
            ("structure_version_resets", Json::Int(self.structure_version.resets)),
            ("traversals", Json::Int(self.traversal_total().value)),
            ("traversals_resets", Json::Int(self.traversal_total().resets)),
            ("nodes", Json::Obj(nodes)),
            ("edges", Json::Obj(edges)),
            ("rng_state", rng_state),
            ("weights", weights),
            ("paths", paths),
        ]);
        if self.enc.is_default() {
            return graph; // an ordinary file is byte for byte what it always was
        }
        // the encoding sits third, right after format_version, which is exactly
        // where Python and Go write it: the document is one document
        let mut pairs = match graph {
            Json::Obj(pairs) => pairs,
            other => return other,
        };
        pairs.insert(
            2,
            (
                "encoding".to_string(),
                Json::obj([
                    ("unit", Json::str(self.enc.unit.name())),
                    ("n", Json::Int(self.enc.n as i64)),
                    ("stride", Json::Int(self.enc.stride as i64)),
                ]),
            ),
        );
        Json::Obj(pairs)
    }

    /// Rebuilds a graph from a `radixnet-graph` document, upgrading one written
    /// before the `BACK` sentinel existed.
    pub fn from_doc(doc: &Json) -> Result<Graph, String> {
        if doc.at("format").as_str() != Some(GRAPH_FORMAT) {
            return Err(format!("not a {GRAPH_FORMAT} document"));
        }
        let upgraded;
        let doc = if doc.at("format_version").as_i64().unwrap_or(1) >= GRAPH_FORMAT_VERSION {
            doc
        } else {
            upgraded = with_back(doc);
            &upgraded
        };
        let nodes = doc.at("nodes");
        let edges = doc.at("edges");
        let labels = nodes.at("labels").to_strings();
        let n = labels.len();
        if n < FIRST || labels[START] != START_LABEL || labels[END] != END_LABEL || labels[BACK] != BACK_LABEL {
            return Err("graph document is missing the START/END/BACK sentinels".to_string());
        }
        let weights = doc.at("weights");
        let window = weights.at("window").as_i64().unwrap_or(10_000).max(1) as usize;
        // files written before the dual frequency function carried no global_scale
        let legacy = weights.get("global_scale").is_none();
        let opts = GraphOptions {
            count_scale: weights
                .at("count_scale")
                .as_f64()
                .unwrap_or(if legacy { 1.0 } else { 0.0 }),
            reward_scale: weights.at("reward_scale").as_f64().unwrap_or(1.0),
            global_scale: weights
                .at("global_scale")
                .as_f64()
                .unwrap_or(if legacy { 0.0 } else { 0.5 }),
            window_scale: weights
                .at("window_scale")
                .as_f64()
                .unwrap_or(if legacy { 0.0 } else { 0.5 }),
            path_scale: weights.at("path_scale").as_f64().unwrap_or(1.0),
            window,
            encoding: read_encoding(doc)?,
        };
        let mut g = Graph::new(doc.at("seed").as_i64().unwrap_or(0), opts)?;
        g.inverted = doc.at("inverted").as_bool().unwrap_or(false);

        // the three sentinels are already there; the rest of the file's nodes follow
        let counts = nodes.at("count").to_i64s();
        let node_resets = nodes.at("count_resets").to_i64s();
        if counts.len() != n || (!node_resets.is_empty() && node_resets.len() != n) {
            return Err("node arrays have inconsistent lengths".to_string());
        }
        g.reset_nodes(&labels, &counts, &node_resets)?;

        let src = edges.at("src").to_i64s();
        let dst = edges.at("dst").to_i64s();
        let w = edges.at("w").to_f64s();
        let edge_count = edges.at("count").to_i64s();
        let edge_resets = edges.at("count_resets").to_i64s();
        let reward = edges.at("reward").to_f64s();
        if src.len() != dst.len() || w.len() != src.len() || edge_count.len() != src.len() {
            return Err("edge arrays have inconsistent lengths".to_string());
        }
        if !reward.is_empty() && reward.len() != src.len() {
            return Err("edge reward array has an inconsistent length".to_string());
        }
        if !edge_resets.is_empty() && edge_resets.len() != src.len() {
            return Err("edge arrays have inconsistent lengths".to_string());
        }
        for i in 0..src.len() {
            let (p, c) = (src[i] as usize, dst[i] as usize);
            if p >= n || c >= n || g.edge(p, c).is_some() {
                return Err(format!("invalid or duplicate edge {p} -> {c}"));
            }
            let e = g.new_edge(p, c, edge_count[i], edge_resets.get(i).copied().unwrap_or(0));
            g.edge_w[e] = w[i];
            g.edge_reward[e] = reward.get(i).copied().unwrap_or(0.0);
        }

        let state = doc.at("rng_state");
        if !state.is_null() {
            let words = state.as_array().get(1).map(|v| v.to_i64s()).unwrap_or_default();
            if words.len() >= 2 {
                let index = *words.last().expect("checked") as usize;
                let words: Vec<u32> = words[..words.len() - 1].iter().map(|&x| x as u32).collect();
                g.rng.set_state(&words, index)?;
            }
        }
        g.version = Counter::new(
            doc.at("version").as_i64().unwrap_or(0),
            doc.at("version_resets").as_i64().unwrap_or(0),
        );
        g.structure_version = Counter::new(
            doc.at("structure_version").as_i64().unwrap_or(0),
            doc.at("structure_version_resets").as_i64().unwrap_or(0),
        );
        g.set_traversals(match doc.get("traversals") {
            Some(v) => Counter::new(
                v.as_i64().unwrap_or(0),
                doc.at("traversals_resets").as_i64().unwrap_or(0),
            ),
            // a format 1 file counted into unbounded integers: their sum bounds every one of them
            None => Counter::new(counts.iter().sum::<i64>() + edge_count.iter().sum::<i64>(), 0),
        });
        g.set_total_traversals(Counter::new(
            weights.at("total_traversals").as_i64().unwrap_or(0),
            weights.at("total_traversals_resets").as_i64().unwrap_or(0),
        ));
        for e in weights.at("window_events").to_i64s() {
            g.replay_window_event(e as usize);
        }
        let paths = doc.at("paths");
        let (prevs, path_edges) = (paths.at("prev").to_i64s(), paths.at("edge").to_i64s());
        let seen = paths.at("seen").to_i64s();
        let correct = paths.at("correct").to_i64s();
        let incorrect = paths.at("incorrect").to_i64s();
        for (i, (&prev, &edge)) in prevs.iter().zip(path_edges.iter()).enumerate() {
            g.load_path(
                prev as usize,
                edge as usize,
                seen.get(i).copied().unwrap_or(0),
                correct.get(i).copied().unwrap_or(0),
                incorrect.get(i).copied().unwrap_or(0),
            );
        }
        g.carry_counters(true); // normalise whatever the file carried, however it was written
        g.invalidate();
        g.recompute_weights();
        Ok(g)
    }
}

/// A graph document with the `BACK` sentinel in it: anything older than format
/// 3 gains an unvisited one at [`BACK`], and every node id from there up shifts
/// by one.  A model that has never caught itself repeating has nothing to say
/// about where it goes round.
fn with_back(doc: &Json) -> Json {
    let nodes = doc.at("nodes");
    let labels = nodes.at("labels").to_strings();
    if labels.len() < BACK || (labels.len() > BACK && labels[BACK] == BACK_LABEL) {
        return doc.clone();
    }
    let insert = |values: Vec<Json>, blank: Json| -> Json {
        let mut out = values;
        out.insert(BACK, blank);
        Json::Arr(out)
    };
    let mut new_nodes: Vec<(String, Json)> = Vec::new();
    for (key, value) in match nodes {
        Json::Obj(pairs) => pairs.clone(),
        _ => Vec::new(),
    } {
        let items = value.as_array().to_vec();
        if items.is_empty() {
            new_nodes.push((key, value));
            continue;
        }
        // the sentinel takes START's activation parameters and its own fixed state
        let blank = match key.as_str() {
            "labels" => Json::str(BACK_LABEL),
            "z" => Json::Num(BACK_Z),
            "count" | "count_resets" => Json::Int(0),
            _ => items[START].clone(),
        };
        new_nodes.push((key, insert(items, blank)));
    }
    let shift = |value: &Json| -> Json {
        Json::ints(
            value
                .to_i64s()
                .into_iter()
                .map(|i| if i as usize >= BACK { i + 1 } else { i }),
        )
    };
    let edges = doc.at("edges");
    let mut new_edges: Vec<(String, Json)> = match edges {
        Json::Obj(pairs) => pairs.clone(),
        _ => Vec::new(),
    };
    for (key, value) in new_edges.iter_mut() {
        if key == "src" || key == "dst" {
            *value = shift(value);
        }
    }
    let mut out: Vec<(String, Json)> = match doc {
        Json::Obj(pairs) => pairs.clone(),
        _ => Vec::new(),
    };
    for (key, value) in out.iter_mut() {
        match key.as_str() {
            "nodes" => *value = Json::Obj(new_nodes.clone()),
            "edges" => *value = Json::Obj(new_edges.clone()),
            "format_version" => *value = Json::Int(GRAPH_FORMAT_VERSION),
            _ => {}
        }
    }
    Json::Obj(out)
}

impl Model {
    /// The model as a `radixnet-count` document.
    pub fn to_doc(&mut self) -> Json {
        Json::obj([
            ("format", Json::str(self.format())),
            ("version", Json::Int(MODEL_FORMAT_VERSION)),
            ("saved_at", Json::str(utc_now())),
            ("kind", Json::str(self.kind())),
            ("meta", self.meta.to_json()),
            (
                "history",
                Json::Arr(self.history.iter().map(EpochRecord::to_json).collect()),
            ),
            ("graph", self.g.to_doc()),
        ])
    }

    /// Rebuilds a model from a `radixnet-count` document.
    pub fn from_doc(doc: &Json) -> Result<Model, String> {
        let format = doc.at("format").as_str().unwrap_or("");
        if format != MODEL_FORMAT {
            return Err(format!("not a {MODEL_FORMAT} model document"));
        }
        let version = doc.at("version").as_i64().unwrap_or(1);
        if version > MODEL_FORMAT_VERSION {
            return Err(format!("unsupported {format} model version {version}"));
        }
        let g = Graph::from_doc(doc.get("graph").ok_or("model document has no graph")?)?;
        let mut model = Model::from_graph(g);
        model.history = doc
            .at("history")
            .as_array()
            .iter()
            .map(EpochRecord::from_json)
            .collect();
        model.meta.merge_json(doc.at("meta")); // a file may carry a counter that was never wrapped
        Ok(model)
    }

    /// Writes the model as JSON, gzipped when `path` ends with `.gz`, through a
    /// temporary file and a rename.
    pub fn save(&mut self, path: &str) -> Result<(), String> {
        let text = self.to_doc().render(0);
        let bytes = if path.ends_with(".gz") {
            crate::gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        write_atomic(path, &bytes)
    }

    /// Reads a model file written by any of the three implementations.
    pub fn load(path: &str) -> Result<Model, String> {
        Model::from_doc(&read_document(path)?)
    }
}

/// Reads a JSON document, gunzipping it when it is gzipped (by its magic, so a
/// `.json` that happens to be compressed still reads).
pub fn read_document(path: &str) -> Result<Json, String> {
    let bytes = std::fs::read(path).map_err(|err| format!("cannot read {path}: {err}"))?;
    let text = if crate::gzip::is_gzip(&bytes) {
        String::from_utf8(crate::gzip::decompress(&bytes)?).map_err(|err| format!("{path} is not UTF-8: {err}"))?
    } else {
        String::from_utf8(bytes).map_err(|err| format!("{path} is not UTF-8: {err}"))?
    };
    parse(&text).map_err(|err| format!("{path}: {err}"))
}

/// Writes bytes through a temporary file next to the target and renames it over
/// the top, so a reader never sees a half-written model.
pub fn write_atomic(path: &str, bytes: &[u8]) -> Result<(), String> {
    let temp = format!("{path}.tmp");
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|err| format!("cannot create {}: {err}", parent.display()))?;
        }
    }
    std::fs::write(&temp, bytes).map_err(|err| format!("cannot write {temp}: {err}"))?;
    std::fs::rename(&temp, path).map_err(|err| format!("cannot replace {path}: {err}"))
}

impl Graph {
    /// Replaces the node table with a file's - the sentinels included, since a
    /// file carries its own.
    pub(crate) fn reset_nodes(&mut self, labels: &[String], counts: &[i64], resets: &[i64]) -> Result<(), String> {
        self.labels.clear();
        self.label_len.clear();
        self.count.clear();
        self.alive.clear();
        self.children.clear();
        self.parents.clear();
        self.count_resets.clear();
        self.index.clear();
        self.n_alive_nodes = 0;
        let enc = self.enc;
        for (i, label) in labels.iter().enumerate() {
            if i >= FIRST && enc.len(label) < enc.n {
                return Err(format!("node {i} label {label:?} is shorter than {}", enc.n));
            }
            self.new_node(label.clone(), counts[i], resets.get(i).copied().unwrap_or(0));
        }
        for (nid, label) in labels.iter().enumerate().skip(FIRST) {
            let u = enc.units(label);
            let mut o = 0;
            while o + enc.n <= u.len() {
                let gram = u.slice(o, o + enc.n);
                if self.index.insert(gram.clone(), Loc { node: nid, off: o }).is_some() {
                    return Err(format!("gram {gram:?} appears in two nodes"));
                }
                o += enc.stride;
            }
        }
        Ok(())
    }

    /// Counts one edge into the sliding window, as the file recorded it.
    pub(crate) fn replay_window_event(&mut self, e: usize) {
        if e < self.window_edge_count.len() {
            self.window.push(e);
            self.window_edge_count[e] += 1;
        }
    }

    /// Sets how many counter increments the graph has made.
    pub(crate) fn set_traversals(&mut self, value: Counter) {
        self.traversals = value;
    }

    /// Sets the sliding window's all-time total.
    pub(crate) fn set_total_traversals(&mut self, value: Counter) {
        self.total_traversals = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TrainOptions;

    fn trained() -> Model {
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        model.workers = 1;
        model.g.workers = 1;
        let texts: Vec<String> = [
            "the cat sat on the mat",
            "the cat sat on the log",
            "a bird flew over the hill",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        model.reward(&texts[..1], 1, 1.5).unwrap();
        model
    }

    #[test]
    fn a_model_survives_a_round_trip_through_its_own_file() {
        let mut model = trained();
        let doc = model.to_doc();
        let text = doc.render(0);
        let mut back = Model::from_doc(&parse(&text).unwrap()).unwrap();

        assert_eq!(back.g.num_nodes(), model.g.num_nodes());
        assert_eq!(back.g.num_edges(), model.g.num_edges());
        assert_eq!(back.g.num_trigrams(), model.g.num_trigrams());
        assert_eq!(back.g.window_traversals(), model.g.window_traversals());
        assert_eq!(back.g.total_traversals().value, model.g.total_traversals().value);
        assert_eq!(back.g.path_totals().contexts, model.g.path_totals().contexts);
        assert_eq!(back.meta.epochs_total.value, model.meta.epochs_total.value);
        assert_eq!(back.history.len(), model.history.len());
        // and the second file is the first one, byte for byte - `version` aside,
        // which is a cache stamp that every load bumps once (Python's does too)
        let strip = |doc: &Json| -> String {
            match doc.at("graph") {
                Json::Obj(pairs) => {
                    Json::Obj(pairs.iter().filter(|(k, _)| k != "version").cloned().collect()).render(0)
                }
                other => other.render(0),
            }
        };
        let again = back.to_doc();
        assert_eq!(strip(&again), strip(&doc));
    }

    #[test]
    fn a_reloaded_model_predicts_what_it_predicted() {
        let mut model = trained();
        let mut back = Model::from_doc(&model.to_doc()).unwrap();
        for prefix in ["the cat", "a bird", "the "] {
            let a = model.predict(prefix, &Default::default()).unwrap();
            let b = back.predict(prefix, &Default::default()).unwrap();
            assert_eq!(a.best.full_text, b.best.full_text, "{prefix}");
            assert_eq!(a.best.cost.to_bits(), b.best.cost.to_bits(), "{prefix}");
        }
        // the RNG continues where it stopped, so the sampled walks agree too
        let a = model.generate(&Default::default()).unwrap();
        let b = back.generate(&Default::default()).unwrap();
        assert_eq!(a[0].text, b[0].text);
    }

    #[test]
    fn it_reads_and_writes_a_gzipped_file() {
        let dir = std::env::temp_dir().join("radixnet-file-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("model.count.json.gz");
        let path = path.to_str().unwrap();
        let mut model = trained();
        model.save(path).unwrap();
        let bytes = std::fs::read(path).unwrap();
        assert!(crate::gzip::is_gzip(&bytes));
        let mut back = Model::load(path).unwrap();
        assert_eq!(back.g.num_edges(), model.g.num_edges());
        assert_eq!(
            back.predict("the cat", &Default::default()).unwrap().best.full_text,
            model.predict("the cat", &Default::default()).unwrap().best.full_text
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_document_of_the_wrong_kind_is_refused() {
        assert!(Model::from_doc(&Json::obj([("format", Json::str("radixnet-negative"))])).is_err());
        assert!(Model::from_doc(&Json::obj([
            ("format", Json::str(MODEL_FORMAT)),
            ("version", Json::Int(99)),
        ]))
        .is_err());
        assert!(Graph::from_doc(&Json::obj([("format", Json::str("nope"))])).is_err());
    }
}
