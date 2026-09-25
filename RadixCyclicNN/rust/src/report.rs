//! What the model says about itself: the statistics, the judged paths and a
//! node against its neighbours, in the shapes the Python and Go CLIs report
//! them (`radixnet/countnet.py`'s `stats`, `path_contexts` and `node_ratios`).

use crate::graph::Graph;
use crate::json::Json;
use crate::model::Model;

/// `stats()` as the CLI and the HTTP API report it.
pub fn stats(model: &Model) -> Json {
    if model.is_negative() {
        return crate::negative::stats_json(model);
    }
    if model.g.is_radix() {
        return crate::radix::stats_json(model);
    }
    if model.is_resonant() {
        return crate::resonance::stats_json(model);
    }
    let g = &model.g;
    let (pos, neg) = g.total_reward();
    let paths = g.path_totals();
    let meta = &model.meta;
    let last_loss = match model.history.last() {
        Some(record) => Json::Num(record.loss),
        None => Json::Null,
    };
    let mut pairs = vec![
        ("kind".to_string(), Json::str(model.kind())),
        ("nodes".to_string(), Json::Int(g.num_nodes() as i64)),
        ("edges".to_string(), Json::Int(g.num_edges() as i64)),
        ("trigrams".to_string(), Json::Int(g.num_trigrams() as i64)),
        // `grams` is `trigrams` under the name that survives an encoding of any n
        ("grams".to_string(), Json::Int(g.num_trigrams() as i64)),
        // what every number below is counted in, and the dial it was read
        // with - a per-word number read as per-character is read wrong
        ("encoding".to_string(), Json::str(g.enc.to_string())),
        ("unit".to_string(), Json::str(g.enc.unit.name())),
        ("units".to_string(), Json::str(model.units())),
        ("ngram".to_string(), Json::Int(g.enc.n as i64)),
        ("stride".to_string(), Json::Int(g.enc.stride as i64)),
        // where a correction lands: the band's blur, null while it is off
        (
            "attention_blur".to_string(),
            g.attention.blur.map(Json::Num).unwrap_or(Json::Null),
        ),
        ("compression_ratio".to_string(), Json::Num(g.compression_ratio())),
        ("inverted".to_string(), Json::Bool(g.inverted)),
        ("backend".to_string(), Json::str("rust")),
        ("device".to_string(), Json::str(model.device_label())),
    ];
    for (key, counter) in [
        ("epochs_total", meta.epochs_total),
        ("trained_chars", meta.trained_chars),
        ("trained_texts", meta.trained_texts),
        ("twonrl_runs", meta.twonrl_runs),
        ("feedback_passes", meta.feedback_passes),
    ] {
        pairs.push((key.to_string(), Json::Int(counter.value)));
        pairs.push((format!("{key}_resets"), Json::Int(counter.resets)));
    }
    pairs.extend([
        ("history_len".to_string(), Json::Int(model.history.len() as i64)),
        ("last_loss".to_string(), last_loss),
        ("rewards_total".to_string(), Json::Num(meta.rewards_total)),
        ("penalties_total".to_string(), Json::Num(meta.penalties_total)),
        ("path_contexts".to_string(), Json::Int(paths.contexts as i64)),
        ("path_judged".to_string(), Json::Int(paths.judged as i64)),
        ("path_seen".to_string(), Json::Int(paths.seen)),
        ("path_correct".to_string(), Json::Int(paths.correct)),
        ("path_incorrect".to_string(), Json::Int(paths.incorrect)),
        ("edge_reward_positive".to_string(), Json::Num(pos)),
        ("edge_reward_negative".to_string(), Json::Num(neg)),
        ("count_scale".to_string(), Json::Num(g.count_scale)),
        ("reward_scale".to_string(), Json::Num(g.reward_scale)),
        ("global_scale".to_string(), Json::Num(g.global_scale)),
        ("window_scale".to_string(), Json::Num(g.window_scale)),
        ("path_scale".to_string(), Json::Num(g.path_scale)),
        ("window".to_string(), Json::Int(g.window_size as i64)),
        ("total_traversals".to_string(), Json::Int(g.total_traversals().value)),
        (
            "total_traversals_resets".to_string(),
            Json::Int(g.total_traversals().resets),
        ),
        ("window_traversals".to_string(), Json::Int(g.window_traversals() as i64)),
    ]);
    if g.enc.unit != crate::encoding::Unit::Chars {
        let words = crate::encoding::vocabulary(&g.enc, g.gram_index());
        pairs.push(("vocabulary".to_string(), Json::Int(words.len() as i64)));
    }
    // this port's and Go's own: how the counters were bumped (the frontend's
    // engine badge reads it); Python has no second way to count
    pairs.push(("counting".to_string(), Json::str(model.counting())));
    Json::Obj(pairs)
}

/// The weight function's settings, as `GET /api/model` and the CLI report them.
pub fn weight_config(g: &Graph) -> Json {
    Json::obj([
        ("function", Json::str("dual-frequency")),
        ("count_scale", Json::Num(g.count_scale)),
        ("global_scale", Json::Num(g.global_scale)),
        ("window_scale", Json::Num(g.window_scale)),
        ("reward_scale", Json::Num(g.reward_scale)),
        ("path_scale", Json::Num(g.path_scale)),
        ("window", Json::Int(g.window_size as i64)),
        ("smoothing", Json::Num(crate::weights::SMOOTHING)),
    ])
}

/// Sets one of those by the name the API and the CLI use; an unknown name is an error.
pub fn configure_weight(g: &mut Graph, name: &str, value: f64) -> Result<(), String> {
    match name.replace('-', "_").as_str() {
        "count_scale" => g.count_scale = value,
        "global_scale" => g.global_scale = value,
        "window_scale" => g.window_scale = value,
        "reward_scale" => g.reward_scale = value,
        "path_scale" => g.path_scale = value,
        "window" => {
            if value < 1.0 {
                return Err(format!("window must be >= 1, got {value}"));
            }
            g.set_window(value as usize);
        }
        other => {
            return Err(format!(
                "unknown weight setting {other:?}; expected count_scale, global_scale, window_scale,                  reward_scale, path_scale or window"
            ))
        }
    }
    Ok(())
}

/// The judged contexts, most judged first; `limit` 0 is all of them and `node`
/// `Some(n)` only the steps leaving that node.
pub fn path_rows(g: &Graph, limit: usize, node: Option<usize>) -> Json {
    let rows: Vec<Json> = g
        .path_contexts(0)
        .into_iter()
        .filter(|(key, _)| node.is_none_or(|n| g.parent_of_edge(key.edge) == Some(n)))
        .take(if limit == 0 { usize::MAX } else { limit })
        .map(|(key, row)| {
            let judged = row.correct + row.incorrect;
            let correct_ratio = if judged > 0 {
                Json::Num(row.correct as f64 / judged as f64)
            } else {
                Json::Null
            };
            let traversals = g.edge_traversals(key.edge).float();
            let seen_ratio = if traversals > 0.0 {
                Json::Num(row.seen as f64 / traversals)
            } else {
                Json::Null
            };
            Json::obj([
                ("prev", Json::Int(key.prev as i64)),
                ("edge", Json::Int(key.edge as i64)),
                ("seen", Json::Int(row.seen)),
                ("correct", Json::Int(row.correct)),
                ("incorrect", Json::Int(row.incorrect)),
                ("correct_ratio", correct_ratio),
                ("seen_ratio", seen_ratio),
                ("term", Json::Num(g.path_term(key.prev, key.edge))),
            ])
        })
        .collect();
    let totals = g.path_totals();
    Json::obj([
        (
            "totals",
            Json::obj([
                ("contexts", Json::Int(totals.contexts as i64)),
                ("judged", Json::Int(totals.judged as i64)),
                ("seen", Json::Int(totals.seen)),
                ("correct", Json::Int(totals.correct)),
                ("incorrect", Json::Int(totals.incorrect)),
            ]),
        ),
        ("paths", Json::Arr(rows)),
    ])
}

/// One side of a node - a row per neighbour, with its share of that side's
/// traffic and of that side's reward *magnitude*, signed, so a penalty reads as
/// a negative share of the pressure on the node.
fn side_rows(g: &Graph, pairs: &[(usize, usize)]) -> Vec<Json> {
    let mut pairs = pairs.to_vec();
    pairs.sort_unstable(); // by neighbour id, so every implementation adds the shares up in one order
    let traversals: Vec<f64> = pairs.iter().map(|&(_, e)| g.edge_traversals(e).float()).collect();
    let mut total = 0.0;
    for t in &traversals {
        total += *t; // an explicit left-to-right sum, as in the Python implementation
    }
    let mut mass = 0.0;
    for &(_, e) in &pairs {
        mass += g.edge_reward(e).abs();
    }
    let mut rows: Vec<(i64, f64, usize, Json)> = Vec::with_capacity(pairs.len());
    for (i, &(n, e)) in pairs.iter().enumerate() {
        let (path_seen, correct, incorrect) = g.edge_paths(e);
        let judged = correct + incorrect;
        let reward = g.edge_reward(e);
        let seen = g.edge_traversals(e);
        let row = Json::obj([
            ("node", Json::Int(n as i64)),
            ("label", Json::str(g.label(n).to_string())),
            ("edge", Json::Int(e as i64)),
            ("seen", Json::Int(seen.value)),
            ("seen_resets", Json::Int(seen.resets)),
            (
                "seen_ratio",
                Json::Num(if total > 0.0 { traversals[i] / total } else { 0.0 }),
            ),
            ("reward", Json::Num(reward)),
            ("reward_ratio", Json::Num(if mass > 0.0 { reward / mass } else { 0.0 })),
            ("path_seen", Json::Int(path_seen)),
            (
                "path_ratio",
                if traversals[i] > 0.0 {
                    Json::Num(path_seen as f64 / traversals[i])
                } else {
                    Json::Null
                },
            ),
            ("correct", Json::Int(correct)),
            ("incorrect", Json::Int(incorrect)),
            (
                "correct_ratio",
                if judged > 0 {
                    Json::Num(correct as f64 / judged as f64)
                } else {
                    Json::Null
                },
            ),
        ]);
        rows.push((seen.value, reward, n, row));
    }
    rows.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.total_cmp(&a.1)).then(a.2.cmp(&b.2)));
    rows.into_iter().map(|(_, _, _, row)| row).collect()
}

/// What a whole side of a node did, over the rows in the order they are reported.
fn side_totals(rows: &[Json]) -> Json {
    let sum_i = |key: &str| rows.iter().map(|r| r.at(key).as_i64().unwrap_or(0)).sum::<i64>();
    let reward: f64 = rows.iter().map(|r| r.at("reward").as_f64().unwrap_or(0.0)).sum();
    let (correct, incorrect) = (sum_i("correct"), sum_i("incorrect"));
    let judged = correct + incorrect;
    Json::obj([
        ("edges", Json::Int(rows.len() as i64)),
        ("seen", Json::Int(sum_i("seen"))),
        ("reward", Json::Num(reward)),
        ("path_seen", Json::Int(sum_i("path_seen"))),
        ("correct", Json::Int(correct)),
        ("incorrect", Json::Int(incorrect)),
        (
            "correct_ratio",
            if judged > 0 {
                Json::Num(correct as f64 / judged as f64)
            } else {
                Json::Null
            },
        ),
    ])
}

/// One node against the nodes around it, or `None` when it is not a live node.
pub fn node_ratios(g: &Graph, node: usize) -> Option<Json> {
    if !g.is_alive(node) {
        return None;
    }
    let rows_in = side_rows(g, &g.parents_of(node));
    let rows_out = side_rows(g, &g.children_pairs(node));
    let count = g.node_count(node);
    Some(Json::obj([
        ("node", Json::Int(node as i64)),
        ("label", Json::str(g.label(node).to_string())),
        ("visits", Json::Int(count.value)),
        ("visit_resets", Json::Int(count.resets)),
        ("in_totals", side_totals(&rows_in)),
        ("out_totals", side_totals(&rows_out)),
        ("from", Json::Arr(rows_in)),
        ("to", Json::Arr(rows_out)),
    ]))
}

/// [`node_ratios`] of the most visited nodes, or of one node.
pub fn node_rows(g: &Graph, limit: usize, node: Option<usize>) -> Json {
    if let Some(node) = node {
        return Json::Arr(node_ratios(g, node).into_iter().collect());
    }
    let mut order: Vec<usize> = (0..g.num_node_ids()).filter(|&i| g.is_alive(i)).collect();
    order.sort_by(|&a, &b| g.node_count(b).value.cmp(&g.node_count(a).value).then(a.cmp(&b)));
    if limit > 0 {
        order.truncate(limit);
    }
    Json::Arr(order.into_iter().filter_map(|i| node_ratios(g, i)).collect())
}

/// Cuts a file's content into training texts - the units a pass fans out over.
///
/// `lines` is one text per non-blank line (the default everywhere),
/// `paragraphs` blocks separated by blank lines, `pages` form-feed separated
/// pages (or every `page_lines` lines when there is no form feed), and `file`
/// the whole content as one text.
pub fn split_texts(content: &str, unit: &str, page_lines: usize) -> Vec<String> {
    let content = content.replace("\r\n", "\n");
    let content = content.strip_prefix('\u{feff}').unwrap_or(&content);
    match unit.trim().to_ascii_lowercase().as_str() {
        "paragraph" | "paragraphs" => {
            let mut texts = Vec::new();
            let mut block: Vec<&str> = Vec::new();
            for line in content.split('\n') {
                if line.trim().is_empty() {
                    if !block.is_empty() {
                        texts.push(block.join(" "));
                        block.clear();
                    }
                } else {
                    block.push(line.trim());
                }
            }
            if !block.is_empty() {
                texts.push(block.join(" "));
            }
            texts
        }
        "page" | "pages" => {
            let page_lines = if page_lines == 0 { 50 } else { page_lines };
            let pages: Vec<String> = if content.contains('\u{c}') {
                content.split('\u{c}').map(str::to_string).collect()
            } else {
                content
                    .split('\n')
                    .collect::<Vec<_>>()
                    .chunks(page_lines)
                    .map(|lines| lines.join("\n"))
                    .collect()
            };
            pages
                .iter()
                .filter_map(|page| {
                    let parts: Vec<&str> = page.split('\n').map(str::trim).filter(|l| !l.is_empty()).collect();
                    (!parts.is_empty()).then(|| parts.join(" "))
                })
                .collect()
        }
        "file" | "whole" | "whole-file" => {
            let text = content.trim_end_matches(['\r', '\n']);
            if text.trim().is_empty() {
                Vec::new()
            } else {
                vec![text.to_string()]
            }
        }
        _ => content
            .split('\n')
            .filter(|l| !l.trim().is_empty())
            .map(str::to_string)
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_corpus_is_cut_the_way_it_is_asked_for() {
        let content = "one\n\ntwo\nthree\n\n";
        assert_eq!(split_texts(content, "lines", 0), vec!["one", "two", "three"]);
        assert_eq!(split_texts(content, "paragraphs", 0), vec!["one", "two three"]);
        assert_eq!(split_texts(content, "file", 0), vec!["one\n\ntwo\nthree"]);
        assert_eq!(split_texts("a\nb\nc\nd", "pages", 2), vec!["a b", "c d"]);
        assert_eq!(split_texts("", "lines", 0), Vec::<String>::new());
    }
}
