//! The language models behind the API, side by side: the chain and the N-gram answer the same routes in the same shapes
//! (`/api/chain/...`, `/api/ngram/...`), through the `Lm` trait; and around them uploads, checkpoints, a model talking
//! to itself or to a local LLM, and a corpus written by one.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::autotemp;
use crate::autotrain::{parse_lines, Ollama};
use crate::chain::{load_chain, Chain, ChainEvaluation};
use crate::guard::Guard;
use crate::http::Request;
use crate::json::Json;
use crate::ngram::{detokenize, split_text_lines, train_corpus, Drill, NGram, Sequence, Teaching};
use crate::rng::Rng;

// ---- the trait ---------------------------------------------------------------------------------------------------------

/// What a route needs of a language model.
pub trait Lm {
    fn name(&self) -> &'static str;
    fn info(&mut self) -> Json;
    fn unit(&self) -> String;
    fn lower(&self) -> bool;
    fn tokens(&self, text: &str) -> Sequence;
    fn prompt_tokens(&self, text: &str) -> Sequence;
    /// How many tokens before the next the model predicts it from (a greedy draw after the same ones is the same).
    fn context_len(&self) -> usize;
    /// Predict every token before learning it, crediting the guess by `credit` (rewards too unless not `reward`).
    fn learning(&mut self, predict: bool, credit: f64, reward: bool);
    fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String>;
    fn train_lines(&mut self, lines: &[String]) -> usize;
    /// (first guesses right, tokens predicted) since the last reset.
    fn online(&self) -> (u64, u64);
    /// What predicting while training measured, and a fresh start.
    fn take_online(&mut self) -> Json;
    fn evaluate(&mut self, test: &[Sequence]) -> Json;
    fn predict(&mut self, text: &str, k: usize) -> Json;
    /// Tokens continuing `toks`, and (for the chain) how deep every walk went.
    fn continue_tokens(
        &mut self,
        toks: &[String],
        max: usize,
        t: f64,
        rng: &mut Rng,
    ) -> (Vec<String>, Option<Vec<usize>>);
    fn credit_text(&mut self, text: &str, amount: f64) -> usize;
    /// Every token of an oracle's `answer` to `text` as a prediction, credited for itself alone: the model's guess
    /// there rewarded when it is the answer's, punished (by `punish`; 0: rewards only) when not.  (right, of).
    fn credit_answer(&mut self, text: &str, answer: &str, reward: f64, punish: f64) -> (usize, usize);
    /// Every token of a text (and `</s>`) with the probability given it, and (for the chain) the depth of its walk.
    fn score(&mut self, text: &str) -> Vec<(String, f64, Option<usize>)>;
    fn teach(&mut self, text: &[String], reply: &[String], max_rounds: usize, threshold: f64) -> Teaching;
    fn drill(&mut self, seqs: &[Sequence], max_rounds: usize, patience: usize, threshold: f64) -> Drill;
    fn ceiling(&self, seqs: &[Sequence]) -> (usize, usize);
    fn save(&self, path: &str) -> Result<(), String>;
    fn history(&mut self) -> &mut Vec<Json>;
}

fn opt(v: Option<f64>) -> Json {
    v.map(Json::from).unwrap_or(Json::Null)
}

pub fn sources_json(sources: &[(String, usize, usize)]) -> Json {
    Json::Array(
        sources
            .iter()
            .map(|(s, c, n)| {
                Json::object()
                    .with("source", s.as_str().into())
                    .with("lines", (*c).into())
                    .with("tokens", (*n).into())
            })
            .collect(),
    )
}

impl Lm for NGram {
    fn name(&self) -> &'static str {
        "ngram"
    }
    fn context_len(&self) -> usize {
        self.order - 1
    }
    fn info(&mut self) -> Json {
        self.stats()
            .with("history", Json::Array(self.history.clone()))
            .with("sources", sources_json(&self.sources))
    }
    fn unit(&self) -> String {
        self.unit.clone()
    }
    fn lower(&self) -> bool {
        self.lower
    }
    fn tokens(&self, text: &str) -> Sequence {
        NGram::tokens(self, text)
    }
    fn prompt_tokens(&self, text: &str) -> Sequence {
        NGram::prompt_tokens(self, text)
    }
    fn learning(&mut self, predict: bool, credit: f64, reward: bool) {
        self.predict = predict;
        self.self_credit = credit;
        self.self_reward = reward;
    }
    fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String> {
        NGram::train_paths(self, paths, held_out)
    }
    fn train_lines(&mut self, lines: &[String]) -> usize {
        NGram::train_lines(self, lines, Some("pasted text"))
    }
    fn online(&self) -> (u64, u64) {
        (self.online.0, self.online.1)
    }
    fn take_online(&mut self) -> Json {
        let (n, acc, ppl, bits) = self.reset_online();
        Json::object()
            .with("predicted", (n as usize).into())
            .with("accuracy", opt(acc))
            .with("perplexity", opt(ppl))
            .with("bits", opt(bits))
    }
    fn evaluate(&mut self, test: &[Sequence]) -> Json {
        let ev = NGram::evaluate(self, test, 1000);
        Json::object()
            .with("tokens", ev.tokens.into())
            .with("perplexity", opt(ev.perplexity))
            .with("bits", opt(ev.bits))
            .with("accuracy", opt(ev.accuracy))
            .with("top5", opt(ev.top5))
    }
    fn predict(&mut self, text: &str, k: usize) -> Json {
        crate::server::ngram_predict(self, &Json::object().with("text", text.into()).with("k", k.into()))
    }
    fn continue_tokens(
        &mut self,
        toks: &[String],
        max: usize,
        t: f64,
        rng: &mut Rng,
    ) -> (Vec<String>, Option<Vec<usize>>) {
        (NGram::continue_tokens(self, toks, max, t, rng), None)
    }
    fn credit_text(&mut self, text: &str, amount: f64) -> usize {
        NGram::credit_text(self, text, amount)
    }
    fn credit_answer(&mut self, text: &str, answer: &str, reward: f64, punish: f64) -> (usize, usize) {
        let (prompt, answer) = (self.prompt_tokens(text), self.tokens(answer));
        NGram::credit_answer(self, &prompt, &answer, reward, punish)
    }
    fn score(&mut self, text: &str) -> Vec<(String, f64, Option<usize>)> {
        self.score_tokens(&NGram::tokens(self, text))
            .into_iter()
            .map(|(t, p)| (t, p, None))
            .collect()
    }
    fn teach(&mut self, text: &[String], reply: &[String], max_rounds: usize, threshold: f64) -> Teaching {
        NGram::teach(self, text, reply, max_rounds, threshold)
    }
    fn drill(&mut self, seqs: &[Sequence], max_rounds: usize, patience: usize, threshold: f64) -> Drill {
        NGram::drill(self, seqs, max_rounds, patience, threshold)
    }
    fn ceiling(&self, seqs: &[Sequence]) -> (usize, usize) {
        NGram::ceiling(self, seqs)
    }
    fn save(&self, path: &str) -> Result<(), String> {
        NGram::save(self, path)
    }
    fn history(&mut self) -> &mut Vec<Json> {
        &mut self.history
    }
}

pub fn chain_eval_json(ev: &ChainEvaluation) -> Json {
    Json::object()
        .with("tokens", ev.tokens.into())
        .with("perplexity", opt(ev.perplexity))
        .with("bits", opt(ev.bits))
        .with("accuracy", opt(ev.accuracy))
        .with("depth", opt(ev.depth))
        .with(
            "depths",
            Json::numbers(&ev.depths.iter().map(|&d| d as f64).collect::<Vec<_>>()),
        )
}

impl Lm for Chain {
    fn name(&self) -> &'static str {
        "chain"
    }
    fn context_len(&self) -> usize {
        self.nodes - 1
    }
    fn info(&mut self) -> Json {
        self.stats()
            .with("history", Json::Array(self.history.clone()))
            .with("sources", sources_json(&self.sources))
    }
    fn unit(&self) -> String {
        self.unit.clone()
    }
    fn lower(&self) -> bool {
        self.lower
    }
    fn tokens(&self, text: &str) -> Sequence {
        Chain::tokens(self, text)
    }
    fn prompt_tokens(&self, text: &str) -> Sequence {
        Chain::prompt_tokens(self, text)
    }
    fn learning(&mut self, predict: bool, credit: f64, reward: bool) {
        self.predict = predict;
        self.self_credit = credit;
        self.self_reward = reward;
    }
    fn train_paths(&mut self, paths: &[String], held_out: usize) -> Result<(usize, Vec<Sequence>), String> {
        Chain::train_paths(self, paths, held_out)
    }
    fn train_lines(&mut self, lines: &[String]) -> usize {
        Chain::train_lines(self, lines, Some("pasted text"))
    }
    fn online(&self) -> (u64, u64) {
        (self.online.0, self.online.1)
    }
    fn take_online(&mut self) -> Json {
        let o = self.reset_online();
        Json::object()
            .with("predicted", (o.predicted as usize).into())
            .with("accuracy", opt(o.accuracy))
            .with("perplexity", opt(o.perplexity))
            .with("bits", opt(o.bits))
            .with("depth", opt(o.depth))
    }
    fn evaluate(&mut self, test: &[Sequence]) -> Json {
        chain_eval_json(&Chain::evaluate(self, test, None))
    }
    fn predict(&mut self, text: &str, k: usize) -> Json {
        let toks = Chain::prompt_tokens(self, text);
        self.explain(&toks, k, None).with("text", text.into())
    }
    fn continue_tokens(
        &mut self,
        toks: &[String],
        max: usize,
        t: f64,
        rng: &mut Rng,
    ) -> (Vec<String>, Option<Vec<usize>>) {
        let (out, depths) = Chain::continue_tokens(self, toks, max, t, rng);
        (out, Some(depths))
    }
    fn credit_text(&mut self, text: &str, amount: f64) -> usize {
        Chain::credit_text(self, text, amount)
    }
    fn credit_answer(&mut self, text: &str, answer: &str, reward: f64, punish: f64) -> (usize, usize) {
        let (prompt, answer) = (self.prompt_tokens(text), self.tokens(answer));
        Chain::credit_answer(self, &prompt, &answer, reward, punish)
    }
    fn score(&mut self, text: &str) -> Vec<(String, f64, Option<usize>)> {
        self.score_tokens(&Chain::tokens(self, text))
            .into_iter()
            .map(|(t, p, d)| (t, p, Some(d)))
            .collect()
    }
    fn teach(&mut self, text: &[String], reply: &[String], max_rounds: usize, threshold: f64) -> Teaching {
        Chain::teach(self, text, reply, max_rounds, threshold)
    }
    fn drill(&mut self, seqs: &[Sequence], max_rounds: usize, patience: usize, threshold: f64) -> Drill {
        Chain::drill(self, seqs, max_rounds, patience, threshold)
    }
    fn ceiling(&self, seqs: &[Sequence]) -> (usize, usize) {
        Chain::ceiling(self, seqs)
    }
    fn save(&self, path: &str) -> Result<(), String> {
        Chain::save(self, path)
    }
    fn history(&mut self) -> &mut Vec<Json> {
        &mut self.history
    }
}

/// The chain the server starts with: 8 nodes of letters, confidence 0.9, trained on `data/corpus` (or
/// `../data/corpus`) when there is one, every tenth line held out.
pub fn default_chain() -> Chain {
    let mut m = Chain::new(8, "char", true, 0.9).expect("valid");
    for dir in ["data/corpus", "../data/corpus"] {
        if Path::new(dir).is_dir() {
            m.predict = false;
            if let Ok((_, test)) = m.train_paths(&[dir.to_string()], 10) {
                let ev = chain_eval_json(&m.evaluate(&test, None));
                m.history
                    .push(ev.with("round", 0usize.into()).with("source", dir.into()));
            }
            m.predict = true;
            break;
        }
    }
    m
}

// ---- the routes both models answer ---------------------------------------------------------------------------------

/// The share of the goal a body's `threshold` asks for (0 to 1; default `THRESHOLD`, 95%).
pub fn threshold_of(b: &Json) -> f64 {
    b.num("threshold", crate::ngram::THRESHOLD).clamp(0.0, 1.0)
}

/// The server paths a training body names: `data` (comma-separated, or a list) and `uploads` (names in the upload
/// directory); neither: the corpus.
pub fn training_paths(b: &Json) -> Result<Vec<String>, String> {
    let mut paths: Vec<String> = match b.get("data") {
        Some(Json::String(s)) => s
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect(),
        Some(Json::Array(a)) => a.iter().filter_map(|v| v.as_str().map(String::from)).collect(),
        _ => Vec::new(),
    };
    if let Some(Json::Array(names)) = b.get("uploads") {
        let dir = uploads_dir();
        for n in names.iter().filter_map(Json::as_str) {
            let p = dir.join(sanitize(n)?);
            if !p.exists() {
                return Err(format!("no upload named {n:?}"));
            }
            paths.push(p.to_string_lossy().into_owned());
        }
    }
    if paths.is_empty() {
        paths.push(
            if Path::new("data/corpus").is_dir() {
                "data/corpus"
            } else {
                "../data/corpus"
            }
            .to_string(),
        );
    }
    Ok(paths)
}

/// What a model writes after `prompt` as text: for base64 the prompt's letters and the new ones decoded together.
pub fn completion_text(m: &dyn Lm, prompt: &str, toks: &[String]) -> String {
    let unit = m.unit();
    if unit == "base64" {
        let mut all = m.prompt_tokens(prompt);
        all.extend(toks.iter().cloned());
        return detokenize(&all, &unit);
    }
    detokenize(toks, &unit)
}

/// A prompt and what a model wrote after it, as one text to credit.
pub fn joined(m: &dyn Lm, prompt: &str, completion: &str) -> String {
    match m.unit().as_str() {
        "base64" => completion.to_string(),
        "char" => format!("{prompt}{completion}"),
        _ => format!("{prompt} {completion}").trim().to_string(),
    }
}

/// The guard's verdict on a sample, in brief.
pub fn verdict_brief(j: &Json) -> Json {
    Json::object()
        .with("verdict", j.get("verdict").cloned().unwrap_or(Json::Null))
        .with("risk", j.get("risk").cloned().unwrap_or(Json::Null))
        .with("coverage", j.get("coverage").cloned().unwrap_or(Json::Null))
        .with("why", j.get("why").cloned().unwrap_or(Json::Null))
        .with("reasons", j.get("reasons").cloned().unwrap_or(Json::Null))
        .with("spans", j.get("spans").cloned().unwrap_or(Json::Null))
}

/// `count` samples continuing `text`; with a guard every one is judged and the rejected ones vetoed - `over` times as
/// many drawn to make up for them, the first at `temperature` and the rest at 0.7 or more, so a vetoed greedy reply
/// has others to give way to; when every one is vetoed the least risky is kept and said so.  Returns the samples
/// (each with its `guard` verdict) and the guard's summary.
/// How samples are drawn: the most tokens, the temperature, the first seed, how many, how many times as many to draw
/// when guarded, and whether a sample that is no answer is drawn again hotter (`auto`).
#[derive(Clone, Copy)]
pub struct Draw {
    pub max: usize,
    pub temperature: f64,
    pub seed: u64,
    pub count: usize,
    pub over: usize,
    pub auto: Option<Auto>,
}

/// Auto-temperature (`autotemp`): the hottest rung, and the draws at each rung.
#[derive(Clone, Copy, Debug)]
pub struct Auto {
    pub top: f64,
    pub tries: usize,
}

/// A request's auto-temperature: `{auto_temperature: true, max_temperature?, tries?}`; `None` when it is off.
pub fn auto_of(b: &Json) -> Option<Auto> {
    b.bool_or("auto_temperature", false).then(|| Auto {
        top: b.num("max_temperature", autotemp::MAX_TEMPERATURE).clamp(0.0, 2.0),
        tries: (b.num("tries", autotemp::TRIES as f64) as usize).clamp(1, 10),
    })
}

pub fn guarded_samples(m: &mut dyn Lm, text: &str, d: &Draw, guard: Option<&mut Guard>) -> (Vec<Json>, Option<Json>) {
    if let Some(a) = d.auto {
        return auto_samples(m, text, d, a, guard);
    }
    let Draw {
        max,
        temperature,
        seed,
        count,
        over,
        auto: _,
    } = *d;
    let Some(g) = guard else {
        return (
            (0..count)
                .map(|i| generate(m, text, max, temperature, seed + i as u64))
                .collect(),
            None,
        );
    };
    let n = count * over.max(1);
    let mut judged: Vec<(Json, Json)> = Vec::with_capacity(n);
    for i in 0..n {
        let t = if i == 0 { temperature } else { temperature.max(0.7) };
        let s = generate(m, text, max, t, seed + i as u64);
        let j = g.judge(s.str_or("completion", "").trim(), None, None, 5);
        judged.push((s, j));
    }
    let mut kept = Vec::new();
    let mut vetoes = Vec::new();
    for (s, j) in &judged {
        if j.str_or("verdict", "") == "reject" {
            vetoes.push(
                Json::object()
                    .with("completion", s.get("completion").cloned().unwrap_or(Json::Null))
                    .with("risk", j.get("risk").cloned().unwrap_or(Json::Null))
                    .with("why", j.get("why").cloned().unwrap_or(Json::Null)),
            );
        } else if kept.len() < count {
            kept.push(s.clone().with("guard", verdict_brief(j)));
        }
    }
    let all_vetoed = kept.is_empty();
    if all_vetoed {
        let mut best = 0;
        for (i, (_, j)) in judged.iter().enumerate() {
            if j.num("risk", 0.0) < judged[best].1.num("risk", 0.0) {
                best = i;
            }
        }
        kept.push(judged[best].0.clone().with("guard", verdict_brief(&judged[best].1)));
    }
    let summary = Json::object()
        .with("judged", n.into())
        .with("vetoed", vetoes.len().into())
        .with("all_vetoed", all_vetoed.into())
        .with("vetoes", Json::Array(vetoes.into_iter().take(5).collect()));
    (kept, Some(summary))
}

/// `count` samples, each drawn up the auto-temperature ladder until it is an answer - not nothing, not without words,
/// not a loop, and (with a guard) not rejected by it - or the ladder ends, when the least bad draw is kept (a loop
/// before a rejected one before one without words before nothing; the coolest of equals).  Every sample says how it
/// was found (`auto`: `found`, the `temperature` it was drawn at, `why` when it is still no answer, and every
/// `attempt`); the guard's summary as `guarded_samples` gives it.
fn auto_samples(
    m: &mut dyn Lm,
    text: &str,
    d: &Draw,
    a: Auto,
    mut guard: Option<&mut Guard>,
) -> (Vec<Json>, Option<Json>) {
    let prompt = m.prompt_tokens(text);
    let k = m.context_len();
    let (mut judged, mut vetoes, mut all_vetoed) = (0usize, Vec::new(), false);
    let mut samples = Vec::with_capacity(d.count);
    for i in 0..d.count {
        // the sample and the guard's verdict on it
        let c = autotemp::climb(
            &prompt,
            k,
            d.temperature,
            a.top,
            a.tries,
            d.seed + i as u64,
            |t, seed| {
                let s = generate(m, text, d.max, t, seed);
                let toks = s
                    .get("tokens")
                    .and_then(Json::as_array)
                    .map(|v| v.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default();
                let completion = s.str_or("completion", "").to_string();
                (toks, completion, (s, None::<Json>))
            },
            |completion, (s, verdict)| {
                let g = guard.as_deref_mut()?;
                let j = g.judge(completion.trim(), None, None, 5);
                judged += 1;
                let rejected = j.str_or("verdict", "") == "reject";
                if rejected {
                    vetoes.push(
                        Json::object()
                            .with("completion", s.get("completion").cloned().unwrap_or(Json::Null))
                            .with("risk", j.get("risk").cloned().unwrap_or(Json::Null))
                            .with("why", j.get("why").cloned().unwrap_or(Json::Null)),
                    );
                }
                *verdict = Some(j);
                rejected.then_some("rejected by the guard")
            },
        );
        all_vetoed |= c.why == Some("rejected by the guard");
        let (s, verdict) = c.pick;
        // a kept sample the guard never saw (a loop, say) is judged now, so a guarded sample always has its verdict
        let verdict = match (verdict, guard.as_deref_mut()) {
            (Some(v), _) => Some(v),
            (None, Some(g)) => {
                judged += 1;
                Some(g.judge(s.str_or("completion", "").trim(), None, None, 5))
            }
            (None, None) => None,
        };
        let attempts: Vec<Json> = c
            .tried
            .iter()
            .map(|x| {
                Json::object()
                    .with("temperature", x.temperature.into())
                    .with("seed", (x.seed as f64).into())
                    .with("completion", x.text.as_str().into())
                    .with("why", x.why.map(Json::from).unwrap_or(Json::Null))
            })
            .collect();
        let mut out = s
            .with("temperature", c.temperature.into())
            .with("seed", (c.seed as f64).into())
            .with(
                "auto",
                Json::object()
                    .with("found", c.found.into())
                    .with("temperature", c.temperature.into())
                    .with("why", c.why.map(Json::from).unwrap_or(Json::Null))
                    .with(
                        "note",
                        autotemp::climb_note(c.found, c.temperature, c.why, &c.tried)
                            .as_str()
                            .into(),
                    )
                    .with("attempts", Json::Array(attempts)),
            );
        if let Some(v) = verdict {
            out = out.with("guard", verdict_brief(&v));
        }
        samples.push(out);
    }
    let summary = guard.map(|_| {
        Json::object()
            .with("judged", judged.into())
            .with("vetoed", vetoes.len().into())
            .with("all_vetoed", all_vetoed.into())
            .with("vetoes", Json::Array(vetoes.into_iter().take(5).collect()))
    });
    (samples, summary)
}

fn generate(m: &mut dyn Lm, text: &str, max: usize, temperature: f64, seed: u64) -> Json {
    let toks = m.prompt_tokens(text);
    let mut rng = Rng::new(seed);
    let (out, depths) = m.continue_tokens(&toks, max, temperature, &mut rng);
    let completion = completion_text(m, text, &out);
    let mut j = Json::object()
        .with("completion", completion.into())
        .with("tokens", Json::strings(&out));
    if let Some(d) = depths {
        j = j.with(
            "depths",
            Json::numbers(&d.iter().map(|&x| x as f64).collect::<Vec<_>>()),
        );
    }
    j
}

fn drill_json(first: (u64, u64), d: &Drill) -> (Json, bool) {
    let mut rows = vec![Json::object()
        .with("round", 1usize.into())
        .with("right", (first.0 as usize).into())
        .with("of", (first.1 as usize).into())];
    for (k, r) in d.rounds.iter().enumerate() {
        rows.push(
            Json::object()
                .with("round", (k + 2).into())
                .with("right", (*r).into())
                .with("of", d.of.into()),
        );
    }
    (Json::Array(rows), d.reached)
}

/// `POST /api/{model}/train`: the files, every token predicted before it is learned; with `until` drilled until N of N.
fn lm_train(m: &mut dyn Lm, b: &Json) -> Result<Json, String> {
    let paths = training_paths(b)?;
    let held_out = b.num("held_out", 10.0) as usize;
    m.learning(
        b.bool_or("predict", true) || b.bool_or("until", false),
        b.num("self_credit", 0.0),
        true,
    );
    m.take_online();
    let (n, test) = m.train_paths(&paths, held_out)?;
    let first = m.online();
    let learned = m.take_online();
    let mut j = Json::object()
        .with("trained", n.into())
        .with("learned", learned.clone());
    if b.bool_or("until", false) {
        let mut seqs = Vec::new();
        train_corpus(
            &paths,
            &m.unit(),
            m.lower(),
            held_out,
            &mut |t| {
                seqs.push(t.to_vec());
                0
            },
            &mut Vec::new(),
        )?;
        let max = (b.num("max_rounds", 20.0) as usize).clamp(2, 1000);
        let d = m.drill(&seqs, max - 1, b.num("patience", max as f64) as usize, threshold_of(b));
        let (rows, reached) = drill_json(first, &d);
        j = j
            .with("rounds", rows)
            .with("reached", reached.into())
            .with("best", d.best.into())
            .with("need", d.need.into());
    }
    let ev = m.evaluate(&test);
    let mut rec = ev
        .clone()
        .with("round", m.history().len().into())
        .with("source", paths.join(",").as_str().into());
    for key in ["predicted", "accuracy", "perplexity", "depth"] {
        if let Some(v) = learned.get(key) {
            let name = if key == "predicted" {
                "learned".to_string()
            } else {
                format!("learned_{key}")
            };
            rec = rec.with(&name, v.clone());
        }
    }
    m.history().push(rec);
    Ok(j.with("held_out", ev).with("model", m.info()))
}

/// `POST /api/{model}/text`: pasted text, a line a sequence; with `until` drilled until N of N.
fn lm_text(m: &mut dyn Lm, b: &Json) -> Result<Json, String> {
    let lines = split_text_lines(b.str_or("text", ""));
    m.learning(
        b.bool_or("predict", true) || b.bool_or("until", false),
        b.num("self_credit", 0.0),
        true,
    );
    m.take_online();
    let n = m.train_lines(&lines);
    let first = m.online();
    let mut j = Json::object()
        .with("trained", n.into())
        .with("learned", m.take_online());
    if b.bool_or("until", false) {
        let seqs: Vec<Sequence> = lines.iter().map(|l| m.tokens(l)).filter(|t| !t.is_empty()).collect();
        let max = (b.num("max_rounds", 20.0) as usize).clamp(2, 1000);
        let d = m.drill(&seqs, max - 1, b.num("patience", max as f64) as usize, threshold_of(b));
        let (rows, reached) = drill_json(first, &d);
        j = j
            .with("rounds", rows)
            .with("reached", reached.into())
            .with("best", d.best.into())
            .with("need", d.need.into());
    }
    Ok(j.with("model", m.info()))
}

/// The routes the chain and the N-gram answer alike (`action` is the path after `/api/{model}/`; `""` is the model);
/// `None` when the action is not one of them.
pub fn lm_route(m: &mut dyn Lm, method: &str, action: &str, req: &Request) -> Option<Result<Json, String>> {
    let b = &req.body;
    let r = match (method, action) {
        ("GET", "") => Ok(m.info()),
        ("POST", "train") => lm_train(m, b),
        ("POST", "text") => lm_text(m, b),
        ("POST", "predict") => Ok(m.predict(b.str_or("text", ""), (b.num("k", 10.0) as usize).clamp(1, 200))),
        ("POST", "generate") => {
            let text = b.str_or("text", "");
            let count = (b.num("count", 1.0) as usize).clamp(1, 50);
            let max = (b.num("max_tokens", 30.0) as usize).clamp(1, 5000);
            let (t, seed) = (b.num("temperature", 0.0), b.num("seed", 1.0) as u64);
            let draw = Draw {
                max,
                temperature: t,
                seed,
                count,
                over: 1,
                auto: auto_of(b),
            };
            let (samples, _) = guarded_samples(m, text, &draw, None);
            let first = samples[0].clone();
            Ok(Json::object()
                .with("text", text.into())
                .with("unit", m.unit().as_str().into())
                .with("completion", first.get("completion").cloned().unwrap_or(Json::Null))
                .with("tokens", first.get("tokens").cloned().unwrap_or(Json::Null))
                .with("samples", Json::Array(samples)))
        }
        ("POST", "credit") => {
            let n = m.credit_text(b.str_or("text", ""), b.num("amount", 1.0));
            Ok(Json::object().with("credited", n.into()).with("model", m.info()))
        }
        ("POST", "feedback") => {
            let a = b.num("amount", 1.0).abs();
            let texts = |key: &str| -> Vec<String> {
                b.get(key)
                    .and_then(Json::as_array)
                    .map(|v| v.iter().filter_map(|t| t.as_str().map(String::from)).collect())
                    .unwrap_or_default()
            };
            let rewarded: usize = texts("good").iter().map(|t| m.credit_text(t, a)).sum();
            let punished: usize = texts("bad").iter().map(|t| m.credit_text(t, -a)).sum();
            Ok(Json::object()
                .with("rewarded", rewarded.into())
                .with("punished", punished.into())
                .with("model", m.info()))
        }
        ("POST", "score") => {
            let text = b.str_or("text", "");
            let scored = m.score(text);
            let logp: f64 = scored.iter().map(|(_, p, _)| p.max(1e-300).ln()).sum();
            let n = scored.len().max(1) as f64;
            let tokens: Vec<Json> = scored
                .iter()
                .map(|(t, p, d)| {
                    let mut j = Json::object().with("token", t.as_str().into()).with("p", (*p).into());
                    if let Some(d) = d {
                        j = j.with("depth", (*d).into());
                    }
                    j
                })
                .collect();
            Ok(Json::object()
                .with("text", text.into())
                .with("unit", m.unit().as_str().into())
                .with("tokens", Json::Array(tokens))
                .with("logp", logp.into())
                .with("perplexity", (-logp / n).exp().into())
                .with("bits", (-logp / n / 2f64.ln()).into()))
        }
        ("POST", "teach") => {
            let (text, reply) = (b.str_or("text", ""), b.str_or("reply", ""));
            let (tt, rt) = (m.tokens(text), m.tokens(reply));
            let t = m.teach(
                &tt,
                &rt,
                (b.num("max_rounds", 100.0) as usize).min(100_000),
                threshold_of(b),
            );
            let mut rng = Rng::new(1);
            let (out, _) = m.continue_tokens(&m.prompt_tokens(text), 2 * t.of + 10, 0.0, &mut rng);
            let completion = completion_text(m, text, &out);
            Ok(t.to_json()
                .with("completion", completion.into())
                .with("model", m.info()))
        }
        ("POST", "save") => {
            let path = b.str_or("path", &format!("{}.json.gz", m.name())).to_string();
            m.save(&path)
                .map(|_| Json::object().with("saved", path.as_str().into()))
        }
        _ => return None,
    };
    Some(r)
}

/// The chain's own routes: a new chain, a file, the threshold, the sweep, the most-seen node contexts.
pub fn chain_route(m: &mut Chain, method: &str, action: &str, req: &Request) -> Result<Json, String> {
    if let Some(r) = lm_route(m, method, action, req) {
        return r;
    }
    let b = &req.body;
    match (method, action) {
        ("POST", "new") => {
            *m = Chain::new(
                b.num("nodes", 8.0) as usize,
                b.str_or("unit", "char"),
                b.bool_or("lower", true),
                b.num("confidence", 0.9),
            )?;
            Ok(m.info())
        }
        ("POST", "load") => {
            *m = load_chain(b.str_or("path", "chain.json.gz"))?;
            Ok(m.info())
        }
        ("POST", "confidence") => {
            let c = b.num("confidence", m.confidence);
            if !(0.0..=1.0).contains(&c) {
                return Err(format!("confidence is 0 to 1, got {c}"));
            }
            m.confidence = c;
            Ok(m.info())
        }
        ("POST", "sweep") => {
            let paths = training_paths(b)?;
            let held_out = (b.num("held_out", 10.0) as usize).max(2);
            let mut test = Vec::new();
            let (unit, lower) = (m.unit.clone(), m.lower);
            let (_, kept) = train_corpus(&paths, &unit, lower, held_out, &mut |_| 0, &mut Vec::new())?;
            test.extend(kept);
            let thresholds: Vec<f64> = match b.get("thresholds").and_then(Json::as_array) {
                Some(a) => a.iter().filter_map(Json::as_f64).collect(),
                None => vec![0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0],
            };
            let rows: Vec<Json> = m
                .sweep(&test, &thresholds)
                .iter()
                .map(|(t, ev)| chain_eval_json(ev).with("confidence", (*t).into()))
                .collect();
            Ok(Json::object()
                .with("rows", Json::Array(rows))
                .with("lines", test.len().into()))
        }
        ("GET", "graph") => {
            let limit = req
                .param("limit")
                .and_then(|v| v.parse().ok())
                .unwrap_or(40usize)
                .clamp(1, 1000);
            let mut all: Vec<(u64, usize, &Vec<u32>, &crate::chain::Node)> = Vec::new();
            for (k, t) in m.tables.iter().enumerate().skip(1) {
                for (ctx, nd) in t {
                    all.push((nd.total, k, ctx, nd));
                }
            }
            all.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));
            let nodes: Vec<Json> = all
                .iter()
                .take(limit)
                .map(|(seen, k, ctx, nd)| {
                    Json::object()
                        .with("depth", (*k).into())
                        .with(
                            "context",
                            Json::strings(&ctx.iter().map(|&i| m.vocab[i as usize].clone()).collect::<Vec<_>>()),
                        )
                        .with("seen", (*seen as f64).into())
                        .with(
                            "guess",
                            nd.best
                                .map(|g| Json::from(m.vocab[g as usize].as_str()))
                                .unwrap_or(Json::Null),
                        )
                        .with("confidence", nd.confidence().into())
                        .with("tested", (nd.tested as f64).into())
                        .with("right", (nd.right as f64).into())
                })
                .collect();
            Ok(Json::object().with("nodes", Json::Array(nodes)))
        }
        ("GET", _) | ("POST", _) => Err("no such route".to_string()),
        _ => Err("method not allowed".to_string()),
    }
}

// ---- conversations -------------------------------------------------------------------------------------------------

/// The model talks to itself: the opening line, then `turns` turns, each continuing the one before it (each turn
/// guarded when a guard is given).
pub fn converse(m: &mut dyn Lm, b: &Json, mut guard: Option<&mut Guard>) -> Json {
    let opening = b.str_or("opening", "hello").to_string();
    let turns = (b.num("turns", 6.0) as usize).clamp(1, 100);
    let max = (b.num("max_tokens", 60.0) as usize).clamp(1, 2000);
    let (t, seed) = (b.num("temperature", 0.0), b.num("seed", 1.0) as u64);
    let mut out = vec![Json::object()
        .with("speaker", "A".into())
        .with("text", opening.as_str().into())];
    let mut last = opening;
    let over = (b.num("over_sample", 3.0) as usize).clamp(1, 20);
    for i in 0..turns {
        let (samples, summary) = guarded_samples(
            m,
            &last,
            &Draw {
                max,
                temperature: t,
                seed: seed + 101 * i as u64,
                count: 1,
                over,
                auto: auto_of(b),
            },
            guard.as_deref_mut(),
        );
        let g = samples.into_iter().next().unwrap_or(Json::Null);
        let text = g.str_or("completion", "").trim().to_string();
        let mut turn = Json::object()
            .with("speaker", if i % 2 == 0 { "B" } else { "A" }.into())
            .with("text", text.as_str().into());
        if let Some(d) = g.get("depths") {
            turn = turn.with("depths", d.clone());
        }
        if let Some(a) = g.get("auto") {
            turn = turn.with("auto", a.clone());
        }
        if let (Some(v), Some(sm)) = (g.get("guard"), summary) {
            turn = turn
                .with("guard", v.clone())
                .with("vetoed", sm.get("vetoed").cloned().unwrap_or(Json::Null));
        }
        out.push(turn);
        if text.is_empty() {
            break;
        }
        last = text;
    }
    Json::object().with("turns", Json::Array(out))
}

fn mark_of(text: &str) -> Option<f64> {
    let digits: String = text
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    digits
        .trim_end_matches('.')
        .parse::<f64>()
        .ok()
        .map(|v| v.clamp(0.0, 10.0))
}

/// A local LLM talks with the model: it says a line on the topic, the model continues it as its reply, the LLM marks
/// the reply out of 10, and with `learn` the reply is credited `(mark - pass_mark) / 5`.  With a guard the replies
/// are guarded, and (`teach_guard`, the default) a reply marked under the pass mark is blamed in it - "marked 3/10",
/// severity `(pass_mark - mark) / 5` - and one marked at or above it cleared.
///
/// With `teacher: "answers"` the LLM is an oracle instead of a judge: it answers each of its lines itself, and every
/// token of that answer is a prediction the model is judged on alone (`Lm::credit_answer`) - its guess there, given
/// the line and the answer so far, rewarded `reward` when right and punished as much when wrong (nothing, with
/// `punish: false`).  Each turn carries the answer and how many of its tokens the model got right; the marks are
/// `10 × right / of`.  Credit only re-ranks what the model already predicts, so `learn_answer` also trains it on the
/// line and the answer.
pub fn chat_llm(m: &mut dyn Lm, b: &Json, mut guard: Option<&mut Guard>) -> Result<Json, String> {
    let client = Ollama::new(
        b.get("url").and_then(Json::as_str),
        b.get("ollama_model").and_then(Json::as_str),
    );
    let topic = b.str_or("topic", "the weather").to_string();
    let turns = (b.num("turns", 4.0) as usize).clamp(1, 50);
    let max = (b.num("max_tokens", 60.0) as usize).clamp(1, 2000);
    let (t, seed) = (b.num("temperature", 0.0), b.num("seed", 1.0) as u64);
    let pass = b.num("pass_mark", 5.0);
    let learn = b.bool_or("learn", true);
    // `teacher: "answers"`: the teacher answers every line itself, and the model is judged token by token against
    // that answer (rewarded `reward` when right, punished as much unless `punish` is false); else it marks the reply
    let answers = b.str_or("teacher", "marks") == "answers";
    let reward = b.num("reward", 1.0).abs();
    let punish = if b.bool_or("punish", true) { reward } else { 0.0 };
    let mut transcript: Vec<Json> = Vec::new();
    let mut said: Vec<String> = Vec::new();
    let mut marks = Vec::new();
    for i in 0..turns {
        let so_far = if said.is_empty() {
            "(nothing yet)".to_string()
        } else {
            said.join("\n")
        };
        let prompt = format!(
            "You are chatting with a small language model about {topic}. Say one short, natural sentence to it (at \
             most 20 words), carrying the conversation on. Reply with the sentence only.\n\nThe conversation so far:\n\
             {so_far}\n\nYour sentence:"
        );
        let line = parse_lines(
            &client.generate(&prompt, 0.7, Some(60), Some(seed + i as u64))?,
            Some(1),
        )
        .into_iter()
        .next()
        .unwrap_or_default();
        if line.is_empty() {
            break;
        }
        transcript.push(
            Json::object()
                .with("speaker", "llm".into())
                .with("text", line.as_str().into()),
        );
        said.push(format!("You: {line}"));
        let over = (b.num("over_sample", 3.0) as usize).clamp(1, 20);
        let (samples, summary) = guarded_samples(
            m,
            &line,
            &Draw {
                max,
                temperature: t,
                seed: seed + 101 * i as u64,
                count: 1,
                over,
                auto: auto_of(b),
            },
            guard.as_deref_mut(),
        );
        let g = samples.into_iter().next().unwrap_or(Json::Null);
        let reply = g.str_or("completion", "").trim().to_string();
        said.push(format!("Model: {reply}"));
        let judge = format!(
            "A small language model was told: \"{line}\"\nIt replied: \"{reply}\"\nMark the reply from 0 (nonsense or \
             nothing) to 10 (a fluent, fitting reply). Answer with the number only."
        );
        if answers {
            // the teacher answers the line itself: its answer is the oracle every token of the model's is judged by
            let ask = format!(
                "Reply to this line as a fluent speaker would, in one short, natural sentence (at most 20 words). Reply \
                 with the sentence only.\n\nThe line: \"{line}\"\nYour reply:"
            );
            let answer = parse_lines(&client.generate(&ask, 0.0, Some(60), Some(seed + i as u64))?, Some(1))
                .into_iter()
                .next()
                .unwrap_or_default();
            let mut turn = Json::object()
                .with("speaker", "model".into())
                .with("text", reply.as_str().into())
                .with("answer", answer.as_str().into());
            if let Some(a) = g.get("auto") {
                turn = turn.with("auto", a.clone());
            }
            if let (Some(v), Some(sm)) = (g.get("guard"), summary) {
                turn = turn
                    .with("guard", v.clone())
                    .with("vetoed", sm.get("vetoed").cloned().unwrap_or(Json::Null));
            }
            // how well the model predicts the teacher's answer, token by token - credited, when learning, token by
            // token: each guess rewarded when right, punished (unless rewards only) when wrong
            let (right, of) = if learn {
                m.credit_answer(&line, &answer, reward, punish)
            } else {
                m.credit_answer(&line, &answer, 0.0, 0.0)
            };
            let share = if of > 0 { right as f64 / of as f64 } else { 0.0 };
            marks.push(10.0 * share);
            // credit only re-ranks what the model already predicts; `learn_answer` also counts the answer - imitation,
            // not reward - so a word it has never seen can come to be predicted at all
            if b.bool_or("learn_answer", false) && !answer.is_empty() {
                let learned = m.train_lines(&[joined(m, &line, &answer)]);
                turn = turn.with("learned", learned.into());
            }
            turn = turn
                .with("right", right.into())
                .with("of", of.into())
                .with("share", share.into());
            said.push(format!("You (the right reply): {answer}"));
            transcript.push(turn);
            continue;
        }
        let mark = mark_of(&client.generate(&judge, 0.0, Some(8), Some(seed + i as u64))?).unwrap_or(0.0);
        marks.push(mark);
        let mut turn = Json::object()
            .with("speaker", "model".into())
            .with("text", reply.as_str().into())
            .with("mark", mark.into());
        if let Some(a) = g.get("auto") {
            turn = turn.with("auto", a.clone());
        }
        if let (Some(v), Some(sm)) = (g.get("guard"), summary) {
            turn = turn
                .with("guard", v.clone())
                .with("vetoed", sm.get("vetoed").cloned().unwrap_or(Json::Null));
        }
        if let Some(gd) = guard.as_deref_mut() {
            if b.bool_or("teach_guard", true) && !reply.is_empty() {
                if mark < pass {
                    gd.blame(&reply, &format!("marked {mark}/10"), (pass - mark) / 5.0);
                    turn = turn.with("blamed", true.into());
                } else {
                    gd.clear(&reply, 1.0);
                }
            }
        }
        if learn {
            let amount = (mark - pass) / 5.0;
            let credited = if amount != 0.0 {
                m.credit_text(&joined(m, &line, &reply), amount)
            } else {
                0
            };
            turn = turn.with("credited", credited.into()).with("amount", amount.into());
        }
        transcript.push(turn);
    }
    let mean = if marks.is_empty() {
        0.0
    } else {
        marks.iter().sum::<f64>() / marks.len() as f64
    };
    Ok(Json::object()
        .with("teacher", if answers { "answers" } else { "marks" }.into())
        .with("transcript", Json::Array(transcript))
        .with("marks", Json::numbers(&marks))
        .with("mean", mean.into())
        .with("model", m.info()))
}

/// Sentences a local LLM writes on a prompt.
pub fn ollama_corpus(b: &Json) -> Result<Json, String> {
    let client = Ollama::new(
        b.get("url").and_then(Json::as_str),
        b.get("ollama_model").and_then(Json::as_str),
    );
    let n = (b.num("lines", 20.0) as usize).clamp(1, 500);
    let prompt = format!(
        "Write {n} different, natural English sentences about {}. Write one sentence per line, with no numbering and \
         nothing else.",
        b.str_or("prompt", "everyday life")
    );
    let text = client.generate(&prompt, 0.8, None, Some(b.num("seed", 1.0) as u64))?;
    Ok(Json::object()
        .with("lines", Json::strings(&parse_lines(&text, Some(n))))
        .with("model", client.model.as_str().into()))
}

// ---- files: uploads and checkpoints --------------------------------------------------------------------------------

fn dir_from(env: &str, under_data: &str, plain: &str) -> PathBuf {
    if let Ok(d) = std::env::var(env) {
        if !d.is_empty() {
            return PathBuf::from(d);
        }
    }
    for data in ["data", "../data"] {
        if Path::new(data).is_dir() && !under_data.is_empty() {
            return Path::new(data).join(under_data);
        }
    }
    PathBuf::from(plain)
}

/// Where uploads go: `$LATTICEFSM_UPLOADS`, else `data/uploads` (or `../data/uploads`), else `uploads`.
pub fn uploads_dir() -> PathBuf {
    dir_from("LATTICEFSM_UPLOADS", "uploads", "uploads")
}

/// Where checkpoints go: `$LATTICEFSM_CHECKPOINTS`, else `checkpoints`.
pub fn checkpoints_dir() -> PathBuf {
    dir_from("LATTICEFSM_CHECKPOINTS", "", "checkpoints")
}

/// A relative file name made safe: every part of `a/b/c` kept to letters, digits, `.`, `_`, `-` and spaces, no part
/// empty, `.` or `..`, none hidden.
pub fn sanitize(name: &str) -> Result<String, String> {
    let parts: Vec<String> = name
        .split(['/', '\\'])
        .filter(|p| !p.is_empty() && *p != "." && *p != "..")
        .map(|p| {
            let clean: String = p
                .chars()
                .map(|c| {
                    if c.is_alphanumeric() || matches!(c, '.' | '_' | '-' | ' ') {
                        c
                    } else {
                        '_'
                    }
                })
                .collect();
            if clean.starts_with('.') {
                format!("_{clean}")
            } else {
                clean
            }
        })
        .collect();
    if parts.is_empty() {
        return Err(format!("not a file name: {name:?}"));
    }
    Ok(parts.join("/"))
}

fn modified(p: &Path) -> f64 {
    fs::metadata(p)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
        .map(|d| d.as_secs() as f64)
        .unwrap_or(0.0)
}

fn list_files(root: &Path, dir: &Path, out: &mut Vec<(String, PathBuf)>) {
    let Ok(rd) = fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            list_files(root, &p, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            out.push((rel.to_string_lossy().replace('\\', "/"), p.clone()));
        }
    }
}

/// Write (or append to) a file under `dir` from a body's `content_base64` or `content`.
fn write_file(dir: &Path, b: &Json) -> Result<(String, PathBuf), String> {
    let name = sanitize(b.str_or("name", ""))?;
    let path = dir.join(&name);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let bytes = match b.get("content_base64").and_then(Json::as_str) {
        Some(s) => crate::b64::decode(s),
        None => b.str_or("content", "").as_bytes().to_vec(),
    };
    let mut f = fs::OpenOptions::new()
        .create(true)
        .write(true)
        .append(b.bool_or("append", false))
        .truncate(!b.bool_or("append", false))
        .open(&path)
        .map_err(|e| format!("{name}: {e}"))?;
    std::io::Write::write_all(&mut f, &bytes).map_err(|e| format!("{name}: {e}"))?;
    Ok((name, path))
}

/// `GET /api/uploads`, `POST /api/uploads`, `POST /api/uploads/delete`.
pub fn uploads_route(method: &str, action: &str, b: &Json) -> Result<Json, String> {
    let dir = uploads_dir();
    match (method, action) {
        ("GET", "") => {
            let mut files = Vec::new();
            list_files(&dir, &dir, &mut files);
            let uploads: Vec<Json> = files
                .iter()
                .map(|(name, p)| {
                    Json::object()
                        .with("name", name.as_str().into())
                        .with("bytes", (fs::metadata(p).map(|m| m.len()).unwrap_or(0) as f64).into())
                        .with("modified", modified(p).into())
                        .with(
                            "kind",
                            crate::corpus::file_kind(name).map(Json::from).unwrap_or(Json::Null),
                        )
                })
                .collect();
            Ok(Json::object()
                .with("dir", dir.to_string_lossy().as_ref().into())
                .with("uploads", Json::Array(uploads)))
        }
        ("POST", "") => {
            let (name, path) = write_file(&dir, b)?;
            let bytes = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            Ok(Json::object()
                .with("name", name.as_str().into())
                .with("bytes", (bytes as f64).into()))
        }
        ("POST", "delete") => {
            let name = sanitize(b.str_or("name", ""))?;
            let path = dir.join(&name);
            fs::remove_file(&path).map_err(|e| format!("{name}: {e}"))?;
            let mut parent = path.parent().map(Path::to_path_buf);
            while let Some(p) = parent {
                if p == dir || fs::remove_dir(&p).is_err() {
                    break;
                }
                parent = p.parent().map(Path::to_path_buf);
            }
            Ok(Json::object().with("deleted", name.as_str().into()))
        }
        ("GET", _) | ("POST", _) => Err("no such route".to_string()),
        _ => Err("method not allowed".to_string()),
    }
}

/// The models a checkpoint can hold.
pub const CHECKPOINT_MODELS: [&str; 5] = ["chain", "ngram", "machine", "network", "guard"];

/// `{model}--{tag}--{unix seconds}.json.gz`, a new name (a counter added if the second is taken).
pub fn checkpoint_name(model: &str, tag: &str) -> Result<String, String> {
    if !CHECKPOINT_MODELS.contains(&model) {
        return Err(format!("model is one of {CHECKPOINT_MODELS:?}, got {model:?}"));
    }
    let tag: String = tag
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
        .take(40)
        .collect();
    let tag = if tag.is_empty() {
        "manual".to_string()
    } else {
        tag.replace("--", "-")
    };
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let dir = checkpoints_dir();
    let mut name = format!("{model}--{tag}--{secs}.json.gz");
    let mut k = 2;
    while dir.join(&name).exists() {
        name = format!("{model}--{tag}--{secs}-{k}.json.gz");
        k += 1;
    }
    Ok(name)
}

/// The model a checkpoint file holds, by its name.
pub fn checkpoint_model(name: &str) -> Option<&'static str> {
    let prefix = name.split("--").next()?;
    CHECKPOINT_MODELS.iter().copied().find(|m| *m == prefix)
}

/// Every checkpoint, newest first.
pub fn checkpoints_list() -> Json {
    let dir = checkpoints_dir();
    let mut files = Vec::new();
    list_files(&dir, &dir, &mut files);
    let mut rows: Vec<(f64, String, PathBuf)> = files
        .into_iter()
        .filter(|(n, _)| !n.contains('/') && (n.ends_with(".json.gz") || n.ends_with(".json")))
        .map(|(n, p)| (modified(&p), n, p))
        .collect();
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(b.1.cmp(&a.1))
    });
    let list: Vec<Json> = rows
        .iter()
        .map(|(t, n, p)| {
            let parts: Vec<&str> = n.split("--").collect();
            Json::object()
                .with("name", n.as_str().into())
                .with("model", checkpoint_model(n).map(Json::from).unwrap_or(Json::Null))
                .with("tag", parts.get(1).map(|s| Json::from(*s)).unwrap_or(Json::Null))
                .with("saved_at", (*t).into())
                .with("bytes", (fs::metadata(p).map(|m| m.len()).unwrap_or(0) as f64).into())
        })
        .collect();
    Json::object()
        .with("dir", dir.to_string_lossy().as_ref().into())
        .with(
            "latest",
            rows.first().map(|r| Json::from(r.1.as_str())).unwrap_or(Json::Null),
        )
        .with("checkpoints", Json::Array(list))
}

/// A checkpoint's path, checked: a plain name in the checkpoint directory.
pub fn checkpoint_path(name: &str) -> Result<PathBuf, String> {
    let clean = sanitize(name)?;
    if clean.contains('/') || clean != name {
        return Err(format!("not a checkpoint name: {name:?}"));
    }
    Ok(checkpoints_dir().join(clean))
}

/// `POST /api/checkpoints/upload`: a checkpoint file sent in (chunks appended as uploads are).
pub fn checkpoint_upload(b: &Json) -> Result<Json, String> {
    let name = sanitize(b.str_or("name", ""))?;
    if name.contains('/') || !(name.ends_with(".json.gz") || name.ends_with(".json")) {
        return Err("a checkpoint is a .json or .json.gz file".to_string());
    }
    let dir = checkpoints_dir();
    fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let (name, _) = write_file(&dir, b)?;
    Ok(Json::object().with("name", name.as_str().into()))
}

// ---- the guard -------------------------------------------------------------------------------------------------------

/// A body's texts: `texts` (a list) or `text` (one).
pub fn texts_of(b: &Json) -> Vec<String> {
    match b.get("texts").and_then(Json::as_array) {
        Some(a) => a.iter().filter_map(|t| t.as_str().map(String::from)).collect(),
        None => b
            .get("text")
            .and_then(Json::as_str)
            .map(|t| vec![t.to_string()])
            .unwrap_or_default(),
    }
}

/// `/api/guard`: the stats; a new guard; its settings; blame, clear, judge, forget, reset; save and load.
pub fn guard_route(g: &mut Guard, method: &str, action: &str, b: &Json) -> Result<Json, String> {
    match (method, action) {
        ("GET", "") => Ok(g.stats()),
        ("POST", "new") => {
            *g = Guard::new(
                b.str_or("unit", "word"),
                b.num("order", 3.0) as usize,
                b.bool_or("lower", true),
                b.num("threshold", 1.0),
                b.num("min_coverage", 0.5),
            )?;
            Ok(g.stats())
        }
        ("POST", "settings") => {
            g.threshold = b.num("threshold", g.threshold).max(0.0);
            g.min_coverage = b.num("min_coverage", g.min_coverage).clamp(0.0, 1.0);
            Ok(g.stats())
        }
        ("POST", "blame") => {
            let (reason, severity) = (b.str_or("reason", ""), b.num("severity", 1.0));
            let n: usize = texts_of(b).iter().map(|t| g.blame(t, reason, severity)).sum();
            Ok(Json::object().with("blamed", n.into()).with("guard", g.stats()))
        }
        ("POST", "clear") => {
            let amount = b.num("amount", 1.0);
            let n: usize = texts_of(b).iter().map(|t| g.clear(t, amount)).sum();
            Ok(Json::object().with("cleared", n.into()).with("guard", g.stats()))
        }
        ("POST", "judge") => {
            let (th, cov) = (
                b.get("threshold").and_then(Json::as_f64),
                b.get("min_coverage").and_then(Json::as_f64),
            );
            let verdicts: Vec<Json> = texts_of(b).iter().map(|t| g.judge(t, th, cov, 5)).collect();
            Ok(Json::object().with("verdicts", Json::Array(verdicts)))
        }
        ("POST", "forget") => {
            let n = g.forget(b.str_or("reason", ""));
            Ok(Json::object().with("forgotten", n.into()).with("guard", g.stats()))
        }
        ("POST", "reset") => {
            g.reset();
            Ok(g.stats())
        }
        ("POST", "save") => {
            let path = b.str_or("path", "guard.json.gz").to_string();
            g.save(&path)
                .map(|_| Json::object().with("saved", path.as_str().into()))
        }
        ("POST", "load") => {
            *g = crate::guard::load_guard(b.str_or("path", "guard.json.gz"))?;
            Ok(g.stats())
        }
        ("GET", _) | ("POST", _) => Err("no such route".to_string()),
        _ => Err("method not allowed".to_string()),
    }
}

/// `POST /api/{model}/generate` with `guard: true`: the samples guarded (`over_sample`, default 3).
pub fn generate_guarded(m: &mut dyn Lm, b: &Json, g: &mut Guard) -> Json {
    let text = b.str_or("text", "");
    let count = (b.num("count", 1.0) as usize).clamp(1, 50);
    let max = (b.num("max_tokens", 30.0) as usize).clamp(1, 5000);
    let (t, seed) = (b.num("temperature", 0.0), b.num("seed", 1.0) as u64);
    let over = (b.num("over_sample", 3.0) as usize).clamp(1, 20);
    let (samples, summary) = guarded_samples(
        m,
        text,
        &Draw {
            max,
            temperature: t,
            seed,
            count,
            over,
            auto: auto_of(b),
        },
        Some(g),
    );
    let first = samples[0].clone();
    Json::object()
        .with("text", text.into())
        .with("unit", m.unit().as_str().into())
        .with("completion", first.get("completion").cloned().unwrap_or(Json::Null))
        .with("tokens", first.get("tokens").cloned().unwrap_or(Json::Null))
        .with("samples", Json::Array(samples))
        .with("guard", summary.unwrap_or(Json::Null))
}

/// `POST /api/{model}/feedback` with `guard: true`: the model credited, and the bad texts blamed in the guard
/// (`reason`, default "feedback") and the good ones cleared.
pub fn feedback_guarded(m: &mut dyn Lm, b: &Json, g: &mut Guard) -> Json {
    let a = b.num("amount", 1.0).abs();
    let list = |key: &str| -> Vec<String> {
        b.get(key)
            .and_then(Json::as_array)
            .map(|v| v.iter().filter_map(|t| t.as_str().map(String::from)).collect())
            .unwrap_or_default()
    };
    let (good, bad) = (list("good"), list("bad"));
    let rewarded: usize = good.iter().map(|t| m.credit_text(t, a)).sum();
    let punished: usize = bad.iter().map(|t| m.credit_text(t, -a)).sum();
    let reason = b.str_or("reason", "feedback");
    let blamed: usize = bad.iter().map(|t| g.blame(t, reason, a)).sum();
    let cleared: usize = good.iter().map(|t| g.clear(t, a)).sum();
    Json::object()
        .with("rewarded", rewarded.into())
        .with("punished", punished.into())
        .with("blamed", blamed.into())
        .with("cleared", cleared.into())
        .with("model", m.info())
}
