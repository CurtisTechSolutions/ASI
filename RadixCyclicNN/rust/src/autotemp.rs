//! Auto-temperature: a search that finds no answer walks again, hotter
//! (`../../SPEC-SearchAndTraining.md` section 10).
//!
//! The Rust twin of `radixnet/autotemp.py`, kept walk for walk.  The
//! temperature is the one dial that trades how likely a walk is for how far it
//! strays: a sampled step weighs every option `p ** (1 / temperature)`
//! (section 1), so a cold walk keeps to the likeliest paths - and, in a cyclic
//! graph, goes round them - while a hot one reaches paths the likely ones crowd
//! out.  When a search comes back with no answer, [`climb`] asks again up a
//! *ladder* of temperatures, a few walks at every rung, and stops at the first
//! answer:
//!
//! * **The search as asked runs first**, exactly as it would have - any mode,
//!   any temperature - and when it answers, nothing else happens.
//! * **Then the ladder** ([`temperature_ladder`]): the asked temperature, then
//!   doubling - from [`AUTO_FLOOR`] when the asked one was greedy - up to a
//!   ceiling, the last rung clamped to it.  Each rung is [`AUTO_DRAWS`] sampled
//!   walks (one at temperature 0, where every walk is the same): an empty or
//!   vetoed walk is mended by another draw as often as by a hotter one, and a
//!   hotter walk is a less likely one, so the temperature rises only when the
//!   draws at this one ran out.
//! * **An answer** is a walk that wrote something and, when the caller asked
//!   for the end of a text, reached END ([`answered`]); a caller with a guard
//!   adds its veto ([`climb`] takes any pick).  A search that offers several
//!   continuations - the beam's top K - has found one when *any* of them
//!   answers: the first that does becomes the prediction ([`first_answer`]),
//!   as the guard promotes the first one it lets through, and nothing is
//!   walked.
//!
//! A deterministic search - Dijkstra, the beam, the resonant model's k-best -
//! reads no temperature at all, so for it the ladder is where sampling starts:
//! the search found nothing at the cheapest paths, and walks at rising
//! temperature look among the others.  Where the graph has no way on at all, no
//! temperature finds one, and the record says so.
//!
//! The walks draw from the generator the search was already drawing from - the
//! model's own, one search after the other - so a climb is reproducible and
//! the three ports draw alike.  Off by default (a ceiling of 0): no extra
//! search, no draw, the answer exactly the one the search gave.

use crate::beam::Prediction;
use crate::encoding::Encoding;
use crate::json::{py_repr, Json};
use crate::model::{GenerateOptions, Model, PredictOptions};
use crate::search::PathResult;

/// Sampled walks drawn at every rung of the ladder before the temperature
/// rises.
///
/// Measured on a count model of 1,500 lines of prose: an empty continuation
/// comes back from 4% of the walks at temperature 0.5, 6% at 1 and 13% at 4 -
/// a hotter walk is no likelier to say something, so a second and a third
/// draw mend more than the next rung does.  Three draws at a rung that answers
/// 94% of the time leave one search in five thousand with nothing to say.
pub const AUTO_DRAWS: usize = 3;

/// The first temperature above greedy: a ladder that starts at 0 climbs to
/// this, then doubles.
pub const AUTO_FLOOR: f64 = 0.5;

/// The error for a ceiling out of range: it must be a finite number `>= 0`
/// (`0` is off).
pub fn check_auto_temperature(ceiling: f64) -> Result<(), String> {
    if ceiling.is_finite() && ceiling >= 0.0 {
        return Ok(());
    }
    Err(format!(
        "auto_temperature must be a finite number >= 0 (0 is off), got {}",
        float_repr(ceiling)
    ))
}

/// A float as Python's `repr` writes it in a message: `-1.0`, `nan`, `inf`.
fn float_repr(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        py_repr(x)
    }
}

/// The temperatures a search climbs through: `start`, then doubling up to
/// `ceiling`.
///
/// The rung after `t` is `min(max(2 * t, AUTO_FLOOR), ceiling)`, so a greedy
/// start climbs to [`AUTO_FLOOR`] first and the last rung is the ceiling
/// itself: `(1, 2)` is `[1, 2]`, `(1, 3)` is `[1, 2, 3]`, `(0, 2)` is
/// `[0, 0.5, 1, 2]`.  A start at or above the ceiling is the one rung - the
/// ladder never cools a walk - and a ceiling of 0 is no ladder.  Temperature
/// divides a log-probability, so the rungs are spaced by a factor rather than
/// a step: each doubling is the same change in how flat a step is.
pub fn temperature_ladder(start: f64, ceiling: f64) -> Vec<f64> {
    // Python's `not ceiling > 0`: a NaN ceiling is no ladder either
    if ceiling.is_nan() || ceiling <= 0.0 {
        return Vec::new();
    }
    let mut rungs = vec![start];
    let mut t = start;
    while t < ceiling {
        t = (2.0 * t).max(AUTO_FLOOR).min(ceiling);
        rungs.push(t);
    }
    rungs
}

/// The sampled walks a rung at temperature `t` draws: [`AUTO_DRAWS`], or one
/// at temperature 0, where a walk is the same every time.
pub fn draws_at(t: f64) -> usize {
    if t == 0.0 {
        1
    } else {
        AUTO_DRAWS
    }
}

/// Python's whitespace (`str.isspace()`): Unicode `White_Space`, and the four
/// ASCII separators `\x1c`-`\x1f` that Rust's does not count.
fn is_python_space(c: char) -> bool {
    c.is_whitespace() || ('\x1c'..='\x1f').contains(&c)
}

/// Is this an answer: did the search write something - more than whitespace -
/// and, when `to_end` asked for the end of a text, did it reach END?
pub fn answered(result: &PathResult, to_end: bool) -> bool {
    !result.text.chars().all(is_python_space) && (result.reached_end || !to_end)
}

/// The search's answer: `found` itself when its best path is one, else the
/// first of the continuations it offered (its `top`, in the order the search
/// ranked them) that is, made the prediction - else `None`.
///
/// The promoted path's text, labels, node ids, cost, steps, END and blame
/// replace the best path's; the search's own `expanded` (the prediction's and
/// the best path's), `top`, `bottom`, `k`, `beam`, `mode` and `traversal` stay
/// as the search left them.  The beam can rank a path cut at the length cap
/// above the ones that reached END - a walk round a cycle is cheap - so a
/// request for the end of a text may find its answer second or third: that is
/// still the search's own answer, and better than any walk at a higher
/// temperature.
pub fn first_answer(found: &Prediction, to_end: bool) -> Option<Prediction> {
    if answered(&found.best, to_end) {
        return Some(found.clone());
    }
    let candidate = found.top.iter().find(|c| answered(c, to_end))?;
    let mut promoted = found.clone();
    promoted.best = PathResult {
        expanded: found.best.expanded,
        ..candidate.clone()
    };
    Some(promoted)
}

/// What a climb did: the ceiling it was given, whether the search as asked had
/// to be retried (`climbed`), whether an answer came back (`found`), the
/// temperature of the walk that gave it (`None` when the answer is a
/// deterministic search's, or there is none) and how many searches ran, the
/// one as asked included (`tries`).
///
/// With nothing found, what comes back is the search as asked, and
/// `temperature` is `None`: no walk answered.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AutoTemperature {
    pub ceiling: f64,
    pub climbed: bool,
    pub found: bool,
    pub temperature: Option<f64>,
    pub tries: usize,
}

impl AutoTemperature {
    /// The record of a climb to `ceiling` that has run the search as asked,
    /// and nothing else yet.
    pub fn new(ceiling: f64) -> AutoTemperature {
        AutoTemperature {
            ceiling,
            tries: 1,
            ..Default::default()
        }
    }

    /// `{"ceiling", "climbed", "found", "temperature", "tries"}` - the record
    /// every answer carries as `auto_temperature` while the ceiling is on.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("ceiling", Json::Num(self.ceiling)),
            ("climbed", Json::Bool(self.climbed)),
            ("found", Json::Bool(self.found)),
            ("temperature", self.temperature.map(Json::Num).unwrap_or(Json::Null)),
            ("tries", Json::Int(self.tries as i64)),
        ])
    }

    /// The record read back from its JSON; `None` for anything that is not an
    /// object.
    pub fn from_json(doc: &Json) -> Option<AutoTemperature> {
        let Json::Obj(_) = doc else { return None };
        Some(AutoTemperature {
            ceiling: doc.at("ceiling").as_f64().unwrap_or(0.0),
            climbed: doc.at("climbed").as_bool().unwrap_or(false),
            found: doc.at("found").as_bool().unwrap_or(false),
            temperature: doc.at("temperature").as_f64(),
            tries: doc.at("tries").as_i64().unwrap_or(0).max(0) as usize,
        })
    }
}

/// Runs the search as asked and, while it gives no answer, sampled walks up
/// the ladder.
///
/// `search(None)` runs the caller's search exactly as it was asked for, and
/// `search(Some(t))` the same request as a sampled walk at temperature `t`;
/// `pick` makes an answer of what came back, or `None` when there is none in
/// it: the model's own test ([`first_answer`]), and whatever the caller adds -
/// a guard's veto.  `temperature` is the one the request carries and `sampled`
/// whether the search as asked was itself a sampled walk (then it is the
/// temperature of an answer it gives).  The ladder is [`temperature_ladder`]
/// from `temperature` to `ceiling`, [`draws_at`] sampled walks at every rung,
/// and the first answer ends it.  When none comes, the result is the search as
/// asked - the most faithful thing there is to return - and the record says
/// nothing was found.  A search that fails ends the climb with its error.
pub fn climb<R>(
    mut search: impl FnMut(Option<f64>) -> Result<R, String>,
    mut pick: impl FnMut(&R) -> Option<R>,
    temperature: f64,
    ceiling: f64,
    sampled: bool,
) -> Result<(R, AutoTemperature), String> {
    let first = search(None)?;
    let mut record = AutoTemperature::new(ceiling);
    if let Some(chosen) = pick(&first) {
        record.found = true;
        record.temperature = sampled.then_some(temperature);
        return Ok((chosen, record));
    }
    record.climbed = true;
    for t in temperature_ladder(temperature, ceiling) {
        for _ in 0..draws_at(t) {
            record.tries += 1;
            if let Some(chosen) = pick(&search(Some(t))?) {
                record.found = true;
                record.temperature = Some(t);
                return Ok((chosen, record));
            }
        }
    }
    Ok((first, record))
}

// -- the two searches every kind has ----------------------------------------------------------

/// The request a climb sends: `None` the one as asked, a temperature the same
/// request as a sampled walk at it.
pub fn at_temperature(o: &PredictOptions, t: Option<f64>) -> PredictOptions {
    match t {
        None => o.clone(),
        Some(t) => PredictOptions {
            mode: "sample".to_string(),
            temperature: t,
            ..o.clone()
        },
    }
}

/// `model.predict(prefix, o)`, climbing the ladder when it finds no answer.
///
/// Returns `(prediction, record)`; the record is `None` when
/// `auto_temperature` is 0 (off), and the search is then exactly the one
/// [`Model::predict`] runs.  Any kind, any mode: the walks up the ladder are
/// the same request in `sample` mode at each rung's temperature, drawn from the
/// model's own generator.
pub fn predict(
    model: &mut Model,
    prefix: &str,
    auto_temperature: f64,
    o: &PredictOptions,
) -> Result<(Prediction, Option<AutoTemperature>), String> {
    check_auto_temperature(auto_temperature)?;
    if auto_temperature == 0.0 {
        return Ok((model.predict(prefix, o)?, None));
    }
    let (found, record) = climb(
        |t| model.predict(prefix, &at_temperature(o, t)),
        |found| first_answer(found, o.to_end),
        o.temperature,
        auto_temperature,
        o.mode.to_lowercase() == "sample",
    )?;
    Ok((found, Some(record)))
}

/// Did a generated text say anything beyond its prefix?  A generated `text`
/// is the whole text, prefix included, so it is measured against the prefix in
/// the model's units.
pub fn wrote(enc: &Encoding, result: &PathResult, prefix: &str) -> bool {
    !result.text.chars().all(is_python_space) && enc.len(&result.text) > enc.len(prefix)
}

/// The temperature of an answer the generation as asked gave: the asked one
/// when it sampled - its mode, `sample` when it names none, as
/// [`Model::generate`] reads it - and `None` for the beam and Dijkstra, which
/// read no temperature.
pub fn asked_temperature(o: &GenerateOptions) -> Option<f64> {
    let sampled = o.mode.is_empty() || o.mode.to_lowercase() == "sample";
    sampled.then_some(o.temperature)
}

/// The generation the `k`-th retry of a climb asks for: `need` sampled walks
/// at temperature `t`, and - when the caller gave a seed - a generator of
/// their own seeded `seed + k`, so a retry at the same temperature is a new
/// draw and the whole climb is reproducible.
pub fn retry_generation(o: &GenerateOptions, need: usize, t: f64, k: usize) -> GenerateOptions {
    GenerateOptions {
        count: need,
        mode: "sample".to_string(),
        temperature: t,
        seed: o.seed.map(|seed| seed.wrapping_add(k as i64)),
        ..o.clone()
    }
}

/// `model.generate(o)`, with every sample that said nothing walked again up
/// the ladder.
///
/// Returns `(results, record)`; the record is `None` when `auto_temperature`
/// is 0 (off), and the generation is then exactly [`Model::generate`]'s.  The
/// generation as asked runs first; when fewer than `o.count` of its texts say
/// anything beyond the prefix ([`wrote`]), the missing ones are drawn as
/// sampled walks up the ladder, [`draws_at`] generations a rung, each asking
/// for as many as are still missing ([`retry_generation`]).  What comes back
/// is then the texts that said something - those of the generation as asked,
/// in its order, then the ones the ladder found - and fewer than `o.count` only
/// when even the ceiling found too few.
pub fn generate(
    model: &mut Model,
    auto_temperature: f64,
    o: &GenerateOptions,
) -> Result<(Vec<PathResult>, Option<AutoTemperature>), String> {
    check_auto_temperature(auto_temperature)?;
    if auto_temperature == 0.0 {
        return Ok((model.generate(o)?, None));
    }
    let enc = model.encoding();
    let results = model.generate(o)?;
    let mut record = AutoTemperature::new(auto_temperature);
    let mut kept: Vec<PathResult> = results.iter().filter(|r| wrote(&enc, r, &o.prefix)).cloned().collect();
    if kept.len() >= o.count {
        record.found = true;
        record.temperature = asked_temperature(o);
        return Ok((results, Some(record)));
    }
    record.climbed = true;
    for t in temperature_ladder(o.temperature, auto_temperature) {
        for _ in 0..draws_at(t) {
            let need = o.count - kept.len();
            let retry = retry_generation(o, need, t, record.tries);
            record.tries += 1;
            let drawn = model.generate(&retry)?;
            kept.extend(drawn.into_iter().filter(|r| wrote(&enc, r, &o.prefix)).take(need));
            if kept.len() >= o.count {
                record.found = true;
                record.temperature = Some(t);
                return Ok((kept, Some(record)));
            }
        }
    }
    Ok((kept, Some(record)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphOptions;
    use crate::model::TrainOptions;

    /// A strong cycle - "ab " comes round again far more often than it ends -
    /// so a greedy walk goes round for ever.
    fn looping() -> Vec<String> {
        let long = format!("{}end", "ab ".repeat(20));
        vec![long.clone(), long.clone(), long, "ab end".to_string()]
    }

    /// A text that ends as often as it goes on, so a walk from its end says
    /// nothing half the time.
    fn ends() -> Vec<String> {
        vec![
            "the cat sat on the mat".to_string(),
            "the cat sat on the mat and slept".to_string(),
        ]
    }

    fn count_model(texts: &[String], epochs: usize) -> Model {
        let mut model = Model::new(1, GraphOptions::default()).unwrap();
        model
            .train(
                texts,
                &TrainOptions {
                    epochs,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    fn walk(text: &str, reached_end: bool) -> PathResult {
        PathResult {
            text: text.to_string(),
            reached_end,
            ..Default::default()
        }
    }

    /// Python's `first_answer` of a plain path: the path itself when it answers.
    fn itself(result: &PathResult) -> Option<PathResult> {
        answered(result, false).then(|| result.clone())
    }

    #[test]
    fn the_rungs_double_up_to_the_ceiling() {
        assert_eq!(temperature_ladder(1.0, 2.0), vec![1.0, 2.0]);
        assert_eq!(temperature_ladder(1.0, 3.0), vec![1.0, 2.0, 3.0]);
        assert_eq!(temperature_ladder(1.0, 1.5), vec![1.0, 1.5]);
        assert_eq!(temperature_ladder(0.7, 2.0), vec![0.7, 1.4, 2.0]);
        assert_eq!(temperature_ladder(0.25, 1.0), vec![0.25, 0.5, 1.0]);
    }

    #[test]
    fn greedy_climbs_to_the_floor_first() {
        assert_eq!(AUTO_FLOOR, 0.5);
        assert_eq!(temperature_ladder(0.0, 2.0), vec![0.0, 0.5, 1.0, 2.0]);
        assert_eq!(temperature_ladder(0.0, 0.3), vec![0.0, 0.3]);
        assert_eq!((draws_at(0.0), draws_at(0.5)), (1, AUTO_DRAWS));
    }

    #[test]
    fn it_never_cools_and_off_is_no_ladder() {
        assert_eq!(temperature_ladder(3.0, 2.0), vec![3.0]);
        assert_eq!(temperature_ladder(2.0, 2.0), vec![2.0]);
        assert!(temperature_ladder(1.0, 0.0).is_empty());
    }

    #[test]
    fn a_ceiling_out_of_range_is_refused() {
        assert!(check_auto_temperature(0.0).is_ok());
        assert!(check_auto_temperature(2.5).is_ok());
        for bad in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(check_auto_temperature(bad).is_err(), "{bad}");
        }
        assert_eq!(
            check_auto_temperature(-1.0).unwrap_err(),
            "auto_temperature must be a finite number >= 0 (0 is off), got -1.0"
        );
    }

    #[test]
    fn an_answer_wrote_something_and_reached_the_end_when_asked() {
        assert!(answered(&walk(" sat", false), false));
        assert!(!answered(&walk("", true), false));
        assert!(!answered(&walk("  \n", true), false));
        assert!(!answered(&walk("\x1c", true), false), "Python's whitespace");
        assert!(answered(&walk(" sat", true), true));
        assert!(!answered(&walk(" sat", false), true));
    }

    #[test]
    fn an_answer_as_asked_is_the_whole_of_it() {
        let mut log: Vec<Option<f64>> = Vec::new();
        let (result, record) = climb(
            |t| {
                log.push(t);
                Ok(walk("said", true))
            },
            itself,
            1.0,
            2.0,
            true,
        )
        .unwrap();
        assert_eq!(result.text, "said");
        assert_eq!(log, vec![None]);
        assert_eq!(
            record.to_json().render(0),
            "{\"ceiling\":2.0,\"climbed\":false,\"found\":true,\"temperature\":1.0,\"tries\":1}"
        );
    }

    #[test]
    fn every_rung_is_three_draws_and_the_first_answer_ends_it() {
        assert_eq!(AUTO_DRAWS, 3);
        let mut log: Vec<Option<f64>> = Vec::new();
        let (result, record) = climb(
            |t| {
                log.push(t);
                Ok(walk(if t == Some(2.0) { "said" } else { "" }, true))
            },
            itself,
            1.0,
            2.0,
            false,
        )
        .unwrap();
        assert_eq!(log, vec![None, Some(1.0), Some(1.0), Some(1.0), Some(2.0)]);
        assert_eq!(result.text, "said");
        assert_eq!(
            (record.climbed, record.found, record.temperature, record.tries),
            (true, true, Some(2.0), 5)
        );
    }

    #[test]
    fn a_greedy_rung_is_one_walk() {
        let mut log: Vec<Option<f64>> = Vec::new();
        climb(
            |t| {
                log.push(t);
                Ok(walk("", true))
            },
            itself,
            0.0,
            1.0,
            true,
        )
        .unwrap();
        let mut expected = vec![None, Some(0.0)];
        expected.extend([Some(0.5); 3]);
        expected.extend([Some(1.0); 3]);
        assert_eq!(log, expected);
    }

    #[test]
    fn nothing_found_gives_back_the_search_as_asked() {
        let mut calls = 0;
        let (result, record) = climb(
            |_| {
                calls += 1;
                Ok(PathResult {
                    labels: if calls == 1 {
                        vec!["as asked".to_string()]
                    } else {
                        Vec::new()
                    },
                    ..Default::default()
                })
            },
            itself,
            1.0,
            2.0,
            false,
        )
        .unwrap();
        assert_eq!(result.labels, vec!["as asked".to_string()]);
        assert_eq!(
            record,
            AutoTemperature {
                ceiling: 2.0,
                climbed: true,
                found: false,
                temperature: None,
                tries: 7
            }
        );
        assert_eq!(record.to_json().at("temperature"), &Json::Null);
        assert_eq!(AutoTemperature::from_json(&record.to_json()), Some(record));
    }

    #[test]
    fn any_test_is_an_answer_test() {
        // a caller's own test - a guard's veto - decides as well as the model's
        let mut texts = ["vetoed", "vetoed", "clean"].into_iter();
        let (result, record) = climb(
            |_| Ok(walk(texts.next().unwrap_or(""), false)),
            |r: &PathResult| (answered(r, false) && r.text != "vetoed").then(|| r.clone()),
            1.0,
            2.0,
            true,
        )
        .unwrap();
        assert_eq!(
            (result.text.as_str(), record.tries, record.temperature),
            ("clean", 3, Some(1.0))
        );
    }

    #[test]
    fn a_search_that_fails_ends_the_climb() {
        let err = climb(
            |t| match t {
                None => Ok(walk("", false)),
                Some(_) => Err("no such mode".to_string()),
            },
            itself,
            1.0,
            2.0,
            false,
        )
        .unwrap_err();
        assert_eq!(err, "no such mode");
    }

    fn sampled(length: usize) -> PredictOptions {
        PredictOptions {
            mode: "sample".to_string(),
            length,
            ..Default::default()
        }
    }

    #[test]
    fn off_is_the_search_it_always_was() {
        let mut model = count_model(&looping(), 2);
        let before = model.g.rng.clone();
        let plain = model.predict("ab", &sampled(12)).unwrap();
        let after = model.g.rng.state();
        model.g.rng = before;
        let (result, record) = predict(&mut model, "ab", 0.0, &sampled(12)).unwrap();
        assert!(record.is_none());
        assert_eq!(
            (&result.best.text, &result.best.node_ids),
            (&plain.best.text, &plain.best.node_ids)
        );
        assert_eq!(model.g.rng.state(), after, "the same draws, no more");
    }

    /// Greedy, with room to end and the end asked for: a walk round the cycle.
    fn greedy() -> PredictOptions {
        PredictOptions {
            temperature: 0.0,
            length: 20,
            max_length: Some(30),
            to_end: true,
            ..sampled(20)
        }
    }

    #[test]
    fn a_greedy_walk_that_goes_round_is_mended_by_a_warmer_one() {
        let mut model = count_model(&looping(), 2);
        let cold = model.predict("ab", &greedy()).unwrap();
        assert!(!cold.best.reached_end, "greedy goes round and never ends");
        let (result, record) = predict(&mut model, "ab", 2.0, &greedy()).unwrap();
        let record = record.unwrap();
        assert!(result.best.reached_end);
        assert!(!result.best.text.trim().is_empty());
        assert!(record.climbed && record.found);
        assert!(record.temperature.unwrap() > 0.0);
        assert!(record.tries > 1);
        assert_eq!(result.mode, "sample");
    }

    fn beam(max_length: usize) -> PredictOptions {
        PredictOptions {
            mode: "beam".to_string(),
            length: 2,
            max_length: Some(max_length),
            to_end: true,
            ..Default::default()
        }
    }

    #[test]
    fn a_deterministic_search_that_finds_nothing_falls_back_to_walks() {
        // the beam reads no temperature: when it finds nothing, the ladder is where sampling starts - and
        // the walks are held to the same cap, so an end no path reaches inside it is not reached by walking
        let mut model = count_model(&looping(), 2);
        let plain = model.predict("ab", &beam(3)).unwrap(); // " end" needs 4
        assert!(first_answer(&plain, true).is_none());
        let (result, record) = predict(&mut model, "ab", 2.0, &beam(3)).unwrap();
        assert_eq!(
            record.unwrap(),
            AutoTemperature {
                ceiling: 2.0,
                climbed: true,
                found: false,
                temperature: None,
                tries: 1 + 2 * AUTO_DRAWS
            }
        );
        assert_eq!(
            (&result.best.text, &result.best.node_ids),
            (&plain.best.text, &plain.best.node_ids),
            "the search as asked"
        );
    }

    #[test]
    fn an_answer_lower_in_the_beam_is_the_beams_own() {
        // with room to end, the beam's cheapest path still goes round to the cap - but its top K hold paths
        // that end, and the first of them is the answer: found as asked, nothing walked
        let mut model = count_model(&looping(), 2);
        let plain = model.predict("ab", &beam(30)).unwrap();
        assert!(
            !plain.best.reached_end,
            "the cheapest path is the cycle, cut at the cap"
        );
        let ending = plain.top.iter().find(|p| p.reached_end).unwrap().clone();
        let (result, record) = predict(&mut model, "ab", 2.0, &beam(30)).unwrap();
        let record = record.unwrap();
        assert_eq!(
            (&result.best.text, &result.best.node_ids, result.best.reached_end),
            (&ending.text, &ending.node_ids, true)
        );
        let texts = |p: &Prediction| p.top.iter().map(|r| r.text.clone()).collect::<Vec<_>>();
        assert_eq!(texts(&result), texts(&plain), "the ranking is the search's");
        assert_eq!(
            (result.expanded, result.best.expanded),
            (plain.expanded, plain.best.expanded),
            "the search's own effort"
        );
        assert_eq!(
            (record.climbed, record.found, record.tries, record.temperature),
            (false, true, 1, None)
        );
    }

    #[test]
    fn no_way_on_is_no_answer_at_any_temperature() {
        let mut model = count_model(&["the cat sat".to_string()], 1);
        let (result, record) = predict(&mut model, "the cat sat", 2.0, &sampled(5)).unwrap();
        assert_eq!(result.best.text, "");
        assert_eq!(
            record.unwrap().to_json().render(0),
            "{\"ceiling\":2.0,\"climbed\":true,\"found\":false,\"temperature\":null,\"tries\":7}"
        );
    }

    #[test]
    fn every_kind_climbs() {
        use crate::kinds::{new_model, train, TrainSettings};
        use crate::radix::TrainConfig;
        for kind in ["radix", "count", "resonant", "negative"] {
            let mut model = new_model(kind, 1, Encoding::default(), &[]).unwrap();
            let settings = TrainSettings {
                config: TrainConfig {
                    epochs: 2,
                    ..Default::default()
                },
                ..Default::default()
            };
            train(&mut model, &looping(), &settings, &mut |_| true).unwrap();
            let (result, record) = predict(&mut model, "ab", 2.0, &greedy()).unwrap();
            let record = record.unwrap();
            assert_eq!(record.found, answered(&result.best, true), "{kind}");
        }
    }

    fn sampling(prefix: &str, count: usize, seed: Option<i64>) -> GenerateOptions {
        GenerateOptions {
            count,
            mode: "sample".to_string(),
            prefix: prefix.to_string(),
            seed,
            ..Default::default()
        }
    }

    const MAT: &str = "the cat sat on the mat";

    #[test]
    fn off_is_the_generation_it_always_was() {
        let mut model = count_model(&ends(), 2);
        let plain = model.generate(&sampling(MAT, 6, Some(4))).unwrap();
        let (results, record) = generate(&mut model, 0.0, &sampling(MAT, 6, Some(4))).unwrap();
        assert!(record.is_none());
        let texts = |rs: &[PathResult]| rs.iter().map(|r| r.text.clone()).collect::<Vec<_>>();
        assert_eq!(texts(&results), texts(&plain));
    }

    #[test]
    fn samples_that_said_nothing_are_drawn_again() {
        let mut model = count_model(&ends(), 2);
        let enc = model.encoding();
        let plain = model.generate(&sampling(MAT, 6, Some(4))).unwrap();
        let silent = plain.iter().filter(|r| !wrote(&enc, r, MAT)).count();
        assert!(silent > 0, "some walks end where the prefix does");
        let (results, record) = generate(&mut model, 2.0, &sampling(MAT, 6, Some(4))).unwrap();
        let record = record.unwrap();
        assert_eq!(results.len(), 6);
        assert!(results
            .iter()
            .all(|r| r.text.starts_with(MAT) && r.text.len() > MAT.len()));
        assert!(record.climbed && record.found);
        // the samples that said something come first, in their order
        let said: Vec<String> = plain
            .iter()
            .filter(|r| wrote(&enc, r, MAT))
            .map(|r| r.text.clone())
            .collect();
        let got: Vec<String> = results.iter().map(|r| r.text.clone()).collect();
        assert_eq!(&got[..said.len()], &said[..]);
        // and the whole climb is reproducible from the seed
        let (again, _) = generate(&mut model, 2.0, &sampling(MAT, 6, Some(4))).unwrap();
        assert_eq!(again.iter().map(|r| r.text.clone()).collect::<Vec<_>>(), got);
    }

    #[test]
    fn a_retry_with_a_seed_draws_from_a_generator_of_its_own() {
        let o = sampling(MAT, 6, Some(4));
        let retry = retry_generation(&o, 2, 1.0, 3);
        assert_eq!(
            (retry.count, retry.mode.as_str(), retry.temperature, retry.seed),
            (2, "sample", 1.0, Some(7))
        );
        assert_eq!(retry_generation(&sampling(MAT, 6, None), 2, 1.0, 3).seed, None);
    }

    #[test]
    fn nothing_asked_for_is_found_as_asked() {
        let mut model = count_model(&ends(), 2);
        let (results, record) = generate(&mut model, 2.0, &sampling(MAT, 0, None)).unwrap();
        assert!(results.is_empty());
        assert_eq!(
            record.unwrap().to_json().render(0),
            "{\"ceiling\":2.0,\"climbed\":false,\"found\":true,\"temperature\":1.0,\"tries\":1}"
        );
    }

    #[test]
    fn too_few_at_the_ceiling_is_what_there_is() {
        let mut model = count_model(&["the cat sat".to_string()], 1);
        let (results, record) = generate(&mut model, 1.0, &sampling("the cat sat", 3, None)).unwrap();
        let record = record.unwrap();
        assert!(results.is_empty());
        assert!(!record.found);
        assert_eq!((record.ceiling, record.tries), (1.0, 1 + AUTO_DRAWS));
    }
}
