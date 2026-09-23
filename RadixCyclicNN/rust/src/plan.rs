//! The lesson plan the teacher writes from a report card (the planner of
//! `radixnet/tutor.py`, `go/radixnet/plan.go`).
//!
//! The English tutor ends a batch of lessons with a report card: how many
//! passed, the mean marks, and how often each kind of mistake was the worst
//! thing in a sentence.  This module hands that card back to the teacher - a
//! local Ollama model or ChatGPT, whichever taught - which answers with the
//! syllabus of the lessons that follow: one point of grammar each, aimed at
//! the mistakes the marking found, and a *brief* for the teacher who writes
//! the next batch's exercises ([`crate::tutor::TutorConfig::brief`]).
//!
//! # The marks are the floor, not the answer
//!
//! [`plan_from_card`] is what the marks alone imply: a lesson per weak point,
//! worst first.  No LLM is involved, so a report card always leads to a plan -
//! and [`plan_lessons`] falls back to it whenever the teacher's answer cannot
//! be read, and patches it in where the answer skipped a weakness the card
//! marked down.  How much harder the next batch gets is never the answer's to
//! decide either: [`upgrade_from_card`] reads it off the marks (D-051) and the
//! teacher is asked to repeat it word for word.  So the same card plans the
//! same lessons, whatever the LLM makes of it.
//!
//! # Why a card is JSON here
//!
//! A card arrives from three places: the tutor's own [`crate::tutor::report_card`],
//! the record at the end of a run, and a request body (`POST /api/tutor/plan`
//! with `report`).  The last is whatever a client sent, so the card is read
//! the way Python reads a dict that survived a JSON round trip - a count may
//! be a float or a string, a mark may be missing - rather than as a struct.

use crate::json::Json;
use crate::llm::{loads_lenient, parse_lines, LlmClient, LlmOptions};
use crate::review::ReviewError;
use crate::tutor::{clip, error_type, ERROR_TYPES, MAX_COMMENT_CHARS};

/// What this module's own lines are filed under.
const LOG: &str = "tutor";

/// The point of grammar a lesson drills to fix each mistake of
/// [`ERROR_TYPES`] (and `other`).
pub const ERROR_FOCUS: &[(&str, &str)] = &[
    ("agreement", "subject-verb agreement"),
    ("tense", "verb tenses"),
    ("article", "articles (a, an, the)"),
    ("preposition", "prepositions"),
    ("plural", "plural nouns"),
    ("pronoun", "pronouns"),
    ("word-order", "word order"),
    ("spelling", "spelling"),
    ("punctuation", "punctuation"),
    ("vocabulary", "word choice"),
    ("fragment", "finishing the sentence"),
    ("nonsense", "writing a sentence that means something"),
    ("other", "sentence structure"),
];

/// How hard the exercises are; a plan moves the student one step up a card
/// it has nothing left to fix in.
pub const LEVELS: &[&str] = &["beginner", "intermediate", "advanced"];

/// Lessons a plan holds unless more are asked for.
pub const DEFAULT_PLAN_LESSONS: usize = 3;

/// A report card at or above both is a student ready for the next level.
pub const STRONG_PASS_RATE: f64 = 0.8;
/// See [`STRONG_PASS_RATE`].
pub const STRONG_SCORE: f64 = 8.0;
/// Half the lessons passed: not a level up, but the openings get longer.
pub const STRETCH_PASS_RATE: f64 = 0.5;

/// How long an opening the teacher writes; an upgrade moves one rung up.
pub const WORDS_LADDER: &[&str] = &["3 to 6", "5 to 8", "7 to 12", "10 to 16"];

/// What the next batch does with the difficulty: hold it, stretch it, or move
/// up a level.
pub const UPGRADE_STEPS: &[&str] = &["hold", "stretch", "advance"];

/// Correct example sentences a held-back batch gets to imitate.
pub const HOLD_DRILLS: i64 = 3;

/// The longest brief kept from an answer.
pub const MAX_BRIEF_CHARS: usize = 600;

/// The grammar point that drills one mistake (`""` for `none` and anything
/// the vocabulary does not know).
pub fn focus_for(error: &str) -> &'static str {
    let kind = error_type(error);
    ERROR_FOCUS
        .iter()
        .find(|(name, _)| *name == kind)
        .map(|(_, focus)| *focus)
        .unwrap_or("")
}

/// Python's `float(value)` of a card field; `None` where it raises.
fn float_of(value: &Json) -> Option<f64> {
    match value {
        Json::Int(n) => Some(*n as f64),
        Json::Num(x) => Some(*x),
        Json::Bool(b) => Some(if *b { 1.0 } else { 0.0 }),
        Json::Str(s) => crate::tutor::parse_float(s),
        _ => None,
    }
}

/// A count out of a report card, however it survived the JSON round trip
/// (`0` when it is unusable) - Python's `max(0, int(float(value)))`.
pub fn count_of(value: &Json) -> i64 {
    match float_of(value) {
        Some(x) if x.is_finite() => (x.trunc() as i64).max(0),
        _ => 0,
    }
}

/// A mean mark of a report card, or `None` when it is missing.
pub fn mark_of(value: &Json) -> Option<f64> {
    float_of(value).filter(|x| !x.is_nan())
}

/// One mistake of a report card: how often it was the worst thing in a
/// sentence, in how many of the lessons, and the grammar point that fixes it.
#[derive(Clone, Debug, PartialEq)]
pub struct WeakPoint {
    pub error: String,
    pub count: i64,
    /// Its share of the lessons; `None` when the card counts none.
    pub share: Option<f64>,
    pub focus: String,
}

impl WeakPoint {
    /// `{"error", "count", "share", "focus"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("error", Json::str(self.error.clone())),
            ("count", Json::Int(self.count)),
            ("share", self.share.map(Json::Num).unwrap_or(Json::Null)),
            ("focus", Json::str(self.focus.clone())),
        ])
    }
}

/// The mistakes of a report card, worst first (ties alphabetically, so the
/// same marks always plan the same lessons), at most `limit` of them.
pub fn weak_points(card: &Json, limit: usize) -> Vec<WeakPoint> {
    let mut counts: Vec<(String, i64)> = Vec::new();
    if let Some(Json::Obj(errors)) = card.get("errors") {
        for (name, count) in errors {
            let error = error_type(name);
            let number = count_of(count);
            if error != "none" && number > 0 {
                match counts.iter_mut().find(|(e, _)| *e == error) {
                    Some(slot) => slot.1 += number,
                    None => counts.push((error, number)),
                }
            }
        }
    }
    let lessons = card.get("lessons").map(count_of).unwrap_or(0);
    counts.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    counts
        .into_iter()
        .take(limit)
        .map(|(error, count)| WeakPoint {
            share: (lessons != 0).then(|| count as f64 / lessons as f64),
            focus: focus_for(&error).to_string(),
            error,
            count,
        })
        .collect()
}

/// The card's pass rate and mean score.
fn rate_and_score(card: &Json) -> (Option<f64>, Option<f64>) {
    (
        card.get("pass_rate").and_then(mark_of),
        card.get("mean_score").and_then(mark_of),
    )
}

/// A level as the planner compares it: trimmed, lower case, `beginner` when blank.
fn level_name(level: &str) -> String {
    let level = level.trim().to_lowercase();
    if level.is_empty() {
        LEVELS[0].to_string()
    } else {
        level
    }
}

/// One step up [`LEVELS`] when the card is strong ([`STRONG_PASS_RATE`]
/// passed at [`STRONG_SCORE`]), else the level it was given.
pub fn next_level(level: &str, card: &Json) -> String {
    let level = level_name(level);
    let (rate, score) = rate_and_score(card);
    let strong = rate.is_some_and(|r| r >= STRONG_PASS_RATE) && score.is_some_and(|s| s >= STRONG_SCORE);
    match LEVELS.iter().position(|l| *l == level) {
        Some(at) if strong => LEVELS[(at + 1).min(LEVELS.len() - 1)].to_string(),
        _ => level,
    }
}

/// `steps` rungs up [`WORDS_LADDER`]; a setting that is not on the ladder is
/// left alone (and a blank one is the first rung).
pub fn next_words(words: &str, steps: usize) -> String {
    let text = words.split_whitespace().collect::<Vec<_>>().join(" ");
    match WORDS_LADDER.iter().position(|w| *w == text) {
        Some(at) => WORDS_LADDER[(at + steps).min(WORDS_LADDER.len() - 1)].to_string(),
        None if text.is_empty() => WORDS_LADDER[0].to_string(),
        None => text,
    }
}

fn percent(value: Option<f64>) -> String {
    match value {
        Some(v) => format!("{:.0}%", v * 100.0),
        None => "too few".to_string(),
    }
}

/// The step the next batch takes, and the settings it takes it with.
#[derive(Clone, Debug, PartialEq)]
pub struct Upgrade {
    /// One of [`UPGRADE_STEPS`].
    pub step: String,
    pub level: String,
    pub words: String,
    pub threshold: f64,
    pub drills: i64,
    /// The step in the teacher's own words, repeated in the brief.
    pub note: String,
}

impl Upgrade {
    /// `{"step", "level", "words", "threshold", "drills", "note"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("step", Json::str(self.step.clone())),
            ("level", Json::str(self.level.clone())),
            ("words", Json::str(self.words.clone())),
            ("threshold", Json::Num(self.threshold)),
            ("drills", Json::Int(self.drills)),
            ("note", Json::str(self.note.clone())),
        ])
    }
}

/// The incremental step the next batch takes, read off the marks (D-051).
///
/// `advance` when the card is strong: a level up, longer openings and a
/// higher pass mark.  `stretch` when [`STRETCH_PASS_RATE`] passed: the same
/// level, but longer openings.  `hold` otherwise: nothing gets harder, the
/// weak points are drilled and the teacher's own example sentences
/// ([`HOLD_DRILLS`]) come with them, because a student who is failing does
/// not need a harder exercise.
pub fn upgrade_from_card(card: &Json, level: &str, words: &str, threshold: f64, drills: i64) -> Upgrade {
    let (rate, score) = rate_and_score(card);
    let mut level = level_name(level);
    let mut words = words.split_whitespace().collect::<Vec<_>>().join(" ");
    if words.is_empty() {
        words = WORDS_LADDER[0].to_string();
    }
    let passed = percent(rate);
    let mark = score.map(|s| format!("{s:.1}")).unwrap_or_else(|| "-".to_string());
    let mut threshold = threshold;
    let mut drills = drills;
    let (step, note) = if rate.is_some_and(|r| r >= STRONG_PASS_RATE) && score.is_some_and(|s| s >= STRONG_SCORE) {
        level = next_level(&level, card);
        words = next_words(&words, 1);
        // Python's min(9.0, threshold + 1.0)
        let raised = threshold + 1.0;
        threshold = if raised < 9.0 { raised } else { 9.0 };
        (
            "advance",
            format!(
                "The last batch passed {passed} at {mark} out of 10, so this one steps up to {level}: openings of \
                 {words} words and a pass mark of {threshold:.0} out of 10."
            ),
        )
    } else if rate.is_some_and(|r| r >= STRETCH_PASS_RATE) {
        words = next_words(&words, 1);
        (
            "stretch",
            format!(
                "The last batch passed {passed}, so this one stays at {level} but stretches the openings to {words} \
                 words."
            ),
        )
    } else {
        drills = drills.max(HOLD_DRILLS);
        (
            "hold",
            format!(
                "The last batch passed {passed}, so this one holds at {level} with openings of {words} words, drills \
                 the mistakes and comes with {drills} correct sentences to imitate."
            ),
        )
    };
    Upgrade {
        step: step.to_string(),
        level,
        words,
        threshold,
        drills,
        note,
    }
}

/// `"a, b and c"` - the teacher's own English, so its brief reads like one.
fn and_list(names: &[String]) -> String {
    match names {
        [] => String::new(),
        [one] => one.clone(),
        [head @ .., last] => format!("{} and {last}", head.join(", ")),
    }
}

/// The brief the marks alone write for the next batch: what to drill, how it
/// steps up, what it is about.
pub fn plan_brief(weak: &[WeakPoint], upgrade: Option<&Upgrade>, topic: &str) -> String {
    let points: Vec<String> = weak
        .iter()
        .map(|p| {
            if p.focus.is_empty() {
                p.error.clone()
            } else {
                p.focus.clone()
            }
        })
        .filter(|name| !name.is_empty())
        .collect();
    let opening = match points.len() {
        0 => "Nothing was marked down, so widen the vocabulary instead of drilling one point of grammar.".to_string(),
        1 => format!("Drill {}: that is what the last batch got wrong.", points[0]),
        _ => format!(
            "Drill {}, worst first: that is what the last batch got wrong.",
            and_list(&points)
        ),
    };
    let mut lines = vec![opening, upgrade.map(|u| u.note.trim().to_string()).unwrap_or_default()];
    if !topic.trim().is_empty() {
        lines.push(format!("Keep the sentences about {}.", topic.trim()));
    }
    lines.retain(|line| !line.is_empty());
    lines.join(" ")
}

/// One lesson of a plan: the point of grammar it drills, what its sentences
/// are about and the mistake it is aimed at.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedLesson {
    pub focus: String,
    pub topic: String,
    /// One sentence for the student's file: why it is being taught.
    pub why: String,
    /// The mistake of [`ERROR_TYPES`] it fixes (`none`: no particular one).
    pub targets: String,
    pub exercises: i64,
    pub drills: i64,
    /// Openings the teacher already wrote, if any.
    pub prefixes: Vec<String>,
}

impl PlannedLesson {
    /// `{"focus", "topic", "why", "targets", "exercises", "drills", "prefixes"}`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("focus", Json::str(self.focus.clone())),
            ("topic", Json::str(self.topic.clone())),
            ("why", Json::str(self.why.clone())),
            ("targets", Json::str(self.targets.clone())),
            ("exercises", Json::Int(self.exercises)),
            ("drills", Json::Int(self.drills)),
            ("prefixes", Json::strs(self.prefixes.clone())),
        ])
    }
}

/// The lessons to run next, written from a report card: what they drill and why.
#[derive(Clone, Debug, PartialEq)]
pub struct LessonPlan {
    pub lessons: Vec<PlannedLesson>,
    /// Where the student stands, in a sentence or two.
    pub summary: String,
    /// The brief for the next batch: what to drill, how it steps up, what it
    /// is about.
    pub prompt: String,
    /// The step the next batch takes ([`upgrade_from_card`]); `None` only on
    /// an answer not yet merged with the marks.
    pub upgrade: Option<Upgrade>,
    /// The level the next lessons should be at.
    pub level: String,
    /// The topic they were planned around.
    pub topic: String,
    /// The weak points of the card behind the plan.
    pub weak: Vec<WeakPoint>,
    /// Who wrote it: a provider, or `report card` (the marks alone).
    pub source: String,
}

impl LessonPlan {
    /// The mistakes the plan drills, in its own order.
    pub fn targets(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for lesson in &self.lessons {
            if lesson.targets != "none" && !lesson.targets.is_empty() && !out.contains(&lesson.targets) {
                out.push(lesson.targets.clone());
            }
        }
        out
    }

    /// `{"summary", "prompt", "upgrade", "level", "topic", "source", "weak",
    /// "targets", "lessons"}` - the Python server's shape.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("summary", Json::str(self.summary.clone())),
            ("prompt", Json::str(self.prompt.clone())),
            (
                "upgrade",
                self.upgrade
                    .as_ref()
                    .map(Upgrade::to_json)
                    .unwrap_or(Json::Obj(Vec::new())),
            ),
            ("level", Json::str(self.level.clone())),
            ("topic", Json::str(self.topic.clone())),
            ("source", Json::str(self.source.clone())),
            ("weak", Json::Arr(self.weak.iter().map(WeakPoint::to_json).collect())),
            ("targets", Json::strs(self.targets())),
            (
                "lessons",
                Json::Arr(self.lessons.iter().map(PlannedLesson::to_json).collect()),
            ),
        ])
    }

    /// The pairs of [`LessonPlan::to_json`], for a record that carries them
    /// after its own.
    pub fn pairs(&self) -> Vec<(String, Json)> {
        match self.to_json() {
            Json::Obj(pairs) => pairs,
            _ => Vec::new(),
        }
    }
}

const PLAN_SCHEMA: &str = "{\"summary\": \"<one or two sentences>\", \"prompt\": \"<the brief for the next batch>\", \
     \"lessons\": [{\"focus\": \"<the point of grammar>\", \"targets\": \"<the mistake it fixes>\", \"topic\": \"<what \
     its sentences are about>\", \"why\": \"<one sentence>\"}, ...]}";

const PLAN_SYSTEM: &str =
    "You are an English teacher planning the next lessons for a beginner student from its report \
     card. The student is given the opening words of a sentence and writes the rest; the report card is the marking \
     of the lessons it has just done - how many it passed, its mean marks out of 10 for grammar, spelling and \
     fluency, and how often each kind of mistake was the worst thing in a sentence. Plan the {n} lessons that repair \
     the most: each one drills a single point of grammar ('focus', two or three words, e.g. 'subject-verb \
     agreement', 'past tense', 'plural nouns'), names the mistake from the report card it is aimed at ('targets', \
     one of: {types}), gives the everyday subject its sentences should be about ('topic', two or three words) and \
     says in one short sentence why it is being taught ('why'). Put the worst mistake first, never plan two lessons \
     for the same mistake. How much harder the next batch gets is decided by the marks and given to you below: it \
     is not yours to change. 'summary' is one or two sentences on where the student stands. 'prompt' is the brief \
     for the teacher who writes the next batch of exercises: two or three sentences addressed to them, naming the \
     points of grammar to drill in order, repeating the step up in difficulty word for word, and saying what the \
     sentences should be about - instructions to a colleague, not a report on the student. Reply with JSON only, no \
     prose, exactly of the form {schema} with exactly {n} lessons.";

/// The planner's system prompt for `count` lessons.
pub fn plan_system(count: usize) -> String {
    PLAN_SYSTEM
        .replace("{types}", &ERROR_TYPES[1..].join(", "))
        .replace("{schema}", PLAN_SCHEMA)
        .replace("{n}", &count.to_string())
}

/// The report card as the teacher reads it: the marks, then the mistakes
/// worst first.
pub fn card_lines(card: &Json) -> Vec<String> {
    let at = |key: &str| card.get(key).unwrap_or(&Json::Null);
    let lessons = count_of(at("lessons"));
    let passed = count_of(at("passed"));
    let mut lines = vec![
        format!("Lessons marked: {lessons}"),
        match mark_of(at("pass_rate")) {
            Some(rate) => format!("Passed: {passed} ({:.0}%)", rate * 100.0),
            None => format!("Passed: {passed}"),
        },
    ];
    let known: Vec<String> = [
        ("overall", "mean_score"),
        ("grammar", "mean_grammar"),
        ("spelling", "mean_spelling"),
        ("fluency", "mean_fluency"),
    ]
    .iter()
    .filter_map(|(name, key)| mark_of(at(key)).map(|value| format!("{name} {value:.1}")))
    .collect();
    lines.push(format!(
        "Mean marks out of 10: {}",
        if known.is_empty() {
            "none recorded".to_string()
        } else {
            known.join(", ")
        }
    ));
    let points = weak_points(card, ERROR_TYPES.len());
    if points.is_empty() {
        lines.push("Mistakes: none were named.".to_string());
    } else {
        let named: Vec<String> = points
            .iter()
            .map(|p| match p.share {
                Some(share) => format!("{} x{} ({:.0}% of the lessons)", p.error, p.count, share * 100.0),
                None => format!("{} x{}", p.error, p.count),
            })
            .collect();
        lines.push(format!("Mistakes, worst first: {}", named.join(", ")));
    }
    lines
}

/// The first of `keys` holding a non-blank string.
fn text_field<'a>(item: &'a Json, keys: &[&str]) -> Option<&'a str> {
    keys.iter()
        .filter_map(|k| item.get(k).and_then(Json::as_str))
        .find(|value| !value.trim().is_empty())
}

/// The openings a planned lesson came with (a list, or lines of text).
fn plan_prefixes(item: &Json, limit: usize) -> Vec<String> {
    for key in ["prefixes", "openings", "examples"] {
        let listed: Vec<String> = match item.get(key) {
            Some(Json::Str(text)) => parse_lines(text, (limit > 0).then_some(limit)),
            Some(Json::Arr(values)) => values.iter().filter_map(|v| v.as_str().map(str::to_string)).collect(),
            _ => continue,
        };
        return listed
            .iter()
            .filter(|text| !text.trim().is_empty())
            .take(limit)
            .map(|text| clip(text, 120))
            .collect();
    }
    Vec::new()
}

/// An LLM answer as a plan of at most `count` lessons; unusable and duplicate
/// entries are dropped.
///
/// Tolerates the shapes an LLM drifts into: `{"lessons": [...]}`, `{"plan":
/// [...]}`, a bare list, plain strings instead of objects, `point` / `skill` /
/// `grammar` instead of `focus`, `reason` instead of `why`, and a `targets`
/// that names the mistake in its own words.  How much harder the next batch
/// gets is not read from the answer at all.
pub fn parse_plan(raw: &str, count: usize, topic: &str, exercises: i64, drills: i64) -> LessonPlan {
    let data = loads_lenient(raw);
    let mut items: Option<Vec<Json>> = None;
    let mut summary = String::new();
    let mut prompt = String::new();
    match &data {
        Some(doc @ Json::Obj(_)) => {
            for key in ["lessons", "plan", "syllabus", "items", "results"] {
                if let Some(Json::Arr(list)) = doc.get(key) {
                    items = Some(list.clone());
                    break;
                }
            }
            if items.is_none() && ["focus", "point", "skill"].iter().any(|k| doc.get(k).is_some()) {
                items = Some(vec![doc.clone()]);
            }
            if let Some(text) = text_field(doc, &["summary", "assessment", "comment"]) {
                summary = clip(text, MAX_COMMENT_CHARS);
            }
            if let Some(text) = text_field(doc, &["prompt", "brief", "instructions", "next_prompt"]) {
                prompt = clip(text, MAX_BRIEF_CHARS);
            }
        }
        Some(Json::Arr(list)) => items = Some(list.clone()),
        _ => {}
    }
    // plain lines are still a syllabus
    let items = items.unwrap_or_else(|| {
        parse_lines(raw, Some(count))
            .into_iter()
            .map(|line| Json::obj([("focus", Json::str(line))]))
            .collect()
    });
    let mut lessons: Vec<PlannedLesson> = Vec::new();
    let mut seen: Vec<String> = Vec::new();
    for item in items {
        let item = match item {
            Json::Str(text) => Json::obj([("focus", Json::Str(text))]),
            obj @ Json::Obj(_) => obj,
            _ => continue,
        };
        let mut focus = text_field(&item, &["focus", "point", "skill", "grammar", "lesson", "title"])
            .map(|v| clip(v, 60))
            .unwrap_or_default();
        let named = text_field(&item, &["targets", "target", "error", "mistake"]).unwrap_or("");
        let targets = if !named.is_empty() {
            error_type(named)
        } else {
            // "plural nouns" is a lesson about plurals whatever it calls itself
            let inferred = error_type(&focus);
            if inferred == "none" || inferred == "other" {
                "none".to_string()
            } else {
                inferred
            }
        };
        if focus.is_empty() {
            focus = focus_for(&targets).to_string();
        }
        if focus.is_empty() {
            continue;
        }
        let key = focus.to_lowercase();
        if seen.contains(&key) {
            continue;
        }
        seen.push(key);
        let why = text_field(&item, &["why", "reason", "because", "rationale", "note"])
            .map(|v| clip(v, MAX_COMMENT_CHARS))
            .unwrap_or_default();
        // item.get("topic", item.get("subject", item.get("theme", "")))
        let subject = ["topic", "subject", "theme"]
            .iter()
            .find_map(|k| item.get(k))
            .and_then(Json::as_str)
            .map(|s| clip(s, 60))
            .unwrap_or_default();
        lessons.push(PlannedLesson {
            focus,
            topic: if subject.is_empty() { topic.to_string() } else { subject },
            why,
            targets,
            exercises,
            drills,
            prefixes: plan_prefixes(&item, count),
        });
        if lessons.len() >= count {
            break;
        }
    }
    LessonPlan {
        lessons,
        summary,
        prompt,
        upgrade: None,
        level: LEVELS[0].to_string(),
        topic: topic.to_string(),
        weak: Vec::new(),
        source: "llm".to_string(),
    }
}

/// What a plan is asked for: the topic and level of the lessons so far, how
/// many lessons to plan and the settings each one carries.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanRequest {
    pub topic: String,
    pub level: String,
    /// How long the last batch's openings were (the ladder steps up from here).
    pub words: String,
    /// The pass mark it was marked against.
    pub threshold: f64,
    pub count: usize,
    pub exercises: i64,
    pub drills: i64,
    /// The teacher's model (`""` = the client's own).
    pub model: String,
    /// 0 = the default 0.7.
    pub temperature: f64,
}

impl Default for PlanRequest {
    /// Python's defaults.
    fn default() -> PlanRequest {
        PlanRequest {
            topic: String::new(),
            level: LEVELS[0].to_string(),
            words: WORDS_LADDER[0].to_string(),
            threshold: 6.0,
            count: DEFAULT_PLAN_LESSONS,
            exercises: 5,
            drills: 0,
            model: String::new(),
            temperature: 0.0,
        }
    }
}

/// The lesson plan the marks alone imply: one lesson per weak point of the
/// card, worst first, with the step up and the brief for the next batch.
///
/// No LLM is involved, so a report card always leads to a plan.  A card with
/// no mistake left to name plans one lesson that keeps the topic and lets the
/// upgrade move the student on.
pub fn plan_from_card(card: &Json, req: &PlanRequest) -> LessonPlan {
    let weak = weak_points(card, req.count.max(1));
    let upgrade = upgrade_from_card(card, &req.level, &req.words, req.threshold, req.drills);
    let total = card.get("lessons").map(count_of).unwrap_or(0);
    let mut lessons: Vec<PlannedLesson> = weak
        .iter()
        .map(|point| {
            let share = point.share.map(|s| format!(" ({:.0}%)", s * 100.0)).unwrap_or_default();
            PlannedLesson {
                focus: if point.focus.is_empty() {
                    point.error.clone()
                } else {
                    point.focus.clone()
                },
                topic: req.topic.clone(),
                why: format!(
                    "{} of {total} lessons{share}{} marked down for {}.",
                    point.count,
                    if point.count == 1 { " was" } else { " were" },
                    point.error
                ),
                targets: point.error.clone(),
                exercises: req.exercises,
                drills: upgrade.drills,
                prefixes: Vec::new(),
            }
        })
        .collect();
    if lessons.is_empty() {
        lessons.push(PlannedLesson {
            focus: String::new(),
            topic: req.topic.clone(),
            why: "No mistake was named, so the next lessons stay on the topic and take the step up instead."
                .to_string(),
            targets: "none".to_string(),
            exercises: req.exercises,
            drills: upgrade.drills,
            prefixes: Vec::new(),
        });
    }
    let passed = card.get("passed").map(count_of).unwrap_or(0);
    let mut summary = format!("{passed} of {total} lessons passed");
    if let Some(score) = card.get("mean_score").and_then(mark_of) {
        summary.push_str(&format!(" at a mean of {score:.1} out of 10"));
    }
    if weak.is_empty() {
        summary.push_str("; nothing was marked down");
    } else {
        let names: Vec<&str> = weak.iter().map(|p| p.error.as_str()).collect();
        summary.push_str(&format!("; the weakest points are {}", names.join(", ")));
    }
    summary.push('.');
    LessonPlan {
        lessons,
        summary,
        prompt: plan_brief(&weak, Some(&upgrade), &req.topic),
        level: upgrade.level.clone(),
        upgrade: Some(upgrade),
        topic: req.topic.clone(),
        weak,
        source: "report card".to_string(),
    }
}

/// Asks the teacher for the next lessons, given the report card of the ones
/// just marked and the step up in difficulty the marks have earned.
///
/// [`plan_from_card`] is the floor.  A weakness the teacher's plan does not
/// target takes the place of a lesson that drills nothing the card marked
/// down (and is appended when there is no such lesson), so every mistake on
/// the card is somebody's lesson; a brief it does not write is the one the
/// marks wrote; an answer that cannot be read - or that names no mistake and
/// says nothing about the student - leaves the whole plan as it stands.
/// `source` says which of the two wrote it.  The upgrade is never the
/// answer's: the marks decide it and the teacher is asked to repeat it.
pub fn plan_lessons(client: &dyn LlmClient, card: &Json, req: &PlanRequest) -> Result<LessonPlan, ReviewError> {
    if req.count < 1 {
        return Err(ReviewError::Invalid("count must be >= 1".to_string()));
    }
    if !matches!(card, Json::Obj(_)) || card.get("lessons").map(count_of).unwrap_or(0) < 1 {
        return Err(ReviewError::Invalid(
            "a report card of at least one lesson is needed to plan the next lessons".to_string(),
        ));
    }
    let fallback = plan_from_card(card, req);
    let mut lines = card_lines(card);
    if !req.topic.trim().is_empty() {
        lines.insert(0, format!("The lessons so far were about: {}", req.topic.trim()));
    }
    let level = if req.level.is_empty() {
        LEVELS[0]
    } else {
        req.level.as_str()
    };
    lines.push(format!(
        "Level so far: {}, openings of {} words, pass mark {:.0} out of 10",
        level.trim(),
        req.words,
        req.threshold
    ));
    let upgrade = fallback.upgrade.clone().expect("the marks always plan a step");
    lines.push(format!(
        "The step up this batch has earned, to repeat in your brief: {}",
        upgrade.note
    ));
    lines.push(format!("Plan the next {} lesson(s) now.", req.count));
    let temperature = if req.temperature > 0.0 { req.temperature } else { 0.7 };
    let o = LlmOptions::default()
        .system(plan_system(req.count))
        .model(req.model.as_str())
        .json()
        .temperature(temperature);
    let raw = client.generate(&lines.join("\n"), &o)?;
    let mut plan = parse_plan(&raw, req.count, &req.topic, req.exercises, req.drills);
    if plan.lessons.is_empty() || (plan.summary.is_empty() && plan.targets().is_empty()) {
        // nothing usable came back: the marks alone still plan the lessons
        crate::log_info!(LOG, "the teacher's plan was unusable: planned from the report card");
        return Ok(fallback);
    }
    plan.source = client.provider().to_string();
    plan.weak = fallback.weak.clone();
    plan.topic = req.topic.clone();
    if plan.summary.is_empty() {
        plan.summary = fallback.summary.clone();
    }
    if plan.prompt.is_empty() {
        plan.prompt = fallback.prompt.clone();
    }
    // the marks decide the step up, not the answer
    plan.level = fallback.level.clone();
    for lesson in &mut plan.lessons {
        if lesson.topic.is_empty() {
            lesson.topic = req.topic.clone();
        }
        lesson.drills = upgrade.drills;
        if lesson.why.is_empty() && lesson.targets != "none" {
            if let Some(known) = fallback.lessons.iter().find(|l| l.targets == lesson.targets) {
                lesson.why = known.why.clone();
            }
        }
    }
    plan.upgrade = Some(upgrade);
    let mut covered: Vec<String> = plan
        .lessons
        .iter()
        .filter(|l| l.targets != "none")
        .map(|l| l.targets.clone())
        .collect();
    let marked: Vec<&str> = fallback.weak.iter().map(|p| p.error.as_str()).collect();
    // a weakness the teacher's plan skipped takes the place of a lesson that
    // drills nothing the card marked down
    let mut spare: std::collections::VecDeque<usize> = plan
        .lessons
        .iter()
        .enumerate()
        .filter(|(_, l)| !marked.contains(&l.targets.as_str()))
        .map(|(i, _)| i)
        .collect();
    for lesson in &fallback.lessons {
        if lesson.targets == "none" || covered.contains(&lesson.targets) {
            continue;
        }
        covered.push(lesson.targets.clone());
        match spare.pop_front() {
            Some(at) => plan.lessons[at] = lesson.clone(),
            None => plan.lessons.push(lesson.clone()),
        }
    }
    crate::log_info!(
        LOG,
        "{} planned {} lesson(s): {}",
        plan.source,
        plan.lessons.len(),
        plan.targets().join(", ")
    );
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::json::parse;
    use crate::llm::LlmError;
    use std::sync::Mutex;

    struct Teacher {
        answer: String,
        asked: Mutex<Vec<(String, LlmOptions)>>,
    }

    impl Teacher {
        fn new(answer: &str) -> Teacher {
            Teacher {
                answer: answer.to_string(),
                asked: Mutex::new(Vec::new()),
            }
        }
    }

    impl LlmClient for Teacher {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://teacher"
        }
        fn model(&self) -> &str {
            "teacher"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.asked.lock().unwrap().push((prompt.to_string(), o.clone()));
            Ok(self.answer.clone())
        }
        fn chat(&self, _messages: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn card() -> Json {
        parse(
            "{\"lessons\": 6, \"passed\": 1, \"pass_rate\": 0.16666666666666666, \"mean_score\": 3.0, \"errors\": \
             {\"agreement\": 4, \"verb tense\": 1, \"none\": 3, \"tense\": \"1\"}}",
        )
        .unwrap()
    }

    #[test]
    fn weak_points_rank_the_card_worst_first() {
        let points = weak_points(&card(), 3);
        let names: Vec<(&str, i64)> = points.iter().map(|p| (p.error.as_str(), p.count)).collect();
        assert_eq!(names, vec![("agreement", 4), ("tense", 2)], "'verb tense' is a tense");
        assert_eq!(points[0].focus, "subject-verb agreement");
        assert_eq!(points[0].share, Some(4.0 / 6.0));
        assert_eq!(
            points[1].to_json().render(0),
            "{\"error\":\"tense\",\"count\":2,\"share\":0.3333333333333333,\"focus\":\"verb tenses\"}"
        );
        assert!(weak_points(&Json::Null, 3).is_empty());
        assert_eq!(focus_for("none"), "");
        assert_eq!(focus_for("style"), "sentence structure");
    }

    #[test]
    fn the_step_up_is_read_off_the_marks() {
        let strong = parse("{\"pass_rate\": 0.9, \"mean_score\": 8.5}").unwrap();
        let up = upgrade_from_card(&strong, "Beginner", "3 to 6", 6.0, 0);
        assert_eq!(
            (up.step.as_str(), up.level.as_str(), up.words.as_str(), up.threshold),
            ("advance", "intermediate", "5 to 8", 7.0)
        );
        assert_eq!(
            up.note,
            "The last batch passed 90% at 8.5 out of 10, so this one steps up to intermediate: openings of 5 to 8 \
             words and a pass mark of 7 out of 10."
        );
        let half = parse("{\"pass_rate\": 0.5, \"mean_score\": 5}").unwrap();
        let up = upgrade_from_card(&half, "", "  10  to 16 ", 6.0, 1);
        assert_eq!((up.step.as_str(), up.words.as_str()), ("stretch", "10 to 16"));
        let weak = upgrade_from_card(&card(), "beginner", "", 6.0, 1);
        assert_eq!((weak.step.as_str(), weak.drills), ("hold", 3));
        assert_eq!(
            weak.note,
            "The last batch passed 17%, so this one holds at beginner with openings of 3 to 6 words, drills the \
             mistakes and comes with 3 correct sentences to imitate."
        );
        assert!(upgrade_from_card(&Json::Null, "x", "7 to 12", 9.5, 0)
            .note
            .contains("too few"));
        assert_eq!(next_level("advanced", &strong), "advanced");
        assert_eq!(next_level("expert", &strong), "expert");
        assert_eq!(next_words("odd", 1), "odd");
        assert_eq!(next_words("", 1), "3 to 6");
        assert_eq!(next_words("7 to 12", 5), "10 to 16");
    }

    #[test]
    fn the_card_is_read_as_the_teacher_reads_it() {
        let lines = card_lines(&card());
        assert_eq!(
            lines,
            vec![
                "Lessons marked: 6",
                "Passed: 1 (17%)",
                "Mean marks out of 10: overall 3.0",
                "Mistakes, worst first: agreement x4 (67% of the lessons), tense x2 (33% of the lessons)",
            ]
        );
        let bare = card_lines(&parse("{\"lessons\": \"2\"}").unwrap());
        assert_eq!(bare[1], "Passed: 0");
        assert_eq!(bare[2], "Mean marks out of 10: none recorded");
        assert_eq!(bare[3], "Mistakes: none were named.");
        assert_eq!(count_of(&Json::str("x")), 0);
        assert_eq!(count_of(&Json::Num(-3.0)), 0);
        assert_eq!(count_of(&Json::Num(2.9)), 2);
        assert_eq!(mark_of(&Json::Null), None);
    }

    #[test]
    fn the_marks_alone_plan_a_lesson_per_weakness() {
        let req = PlanRequest {
            topic: "animals".to_string(),
            count: 2,
            exercises: 3,
            ..Default::default()
        };
        let plan = plan_from_card(&card(), &req);
        assert_eq!(plan.source, "report card");
        assert_eq!(plan.targets(), vec!["agreement", "tense"]);
        assert_eq!(
            plan.lessons[0].why,
            "4 of 6 lessons (67%) were marked down for agreement."
        );
        assert_eq!(plan.lessons[0].drills, 3, "a held batch gets drills");
        assert_eq!(
            plan.summary,
            "1 of 6 lessons passed at a mean of 3.0 out of 10; the weakest points are agreement, tense."
        );
        assert!(plan
            .prompt
            .starts_with("Drill subject-verb agreement and verb tenses, worst first:"));
        assert!(plan.prompt.ends_with("Keep the sentences about animals."));
        let clean = plan_from_card(&parse("{\"lessons\": 2, \"passed\": 2}").unwrap(), &req);
        assert_eq!(clean.lessons.len(), 1);
        assert_eq!(clean.lessons[0].targets, "none");
        assert!(clean.summary.ends_with("; nothing was marked down."));
        let doc = plan.to_json();
        assert_eq!(doc.at("upgrade").at("step").as_str(), Some("hold"));
        assert_eq!(doc.at("targets").to_strings(), vec!["agreement", "tense"]);
    }

    #[test]
    fn an_answer_drifts_and_is_still_read() {
        let raw = "Here: {\"summary\": \" The student struggles. \", \"brief\": \"Drill.\", \"plan\": [\"plural \
                   nouns\", {\"point\": \"past tense\", \"mistake\": \"verb_tense\", \"subject\": \"yesterday\", \
                   \"reason\": \"Tenses drifted.\", \"openings\": \"1. yesterday we\\n2. last week they\"}, \
                   {\"focus\": \"Plural Nouns\"}, {\"targets\": \"article\"}, 7]}";
        let plan = parse_plan(raw, 5, "life", 4, 1);
        let got: Vec<(&str, &str, &str)> = plan
            .lessons
            .iter()
            .map(|l| (l.focus.as_str(), l.targets.as_str(), l.topic.as_str()))
            .collect();
        assert_eq!(
            got,
            vec![
                ("plural nouns", "plural", "life"),
                ("past tense", "tense", "yesterday"),
                ("articles (a, an, the)", "article", "life"),
            ]
        );
        assert_eq!(plan.lessons[1].prefixes, vec!["yesterday we", "last week they"]);
        assert_eq!(plan.lessons[1].why, "Tenses drifted.");
        assert_eq!(plan.summary, "The student struggles.");
        assert_eq!(plan.prompt, "Drill.");
        let lines = parse_plan("1. word order\n2. spelling\n3. spelling", 2, "", 5, 0);
        assert_eq!(lines.targets(), vec!["word-order", "spelling"]);
        assert!(parse_plan("", 3, "", 5, 0).lessons.is_empty());
    }

    #[test]
    fn the_teacher_plans_and_the_marks_fill_the_gaps() {
        let teacher = Teacher::new(
            "{\"summary\": \"Verbs are weak.\", \"lessons\": [{\"focus\": \"word order\", \"targets\": \
             \"word-order\"}, {\"focus\": \"subject-verb agreement\", \"targets\": \"agreement\", \"topic\": \"pets\"}]}",
        );
        let req = PlanRequest {
            topic: "animals".to_string(),
            count: 2,
            threshold: 9.5,
            ..Default::default()
        };
        let plan = plan_lessons(&teacher, &card(), &req).unwrap();
        assert_eq!(plan.source, "ollama");
        // word order drills nothing the card marked down: tense takes its place
        assert_eq!(plan.targets(), vec!["tense", "agreement"]);
        assert_eq!(plan.lessons[0].why, "2 of 6 lessons (33%) were marked down for tense.");
        assert_eq!(plan.lessons[1].topic, "pets");
        assert_eq!(
            plan.lessons[1].why,
            "4 of 6 lessons (67%) were marked down for agreement."
        );
        assert_eq!(plan.summary, "Verbs are weak.");
        assert!(
            plan.prompt.starts_with("Drill subject-verb agreement"),
            "the marks' brief"
        );
        let asked = teacher.asked.lock().unwrap();
        let (prompt, o) = &asked[0];
        assert!(prompt.starts_with("The lessons so far were about: animals\nLessons marked: 6\n"));
        assert!(prompt.contains("\nLevel so far: beginner, openings of 3 to 6 words, pass mark 10 out of 10\n"));
        assert!(prompt.ends_with("\nPlan the next 2 lesson(s) now."));
        assert!(o.json);
        assert!(o.system.contains("Plan the 2 lessons") && o.system.ends_with("with exactly 2 lessons."));
        assert!(o.system.contains("one of: agreement, tense, article,"));
        assert_eq!(o.options, vec![("temperature".to_string(), Json::Num(0.7))]);
        drop(asked);
        let unusable = Teacher::new("I cannot.");
        let floor = plan_lessons(&unusable, &card(), &req).unwrap();
        assert_eq!(floor.source, "report card");
        assert!(plan_lessons(&teacher, &parse("{\"lessons\": 0}").unwrap(), &req).is_err());
        let none = PlanRequest { count: 0, ..req };
        assert!(matches!(
            plan_lessons(&teacher, &card(), &none),
            Err(ReviewError::Invalid(_))
        ));
    }
}
