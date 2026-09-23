//! The tutoring loop: rounds of exercises, completions, grades and the
//! learning they lead to, batch after batch (`TutorTrainer` of
//! `radixnet/tutor.py`).
//!
//! # What a mark is worth (D-050)
//!
//! A grade is a rating, not a like.  A failed sentence is pushed away by how
//! bad its mark was ([`TutorTrainer::weight_of`]: `min_weight` for a near
//! miss, 1 for a hopeless one) and a passed one kept by how good it was
//! ([`TutorTrainer::reward_of`]: its mark over 10), while what the teacher
//! wrote - a correction, a model answer, a drill - is correct by construction
//! and always weighs [`TEACHER_WEIGHT`].  The weights reach the graph through
//! the model's own kind ([`crate::kinds`]): one pass per group of equally
//! weighted texts, the count and phase models' reward or penalty scaled by
//! the weight, the sine model's learning rates - as each of Python's kinds
//! applies them.
//!
//! A failure the teacher corrected is not 2NRL at all on the count model: the
//! sentence and its correction are aligned and only what changed moves
//! (`Model::correct`, `diff_corrections`), so the words the network got right
//! keep what they earned.  The sine and phase models have no `correct` in
//! Python, so there a corrected failure is 2NRL garbage and its correction
//! good English, as with `--no-diff-corrections`.
//!
//! # The auto run (D-052)
//!
//! One batch is `rounds` rounds and the report card over them.  With
//! `batches` above 1 - or 0, which keeps going until it is stopped - the run
//! drives itself: the card is planned from ([`crate::plan::plan_lessons`]),
//! the plan is applied ([`TutorTrainer::apply_plan`]: its brief becomes the
//! next batch's standing instruction and the step up the marks earned
//! (D-051) its difficulty), and the next batch is taught to it.  A batch that
//! cannot be planned ends the run rather than repeating itself.
//!
//! # Where the networks live
//!
//! [`Networks`] is the one thing the CLI and the server do differently: the
//! command owns the model (and the negative network) for the length of the
//! run, while the server keeps each behind its own lock and takes it for a
//! step at a time - a completion, a learning pass, a blame - so the
//! frontend's reads get in between and nothing is held while the teacher
//! thinks, which is most of a round.  The chat loop ([`crate::chat`]) uses
//! the same type.

use std::time::Instant;

use crate::blame::{TeachOptions, TeachReport};
use crate::checkpoint::Checkpoints;
use crate::correct::{CorrectOptions, Correction};
use crate::json::Json;
use crate::kinds;
use crate::llm::{normalise_provider, LlmClient, DEFAULT_PROVIDER, PROVIDERS};
use crate::model::{EpochRecord, Model, PredictOptions};
use crate::plan::{count_of, mark_of, plan_lessons, LessonPlan, PlanRequest, DEFAULT_PLAN_LESSONS};
use crate::radix::Feedback;
use crate::review::ReviewError;
use crate::service::Service;

use super::{
    card_pairs, clip, default_tutor_model, drill_sentences, explain_mistakes, fmean, grade_completions, report_card,
    teach_lessons, write_exercises, Exercise, ExerciseRequest, ExplainOptions, Grade, GradeOptions, Lesson,
    TutorCorrection, MAX_VARIANTS, MODES, TEACHER_WEIGHT, TWONRL_PER,
};

/// What this module's own lines are filed under.
const LOG: &str = "tutor";

// -- the settings ---------------------------------------------------------------------------------

/// The settings of a tutoring run (Python's `TutorConfig`, field for field).
#[derive(Clone, Debug, PartialEq)]
pub struct TutorConfig {
    pub topic: String,
    pub rounds: usize,
    /// Sentence openings per round.
    pub exercises: usize,
    /// Completions the network writes per exercise.
    pub attempts: usize,
    /// Pin every exercise to one point of grammar.
    pub focus: Option<String>,
    pub level: String,
    /// How long a prefix the teacher writes.
    pub words: String,
    /// Standing instructions for the exercise writer: the previous batch's
    /// plan ([`LessonPlan::prompt`]).
    pub brief: String,
    /// `ollama` | `chatgpt`: who teaches.
    pub tutor_provider: String,
    pub tutor_model: String,
    /// Who marks (the teacher's provider unless named).
    pub grader_provider: String,
    /// A different model for the marking.
    pub grader_model: Option<String>,
    /// `dijkstra` | `beam` | `sample`.
    pub mode: String,
    pub length: usize,
    pub max_length: usize,
    pub temperature: f64,
    pub to_end: bool,
    pub beam: Option<usize>,
    pub threshold: f64,
    pub grammar_weight: f64,
    pub batch: usize,
    /// Drill the previous round's weakest points.
    pub adapt: bool,
    /// Extra correct example sentences per round.
    pub drills: usize,
    /// Sentences with the same mistake per failure, for the negative network
    /// (0 = do not ask).
    pub variants: usize,
    /// Their share of the failure's severity.
    pub variant_weight: f64,
    /// Lessons to plan from the final report card (0 = no plan).
    pub plan: usize,
    /// The auto run: batches of `rounds` rounds, each planned from the last
    /// (0 = until stopped).
    pub batches: usize,
    /// A failed lesson also learns the teacher's model answer.
    pub teach_answer: bool,
    /// `false`: a dry run - the grades are reported, nothing is trained.
    pub learn: bool,
    /// `round` | `lesson`.
    pub twonrl_per: String,
    /// Teach a correction from its diff, not as two whole sentences.
    pub diff_corrections: bool,
    /// What the unchanged part of a correction still earns.
    pub keep_weight: f64,
    /// The negative-phase weight of a near miss (a hopeless answer weighs 1).
    pub min_weight: f64,
    pub neg_epochs: usize,
    pub pos_epochs: usize,
    /// The sine model's learning rates (the count and phase models have none).
    pub neg_lr: f64,
    pub pos_lr: f64,
    /// Transitions per backend step on the sine model.
    pub batch_size: usize,
    /// The magnitude of a penalty or reward on the count and phase models
    /// (`None` = 1).
    pub strength: Option<f64>,
    /// Keep teaching earlier corrections.
    pub replay: bool,
    /// How many of them to keep (0 = no limit).
    pub replay_limit: usize,
    /// Rounds between checkpoints (0 = none).
    pub checkpoint_every: usize,
}

impl Default for TutorConfig {
    /// The Python defaults, the teacher's model resolved from the environment.
    fn default() -> TutorConfig {
        TutorConfig {
            topic: "everyday life".to_string(),
            rounds: 3,
            exercises: 5,
            attempts: 1,
            focus: None,
            level: "beginner".to_string(),
            words: "3 to 6".to_string(),
            brief: String::new(),
            tutor_provider: DEFAULT_PROVIDER.to_string(),
            tutor_model: default_tutor_model(DEFAULT_PROVIDER),
            grader_provider: DEFAULT_PROVIDER.to_string(),
            grader_model: None,
            mode: "dijkstra".to_string(),
            length: 20,
            max_length: 80,
            temperature: 1.0,
            to_end: true,
            beam: None,
            threshold: 6.0,
            grammar_weight: 0.6,
            batch: 10,
            adapt: true,
            drills: 0,
            variants: super::DEFAULT_VARIANTS,
            variant_weight: 0.5,
            plan: 0,
            batches: 1,
            teach_answer: true,
            learn: true,
            twonrl_per: "round".to_string(),
            diff_corrections: true,
            keep_weight: 0.0,
            min_weight: 0.25,
            neg_epochs: 2,
            pos_epochs: 3,
            neg_lr: 0.5,
            pos_lr: 0.1,
            batch_size: 4,
            strength: None,
            replay: true,
            replay_limit: 64,
            checkpoint_every: 0,
        }
    }
}

impl TutorConfig {
    /// Normalises the providers and fills in the models they imply, the way
    /// Python's `__post_init__` does (so reports name the real model).
    pub fn resolve(&mut self) -> Result<(), String> {
        let bad = || {
            format!(
                "tutor_provider and grader_provider must be one of {}",
                PROVIDERS.join(", ")
            )
        };
        self.tutor_provider = normalise_provider(&self.tutor_provider).map_err(|_| bad())?.to_string();
        self.grader_provider = if self.grader_provider.trim().is_empty() {
            self.tutor_provider.clone()
        } else {
            normalise_provider(&self.grader_provider)
                .map_err(|_| bad())?
                .to_string()
        };
        if self.tutor_model.trim().is_empty() {
            self.tutor_model = default_tutor_model(&self.tutor_provider);
        }
        if self.grader_model.as_deref().is_some_and(|m| m.trim().is_empty()) {
            self.grader_model = None;
        }
        Ok(())
    }

    /// The model the marking runs on: `grader_model`, else the teacher's
    /// model on the teacher's provider, else that provider's default.
    pub fn resolved_grader_model(&self) -> String {
        if let Some(model) = self.grader_model.as_deref().filter(|m| !m.is_empty()) {
            return model.to_string();
        }
        if self.grader_provider == self.tutor_provider {
            return self.tutor_model.clone();
        }
        default_tutor_model(&self.grader_provider)
    }

    /// Refuses settings the loop cannot run with - Python's messages, in
    /// Python's order.
    pub fn validate(&self) -> Result<(), String> {
        let known = |p: &str| PROVIDERS.contains(&p);
        if !known(&self.tutor_provider) || !known(&self.grader_provider) {
            return Err(format!(
                "tutor_provider and grader_provider must be one of {}",
                PROVIDERS.join(", ")
            ));
        }
        let unit = |x: f64| (0.0..=1.0).contains(&x);
        // every rule with its refusal; the first that fails is the one reported
        let checks: Vec<(bool, String)> = vec![
            (self.topic.trim().is_empty(), "topic must be a non-empty string".into()),
            (self.rounds < 1, "rounds must be >= 1".into()),
            (self.exercises < 1, "exercises must be >= 1".into()),
            (self.attempts < 1, "attempts must be >= 1".into()),
            (
                !MODES.contains(&self.mode.as_str()),
                format!("mode must be one of {}", MODES.join(", ")),
            ),
            (
                !TWONRL_PER.contains(&self.twonrl_per.as_str()),
                format!("twonrl_per must be one of {}", TWONRL_PER.join(", ")),
            ),
            (self.max_length < 1, "length must be >= 0 and max_length >= 1".into()),
            (
                self.temperature.is_nan() || self.temperature < 0.0,
                "temperature must be >= 0".into(),
            ),
            (self.beam.is_some_and(|b| b < 1), "beam must be >= 1".into()),
            (
                !(0.0..=10.0).contains(&self.threshold),
                "threshold must lie in [0, 10]".into(),
            ),
            (!unit(self.grammar_weight), "grammar_weight must lie in [0, 1]".into()),
            (!unit(self.min_weight), "min_weight must lie in [0, 1]".into()),
            (!unit(self.keep_weight), "keep_weight must lie in [0, 1]".into()),
            (self.batch < 1, "batch must be >= 1".into()),
            (
                self.variants > MAX_VARIANTS,
                format!("variants must lie in [0, {MAX_VARIANTS}]"),
            ),
            (
                self.variant_weight.is_nan() || self.variant_weight < 0.0,
                "variant_weight must be >= 0".into(),
            ),
            (
                self.neg_lr.is_nan() || self.pos_lr.is_nan() || self.neg_lr < 0.0 || self.pos_lr < 0.0,
                "learning rates must be >= 0".into(),
            ),
            (self.batch_size < 1, "batch_size must be >= 1".into()),
            (
                self.strength.is_some_and(|s| s.is_nan() || s < 0.0),
                "strength must be >= 0".into(),
            ),
        ];
        match checks.into_iter().find(|(failed, _)| *failed) {
            Some((_, message)) => Err(message),
            None => Ok(()),
        }
    }

    /// The settings as a document (`dataclasses.asdict` of Python's config).
    pub fn to_json(&self) -> Json {
        let text = |s: &str| Json::str(s);
        let maybe = |s: &Option<String>| s.as_deref().map(Json::str).unwrap_or(Json::Null);
        let int = |n: usize| Json::Int(n as i64);
        Json::obj([
            ("topic", text(&self.topic)),
            ("rounds", int(self.rounds)),
            ("exercises", int(self.exercises)),
            ("attempts", int(self.attempts)),
            ("focus", maybe(&self.focus)),
            ("level", text(&self.level)),
            ("words", text(&self.words)),
            ("brief", text(&self.brief)),
            ("tutor_provider", text(&self.tutor_provider)),
            ("tutor_model", text(&self.tutor_model)),
            ("grader_provider", text(&self.grader_provider)),
            ("grader_model", maybe(&self.grader_model)),
            ("mode", text(&self.mode)),
            ("length", int(self.length)),
            ("max_length", int(self.max_length)),
            ("temperature", Json::Num(self.temperature)),
            ("to_end", Json::Bool(self.to_end)),
            ("beam", self.beam.map(|b| Json::Int(b as i64)).unwrap_or(Json::Null)),
            ("threshold", Json::Num(self.threshold)),
            ("grammar_weight", Json::Num(self.grammar_weight)),
            ("batch", int(self.batch)),
            ("adapt", Json::Bool(self.adapt)),
            ("drills", int(self.drills)),
            ("variants", int(self.variants)),
            ("variant_weight", Json::Num(self.variant_weight)),
            ("plan", int(self.plan)),
            ("batches", int(self.batches)),
            ("teach_answer", Json::Bool(self.teach_answer)),
            ("learn", Json::Bool(self.learn)),
            ("twonrl_per", text(&self.twonrl_per)),
            ("diff_corrections", Json::Bool(self.diff_corrections)),
            ("keep_weight", Json::Num(self.keep_weight)),
            ("min_weight", Json::Num(self.min_weight)),
            ("neg_epochs", int(self.neg_epochs)),
            ("pos_epochs", int(self.pos_epochs)),
            ("neg_lr", Json::Num(self.neg_lr)),
            ("pos_lr", Json::Num(self.pos_lr)),
            ("batch_size", int(self.batch_size)),
            ("strength", self.strength.map(Json::Num).unwrap_or(Json::Null)),
            ("replay", Json::Bool(self.replay)),
            ("replay_limit", int(self.replay_limit)),
            ("checkpoint_every", int(self.checkpoint_every)),
        ])
    }

    /// The magnitude of one reward or penalty.
    fn base_strength(&self) -> f64 {
        self.strength.unwrap_or(1.0)
    }

    /// What one set of grades teaches, for the model's kind to read: the
    /// epochs, rates, batch size and strength of every pass, and the marks
    /// as weights (Python's `two_nrl(..., bad_weights=bad_weights or None,
    /// good_weights=... or None)`: no marks, no weights).
    fn feedback(&self, bad_weights: &[f64], good_weights: &[f64]) -> Feedback {
        let given = |weights: &[f64]| (!weights.is_empty()).then(|| weights.to_vec());
        Feedback {
            neg_epochs: self.neg_epochs,
            pos_epochs: self.pos_epochs,
            neg_lr: self.neg_lr,
            pos_lr: self.pos_lr,
            strength: self.base_strength(),
            batch_size: Some(self.batch_size),
            auto_compress: None,
            clip: None,
            shuffle: None,
            good_weights: given(good_weights),
            bad_weights: given(bad_weights),
        }
    }
}

// -- where the networks live ----------------------------------------------------------------------

/// Where a teaching loop's networks live: owned by a command for the length of
/// the run, or behind the server's locks and taken for a step at a time.
pub enum Networks<'a> {
    /// A command's own models.
    Owned {
        model: &'a mut Model,
        negative: Option<&'a mut Model>,
    },
    /// The server's: the running model, and - when `negative` - its negative
    /// network, which the caller has loaded ([`Service::ensure_negative`]).
    Shared { svc: &'a Service, negative: bool },
}

impl Networks<'_> {
    /// Runs `f` on the positive model.
    pub fn with_model<T>(&mut self, f: impl FnOnce(&mut Model) -> T) -> T {
        match self {
            Networks::Owned { model, .. } => f(model),
            Networks::Shared { svc, .. } => svc.with_model(f),
        }
    }

    /// Whether a negative network takes part.
    pub fn has_negative(&self) -> bool {
        match self {
            Networks::Owned { negative, .. } => negative.is_some(),
            Networks::Shared { negative, .. } => *negative,
        }
    }

    /// Runs `f` on the negative network; `None` when there is none.
    pub fn with_negative<T>(&mut self, f: impl FnOnce(&mut Model) -> T) -> Result<Option<T>, String> {
        match self {
            Networks::Owned { negative, .. } => Ok(negative.as_deref_mut().map(f)),
            Networks::Shared { svc, negative: true } => svc.with_negative(f).map(Some).map_err(|e| e.message),
            Networks::Shared { .. } => Ok(None),
        }
    }

    /// Runs `f` on both at once, the model's lock taken first (the rule of
    /// [`Service::ensure_negative`]).
    pub fn with_both<T>(&mut self, f: impl FnOnce(&mut Model, Option<&mut Model>) -> T) -> Result<T, String> {
        match self {
            Networks::Owned { model, negative } => Ok(f(model, negative.as_deref_mut())),
            Networks::Shared { svc, negative: false } => Ok(svc.with_model(|m| f(m, None))),
            Networks::Shared { svc, negative: true } => {
                // loaded before the model is locked, never inside it
                svc.ensure_negative().map_err(|e| e.message)?;
                svc.with_model(|m| svc.with_negative(|n| f(m, Some(n))))
                    .map_err(|e| e.message)
            }
        }
    }
}

// -- weighted learning ----------------------------------------------------------------------------

/// Texts of equal (3-decimal) weight grouped, heaviest first; weights of 0
/// are dropped (Python's `_weight_groups`).  A rating per text becomes one
/// pass per distinct weight, scaled by it.
pub fn weight_groups(texts: &[String], weights: &[f64], name: &str) -> Result<Vec<(f64, Vec<String>)>, String> {
    if weights.len() != texts.len() {
        return Err(format!(
            "{name} has {} entries for {} texts",
            weights.len(),
            texts.len()
        ));
    }
    let mut groups: Vec<(f64, Vec<String>)> = Vec::new();
    for (text, &weight) in texts.iter().zip(weights) {
        if !weight.is_finite() || weight < 0.0 {
            return Err(format!(
                "{name} must be finite and >= 0, got {}",
                crate::json::py_repr(weight)
            ));
        }
        if weight > 0.0 {
            // Python's round(weight, 3): the decimal rounding, read back
            let key: f64 = format!("{weight:.3}").parse().unwrap_or(weight);
            match groups.iter_mut().find(|(w, _)| *w == key) {
                Some(group) => group.1.push(text.clone()),
                None => groups.push((key, vec![text.clone()])),
            }
        }
    }
    groups.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    Ok(groups)
}

/// The loss of the last record, `None` when there is none.
fn last_loss(records: &[EpochRecord]) -> Option<f64> {
    records.last().map(|r| r.loss)
}

/// `(texts, weights)` in first-seen order; a text offered twice keeps its
/// largest weight (Python's `_merge_weighted`).
fn merge_weighted(pairs: impl IntoIterator<Item = (String, f64)>) -> (Vec<String>, Vec<f64>) {
    let mut texts: Vec<String> = Vec::new();
    let mut weights: Vec<f64> = Vec::new();
    for (text, weight) in pairs {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        match texts.iter().position(|t| t == text) {
            Some(at) => {
                // max(merged.get(text, 0.0), weight)
                if weight > weights[at] {
                    weights[at] = weight;
                }
            }
            None => {
                texts.push(text.to_string());
                weights.push(if weight > 0.0 { weight } else { 0.0 });
            }
        }
    }
    (texts, weights)
}

// -- what a set of grades teaches ------------------------------------------------------------------

/// What the grades of a set of lessons mean for the network.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Graded {
    pub bad: Vec<String>,
    pub bad_weights: Vec<f64>,
    pub good: Vec<String>,
    pub good_weights: Vec<f64>,
}

/// What one set of grades did to the network (the keys a round record carries).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Learned {
    pub bad: usize,
    pub good: usize,
    /// What ran, in order: `correct`, `2nrl`, `reward`, `punish`, joined by `+`.
    pub action: Option<String>,
    pub neg_loss: Option<f64>,
    pub pos_loss: Option<f64>,
    pub mean_weight: Option<f64>,
    pub mean_reward: Option<f64>,
    pub corrections: usize,
    pub edits: usize,
    pub penalised: usize,
    pub rewarded: usize,
    pub marked_correct: usize,
    pub marked_incorrect: usize,
}

impl Learned {
    /// `{"bad", "good", "action", "neg_loss", "pos_loss", "mean_weight",
    /// "mean_reward", "corrections", "edits", "penalised", "rewarded",
    /// "marked_correct", "marked_incorrect"}`.
    pub fn pairs(&self) -> Vec<(String, Json)> {
        let num = |x: Option<f64>| x.map(Json::Num).unwrap_or(Json::Null);
        let int = |n: usize| Json::Int(n as i64);
        vec![
            ("bad".to_string(), int(self.bad)),
            ("good".to_string(), int(self.good)),
            (
                "action".to_string(),
                self.action.clone().map(Json::Str).unwrap_or(Json::Null),
            ),
            ("neg_loss".to_string(), num(self.neg_loss)),
            ("pos_loss".to_string(), num(self.pos_loss)),
            ("mean_weight".to_string(), num(self.mean_weight)),
            ("mean_reward".to_string(), num(self.mean_reward)),
            ("corrections".to_string(), int(self.corrections)),
            ("edits".to_string(), int(self.edits)),
            ("penalised".to_string(), int(self.penalised)),
            ("rewarded".to_string(), int(self.rewarded)),
            ("marked_correct".to_string(), int(self.marked_correct)),
            ("marked_incorrect".to_string(), int(self.marked_incorrect)),
        ]
    }
}

/// `a+b+c` in order and without repeats, `None` when nothing ran.
fn join_actions(actions: &[String]) -> Option<String> {
    let mut out: Vec<&str> = Vec::new();
    for action in actions {
        if !action.is_empty() && !out.contains(&action.as_str()) {
            out.push(action);
        }
    }
    (!out.is_empty()).then(|| out.join("+"))
}

/// Whether this model can be taught a correction from its diff - Python's
/// `callable(getattr(model, "correct", None))`: the count model can; the sine
/// and phase models have no `correct`, so their corrected failures go through
/// 2NRL whole; and the negative network learns a correction by blaming it
/// instead, which the tutor does beside the model.
fn can_correct(model: &Model) -> bool {
    model.kind() == "count"
}

/// Teaches one correction from its diff (`Model::correct`): the steps of the
/// attempt the teacher struck out lose `strength * weight`, the fix gains
/// `strength` at the teacher's full rate, and the rest of the correction earns
/// `keep` times as much - so only what changed moves.
fn teach_correction(
    model: &mut Model,
    correction: &TutorCorrection,
    strength: f64,
    keep: f64,
) -> Result<Correction, String> {
    model.correct(
        &correction.wrong,
        &correction.right,
        &CorrectOptions {
            strength,
            weight: correction.weight,
            reward: TEACHER_WEIGHT,
            keep,
            count: true,
        },
    )
}

// -- the loop -------------------------------------------------------------------------------------

/// Runs the lessons: the teacher sets and marks the exercises, the network
/// completes them and learns from the grades.
///
/// `client` is the teacher; `grader` marks (the teacher itself unless the
/// configuration names another provider or endpoint).  `stop` is asked
/// between steps, `progress` sees every lesson, round, report, plan, batch
/// and note record as it happens, and `checkpoints` saves the model every
/// `checkpoint_every` rounds.
pub struct TutorTrainer<'a> {
    client: &'a dyn LlmClient,
    grader: &'a dyn LlmClient,
    pub config: TutorConfig,
    /// Every record emitted, in order.
    pub history: Vec<Json>,
    /// Every lesson taught, in order.
    pub lessons: Vec<Lesson>,
    /// The corrections and good sentences it keeps teaching.
    pub replay: Vec<String>,
    /// The last round's weakest points, drilled next when `adapt` is on.
    pub weak: Vec<String>,
    stop: Box<dyn Fn() -> bool + 'a>,
    progress: Box<dyn FnMut(&Json) + 'a>,
    checkpoints: Option<&'a Checkpoints>,
}

impl<'a> TutorTrainer<'a> {
    /// The loop, once its settings are known to be sound.
    pub fn new(
        client: &'a dyn LlmClient,
        grader: &'a dyn LlmClient,
        mut config: TutorConfig,
    ) -> Result<TutorTrainer<'a>, String> {
        config.resolve()?;
        config.validate()?;
        Ok(TutorTrainer {
            client,
            grader,
            config,
            history: Vec::new(),
            lessons: Vec::new(),
            replay: Vec::new(),
            weak: Vec::new(),
            stop: Box::new(|| false),
            progress: Box::new(|_| {}),
            checkpoints: None,
        })
    }

    /// What is asked between steps; `true` ends the run cleanly.
    pub fn with_stop(mut self, stop: impl Fn() -> bool + 'a) -> Self {
        self.stop = Box::new(stop);
        self
    }

    /// What sees every record as it happens.
    pub fn with_progress(mut self, progress: impl FnMut(&Json) + 'a) -> Self {
        self.progress = Box::new(progress);
        self
    }

    /// Where the model is checkpointed every `checkpoint_every` rounds.
    pub fn with_checkpoints(mut self, checkpoints: Option<&'a Checkpoints>) -> Self {
        self.checkpoints = checkpoints;
        self
    }

    fn stopped(&self) -> bool {
        (self.stop)()
    }

    fn emit(&mut self, record: Json) {
        (self.progress)(&record);
        self.history.push(record);
    }

    fn note(&mut self, key: &str, at: usize, message: String) {
        crate::log_warn!(LOG, "{message}");
        self.emit(Json::obj([
            ("kind", Json::str("note")),
            (key, Json::Int(at as i64)),
            ("message", Json::Str(message)),
        ]));
    }

    // -- the three steps of a lesson --------------------------------------------------------------

    /// Step 1: the teacher writes this round's sentence openings.
    pub fn set_exercises(&self, round: usize) -> Result<Vec<Exercise>, ReviewError> {
        let cfg = &self.config;
        let mut exercises = write_exercises(
            self.client,
            &ExerciseRequest {
                topic: cfg.topic.clone(),
                count: cfg.exercises,
                focus: cfg.focus.clone().unwrap_or_default(),
                level: cfg.level.clone(),
                weak: if cfg.adapt { self.weak.clone() } else { Vec::new() },
                words: cfg.words.clone(),
                brief: cfg.brief.clone(),
                model: cfg.tutor_model.clone(),
                temperature: 0.0,
            },
        )?;
        for (position, exercise) in exercises.iter_mut().enumerate() {
            exercise.id = format!("r{round}e{}", position + 1);
        }
        Ok(exercises)
    }

    /// Step 2: the network continues the prefix (attempt 0 in the configured
    /// mode, later ones sampled).
    pub fn complete(&self, model: &mut Model, exercise: &Exercise, attempt: usize) -> Result<Lesson, String> {
        let cfg = &self.config;
        let mode = if attempt == 0 { cfg.mode.as_str() } else { "sample" };
        let started = Instant::now();
        let found = model.predict(
            &exercise.cue(),
            &PredictOptions {
                length: cfg.length,
                mode: mode.to_string(),
                temperature: cfg.temperature,
                max_length: Some(cfg.max_length),
                to_end: cfg.to_end && mode != "sample",
                beam: cfg.beam.unwrap_or(0),
                ..Default::default()
            },
        )?;
        let mut lesson = Lesson::new(exercise.clone(), attempt, mode, &found.best.text);
        lesson.cost = found.best.cost;
        lesson.probability = found.best.probability();
        lesson.reached_end = found.best.reached_end;
        lesson.seconds = started.elapsed().as_secs_f64();
        Ok(lesson)
    }

    /// Step 3: the marker marks the completions (one call per `batch`).
    pub fn grade(&self, lessons: &mut [Lesson]) -> Result<(), ReviewError> {
        let cfg = &self.config;
        grade_completions(
            self.grader,
            lessons,
            &GradeOptions {
                topic: cfg.topic.clone(),
                threshold: cfg.threshold,
                grammar_weight: cfg.grammar_weight,
                model: cfg.resolved_grader_model(),
                batch: cfg.batch,
                temperature: 0.0,
                graded_by: self.grader.provider().to_string(),
            },
        )
    }

    /// Step 4: why are the failures wrong, and what else is wrong in the same
    /// way?  Only asked with a negative network attached and `variants` above
    /// 0 - it is the only thing that learns from the answer - and a teacher
    /// that cannot answer costs the widening, not the round.
    ///
    /// `(explained, similar)`: the failures that came back with a reason, and
    /// the sentences with the same mistake.
    pub fn widen(&mut self, lessons: &mut [Lesson], round: usize, negative: bool) -> (usize, usize) {
        let cfg = self.config.clone();
        if !negative || cfg.variants < 1 || self.stopped() {
            return (0, 0);
        }
        let mut failed: Vec<&mut Lesson> = lessons
            .iter_mut()
            .filter(|l| !l.grade.passed && !l.sentence.trim().is_empty() && !l.empty())
            .collect();
        if failed.is_empty() {
            return (0, 0);
        }
        let asked = explain_mistakes(
            self.client,
            &mut failed,
            &ExplainOptions {
                topic: cfg.topic.clone(),
                count: cfg.variants,
                weight: cfg.variant_weight,
                model: cfg.tutor_model.clone(),
                batch: cfg.batch,
                temperature: 0.0,
            },
        );
        if let Err(err) = asked {
            // the round stands without the widening
            self.note("round", round, format!("no similar mistakes: {err}"));
            return (0, 0);
        }
        let explained = failed.iter().filter(|l| !l.why.is_empty()).count();
        let similar = failed.iter().map(|l| l.variants.len()).sum();
        (explained, similar)
    }

    // -- what a grade is worth --------------------------------------------------------------------

    /// The negative-phase weight of a failed sentence: `min_weight` for a near
    /// miss, 1 for a hopeless one.
    pub fn weight_of(&self, grade: &Grade) -> f64 {
        let cfg = &self.config;
        let Some(score) = grade.score else { return 1.0 };
        if cfg.threshold <= 0.0 {
            return 1.0;
        }
        let badness = ((cfg.threshold - score) / cfg.threshold).clamp(0.0, 1.0);
        cfg.min_weight + (1.0 - cfg.min_weight) * badness
    }

    /// The positive-phase weight of a sentence the network wrote: its mark,
    /// `score / 10`.  A 10 out of 10 is kept in full and a bare pass at a
    /// fraction of it, so passing is a rating, not a like.
    pub fn reward_of(&self, grade: &Grade) -> f64 {
        match grade.score {
            Some(score) => (score / 10.0).clamp(0.0, 1.0),
            None => 0.0,
        }
    }

    /// Whether a correction is taught from its diff (`diff_corrections`, and
    /// a model that can learn one).
    pub fn diffs(&self, model: Option<&Model>) -> bool {
        self.config.diff_corrections && model.is_some_and(can_correct)
    }

    /// The failed lessons a diff can teach: what the network wrote, what the
    /// teacher wrote, how bad it was.  A lesson that wrote nothing has no
    /// mistake to align, and one the teacher left uncorrected has nothing to
    /// align it against; both go the old way, through [`TutorTrainer::texts_of`].
    pub fn corrections_of(&self, lessons: &[&Lesson], diffs: bool) -> Vec<TutorCorrection> {
        if !diffs {
            return Vec::new();
        }
        let mut out: Vec<TutorCorrection> = Vec::new();
        for lesson in lessons {
            let grade = &lesson.grade;
            if grade.passed || lesson.continuation.trim().is_empty() || grade.correction.trim().is_empty() {
                continue;
            }
            let pair = TutorCorrection {
                wrong: lesson.sentence.trim().to_string(),
                right: grade.correction.trim().to_string(),
                weight: self.weight_of(grade),
            };
            match out.iter_mut().find(|c| c.wrong == pair.wrong && c.right == pair.right) {
                // the same mistake twice (several attempts) keeps its worst mark
                Some(known) => {
                    if pair.weight > known.weight {
                        *known = pair;
                    }
                }
                None => out.push(pair),
            }
        }
        out
    }

    /// Failures as garbage weighted by how bad the mark was, the sentences
    /// that passed rewarded in proportion to their mark and the teacher's own
    /// English at full weight.  A failure the diff teaches is left out of both.
    pub fn texts_of(&self, lessons: &[&Lesson], diffs: bool) -> Graded {
        let cfg = &self.config;
        let diffed = self.corrections_of(lessons, diffs);
        let mut out = Graded::default();
        let mut good: Vec<(String, f64)> = Vec::new();
        for lesson in lessons {
            let grade = &lesson.grade;
            let sentence = lesson.sentence.trim();
            let answer = lesson.exercise.answer.trim();
            if grade.passed {
                if !sentence.is_empty() {
                    good.push((sentence.to_string(), self.reward_of(grade)));
                }
                continue;
            }
            let correction = grade.correction.trim();
            if diffed.iter().any(|c| c.wrong == sentence && c.right == correction) {
                // the diff teaches this one, sentence against correction
                if cfg.teach_answer && !answer.is_empty() {
                    good.push((answer.to_string(), TEACHER_WEIGHT));
                }
                continue;
            }
            // nothing written is nothing to punish: the prefix itself is correct
            if !lesson.continuation.trim().is_empty() {
                out.bad.push(sentence.to_string());
                out.bad_weights.push(self.weight_of(grade));
            }
            if !correction.is_empty() {
                good.push((correction.to_string(), TEACHER_WEIGHT));
            }
            if cfg.teach_answer && !answer.is_empty() {
                good.push((answer.to_string(), TEACHER_WEIGHT));
            }
        }
        (out.good, out.good_weights) = merge_weighted(good);
        out
    }

    /// One set of grades: the corrections taught from their diffs, then 2NRL
    /// over whatever is left - garbage weighted by how bad it was, good English
    /// by how good, earlier corrections replayed at the full rate.
    pub fn learn(
        &mut self,
        model: &mut Model,
        graded: &Graded,
        corrections: &[TutorCorrection],
    ) -> Result<Learned, String> {
        let cfg = self.config.clone();
        // dict(zip(good, good_weights)): a text given twice weighs what it was given last
        let mut weights_of: Vec<(&str, f64)> = Vec::new();
        for (text, weight) in graded.good.iter().zip(&graded.good_weights) {
            match weights_of.iter_mut().find(|(t, _)| t == text) {
                Some(slot) => slot.1 = *weight,
                None => weights_of.push((text, *weight)),
            }
        }
        let (mut good_all, mut good_all_weights) = merge_weighted(graded.good.iter().map(|text| {
            let weight = weights_of
                .iter()
                .find(|(t, _)| t == text)
                .map(|(_, w)| *w)
                .unwrap_or(TEACHER_WEIGHT);
            (text.clone(), weight)
        }));
        if cfg.replay {
            let mut extra: Vec<String> = self.replay.iter().filter(|t| !good_all.contains(t)).cloned().collect();
            if cfg.replay_limit > 0 && extra.len() > cfg.replay_limit {
                extra.drain(..extra.len() - cfg.replay_limit);
            }
            good_all_weights.extend(std::iter::repeat_n(TEACHER_WEIGHT, extra.len()));
            good_all.extend(extra);
        }
        let mut result = Learned {
            bad: graded.bad.len(),
            good: good_all.len(),
            mean_weight: fmean(&graded.bad_weights),
            mean_reward: fmean(&good_all_weights),
            ..Default::default()
        };
        let mut actions: Vec<String> = Vec::new();
        let strength = cfg.base_strength();
        for correction in corrections {
            if self.stopped() {
                break;
            }
            let moved = teach_correction(model, correction, strength, cfg.keep_weight)?;
            result.corrections += 1;
            result.edits += moved.edits;
            result.penalised += moved.penalised;
            result.rewarded += moved.rewarded;
            result.marked_correct += moved.marked_correct;
            result.marked_incorrect += moved.marked_incorrect;
            if moved.loss.is_some() {
                result.pos_loss = moved.loss;
            }
            if !self.replay.contains(&correction.right) {
                self.replay.push(correction.right.clone());
            }
        }
        if result.corrections > 0 {
            actions.push("correct".to_string());
        }
        if graded.bad.is_empty() && good_all.is_empty() {
            result.action = join_actions(&actions);
            self.trim_replay();
            return Ok(result);
        }
        // every pass through the model's own kind, the marks as weights; the
        // stop is looked at after every epoch, as Python's stop event is
        let stop = &self.stop;
        let feedback = cfg.feedback(&graded.bad_weights, &good_all_weights);
        let mut go_on = |_: &EpochRecord| !stop();
        if !graded.bad.is_empty() && !good_all.is_empty() {
            let (negative, positive) = kinds::two_nrl(model, &graded.bad, &good_all, &feedback, &mut go_on)?;
            actions.push("2nrl".to_string());
            result.neg_loss = last_loss(&negative);
            result.pos_loss = last_loss(&positive);
        } else if !good_all.is_empty() {
            let records = kinds::reward(model, &good_all, &feedback, &mut go_on)?;
            actions.push("reward".to_string());
            result.pos_loss = last_loss(&records);
        } else {
            let records = kinds::punish(model, &graded.bad, &feedback, &mut go_on)?;
            actions.push("punish".to_string());
            result.neg_loss = last_loss(&records);
        }
        result.action = join_actions(&actions);
        for text in &graded.good {
            if !self.replay.contains(text) {
                self.replay.push(text.clone());
            }
        }
        self.trim_replay();
        Ok(result)
    }

    /// Keeps the replay buffer to `replay_limit` texts (0 = no limit).
    fn trim_replay(&mut self) {
        let limit = self.config.replay_limit;
        if limit > 0 && self.replay.len() > limit {
            self.replay.drain(..self.replay.len() - limit);
        }
    }

    /// The graded texts with the round's drill sentences added at the full rate.
    fn with_drills(graded: Graded, drills: &[String]) -> Graded {
        let mut out = graded;
        out.good.extend(drills.iter().cloned());
        out.good_weights
            .extend(std::iter::repeat_n(TEACHER_WEIGHT, drills.len()));
        out
    }

    /// Applies the round's grades: once for the whole round, or lesson by
    /// lesson (the drill sentences are taught once either way).  A dry run
    /// reports what would have been learned without touching the network.
    fn learn_lessons(&mut self, nets: &mut Networks, lessons: &[Lesson], drills: &[String]) -> Result<Learned, String> {
        let diffs = self.config.diff_corrections && nets.with_model(|m| can_correct(m));
        let all: Vec<&Lesson> = lessons.iter().collect();
        if !self.config.learn {
            let graded = self.texts_of(&all, diffs);
            let corrections = self.corrections_of(&all, diffs);
            let (good, good_weights) = merge_weighted(
                graded
                    .good
                    .iter()
                    .cloned()
                    .zip(graded.good_weights.iter().copied())
                    .chain(drills.iter().map(|t| (t.clone(), TEACHER_WEIGHT))),
            );
            let edits = corrections
                .iter()
                .map(|c| crate::diff::summary(&c.wrong, &c.right, 0, &crate::encoding::Encoding::default()).len())
                .sum();
            return Ok(Learned {
                bad: graded.bad.len(),
                good: good.len(),
                mean_weight: fmean(&graded.bad_weights),
                mean_reward: fmean(&good_weights),
                corrections: corrections.len(),
                edits,
                ..Default::default()
            });
        }
        if self.config.twonrl_per != "lesson" {
            let graded = Self::with_drills(self.texts_of(&all, diffs), drills);
            let corrections = self.corrections_of(&all, diffs);
            return nets.with_model(|m| self.learn(m, &graded, &corrections));
        }
        let mut merged = Learned::default();
        let mut actions: Vec<String> = Vec::new();
        let mut bad_seen: Vec<f64> = Vec::new();
        let mut good_seen: Vec<f64> = Vec::new();
        let mut drills: &[String] = drills;
        for lesson in lessons {
            if self.stopped() {
                break;
            }
            let one = [lesson];
            let graded = self.texts_of(&one, diffs);
            let corrections = self.corrections_of(&one, diffs);
            bad_seen.extend(&graded.bad_weights);
            good_seen.extend(&graded.good_weights);
            let graded = Self::with_drills(graded, drills);
            drills = &[]; // taught once
            let outcome = nets.with_model(|m| self.learn(m, &graded, &corrections))?;
            if let Some(action) = &outcome.action {
                actions.extend(action.split('+').map(str::to_string));
            }
            merged.bad += outcome.bad;
            merged.good += outcome.good;
            merged.corrections += outcome.corrections;
            merged.edits += outcome.edits;
            merged.penalised += outcome.penalised;
            merged.rewarded += outcome.rewarded;
            merged.marked_correct += outcome.marked_correct;
            merged.marked_incorrect += outcome.marked_incorrect;
            if outcome.neg_loss.is_some() {
                merged.neg_loss = outcome.neg_loss;
            }
            if outcome.pos_loss.is_some() {
                merged.pos_loss = outcome.pos_loss;
            }
        }
        merged.action = join_actions(&actions);
        merged.mean_weight = fmean(&bad_seen);
        merged.mean_reward = fmean(&good_seen);
        Ok(merged)
    }

    /// Hands the round's failures to the negative network: the teacher named
    /// the mistake, so it is the reason (`None` without a negative network).
    fn teach_negative(&self, nets: &mut Networks, lessons: &[Lesson]) -> Result<Option<TeachReport>, String> {
        if !nets.has_negative() || lessons.is_empty() {
            return Ok(None);
        }
        let threshold = self.config.threshold;
        match nets.with_negative(|negative| {
            teach_lessons(negative, lessons, threshold, true, "tutor", &TeachOptions::default())
        })? {
            Some(report) => Ok(Some(report?)),
            None => Ok(None),
        }
    }

    /// One lesson as the record the frontend shows.
    fn lesson_record(round: usize, lesson: &Lesson, batch: usize) -> Json {
        let grade = &lesson.grade;
        let mark = |x: Option<f64>| x.map(Json::Num).unwrap_or(Json::Null);
        Json::obj([
            ("kind", Json::str("lesson")),
            ("batch", Json::Int(batch as i64)),
            ("round", Json::Int(round as i64)),
            ("exercise", Json::str(lesson.exercise.id.clone())),
            ("prefix", Json::str(lesson.exercise.prefix.clone())),
            ("focus", Json::str(lesson.exercise.focus.clone())),
            ("attempt", Json::Int(lesson.attempt as i64 + 1)),
            ("mode", Json::str(lesson.mode.clone())),
            ("continuation", Json::str(clip(&lesson.continuation, 400))),
            ("sentence", Json::str(clip(&lesson.sentence, 400))),
            ("score", mark(grade.score)),
            ("grammar", mark(grade.grammar)),
            ("spelling", mark(grade.spelling)),
            ("fluency", mark(grade.fluency)),
            ("passed", Json::Bool(grade.passed)),
            ("error", Json::str(grade.error.clone())),
            ("correction", Json::str(clip(&grade.correction, 400))),
            ("comment", Json::str(grade.comment.clone())),
            ("graded_by", Json::str(grade.graded_by.clone())),
            ("probability", Json::Num(lesson.probability)),
            ("seconds", Json::Num(lesson.seconds)),
            (
                "changes",
                Json::Arr(lesson.changes().iter().map(crate::diff::Edit::to_json).collect()),
            ),
            ("why", Json::str(lesson.why.clone())),
            (
                "variants",
                Json::Arr(lesson.variants.iter().map(TutorCorrection::to_json).collect()),
            ),
        ])
    }

    /// One round: exercises, completions, grades, and the learning they lead
    /// to; returns its record (not yet emitted) and its lessons.
    pub fn run_round(
        &mut self,
        nets: &mut Networks,
        round: usize,
        batch: usize,
    ) -> Result<(Json, Vec<Lesson>), String> {
        let started = Instant::now();
        let exercises = self.set_exercises(round)?;
        let mut lessons: Vec<Lesson> = Vec::new();
        for exercise in &exercises {
            if self.stopped() {
                break;
            }
            for attempt in 0..self.config.attempts {
                lessons.push(nets.with_model(|m| self.complete(m, exercise, attempt))?);
            }
        }
        if !lessons.is_empty() && !self.stopped() {
            self.grade(&mut lessons)?;
        }
        let (explained, similar) = self.widen(&mut lessons, round, nets.has_negative());
        for lesson in &lessons {
            let record = Self::lesson_record(round, lesson, batch);
            self.emit(record);
        }
        self.lessons.extend(lessons.iter().cloned());
        let card = report_card(&lessons);
        let weakest = card.at("weakest").to_strings();
        let mut drills: Vec<String> = Vec::new();
        if self.config.drills > 0 && !lessons.is_empty() && !self.stopped() {
            let cfg = &self.config;
            match drill_sentences(self.client, &cfg.topic, cfg.drills, &weakest, &cfg.tutor_model) {
                Ok(written) => drills = written,
                // the lesson stands without its drill sentences
                Err(err) => self.note("round", round, format!("no drill sentences: {err}")),
            }
        }
        let learned = if self.stopped() {
            None
        } else {
            Some(self.learn_lessons(nets, &lessons, &drills)?)
        };
        let blamed = self.teach_negative(nets, &lessons)?;
        if self.config.adapt {
            self.weak = weakest;
        }
        let cfg = &self.config;
        let mut pairs: Vec<(String, Json)> = vec![
            ("kind".to_string(), Json::str("round")),
            ("batch".to_string(), Json::Int(batch as i64)),
            ("round".to_string(), Json::Int(round as i64)),
            ("topic".to_string(), Json::str(cfg.topic.clone())),
            (
                "focus".to_string(),
                cfg.focus.clone().map(Json::Str).unwrap_or(Json::Null),
            ),
            ("exercises".to_string(), Json::Int(exercises.len() as i64)),
            ("drills".to_string(), Json::Int(drills.len() as i64)),
            ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
        ];
        pairs.extend(card_pairs(card));
        if let Some(learned) = &learned {
            pairs.extend(learned.pairs());
        }
        if let Some(report) = blamed {
            let doc = report.to_json();
            pairs.push(("negative_blamed".to_string(), Json::Int(report.blamed as i64)));
            pairs.push(("negative_edges".to_string(), Json::Int(report.edges as i64)));
            pairs.push(("negative_reasons".to_string(), doc.at("reasons").clone()));
            pairs.push(("explained".to_string(), Json::Int(explained as i64)));
            pairs.push(("similar".to_string(), Json::Int(similar as i64)));
        }
        crate::log_info!(
            LOG,
            "round {round}: {}/{} passed, learned: {}",
            lessons.iter().filter(|l| l.grade.passed).count(),
            lessons.len(),
            learned
                .as_ref()
                .and_then(|l| l.action.clone())
                .unwrap_or_else(|| "nothing".to_string())
        );
        Ok((Json::Obj(pairs), lessons))
    }

    /// Every batch of rounds, each ending in its own report card; the records
    /// go to `progress` as they happen and come back in order (the lessons and
    /// notes among them only through `progress` and [`TutorTrainer::history`]).
    ///
    /// With `batches` above 1 (or 0, until stopped) every card is planned from
    /// and the plan applied before the next batch; a batch that cannot be
    /// planned ends the run rather than repeating itself.
    pub fn run(&mut self, nets: &mut Networks) -> Result<Vec<Json>, String> {
        let mut records: Vec<Json> = Vec::new();
        let mut round = 0usize;
        let mut batch = 0usize;
        while self.config.batches == 0 || batch < self.config.batches {
            if batch > 0 && self.stopped() {
                break; // stopped between batches: no empty card for one that never ran
            }
            batch += 1;
            let taught = self.lessons.len(); // this batch's lessons, so its card is its own
            let mut here = 0usize;
            for _ in 0..self.config.rounds {
                if self.stopped() {
                    break;
                }
                round += 1;
                here += 1;
                let (record, _) = self.run_round(nets, round, batch)?;
                self.emit(record.clone());
                self.checkpoint(nets, round, batch, &record)?;
                records.push(record);
            }
            let mut summary = vec![
                ("kind".to_string(), Json::str("report")),
                ("batch".to_string(), Json::Int(batch as i64)),
                ("rounds".to_string(), Json::Int(here as i64)),
                ("topic".to_string(), Json::str(self.config.topic.clone())),
            ];
            summary.extend(card_pairs(report_card(&self.lessons[taught..])));
            let summary = Json::Obj(summary);
            self.emit(summary.clone()); // a batch always ends with its card, stopped or not
            records.push(summary.clone());
            let last = self.config.batches > 0 && batch >= self.config.batches;
            let wanted = self.config.plan > 0 || !last; // a batch with a successor is always planned
            if self.stopped() || !wanted || self.lessons.len() == taught {
                break;
            }
            let count = if self.config.plan > 0 {
                self.config.plan
            } else {
                DEFAULT_PLAN_LESSONS
            };
            let plan = match self.plan(&summary, count) {
                Ok(plan) => plan,
                Err(ReviewError::Llm(err)) => {
                    // the lessons stand without a plan for the next ones
                    self.note("batch", batch, format!("no lesson plan: {err}"));
                    break;
                }
                Err(err) => return Err(err.to_string()),
            };
            let mut record = vec![
                ("kind".to_string(), Json::str("plan")),
                ("batch".to_string(), Json::Int(batch as i64)),
                ("rounds".to_string(), Json::Int(here as i64)),
            ];
            record.extend(plan.pairs());
            let record = Json::Obj(record);
            self.emit(record.clone());
            records.push(record);
            if last {
                break;
            }
            let applied = self.apply_plan(&plan)?;
            if self.stopped() {
                break; // stopped while it was planning: do not announce a batch that will not run
            }
            let mut started = vec![
                ("kind".to_string(), Json::str("batch")),
                ("batch".to_string(), Json::Int(batch as i64 + 1)),
            ];
            if let Json::Obj(pairs) = applied {
                started.extend(pairs);
            }
            let started = Json::Obj(started);
            self.emit(started.clone());
            records.push(started);
        }
        Ok(records)
    }

    /// Saves the model when this round is due a checkpoint.
    fn checkpoint(&self, nets: &mut Networks, round: usize, batch: usize, record: &Json) -> Result<(), String> {
        let every = self.config.checkpoint_every;
        let Some(manager) = self.checkpoints else {
            return Ok(());
        };
        if every == 0 || round % every != 0 {
            return Ok(());
        }
        let metrics = Json::obj([
            ("round", Json::Int(round as i64)),
            ("batch", Json::Int(batch as i64)),
            ("mean_score", record.at("mean_score").clone()),
            ("pass_rate", record.at("pass_rate").clone()),
        ]);
        nets.with_model(|m| manager.save(m, round as i64, "tutor", Some(metrics)))?;
        Ok(())
    }

    /// Teaches what comes next to a plan: its brief, and the step up the marks
    /// earned.  The brief becomes [`TutorConfig::brief`] - handed to the
    /// exercise writer with every round from now on - and the upgrade sets the
    /// level, the openings, the pass mark and the drill sentences.  The
    /// single-focus pin is released: the brief carries the points of grammar,
    /// in order, and a pin from an earlier batch would silently overrule it.
    pub fn apply_plan(&mut self, plan: &LessonPlan) -> Result<Json, String> {
        let upgrade = plan
            .upgrade
            .as_ref()
            .map(|u| u.to_json())
            .unwrap_or(Json::Obj(Vec::new()));
        let cfg = &mut self.config;
        if !plan.prompt.is_empty() {
            cfg.brief = plan.prompt.clone();
        }
        cfg.focus = None;
        let text = |key: &str| {
            upgrade
                .get(key)
                .filter(|v| crate::llm::fields::truthy(v))
                .map(crate::llm::fields::py_str_of)
        };
        if let Some(level) = text("level") {
            cfg.level = level;
        }
        if let Some(words) = text("words") {
            cfg.words = words;
        }
        if let Some(threshold) = upgrade.get("threshold").and_then(mark_of) {
            cfg.threshold = threshold;
        }
        if let Some(drills) = upgrade.get("drills").filter(|v| !v.is_null()) {
            cfg.drills = count_of(drills) as usize;
        }
        cfg.validate()?;
        Ok(Json::obj([
            (
                "step",
                upgrade.get("step").cloned().unwrap_or_else(|| Json::str("hold")),
            ),
            ("brief", Json::str(cfg.brief.clone())),
            ("topic", Json::str(cfg.topic.clone())),
            ("level", Json::str(cfg.level.clone())),
            ("words", Json::str(cfg.words.clone())),
            ("threshold", Json::Num(cfg.threshold)),
            ("drills", Json::Int(cfg.drills as i64)),
            ("note", upgrade.get("note").cloned().unwrap_or_else(|| Json::str(""))),
        ]))
    }

    /// The next lessons, planned by the teacher from a report card - by
    /// default the run's own.
    pub fn plan(&self, card: &Json, count: usize) -> Result<LessonPlan, ReviewError> {
        let cfg = &self.config;
        plan_lessons(
            self.client,
            card,
            &PlanRequest {
                topic: cfg.topic.clone(),
                level: cfg.level.clone(),
                words: cfg.words.clone(),
                threshold: cfg.threshold,
                count: if count > 0 { count } else { DEFAULT_PLAN_LESSONS },
                exercises: cfg.exercises as i64,
                drills: cfg.drills as i64,
                model: cfg.tutor_model.clone(),
                temperature: 0.0,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{Teacher, CORPUS};
    use super::*;
    use crate::negative::NegativeOptions;
    use crate::TrainOptions;

    fn trained() -> Model {
        let corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
        let mut model = Model::new(7, crate::GraphOptions::default()).unwrap();
        model
            .train(
                &corpus,
                &TrainOptions {
                    epochs: 6,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    fn config() -> TutorConfig {
        TutorConfig {
            topic: "animals".to_string(),
            rounds: 1,
            exercises: 2,
            mode: "beam".to_string(),
            threshold: 9.5,
            neg_epochs: 1,
            pos_epochs: 1,
            tutor_model: "fake".to_string(),
            ..Default::default()
        }
    }

    #[test]
    fn the_defaults_are_python_s_and_nonsense_is_refused() {
        let d = TutorConfig::default();
        assert!(d.validate().is_ok());
        let doc = d.to_json();
        let keys: Vec<&str> = match &doc {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(keys.len(), 42);
        assert_eq!((keys[0], keys[41]), ("topic", "checkpoint_every"));
        assert_eq!(doc.at("mode").as_str(), Some("dijkstra"));
        assert!(doc.at("strength").is_null() && doc.at("beam").is_null() && doc.at("focus").is_null());
        for (bad, words) in [
            (
                TutorConfig {
                    topic: " ".into(),
                    ..Default::default()
                },
                "topic",
            ),
            (
                TutorConfig {
                    mode: "nope".into(),
                    ..Default::default()
                },
                "mode must be one of dijkstra, beam, sample",
            ),
            (
                TutorConfig {
                    twonrl_per: "hourly".into(),
                    ..Default::default()
                },
                "twonrl_per",
            ),
            (
                TutorConfig {
                    threshold: 99.0,
                    ..Default::default()
                },
                "threshold",
            ),
            (
                TutorConfig {
                    keep_weight: 1.5,
                    ..Default::default()
                },
                "keep_weight",
            ),
            (
                TutorConfig {
                    variants: 11,
                    ..Default::default()
                },
                "variants must lie in [0, 10]",
            ),
        ] {
            assert!(bad.validate().unwrap_err().contains(words), "{words}");
        }
        let mut other = TutorConfig {
            grader_provider: "OpenAI".into(),
            ..Default::default()
        };
        other.resolve().unwrap();
        assert_eq!(other.grader_provider, "chatgpt");
        assert_eq!(other.resolved_grader_model(), crate::chatgpt::default_model());
        let mut bad = TutorConfig {
            tutor_provider: "gemini".into(),
            ..Default::default()
        };
        assert!(bad.resolve().is_err());
    }

    #[test]
    fn weights_group_heaviest_first() {
        let texts: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let groups = weight_groups(&texts, &[0.5, 1.0, 0.0, 0.50004], "w").unwrap();
        assert_eq!(
            groups,
            vec![
                (1.0, vec!["b".to_string()]),
                (0.5, vec!["a".to_string(), "d".to_string()])
            ]
        );
        assert!(weight_groups(&texts, &[1.0], "w")
            .unwrap_err()
            .contains("has 1 entries for 4 texts"));
        assert!(weight_groups(&texts[..1], &[-1.0], "w").is_err());
        let (texts, weights) = merge_weighted(vec![
            (" x ".to_string(), 0.2),
            ("y".to_string(), 1.0),
            ("x".to_string(), 0.7),
            ("  ".to_string(), 1.0),
        ]);
        assert_eq!(
            (texts, weights),
            (vec!["x".to_string(), "y".to_string()], vec![0.7, 1.0])
        );
        assert_eq!(
            join_actions(&["correct".into(), "2nrl".into(), "correct".into()]).as_deref(),
            Some("correct+2nrl")
        );
        assert_eq!(join_actions(&[]), None);
    }

    #[test]
    fn a_mark_decides_how_much_is_learned() {
        let teacher = Teacher::new();
        let trainer = TutorTrainer::new(&teacher, &teacher, config()).unwrap();
        let grade = |score: Option<f64>| Grade {
            score,
            ..Grade::default()
        };
        assert_eq!(trainer.weight_of(&grade(None)), 1.0);
        assert_eq!(trainer.weight_of(&grade(Some(9.5))), 0.25);
        assert_eq!(trainer.weight_of(&grade(Some(0.0))), 1.0);
        assert_eq!(trainer.reward_of(&grade(Some(9.0))), 0.9);
        assert_eq!(trainer.reward_of(&grade(None)), 0.0);
    }

    #[test]
    fn a_round_asks_marks_and_learns() {
        let teacher = Teacher::new();
        let mut model = trained();
        let before = model.meta.feedback_passes.value;
        let seen = std::cell::RefCell::new(Vec::new());
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                rounds: 2,
                drills: 2,
                ..config()
            },
        )
        .unwrap()
        .with_progress(|r| seen.borrow_mut().push(r.at("kind").as_str().unwrap_or("").to_string()));
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let kinds: Vec<&str> = records.iter().filter_map(|r| r.at("kind").as_str()).collect();
        assert_eq!(kinds, vec!["round", "round", "report"]);
        assert_eq!(
            *seen.borrow(),
            vec!["lesson", "lesson", "round", "lesson", "lesson", "round", "report"]
        );
        assert_eq!(teacher.prompts("exercises").len(), 2);
        assert_eq!(teacher.prompts("grades").len(), 2);
        assert_eq!(teacher.prompts("drills").len(), 2);
        let round = &records[0];
        assert_eq!(round.at("lessons").as_i64(), Some(2));
        assert_eq!(round.at("drills").as_i64(), Some(2));
        assert!(round.at("action").as_str().is_some());
        assert!(round.get("negative_blamed").is_none());
        // the second round drills what the first got wrong
        let written = teacher.prompts("exercises");
        assert!(written[1].contains("The student keeps making these mistakes, so drill them: agreement."));
        // every failure came back corrected: it is taught from its diff, the rest is rewarded by its mark
        assert_eq!(round.at("action").as_str(), Some("correct+reward"));
        assert_eq!(round.at("corrections").as_i64(), Some(2));
        assert_eq!(round.at("bad").as_i64(), Some(0));
        assert!(model.meta.feedback_passes.value > before);
        let report = &records[2];
        assert_eq!(
            (report.at("rounds").as_i64(), report.at("lessons").as_i64()),
            (Some(2), Some(4))
        );
        assert_eq!(report.at("weakest").to_strings()[0], "agreement");
        assert!(!trainer.replay.is_empty());
    }

    #[test]
    fn a_dry_run_changes_nothing() {
        let teacher = Teacher::new();
        let mut model = trained();
        let before = crate::report::stats(&model).render(0);
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                learn: false,
                ..config()
            },
        )
        .unwrap();
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        assert!(records[0].at("action").is_null());
        assert_eq!(
            records[0].at("corrections").as_i64(),
            Some(2),
            "what would have been corrected"
        );
        assert!(records[0].at("edits").as_i64().unwrap() > 0);
        assert_eq!(crate::report::stats(&model).render(0), before);
    }

    #[test]
    fn lesson_by_lesson_learning_adds_up() {
        let teacher = Teacher::new();
        let mut model = trained();
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                twonrl_per: "lesson".into(),
                ..config()
            },
        )
        .unwrap();
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let round = &records[0];
        assert_eq!(round.at("action").as_str(), Some("correct+reward"));
        assert_eq!(round.at("corrections").as_i64(), Some(2));
    }

    #[test]
    fn without_diffs_a_failure_is_weighted_garbage() {
        let teacher = Teacher::new();
        let mut model = trained();
        let before = model.meta.twonrl_runs.value;
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                diff_corrections: false,
                ..config()
            },
        )
        .unwrap();
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let round = &records[0];
        assert_eq!(round.at("action").as_str(), Some("2nrl"));
        assert_eq!(
            (round.at("bad").as_i64(), round.at("corrections").as_i64()),
            (Some(2), Some(0))
        );
        // a near miss weighs less than a hopeless answer
        let weight = round.at("mean_weight").as_f64().unwrap();
        assert!(weight > 0.25 && weight <= 1.0, "{weight}");
        assert_eq!(model.meta.twonrl_runs.value, before + 1);
    }

    #[test]
    fn a_negative_network_learns_why_and_the_same_mistake_again() {
        let teacher = Teacher::new();
        let mut model = trained();
        let mut negative = Model::new_negative(0, &NegativeOptions::default()).unwrap();
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                variants: 2,
                variant_weight: 0.25,
                ..config()
            },
        )
        .unwrap();
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: Some(&mut negative),
            })
            .unwrap();
        let round = &records[0];
        assert_eq!(round.at("explained").as_i64(), Some(2));
        assert_eq!(round.at("similar").as_i64(), Some(4));
        assert_eq!(round.at("negative_blamed").as_i64(), Some(6));
        // "the cat sat on the mat" is correct English failed by the pass mark alone: no mistake named
        assert_eq!(round.at("negative_reasons").render(0), "{\"other\":3,\"agreement\":3}");
        assert_eq!(teacher.prompts("why").len(), 1);
        let lessons: Vec<&Json> = trainer
            .history
            .iter()
            .filter(|r| r.at("kind").as_str() == Some("lesson"))
            .collect();
        assert!(lessons
            .iter()
            .all(|l| l.at("variants").as_array().len() == 2 && !l.at("why").as_str().unwrap().is_empty()));
        let log = &negative.neg.as_ref().unwrap().log;
        assert_eq!(log.iter().filter(|e| e.source == "tutor:similar").count(), 4);
    }

    #[test]
    fn an_auto_run_plans_and_teaches_itself() {
        let teacher = Teacher::new();
        let mut model = trained();
        let mut trainer = TutorTrainer::new(
            &teacher,
            &teacher,
            TutorConfig {
                batches: 3,
                learn: false,
                ..config()
            },
        )
        .unwrap();
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let kinds: Vec<&str> = records.iter().filter_map(|r| r.at("kind").as_str()).collect();
        assert_eq!(
            kinds,
            vec!["round", "report", "plan", "batch", "round", "report", "plan", "batch", "round", "report"]
        );
        let started: Vec<&Json> = records
            .iter()
            .filter(|r| r.at("kind").as_str() == Some("batch"))
            .collect();
        assert_eq!(started[0].at("batch").as_i64(), Some(2));
        assert_eq!(started[0].at("step").as_str(), Some("hold"));
        assert_eq!(
            started[0].at("drills").as_i64(),
            Some(3),
            "a held batch imitates drills"
        );
        let written = teacher.prompts("exercises");
        assert_eq!(written.len(), 3);
        assert!(written[2].contains(&format!(
            "The plan for this batch of lessons: {}",
            started[1].at("brief").as_str().unwrap()
        )));
        assert_eq!(trainer.config.focus, None);
        let plan = &records[2];
        assert_eq!(plan.at("source").as_str(), Some("ollama"));
        assert_eq!(
            plan.at("lessons").as_array()[0].at("targets").as_str(),
            Some("agreement")
        );
    }

    #[test]
    fn a_stop_ends_the_run_with_its_card() {
        let teacher = Teacher::new();
        let mut model = trained();
        let rounds = std::cell::Cell::new(0);
        let mut trainer = TutorTrainer::new(&teacher, &teacher, TutorConfig { batches: 0, ..config() })
            .unwrap()
            .with_stop(|| rounds.get() >= 1)
            .with_progress(|r| {
                if r.at("kind").as_str() == Some("round") {
                    rounds.set(rounds.get() + 1)
                }
            });
        let records = trainer
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let kinds: Vec<&str> = records.iter().filter_map(|r| r.at("kind").as_str()).collect();
        assert_eq!(kinds, vec!["round", "report"]);
    }
}
