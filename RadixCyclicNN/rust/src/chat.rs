//! The model in conversation with an LLM, and the LLM marking the
//! conversation (`radixnet/chat.py`, `go/radixnet/chat.go`,
//! `go/server/chat.go`).
//!
//! Every other teacher in this project talks *at* the network: the English
//! tutor writes a prefix and marks the completion ([`crate::tutor`]), the
//! critic reviews texts the model wrote alone ([`crate::critic`]).  The one
//! thing a language model is actually for - holding up its end of a
//! conversation - was only ever tested against *itself*
//! ([`crate::dialogue`], two voices of the same network), where nothing can
//! tell it that its answer did not follow on.  This loop puts a second party
//! on the other side of the line.  One conversation is:
//!
//! 1. the **partner** - a local Ollama model by default, ChatGPT when the
//!    provider says so - says a line, told to keep it short, plain and easy to
//!    carry on from, because that is what a character-level model can reply
//!    to ([`crate::review::chat_line`]);
//! 2. the **model** replies the way it replies to anything: the tail of that
//!    line is located in the graph and continued ([`crate::dialogue::reply`]),
//!    so a reply is a real walk of the network and not a prompt trick.  With a
//!    negative network in hand, the pair vetoes a reply before it is spoken
//!    ([`crate::duo`]);
//! 3. they take turns for `turns` exchanges;
//! 4. the **judge** marks every reply out of 10 against the line it answered,
//!    and the conversation as a whole ([`crate::review::review_conversation`]).
//!
//! What the marks buy: the failures **blame** the negative network and the
//! passes **clear** it ([`crate::review::teach_reviews`]), and - unlike the
//! critic - this loop **trains the positive model too**: the replies that
//! passed are rewarded, the ones that failed punished (2NRL), and the
//! partner's own lines join the positive phase, because they are exactly what
//! a good reply in this conversation would have looked like.
//!
//! That last point is the whole idea.  A model that only ever learns from a
//! corpus has no way of finding out that what it said did not answer the
//! question; here something answers back, and says so.
//!
//! # Where the networks live
//!
//! As for the tutor ([`crate::tutor::Networks`]): the command owns both for
//! the run; the server takes each lock for a step - a reply (the model's lock,
//! then the negative network's when it guards), a blame, a learning pass - and
//! holds nothing while the partner or the judge thinks.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::blame::{TeachOptions, TeachReport};
use crate::cli::{negative_stats, Ctx};
use crate::dialogue::{repeats, reply, Heard, ReplyOptions, Turn, EXPLORE};
use crate::duo::{Filter, FilterConfig};
use crate::http::{accepted, Answer, ApiError, Request, Server};
use crate::json::Json;
use crate::kinds;
use crate::llm::fields::Fields;
use crate::llm::{new_client, normalise_provider, LlmClient, CHATGPT, DEFAULT_PROVIDER, PROVIDERS};
use crate::model::Model;
use crate::mt19937::Mt19937;
use crate::negative::python_repr;
use crate::radix::Feedback;
use crate::review::{chat_line, review_conversation, teach_reviews, Review, ReviewSummary, CHAT_SPEAKERS};
use crate::service::Service;
use crate::tutor::Networks;

/// What this module's own lines are filed under.
const LOG: &str = "chat";

// -- the settings ---------------------------------------------------------------------------------

/// How the conversation is held, marked and learned from (Python's `ChatConfig`).
#[derive(Clone, Debug, PartialEq)]
pub struct ChatConfig {
    /// Conversations to hold; 0 keeps going until something stops it.
    pub conversations: usize,
    /// Replies the model gives per conversation (the partner speaks between them).
    pub turns: usize,
    /// What to talk about; empty lets the partner choose.
    pub topic: String,
    /// The first line, given; empty lets the partner open.
    pub opening: String,
    /// Who the partner is being ("a curious child", "a vet", ...).
    pub persona: String,
    /// Characters of the previous line a reply picks up.
    pub context: usize,
    /// Characters the model may add to a context in one reply.
    pub max_length: usize,
    /// `beam` (the most likely unheard reply) or `sample`.
    pub mode: String,
    /// Candidates considered per reply.
    pub k: usize,
    /// Sampling temperature of the model's replies.
    pub temperature: f64,
    /// Sampling temperature of the partner's lines.
    pub partner_temperature: f64,
    /// The pass mark out of 10: below it a reply is a failure.
    pub threshold: f64,
    pub provider: String,
    /// The partner's model; empty is the client's own.
    pub partner_model: String,
    /// A different model for marking; empty is the partner's.
    pub judge_model: String,
    /// Let the negative network veto a reply before it is spoken.
    pub guard: bool,
    /// Blame the failed replies (and clear with the passed ones).
    pub blame: bool,
    pub clear_passes: bool,
    /// Train the positive model on the marked replies (2NRL), and let its
    /// rethinks teach the graph where it goes round.
    pub learn: bool,
    /// Put the partner's own lines in the positive phase.
    pub teach_partner: bool,
    /// Do not let the model say what the conversation has already heard.
    pub avoid_repeats: bool,
    /// Do not let one reply repeat its own words.
    pub avoid_word_repeats: bool,
    /// Times a reply that caught itself repeating may back up.
    pub explore: usize,
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
    /// Blame epochs per conversation.
    pub epochs: usize,
    /// Seed of the first conversation's sampling; later ones advance it.
    pub seed: Option<i64>,
}

impl Default for ChatConfig {
    /// The Python defaults.
    fn default() -> ChatConfig {
        ChatConfig {
            conversations: 1,
            turns: 4,
            topic: String::new(),
            opening: String::new(),
            persona: String::new(),
            context: 12,
            max_length: 60,
            mode: "beam".to_string(),
            k: 5,
            temperature: 1.0,
            partner_temperature: 0.8,
            threshold: 6.0,
            provider: DEFAULT_PROVIDER.to_string(),
            partner_model: String::new(),
            judge_model: String::new(),
            guard: true,
            blame: true,
            clear_passes: true,
            learn: true,
            teach_partner: true,
            avoid_repeats: true,
            avoid_word_repeats: true,
            explore: EXPLORE,
            neg_epochs: 2,
            pos_epochs: 3,
            neg_lr: 0.5,
            pos_lr: 0.1,
            batch_size: 4,
            strength: None,
            epochs: 1,
            seed: None,
        }
    }
}

impl ChatConfig {
    /// Refuses settings the loop cannot run with - Python's messages, in its order.
    pub fn validate(&self) -> Result<(), String> {
        let negative = |x: f64| x.is_nan() || x < 0.0;
        if self.turns < 1 {
            return Err("turns must be >= 1".to_string());
        }
        if self.max_length < 1 {
            return Err("max_length must be >= 1".to_string());
        }
        if self.mode != "beam" && self.mode != "sample" {
            return Err("mode must be 'beam' or 'sample'".to_string());
        }
        if self.k < 1 {
            return Err("k must be >= 1".to_string());
        }
        if negative(self.temperature) || negative(self.partner_temperature) {
            return Err("temperature must be >= 0".to_string());
        }
        if !(0.0..=10.0).contains(&self.threshold) {
            return Err("threshold must lie in [0, 10]".to_string());
        }
        if negative(self.neg_lr) || negative(self.pos_lr) {
            return Err("learning rates must be >= 0".to_string());
        }
        if self.batch_size < 1 {
            return Err("batch_size must be >= 1".to_string());
        }
        if self.strength.is_some_and(negative) {
            return Err("strength must be >= 0".to_string());
        }
        normalise_provider(&self.provider)?;
        Ok(())
    }

    /// The settings as a document (`dataclasses.asdict` of Python's config).
    pub fn to_json(&self) -> Json {
        let int = |n: usize| Json::Int(n as i64);
        let text = |s: &str| Json::str(s);
        Json::obj([
            ("conversations", int(self.conversations)),
            ("turns", int(self.turns)),
            ("topic", text(&self.topic)),
            ("opening", text(&self.opening)),
            ("persona", text(&self.persona)),
            ("context", int(self.context)),
            ("max_length", int(self.max_length)),
            ("mode", text(&self.mode)),
            ("k", int(self.k)),
            ("temperature", Json::Num(self.temperature)),
            ("partner_temperature", Json::Num(self.partner_temperature)),
            ("threshold", Json::Num(self.threshold)),
            ("provider", text(&self.provider)),
            ("partner_model", text(&self.partner_model)),
            ("judge_model", text(&self.judge_model)),
            ("guard", Json::Bool(self.guard)),
            ("blame", Json::Bool(self.blame)),
            ("clear_passes", Json::Bool(self.clear_passes)),
            ("learn", Json::Bool(self.learn)),
            ("teach_partner", Json::Bool(self.teach_partner)),
            ("avoid_repeats", Json::Bool(self.avoid_repeats)),
            ("avoid_word_repeats", Json::Bool(self.avoid_word_repeats)),
            ("explore", int(self.explore)),
            ("neg_epochs", int(self.neg_epochs)),
            ("pos_epochs", int(self.pos_epochs)),
            ("neg_lr", Json::Num(self.neg_lr)),
            ("pos_lr", Json::Num(self.pos_lr)),
            ("batch_size", int(self.batch_size)),
            ("strength", self.strength.map(Json::Num).unwrap_or(Json::Null)),
            ("epochs", int(self.epochs)),
            ("seed", self.seed.map(Json::Int).unwrap_or(Json::Null)),
        ])
    }

    /// Whether the loop needs a negative network: to blame, or to guard.
    pub fn wants_negative(&self) -> bool {
        self.blame || self.guard
    }

    /// The judge's model: its own, else the partner's (`""`: the client's own).
    fn judge_model(&self) -> &str {
        if self.judge_model.is_empty() {
            &self.partner_model
        } else {
            &self.judge_model
        }
    }
}

// -- one conversation -----------------------------------------------------------------------------

/// One conversation as it was held.
#[derive(Clone, Debug, Default)]
pub struct Held {
    /// Every line, as `(speaker, text)`.
    pub transcript: Vec<(String, String)>,
    /// The `(said to it, its reply)` pairs the judge marks.
    pub exchanges: Vec<(String, String)>,
    pub turns: Vec<Turn>,
    /// The replies the model could only repeat: punished with the failures,
    /// whatever the judge made of them.
    pub repeats: Vec<String>,
    /// Why it ended early (`""` when it ran its course).
    pub stalled: String,
    /// The distinct replies the guard would not let through.
    pub vetoed: usize,
}

/// The conversation loop: the partner talks, the model replies, the judge
/// marks, both networks learn.
///
/// `client` plays the partner and `judge` marks (the partner itself unless
/// the configuration names another model or endpoint).  `stop` is asked
/// before every reply and every conversation; `progress` sees every exchange,
/// conversation and report record as it happens.
pub struct Chat<'a> {
    client: &'a dyn LlmClient,
    judge: &'a dyn LlmClient,
    pub config: ChatConfig,
    /// The conversation records, in order.
    pub history: Vec<Json>,
    /// Conversations held so far.
    pub number: usize,
    stop: Box<dyn Fn() -> bool + 'a>,
    progress: Box<dyn FnMut(&Json) + 'a>,
}

impl<'a> Chat<'a> {
    /// The loop, once its settings are known to be sound.
    pub fn new(
        client: &'a dyn LlmClient,
        judge: &'a dyn LlmClient,
        mut config: ChatConfig,
    ) -> Result<Chat<'a>, String> {
        config.validate()?;
        config.provider = normalise_provider(&config.provider)?.to_string();
        Ok(Chat {
            client,
            judge,
            config,
            history: Vec::new(),
            number: 0,
            stop: Box::new(|| false),
            progress: Box::new(|_| {}),
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

    fn stopped(&self) -> bool {
        (self.stop)()
    }

    /// The partner's next line; only the partner's thinking happens with
    /// nothing locked.
    fn partner_line(&self, transcript: &[(String, String)]) -> Result<String, String> {
        let cfg = &self.config;
        Ok(chat_line(
            self.client,
            transcript,
            &cfg.topic,
            &cfg.persona,
            &cfg.partner_model,
            cfg.partner_temperature,
        )?)
    }

    /// Holds one conversation.
    pub fn converse(&mut self, nets: &mut Networks) -> Result<Held, String> {
        let cfg = self.config.clone();
        let seed = cfg.seed.map(|s| s + self.number as i64);
        let mut rng = if cfg.mode == "sample" {
            seed.map(Mt19937::new)
        } else {
            None
        };
        // the pair guards only when it has something to veto with
        let guarding = cfg.guard && nets.with_negative(|n| crate::duo::ready(n))? == Some(true);
        let mut judged: Vec<(String, bool)> = Vec::new();
        let mut held = Held::default();
        let mut heard = Heard::new(&[]); // what has been said, on both sides
        let mut line = cfg.opening.trim().to_string();
        if line.is_empty() {
            line = self.partner_line(&held.transcript)?;
        }
        if line.is_empty() {
            held.stalled = "the partner said nothing".to_string();
            return Ok(held);
        }
        held.transcript.push((CHAT_SPEAKERS[0].to_string(), line.clone()));
        heard.remember(&line, "");
        for index in 0..cfg.turns {
            if self.stopped() {
                held.stalled = "stopped".to_string();
                break;
            }
            let at = held.transcript.len();
            let spoken = nets.with_both(|model, mut negative| {
                let mut veto = |positive: &mut Model, text: &str| -> bool {
                    // a candidate offered again after a shorter context is judged once
                    if let Some((_, refused)) = judged.iter().find(|(t, _)| t == text) {
                        return *refused;
                    }
                    let Some(negative) = negative.as_deref_mut() else {
                        return false;
                    };
                    let mut pair = Filter {
                        positive,
                        negative,
                        config: FilterConfig::default(),
                    };
                    let refused = pair.judge(text).decision == "reject";
                    judged.push((text.to_string(), refused));
                    refused
                };
                let options = ReplyOptions {
                    heard: &heard,
                    index: at,
                    speaker: CHAT_SPEAKERS[1],
                    mode: &cfg.mode,
                    max_length: cfg.max_length,
                    context: cfg.context,
                    temperature: cfg.temperature,
                    k: cfg.k,
                    beam: 0,
                    step_penalty: 0.0,
                    avoid_repeats: cfg.avoid_repeats,
                    avoid_word_repeats: cfg.avoid_word_repeats,
                    explore: cfg.explore,
                    learn: cfg.learn,
                    veto: if guarding {
                        Some(&mut veto as &mut crate::dialogue::Veto)
                    } else {
                        None
                    },
                    trace: None,
                };
                reply(model, None, &line, options, &mut rng)
            })??;
            let Some(turn) = spoken else {
                // the guard can silence it outright, and that is worth saying
                held.stalled = if judged.iter().any(|(_, refused)| *refused) {
                    "the guard vetoed everything it could say"
                } else {
                    "the model had nothing to say"
                }
                .to_string();
                break;
            };
            held.transcript.push((CHAT_SPEAKERS[1].to_string(), turn.text.clone()));
            held.exchanges.push((line.clone(), turn.text.clone()));
            heard.remember(&turn.text, if turn.context.is_empty() { "" } else { &turn.reply });
            let record = Json::obj([
                ("kind", Json::str("exchange")),
                ("conversation", Json::Int(self.number as i64)),
                ("exchange", Json::Int(index as i64 + 1)),
                ("said", Json::str(line.clone())),
                ("reply", Json::str(turn.text.clone())),
                ("context", Json::str(turn.context.clone())),
                ("fresh", Json::Bool(turn.fresh)),
                ("repeat", Json::Bool(turn.repeat)),
                ("stutter", Json::Bool(turn.stutter)),
                ("vetoed", Json::Int(turn.vetoed as i64)),
                ("cost", Json::Num(turn.cost)),
                ("probability", Json::Num(turn.probability)),
                (
                    "rethink",
                    turn.rethink.as_ref().map(|r| r.to_json()).unwrap_or(Json::Null),
                ),
            ]);
            (self.progress)(&record);
            held.turns.push(turn);
            if index + 1 == cfg.turns {
                break; // the last word is the model's: no line after it to reply to
            }
            line = self.partner_line(&held.transcript)?;
            if line.is_empty() {
                held.stalled = "the partner went quiet".to_string();
                break;
            }
            held.transcript.push((CHAT_SPEAKERS[0].to_string(), line.clone()));
            heard.remember(&line, "");
        }
        held.repeats = repeats(&held.turns);
        held.vetoed = if guarding {
            judged.iter().filter(|(_, refused)| *refused).count()
        } else {
            0
        };
        Ok(held)
    }

    /// Holds one conversation, marks it and learns from it; returns its record.
    pub fn run_conversation(&mut self, nets: &mut Networks) -> Result<Json, String> {
        self.number += 1;
        let started = Instant::now();
        let held = self.converse(nets)?;
        let cfg = self.config.clone();
        let review: Option<ReviewSummary> = if held.exchanges.is_empty() {
            None
        } else {
            // the judge's thinking, also with nothing locked
            Some(review_conversation(
                self.judge,
                &held.exchanges,
                &cfg.topic,
                cfg.judge_model(),
                cfg.threshold,
            )?)
        };
        let reviews: &[Review] = review.as_ref().map(|r| r.reviews.as_slice()).unwrap_or(&[]);
        let taught: Option<TeachReport> = if cfg.blame && !reviews.is_empty() {
            let o = TeachOptions {
                epochs: cfg.epochs,
                ..Default::default()
            };
            match nets.with_negative(|negative| {
                teach_reviews(negative, reviews, cfg.threshold, cfg.clear_passes, "chat", &o)
            })? {
                Some(report) => Some(report?),
                None => None,
            }
        } else {
            None
        };
        let learned = if cfg.learn {
            Some(self.learn(nets, review.as_ref(), &held)?)
        } else {
            None
        };
        let judge = match &review {
            Some(r) => Json::str(r.model.clone()),
            None if cfg.judge_model().is_empty() => Json::Null,
            None => Json::str(cfg.judge_model()),
        };
        let overall = review.as_ref().and_then(|r| r.overall.as_ref());
        let taught_doc = taught.as_ref().map(TeachReport::to_json);
        let count = |key: &str| taught_doc.as_ref().map(|d| d.at(key).clone()).unwrap_or(Json::Int(0));
        let mut pairs: Vec<(String, Json)> = vec![
            ("kind".to_string(), Json::str("conversation")),
            ("conversation".to_string(), Json::Int(self.number as i64)),
            ("topic".to_string(), Json::str(cfg.topic.clone())),
            ("judge".to_string(), judge),
            ("threshold".to_string(), Json::Num(cfg.threshold)),
            (
                "transcript".to_string(),
                Json::Arr(
                    held.transcript
                        .iter()
                        .map(|(speaker, text)| {
                            Json::obj([
                                ("speaker", Json::str(speaker.clone())),
                                ("text", Json::str(text.clone())),
                            ])
                        })
                        .collect(),
                ),
            ),
            ("exchanges".to_string(), Json::Int(held.exchanges.len() as i64)),
            (
                "reviews".to_string(),
                Json::Arr(reviews.iter().map(Review::to_json).collect()),
            ),
            (
                "mean_rating".to_string(),
                review
                    .as_ref()
                    .and_then(|r| r.mean_rating)
                    .map(Json::Num)
                    .unwrap_or(Json::Null),
            ),
            (
                "pass_rate".to_string(),
                review
                    .as_ref()
                    .and_then(|r| r.pass_rate)
                    .map(Json::Num)
                    .unwrap_or(Json::Null),
            ),
            (
                "passed".to_string(),
                Json::Int(review.as_ref().map(|r| r.good.len()).unwrap_or(0) as i64),
            ),
            (
                "failed".to_string(),
                Json::Int(review.as_ref().map(|r| r.bad.len()).unwrap_or(0) as i64),
            ),
            (
                "overall_rating".to_string(),
                overall.and_then(|o| o.rating).map(Json::Num).unwrap_or(Json::Null),
            ),
            (
                "overall_critique".to_string(),
                overall.map(|o| Json::str(o.critique.clone())).unwrap_or(Json::Null),
            ),
            ("stalled".to_string(), Json::str(held.stalled.clone())),
            ("vetoed".to_string(), Json::Int(held.vetoed as i64)),
            ("blamed".to_string(), count("blamed")),
            ("cleared".to_string(), count("cleared")),
            ("edges".to_string(), count("edges")),
            (
                "reasons".to_string(),
                taught_doc
                    .as_ref()
                    .map(|d| d.at("reasons").clone())
                    .unwrap_or(Json::Obj(Vec::new())),
            ),
            // replies the model could only repeat, punished with the failures
            ("repeats".to_string(), Json::Int(held.repeats.len() as i64)),
            // the learning keys are always present, so a record has one shape
            ("action".to_string(), Json::Null),
            ("bad".to_string(), Json::Int(0)),
            ("good".to_string(), Json::Int(0)),
            ("neg_loss".to_string(), Json::Null),
            ("pos_loss".to_string(), Json::Null),
            ("seconds".to_string(), Json::Num(started.elapsed().as_secs_f64())),
        ];
        if let Some(Json::Obj(learned)) = learned {
            for (key, value) in learned {
                match pairs.iter_mut().find(|(k, _)| *k == key) {
                    Some(slot) => slot.1 = value,
                    None => pairs.push((key, value)),
                }
            }
        }
        let record = Json::Obj(pairs);
        crate::log_info!(
            LOG,
            "conversation {}: {}/{} replies failed{}",
            self.number,
            record.at("failed").as_i64().unwrap_or(0),
            held.exchanges.len(),
            if held.stalled.is_empty() {
                String::new()
            } else {
                format!(" ({})", held.stalled)
            }
        );
        self.history.push(record.clone());
        Ok(record)
    }

    /// 2NRL on the marked replies: what failed is punished, what passed (and
    /// the partner's own lines) rewarded.  A reply the model could only repeat
    /// is punished whatever the judge made of it: it said nothing new, and
    /// rewarding it would only make the duplicate likelier next time.
    fn learn(&self, nets: &mut Networks, review: Option<&ReviewSummary>, held: &Held) -> Result<Json, String> {
        let cfg = &self.config;
        let blank = |t: &&String| !t.trim().is_empty();
        let mut good: Vec<String> = review
            .map(|r| r.good.iter().filter(blank).cloned().collect())
            .unwrap_or_default();
        let bad: Vec<String> = review
            .map(|r| r.bad.iter().filter(blank).cloned().collect())
            .unwrap_or_default();
        if cfg.teach_partner {
            // the partner's own lines are what a good reply in this conversation would have looked like
            good.extend(
                held.transcript
                    .iter()
                    .filter(|(speaker, text)| speaker == CHAT_SPEAKERS[0] && !text.trim().is_empty())
                    .map(|(_, text)| text.clone()),
            );
        }
        let said_twice: Vec<String> = held.repeats.iter().filter(blank).cloned().collect();
        let good: Vec<String> = unique(good).into_iter().filter(|t| !said_twice.contains(t)).collect();
        let bad: Vec<String> = unique(bad.into_iter().chain(said_twice.iter().cloned()).collect())
            .into_iter()
            .filter(|t| !good.contains(t))
            .collect();
        // every pass through the model's own kind (Python's `two_nrl(...,
        // neg_lr=, pos_lr=, strength=, batch_size=, stop_event=)`): the sine
        // model reads the rates and batch size, the count and phase models the
        // strength
        let o = Feedback {
            neg_epochs: cfg.neg_epochs,
            pos_epochs: cfg.pos_epochs,
            neg_lr: cfg.neg_lr,
            pos_lr: cfg.pos_lr,
            strength: cfg.strength.unwrap_or(1.0),
            batch_size: Some(cfg.batch_size),
            ..Feedback::two_nrl()
        };
        let mut go_on = |_: &crate::model::EpochRecord| !self.stopped();
        let (action, neg_loss, pos_loss) = if bad.is_empty() && good.is_empty() {
            (None, None, None)
        } else if !bad.is_empty() && !good.is_empty() {
            let (negative, positive) = nets.with_model(|m| kinds::two_nrl(m, &bad, &good, &o, &mut go_on))?;
            (
                Some("2nrl"),
                negative.last().map(|r| r.loss),
                positive.last().map(|r| r.loss),
            )
        } else if !good.is_empty() {
            let records = nets.with_model(|m| kinds::reward(m, &good, &o, &mut go_on))?;
            (Some("reward"), None, records.last().map(|r| r.loss))
        } else {
            let records = nets.with_model(|m| kinds::punish(m, &bad, &o, &mut go_on))?;
            (Some("punish"), records.last().map(|r| r.loss), None)
        };
        let num = |x: Option<f64>| x.map(Json::Num).unwrap_or(Json::Null);
        Ok(Json::obj([
            ("action", action.map(Json::str).unwrap_or(Json::Null)),
            ("bad", Json::Int(bad.len() as i64)),
            ("good", Json::Int(good.len() as i64)),
            ("repeats", Json::Int(said_twice.len() as i64)),
            ("neg_loss", num(neg_loss)),
            ("pos_loss", num(pos_loss)),
        ]))
    }

    /// Holds the configured conversations - or, with 0, keeps going until
    /// `stop` says so - asking `stop` before every one, so stopping finishes
    /// the conversation in progress.  The records are the conversations, then
    /// one report card.
    pub fn run(&mut self, nets: &mut Networks) -> Result<Vec<Json>, String> {
        let limit = self.config.conversations;
        let mut records: Vec<Json> = Vec::new();
        while limit == 0 || records.len() < limit {
            if self.stopped() {
                break;
            }
            let record = self.run_conversation(nets)?;
            (self.progress)(&record);
            records.push(record);
        }
        let card = report_card(&records);
        (self.progress)(&card);
        records.push(card);
        Ok(records)
    }
}

/// First-seen order, each text once (Python's `dict.fromkeys`).
fn unique(texts: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for text in texts {
        if !out.contains(&text) {
            out.push(text);
        }
    }
    out
}

/// What a run of conversations came to: how well the model held them up, and
/// what it was taught.  `mean_rating` averages the per-reply marks over the
/// conversations that produced one, `overall_rating` the judge's verdict on
/// each conversation as a whole, and `trend` is the last conversation's mean
/// minus the first's - positive when the model is answering better than it
/// did at the start.
pub fn report_card(records: &[Json]) -> Json {
    let held: Vec<&Json> = records
        .iter()
        .filter(|r| r.at("kind").as_str() == Some("conversation"))
        .collect();
    let numbers = |key: &str| -> Vec<f64> { held.iter().filter_map(|r| r.at(key).as_f64()).collect() };
    let ratings = numbers("mean_rating");
    let rates = numbers("pass_rate");
    let overall = numbers("overall_rating");
    let total = |key: &str| -> i64 { held.iter().map(|r| r.at(key).as_i64().unwrap_or(0)).sum() };
    let mut reasons: Vec<(String, i64)> = Vec::new();
    for record in &held {
        if let Json::Obj(pairs) = record.at("reasons") {
            for (reason, count) in pairs {
                let count = count.as_i64().unwrap_or(0);
                match reasons.iter_mut().find(|(r, _)| r == reason) {
                    Some(slot) => slot.1 += count,
                    None => reasons.push((reason.clone(), count)),
                }
            }
        }
    }
    reasons.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mean = |values: &[f64]| -> Json {
        if values.is_empty() {
            Json::Null
        } else {
            Json::Num(values.iter().sum::<f64>() / values.len() as f64)
        }
    };
    Json::obj([
        ("kind", Json::str("report")),
        ("conversations", Json::Int(held.len() as i64)),
        ("exchanges", Json::Int(total("exchanges"))),
        ("passed", Json::Int(total("passed"))),
        ("failed", Json::Int(total("failed"))),
        ("vetoed", Json::Int(total("vetoed"))),
        (
            "stalled",
            Json::Int(
                held.iter()
                    .filter(|r| r.at("stalled").as_str().is_some_and(|s| !s.is_empty()))
                    .count() as i64,
            ),
        ),
        ("blamed", Json::Int(total("blamed"))),
        ("cleared", Json::Int(total("cleared"))),
        ("edges", Json::Int(total("edges"))),
        ("mean_rating", mean(&ratings)),
        ("pass_rate", mean(&rates)),
        ("overall_rating", mean(&overall)),
        (
            "trend",
            if ratings.len() > 1 {
                Json::Num(ratings[ratings.len() - 1] - ratings[0])
            } else {
                Json::Null
            },
        ),
        (
            "reasons",
            Json::Obj(reasons.into_iter().map(|(r, n)| (r, Json::Int(n))).collect()),
        ),
    ])
}

// -- the command line -----------------------------------------------------------------------------

/// A record as the command reports it while it runs, on stderr.
fn say(record: &Json) {
    let text = |key: &str| record.at(key).as_str().unwrap_or("").to_string();
    match record.at("kind").as_str() {
        Some("exchange") => {
            crate::log_info!(LOG, "{}: {}", CHAT_SPEAKERS[0], text("said"));
            crate::log_info!(LOG, "{}: {}", CHAT_SPEAKERS[1], text("reply"));
            let mut detail = if text("context").is_empty() {
                "    (a fresh line)".to_string()
            } else {
                format!("    picked up {}", python_repr(&text("context")))
            };
            let vetoed = record.at("vetoed").as_i64().unwrap_or(0);
            if vetoed > 0 {
                detail.push_str(&format!("  [{vetoed} vetoed]"));
            }
            crate::log_info!(LOG, "{detail}");
        }
        Some("report") => crate::log_info!(
            LOG,
            "{} conversation(s), {} repl(ies): {} passed, {} failed",
            record.at("conversations").as_i64().unwrap_or(0),
            record.at("exchanges").as_i64().unwrap_or(0),
            record.at("passed").as_i64().unwrap_or(0),
            record.at("failed").as_i64().unwrap_or(0)
        ),
        _ => {}
    }
}

/// The settings of `radixnet chat`, flag for flag Python's; the seed is the
/// global `--seed` (0 when it is not given, as Python's `effective_seed`).
fn config_from_args(ctx: &Ctx) -> Result<ChatConfig, String> {
    let args = &ctx.args;
    let d = ChatConfig::default();
    let count = |name: &str, default: usize, minimum: i64| crate::llm::count_flag(ctx, name, default as i64, minimum);
    let number = |name: &str, default: f64| crate::llm::nonneg_flag(ctx, name, default);
    let mode = args.str("mode", &d.mode);
    if mode != "beam" && mode != "sample" {
        return Err(format!("--mode must be beam or sample, got {mode:?}"));
    }
    let partner_model = args
        .get("partner-model")
        .filter(|m| !m.is_empty())
        .or_else(|| args.get("ollama-model"))
        .unwrap_or("")
        .to_string();
    let config = ChatConfig {
        conversations: count("conversations", d.conversations, 0)?,
        turns: count("turns", d.turns, 1)?,
        topic: args.str("topic", ""),
        opening: args.str("opening", ""),
        persona: args.str("persona", ""),
        context: count("context", d.context, 0)?,
        max_length: count("max-length", d.max_length, 1)?,
        mode,
        k: count("k", d.k, 1)?,
        temperature: number("temperature", d.temperature)?,
        partner_temperature: number("partner-temperature", d.partner_temperature)?,
        threshold: number("threshold", d.threshold)?,
        provider: normalise_provider(&args.str("provider", DEFAULT_PROVIDER))?.to_string(),
        partner_model,
        judge_model: args.str("judge-model", ""),
        guard: !args.on("no-guard"),
        blame: !args.on("no-blame"),
        clear_passes: !args.on("no-clear"),
        learn: !args.on("no-learn"),
        teach_partner: !args.on("no-teach-partner"),
        avoid_repeats: !args.on("allow-repeats"),
        avoid_word_repeats: !args.on("allow-word-repeats"),
        explore: count("explore", d.explore, 0)?,
        neg_epochs: count("neg-epochs", d.neg_epochs, 0)?,
        pos_epochs: count("pos-epochs", d.pos_epochs, 0)?,
        neg_lr: number("neg-lr", d.neg_lr)?,
        pos_lr: number("pos-lr", d.pos_lr)?,
        batch_size: count("batch-size", d.batch_size, 1)?,
        strength: match args.get("strength") {
            Some(_) => Some(number("strength", 1.0)?),
            None => None,
        },
        epochs: count("epochs", d.epochs, 0)?,
        seed: Some(ctx.seed),
    };
    config.validate()?;
    Ok(config)
}

/// `radixnet chat`: an LLM converses with the model and marks every reply.
/// What failed blames the negative network, what passed clears it, and 2NRL
/// trains the model on both, with the partner's own lines joining the
/// positive phase.  `--no-learn` marks without training; `--no-guard`,
/// `--no-blame` and `--no-clear` keep the negative network out of it.
pub fn cli(ctx: &Ctx) -> Result<(), String> {
    let args = &ctx.args;
    if normalise_provider(&args.str("provider", DEFAULT_PROVIDER))? == CHATGPT && !crate::chatgpt::key_configured() {
        return Err(
            "no OpenAI API key: set OPENAI_API_KEY (or OPENAI_API_KEY_FILE) to let ChatGPT converse, or use \
                    --provider ollama"
                .to_string(),
        );
    }
    let config = config_from_args(ctx)?;
    let timeout = crate::llm::seconds(crate::llm::nonneg_flag(ctx, "timeout", 0.0)?);
    let url = args.str("url", "");
    let client = new_client(&config.provider, &url, &config.partner_model, timeout)?;
    let judge_url = args.str("judge-url", "");
    let judge: Option<Box<dyn LlmClient>> = if !config.judge_model.is_empty() || !judge_url.is_empty() {
        let at = if judge_url.is_empty() { &url } else { &judge_url };
        Some(new_client(&config.provider, at, &config.judge_model, timeout)?)
    } else {
        None
    };
    let judge: &dyn LlmClient = judge.as_deref().unwrap_or(client.as_ref());
    let mut model = ctx.open(true)?;
    let mut negative = if config.wants_negative() {
        Some(ctx.open_negative(false)?)
    } else {
        None
    };
    let out = ctx.out.clone().unwrap_or_else(|| ctx.model_path.clone());
    crate::log_info!(
        LOG,
        "partner {}: {} at {}; judge {} at {}; {} conversation(s) x {} repl(ies), pass at {}/10",
        config.provider,
        client.model(),
        client.url(),
        judge.model(),
        judge.url(),
        if config.conversations == 0 {
            "until stopped".to_string()
        } else {
            config.conversations.to_string()
        },
        config.turns,
        crate::llm::format_g(config.threshold)
    );
    let records = {
        let mut chat = Chat::new(client.as_ref(), judge, config.clone())?.with_progress(say);
        let mut nets = Networks::Owned {
            model: &mut model,
            negative: negative.as_mut(),
        };
        chat.run(&mut nets)?
    };
    let card = records
        .last()
        .filter(|r| r.at("kind").as_str() == Some("report"))
        .cloned()
        .unwrap_or(Json::Obj(Vec::new()));
    let saved = if config.learn {
        model.save(&out)?;
        crate::log_info!(LOG, "saved {out}");
        Json::str(out)
    } else {
        crate::log_info!(LOG, "nothing was learned (--no-learn): the model is untouched");
        Json::Null
    };
    let mut doc = vec![
        ("config".to_string(), config.to_json()),
        ("url".to_string(), Json::str(client.url())),
        ("partner".to_string(), Json::str(client.model())),
        ("judge".to_string(), Json::str(judge.model())),
        ("records".to_string(), Json::Arr(records)),
        ("report".to_string(), card),
        ("interrupted".to_string(), Json::Bool(false)),
        ("saved".to_string(), saved),
        ("speakers".to_string(), Json::strs(CHAT_SPEAKERS)),
    ];
    if let Some(negative) = negative.as_mut() {
        let path = ctx.negative_path();
        negative.save(&path)?;
        doc.push((
            "negative".to_string(),
            Json::obj([
                ("path", Json::str(path.clone())),
                ("saved", crate::llm::saved(&path)),
                ("reasons", crate::review::reason_rows(negative)),
                ("stats", negative_stats(negative)),
            ]),
        ));
    }
    ctx.emit(Json::Obj(doc));
    Ok(())
}

// -- the server -----------------------------------------------------------------------------------

/// What the server keeps for this area: the exchange, conversation and report
/// records of every chat run, oldest first.
#[derive(Default)]
pub struct State {
    history: Mutex<Vec<Json>>,
}

impl State {
    /// Reads the `serve` command's flags for this area (it has none).
    pub fn configure(&mut self, _args: &crate::cli::Args) -> Result<(), String> {
        Ok(())
    }

    /// Every record so far.
    pub fn history(&self) -> Vec<Json> {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).clone()
    }

    /// The report at the end of the last run, `None` before one has finished.
    pub fn card(&self) -> Option<Json> {
        let history = self.history.lock().unwrap_or_else(|e| e.into_inner());
        history
            .iter()
            .rev()
            .find(|r| r.at("kind").as_str() == Some("report"))
            .cloned()
    }

    fn push(&self, record: Json) {
        self.history.lock().unwrap_or_else(|e| e.into_inner()).push(record);
    }
}

/// The conversation settings of a request body, defaults from [`ChatConfig`],
/// read in Python's order so the first bad field is the one refused.
fn config_from_fields(f: &Fields) -> Result<ChatConfig, ApiError> {
    let d = ChatConfig::default();
    let count = |name: &str, default: usize, minimum: usize| f.count_or(name, default, minimum);
    let number = |name: &str, default: f64| f.number_or(name, default, Some(0.0));
    let conversations = count("conversations", d.conversations, 0)?;
    let turns = count("turns", d.turns, 1)?;
    let topic = f.text_or("topic", &d.topic)?;
    let opening = f.text_or("opening", &d.opening)?;
    let persona = f.text_or("persona", &d.persona)?;
    let context = count("context", d.context, 0)?;
    let max_length = count("max_length", d.max_length, 1)?;
    let mode = f.text_or("mode", &d.mode)?.trim().to_lowercase();
    let k = count("k", d.k, 1)?;
    let temperature = number("temperature", d.temperature)?;
    let partner_temperature = number("partner_temperature", d.partner_temperature)?;
    let threshold = number("threshold", d.threshold)?;
    let mut raw = f.text("provider")?.filter(|p| !p.is_empty());
    if raw.is_none() {
        raw = f.text("partner_provider")?.filter(|p| !p.is_empty());
    }
    let provider = match raw {
        Some(raw) => normalise_provider(&raw)
            .map_err(|_| {
                ApiError::bad_request(format!(
                    "'provider' must be one of {} (got {})",
                    PROVIDERS.join(", "),
                    python_repr(&raw)
                ))
            })?
            .to_string(),
        None => d.provider.clone(),
    };
    let partner_model = match f.text("partner_model")?.filter(|m| !m.is_empty()) {
        Some(m) => m,
        None => f.text("model")?.unwrap_or_default(),
    };
    let config = ChatConfig {
        conversations,
        turns,
        topic,
        opening,
        persona,
        context,
        max_length,
        mode,
        k,
        temperature,
        partner_temperature,
        threshold,
        provider,
        partner_model,
        judge_model: f.text("judge_model")?.unwrap_or_default(),
        guard: f.flag("guard", d.guard)?,
        blame: f.flag("blame", d.blame)?,
        clear_passes: f.flag("clear_passes", d.clear_passes)?,
        learn: f.flag("learn", d.learn)?,
        teach_partner: f.flag("teach_partner", d.teach_partner)?,
        avoid_repeats: f.flag("avoid_repeats", d.avoid_repeats)?,
        avoid_word_repeats: f.flag("avoid_word_repeats", d.avoid_word_repeats)?,
        explore: count("explore", d.explore, 0)?,
        neg_epochs: count("neg_epochs", d.neg_epochs, 0)?,
        pos_epochs: count("pos_epochs", d.pos_epochs, 0)?,
        neg_lr: number("neg_lr", d.neg_lr)?,
        pos_lr: number("pos_lr", d.pos_lr)?,
        batch_size: count("batch_size", d.batch_size, 1)?,
        strength: f.number("strength", Some(0.0))?,
        epochs: count("epochs", d.epochs, 0)?,
        seed: f.integer("seed", None)?,
    };
    config.validate().map_err(ApiError::bad_request)?;
    Ok(config)
}

/// This area's routes:
/// `POST /api/chat/start`
/// `GET /api/chat/history`
pub fn routes(server: &mut Server<Service>) {
    server.route("POST", "/api/chat/start", start);
    server.route("GET", "/api/chat/history", history);
}

/// Starts a `chat` job: the partner talks to the model, the judge marks every
/// reply, and both networks learn.  Answers 202 with the job, the settings,
/// the partner's endpoint and model, the judge's model and the speakers; the
/// records follow on `/api/job` and `/api/chat/history`.
fn start(svc: &Arc<Service>, r: &Request) -> Answer {
    let f = Fields::of(r);
    let config = config_from_fields(&f)?;
    let timeout = f.number("timeout", Some(1.0))?.and_then(crate::llm::seconds);
    let partner_model = Some(config.partner_model.as_str()).filter(|m| !m.is_empty());
    let client = svc
        .llm
        .client(&config.provider, f.text("url")?.as_deref(), partner_model, timeout)?;
    let judge_url = f.text("judge_url")?.filter(|u| !u.is_empty());
    let judge: Option<Box<dyn LlmClient>> = if !config.judge_model.is_empty() || judge_url.is_some() {
        let model = Some(config.judge_model.as_str()).filter(|m| !m.is_empty());
        Some(svc.llm.client(&config.provider, judge_url.as_deref(), model, timeout)?)
    } else {
        None
    };
    svc.ensure_idle()?;
    let negative = config.wants_negative();
    if negative {
        // loaded now, so a file that is not a negative network is this request's 400
        svc.ensure_negative()?;
    }
    let (url, partner) = (client.url().to_string(), client.model().to_string());
    let judge_name = judge.as_deref().unwrap_or(client.as_ref()).model().to_string();
    svc.start_job("chat");
    let worker = Arc::clone(svc);
    let settings = config.clone();
    std::thread::spawn(move || {
        let marker: &dyn LlmClient = judge.as_deref().unwrap_or(client.as_ref());
        let outcome = Chat::new(client.as_ref(), marker, settings).and_then(|chat| {
            let mut chat = chat.with_stop(|| worker.stopping()).with_progress(|record| {
                worker.chat.push(record.clone());
                worker.job_progress(record.clone());
            });
            chat.run(&mut Networks::Shared { svc: &worker, negative })
        });
        worker.finish_job(outcome.map(|_| worker.job_records()));
    });
    Ok(accepted(Json::obj([
        ("job", svc.job_json()),
        ("config", config.to_json()),
        ("url", Json::str(url)),
        ("partner", Json::str(partner)),
        ("judge", Json::str(judge_name)),
        ("speakers", Json::strs(CHAT_SPEAKERS)),
    ])))
}

/// `GET /api/chat/history`: the records of every chat run.
fn history(svc: &Arc<Service>, _r: &Request) -> Answer {
    Ok(Json::obj([("history", Json::Arr(svc.chat.history()))]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{LlmError, LlmOptions};
    use crate::negative::NegativeOptions;
    use crate::TrainOptions;

    /// A partner that says the corpus back line by line and a judge that
    /// passes a reply holding "cat", as `tests/test_chat.py`'s fake does.
    struct Partner {
        lines: Mutex<usize>,
        asked: Mutex<Vec<(String, LlmOptions)>>,
        quiet: bool,
    }

    impl Partner {
        fn new() -> Partner {
            Partner {
                lines: Mutex::new(0),
                asked: Mutex::new(Vec::new()),
                quiet: false,
            }
        }
    }

    const SAID: [&str; 3] = [
        "the cat sat on the mat",
        "the dog ran in the park",
        "the cat likes the dog",
    ];

    impl LlmClient for Partner {
        fn provider(&self) -> &'static str {
            "ollama"
        }
        fn url(&self) -> &str {
            "http://partner"
        }
        fn model(&self) -> &str {
            "partner"
        }
        fn models(&self) -> Result<Vec<Json>, LlmError> {
            Ok(Vec::new())
        }
        fn generate(&self, prompt: &str, o: &LlmOptions) -> Result<String, LlmError> {
            self.asked.lock().unwrap().push((prompt.to_string(), o.clone()));
            if o.system.contains("marking a conversation") {
                let reviews: Vec<String> = prompt
                    .split("\n\n")
                    .filter_map(|block| {
                        let rest = block.trim_start().strip_prefix('[')?;
                        let (index, rest) = rest.split_once(']')?;
                        let reply = rest.split("Model: ").nth(1).unwrap_or("");
                        let rating = if reply.contains("cat") { 8 } else { 2 };
                        Some(format!(
                            "{{\"index\": {index}, \"rating\": {rating}, \"critique\": \"{}\"}}",
                            if rating > 5 {
                                "follows on"
                            } else {
                                "it repeats the same word over and over"
                            }
                        ))
                    })
                    .collect();
                return Ok(format!(
                    "{{\"reviews\": [{}], \"overall\": {{\"rating\": 5, \"critique\": \"so-so\"}}}}",
                    reviews.join(", ")
                ));
            }
            if self.quiet {
                return Ok(String::new());
            }
            let mut n = self.lines.lock().unwrap();
            let line = SAID[*n % SAID.len()];
            *n += 1;
            Ok(format!("Partner: {line}"))
        }
        fn chat(&self, _m: &[Json], _o: &LlmOptions) -> Result<String, LlmError> {
            Ok(String::new())
        }
    }

    fn trained() -> Model {
        let corpus: Vec<String> = SAID.iter().map(|s| s.to_string()).collect();
        let mut model = Model::new(3, crate::GraphOptions::default()).unwrap();
        model
            .train(
                &corpus,
                &TrainOptions {
                    epochs: 4,
                    ..Default::default()
                },
            )
            .unwrap();
        model
    }

    fn config() -> ChatConfig {
        ChatConfig {
            conversations: 2,
            turns: 2,
            topic: "animals".to_string(),
            neg_epochs: 1,
            pos_epochs: 1,
            ..Default::default()
        }
    }

    #[test]
    fn the_defaults_are_python_s_and_nonsense_is_refused() {
        let d = ChatConfig::default();
        assert!(d.validate().is_ok());
        let doc = d.to_json();
        assert_eq!(doc.at("partner_temperature").as_f64(), Some(0.8));
        assert!(doc.at("seed").is_null() && doc.at("strength").is_null());
        for (bad, words) in [
            (
                ChatConfig {
                    turns: 0,
                    ..Default::default()
                },
                "turns must be >= 1",
            ),
            (
                ChatConfig {
                    mode: "dijkstra".into(),
                    ..Default::default()
                },
                "mode must be 'beam' or 'sample'",
            ),
            (
                ChatConfig {
                    threshold: 11.0,
                    ..Default::default()
                },
                "threshold",
            ),
            (
                ChatConfig {
                    provider: "nobody".into(),
                    ..Default::default()
                },
                "provider must be one of",
            ),
        ] {
            assert!(bad.validate().unwrap_err().contains(words), "{words}");
        }
    }

    #[test]
    fn a_conversation_is_held_marked_and_learned_from() {
        let partner = Partner::new();
        let mut model = trained();
        let mut negative = Model::new_negative(3, &NegativeOptions::default()).unwrap();
        let kinds = std::cell::RefCell::new(Vec::new());
        let mut chat = Chat::new(&partner, &partner, config())
            .unwrap()
            .with_progress(|r| kinds.borrow_mut().push(r.at("kind").as_str().unwrap_or("").to_string()));
        let records = chat
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: Some(&mut negative),
            })
            .unwrap();
        assert_eq!(records.len(), 3);
        let first = &records[0];
        assert_eq!(first.at("kind").as_str(), Some("conversation"));
        assert_eq!(first.at("exchanges").as_i64(), Some(2));
        let transcript = first.at("transcript").as_array();
        assert_eq!(transcript.len(), 4, "partner, model, partner, model");
        assert_eq!(transcript[0].at("text").as_str(), Some("the cat sat on the mat"));
        assert_eq!(first.at("judge").as_str(), Some("partner"));
        assert_eq!(first.at("overall_critique").as_str(), Some("so-so"));
        let keys: Vec<&str> = match first {
            Json::Obj(pairs) => pairs.iter().map(|(k, _)| k.as_str()).collect(),
            _ => Vec::new(),
        };
        assert_eq!(
            keys[20..],
            ["repeats", "action", "bad", "good", "neg_loss", "pos_loss", "seconds"]
        );
        assert!(first.at("action").as_str().is_some(), "it learned");
        let prompts = partner.asked.lock().unwrap();
        assert_eq!(
            prompts[0].0, "Open the conversation about animals with one short, plain line.",
            "the partner opens"
        );
        assert!(prompts[1]
            .0
            .starts_with("The conversation so far:\nPartner: the cat sat on the mat\nModel: "));
        drop(prompts);
        assert_eq!(
            *kinds.borrow(),
            vec![
                "exchange",
                "exchange",
                "conversation",
                "exchange",
                "exchange",
                "conversation",
                "report"
            ]
        );
        let card = &records[2];
        assert_eq!(card.at("conversations").as_i64(), Some(2));
        assert_eq!(card.at("exchanges").as_i64(), Some(4));
        if card.at("failed").as_i64().unwrap() > 0 {
            assert!(negative.neg.as_ref().unwrap().blame_total > 0.0);
            assert!(negative.neg.as_ref().unwrap().log.iter().all(|e| e.source == "chat"));
        }
    }

    #[test]
    fn a_given_opening_is_spoken_and_no_learn_changes_nothing() {
        let partner = Partner::new();
        let mut model = trained();
        let before = crate::report::stats(&model).render(0);
        let mut chat = Chat::new(
            &partner,
            &partner,
            ChatConfig {
                conversations: 1,
                turns: 1,
                opening: "  tell me about the cat ".into(),
                learn: false,
                blame: false,
                ..config()
            },
        )
        .unwrap();
        let records = chat
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        let held = &records[0];
        assert_eq!(
            held.at("transcript").as_array()[0].at("text").as_str(),
            Some("tell me about the cat")
        );
        assert!(held.at("action").is_null());
        assert_eq!(crate::report::stats(&model).render(0), before);
        assert_eq!(partner.asked.lock().unwrap().len(), 1, "only the judge was asked");
    }

    #[test]
    fn a_quiet_partner_ends_the_conversation() {
        let mut partner = Partner::new();
        partner.quiet = true;
        let mut model = trained();
        let mut chat = Chat::new(
            &partner,
            &partner,
            ChatConfig {
                conversations: 1,
                ..config()
            },
        )
        .unwrap();
        let records = chat
            .run(&mut Networks::Owned {
                model: &mut model,
                negative: None,
            })
            .unwrap();
        assert_eq!(records[0].at("stalled").as_str(), Some("the partner said nothing"));
        assert!(records[0].at("judge").is_null());
        assert_eq!(records[1].at("stalled").as_i64(), Some(1));
    }

    #[test]
    fn the_report_card_adds_up() {
        let conversation = |mean: f64, reasons: &str, stalled: &str| {
            crate::json::parse(&format!(
                "{{\"kind\": \"conversation\", \"exchanges\": 2, \"passed\": 1, \"failed\": 1, \"vetoed\": 1, \
                 \"blamed\": 1, \"cleared\": 0, \"edges\": 3, \"mean_rating\": {mean}, \"pass_rate\": 0.5, \
                 \"overall_rating\": null, \"stalled\": \"{stalled}\", \"reasons\": {reasons}}}"
            ))
            .unwrap()
        };
        let records = vec![
            conversation(4.0, "{\"b\": 1, \"a\": 1}", ""),
            conversation(7.0, "{\"a\": 2}", "stopped"),
        ];
        let card = report_card(&records);
        assert_eq!(card.at("conversations").as_i64(), Some(2));
        assert_eq!(card.at("stalled").as_i64(), Some(1));
        assert_eq!(card.at("mean_rating").as_f64(), Some(5.5));
        assert_eq!(card.at("trend").as_f64(), Some(3.0));
        assert!(card.at("overall_rating").is_null());
        assert_eq!(card.at("reasons").render(0), "{\"a\":3,\"b\":1}");
    }

    #[test]
    fn a_request_body_is_read_as_python_reads_it() {
        let body = crate::json::parse(
            "{\"turns\": 2, \"mode\": \" SAMPLE \", \"partner_provider\": \"gpt\", \"model\": \"m\", \"seed\": 4}",
        )
        .unwrap();
        let config = config_from_fields(&Fields::new(&body)).unwrap();
        assert_eq!(
            (
                config.mode.as_str(),
                config.provider.as_str(),
                config.partner_model.as_str(),
                config.seed
            ),
            ("sample", "chatgpt", "m", Some(4))
        );
        for (bad, words) in [
            ("{\"turns\": 0}", "'turns' must be >= 1 (got 0)"),
            (
                "{\"provider\": \"nobody\"}",
                "'provider' must be one of ollama, chatgpt (got 'nobody')",
            ),
            ("{\"mode\": \"x\"}", "mode must be 'beam' or 'sample'"),
        ] {
            let body = crate::json::parse(bad).unwrap();
            let err = config_from_fields(&Fields::new(&body)).unwrap_err();
            assert_eq!((err.status, err.message.as_str()), (400, words));
        }
    }
}
