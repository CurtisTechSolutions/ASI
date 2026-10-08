//! Talking with the model by voice: every utterance is heard, learned, answered
//! and spoken (D-089; Python's `modelkit/voicechat.py`).
//!
//! The Voice tab listens all the time.  Each thing the person says reaches
//! here as one **turn**: the recording, and what the browser heard in it.  A
//! turn is
//!
//! 1. **heard** - [`crate::speech::teach`] makes the utterance into the texts
//!    the model learns, the transcript and the waveform behind one unique
//!    token (D-035); a model of acoustic units hears the recording as its
//!    units instead (D-082), the one text made of sound it can learn;
//! 2. **trained** - the model is trained on those texts at once, so the reply
//!    already knows them;
//! 3. **answered** - the model replies as it does in every conversation here
//!    ([`crate::dialogue::reply`]: the end of the line picked up and
//!    continued), or Ollama answers for it (`answer = "ollama"`), or Ollama
//!    steps in only when the model has nothing to say to the person
//!    (`"auto"`: no reply, or a fresh text that picked up none of the line);
//! 4. **spoken** - the reply goes through the output decoder (D-088) and its
//!    audio comes back in chunks the browser plays as they arrive;
//! 5. **taught** - a reply Ollama wrote is taught to the model, so that next
//!    time it may answer by itself.
//!
//! [`turn`] does all of it and reports every step to `write` as it happens
//! (the JSON Lines of `POST /api/voice/turn/stream`).  The model's own parts -
//! training and replying - come in as closures, so the service takes its
//! locks around exactly those and Ollama is asked outside them.

use std::sync::Arc;
use std::time::Instant;

use crate::cli::Ctx;
use crate::dialogue::{reply, Heard, ReplyOptions, Turn, Veto, EXPLORE};
use crate::duo::Filter;
use crate::http::{Answer, ApiError, Request, Server, Sink};
use crate::llm::LlmClient;
use crate::multipart::base64::{self, is_python_space};
use crate::multipart::Form;
use crate::ollama::OllamaClient;
use crate::review::reply_line;
use crate::service::Service;
use crate::speech::{self, asr, extend, py_round, teach, AsrReport, TeachOptions, DEFAULT_RATE};
use crate::thinking::THINK_DEPTH;
use crate::voice::{decoder_name, say, spelled, SayOptions};
use radixnet::encoding::{Encoding, Unit};
use radixnet::json::{py_repr, Json};
use radixnet::model::Model;
use radixnet::mt19937::Mt19937;
use radixnet::negative::python_repr;

const LOG: &str = "voice";

/// Who speaks: the person, and the model (or Ollama on its behalf).
pub const SPEAKERS: (&str, &str) = ("You", "Model");

/// Who answers: the model with Ollama stepping in when it has nothing to say,
/// the model alone, Ollama alone, nobody.
pub const ANSWERS: [&str; 4] = ["auto", "model", "ollama", "none"];

/// Samples per `audio` event: half a second at 16 kHz, so playback starts long
/// before the reply has ended.
pub const CHUNK_SAMPLES: usize = 8000;

/// Trains the model on texts and says what it did (`{"texts", "epochs", ...}`).
pub type Learn<'a> = dyn FnMut(&[String]) -> Result<Json, String> + 'a;

/// The model's own answer to a line, given what the conversation has heard.
pub type ModelReply<'a> = dyn FnMut(&str, &Heard) -> Result<Option<Turn>, String> + 'a;

/// Where the events of a turn go, one at a time, as they happen.
pub type Write<'a> = dyn FnMut(Json) + 'a;

/// One turn's settings: how it is heard, learned, answered and spoken
/// (Python's `VoiceOptions`).  The counts are kept as they were given, so that
/// [`VoiceOptions::validate`] refuses them with Python's words.
#[derive(Clone, Debug)]
pub struct VoiceOptions {
    /// How the utterance is heard: the transcript (it wins over the server's
    /// own transcription), the transcription backend, the waveform text's
    /// sample rate and codec, whether the sound is learned, the token.
    pub teach: TeachOptions,
    pub train: bool,
    pub epochs: i64,
    pub lr: f64,
    pub batch_size: i64,
    pub answer: String,
    /// Teach the model a reply Ollama wrote for it.
    pub learn_reply: bool,
    pub persona: String,
    pub topic: String,
    pub mode: String,
    pub max_length: i64,
    pub context: i64,
    pub k: i64,
    pub beam: Option<i64>,
    pub step_penalty: f64,
    pub temperature: f64,
    pub seed: Option<i64>,
    pub avoid_repeats: bool,
    pub avoid_word_repeats: bool,
    pub explore: i64,
    /// What a rethink finds out is taught to the graph, as in every conversation here.
    pub learn: bool,
    pub ollama_temperature: f64,
    pub speak: bool,
    pub voice_rate: i64,
    pub pitch: f64,
    pub tempo: f64,
    pub gain: f64,
    pub polish: i64,
    /// How acoustic units are spoken ([`crate::voice::VOCODERS`]).
    pub vocoder: String,
    pub chunk: i64,
}

impl Default for VoiceOptions {
    /// The Python defaults.
    fn default() -> VoiceOptions {
        VoiceOptions {
            teach: TeachOptions::default(),
            train: true,
            epochs: 2,
            lr: 0.5,
            batch_size: 8,
            answer: "auto".to_string(),
            learn_reply: true,
            persona: String::new(),
            topic: String::new(),
            mode: "beam".to_string(),
            max_length: 60,
            context: 12,
            k: 5,
            beam: None,
            step_penalty: 0.0,
            temperature: 1.0,
            seed: None,
            avoid_repeats: true,
            avoid_word_repeats: true,
            explore: EXPLORE as i64,
            learn: true,
            ollama_temperature: 0.8,
            speak: true,
            voice_rate: phonetok::synth::RATE as i64,
            pitch: 120.0,
            tempo: 1.0,
            gain: 0.5,
            polish: 0,
            vocoder: "auto".to_string(),
            chunk: CHUNK_SAMPLES as i64,
        }
    }
}

impl VoiceOptions {
    /// Python's `VoiceOptions.validate`, word for word.
    pub fn validate(&self) -> Result<(), String> {
        if !ANSWERS.contains(&self.answer.as_str()) {
            return Err(format!(
                "'answer' must be one of {}, got {}",
                ANSWERS.join(", "),
                python_repr(&self.answer)
            ));
        }
        if self.mode != "beam" && self.mode != "sample" {
            return Err(format!(
                "'mode' must be beam or sample, got {}",
                python_repr(&self.mode)
            ));
        }
        if self.epochs < 0 || self.batch_size < 1 || self.lr < 0.0 {
            return Err("'epochs' and 'lr' must be at least 0 and 'batch_size' at least 1".to_string());
        }
        if self.pitch <= 0.0 || self.tempo <= 0.0 || self.gain < 0.0 || self.voice_rate < 1 {
            return Err(
                "'pitch' and 'tempo' must be above 0, 'gain' at least 0 and 'voice_rate' at least 1".to_string(),
            );
        }
        if self.chunk < 1 || self.polish < 0 || self.max_length < 0 || self.context < 0 || self.k < 1 {
            return Err(
                "'chunk' and 'k' must be at least 1; 'polish', 'max_length' and 'context' at least 0".to_string(),
            );
        }
        crate::voice::vocoder_choice(&self.vocoder)?;
        Ok(())
    }

    /// The voice the reply is spoken with.
    pub fn say_options(&self) -> SayOptions {
        SayOptions {
            rate: self.voice_rate.max(1) as u32,
            pitch: self.pitch,
            tempo: self.tempo,
            gain: self.gain,
            polish: self.polish.max(0) as usize,
            vocoder: self.vocoder.clone(),
        }
    }
}

/// What one turn came to: the document the API answers with, and the reply's audio.
#[derive(Clone, Debug)]
pub struct Outcome {
    pub document: Json,
    /// The reply as 16-bit PCM (empty when nothing was spoken).
    pub pcm: Vec<u8>,
    pub rate: u32,
}

impl Outcome {
    /// The reply as a WAV.
    pub fn wav(&self) -> Vec<u8> {
        phonetok::synth::wav_bytes(&self.pcm, self.rate)
    }
}

/// `" ".join(text.split())`.
fn squash(text: &str) -> String {
    text.split(is_python_space)
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

/// A conversation so far as `(speaker, text)` pairs: blank texts dropped, the
/// rest squashed, a nameless speaker the person (the tail of Python's
/// `history_pairs`, once it has taken the pairs apart).
fn tidy_pairs<I: IntoIterator<Item = (String, String)>>(pairs: I) -> Vec<(String, String)> {
    pairs
        .into_iter()
        .filter_map(|(speaker, text)| {
            let text = squash(&text);
            if text.is_empty() {
                return None;
            }
            let speaker = speaker.trim_matches(is_python_space);
            let speaker = if speaker.is_empty() { SPEAKERS.0 } else { speaker };
            Some((speaker.to_string(), text))
        })
        .collect()
}

/// A conversation so far as `(speaker, text)` pairs, from `[speaker, text]`
/// pairs or `{"speaker", "text"}` records (Python's `history_pairs`).
pub fn history_pairs(history: &[Json]) -> Result<Vec<(String, String)>, String> {
    let mut pairs = Vec::with_capacity(history.len());
    for (i, item) in history.iter().enumerate() {
        let (speaker, text) = match item {
            Json::Obj(_) => (
                item.get("speaker").cloned().unwrap_or_else(|| Json::str("")),
                item.get("text").cloned().unwrap_or_else(|| Json::str("")),
            ),
            Json::Arr(items) if items.len() == 2 => (items[0].clone(), items[1].clone()),
            _ => return Err(format!("history[{i}] must be [speaker, text] or {{speaker, text}}")),
        };
        match (speaker, text) {
            (Json::Str(speaker), Json::Str(text)) => pairs.push((speaker, text)),
            _ => return Err(format!("history[{i}]: the speaker and the text must be strings")),
        }
    }
    Ok(tidy_pairs(pairs))
}

/// What an utterance was heard as: the transcript, the texts to learn, the
/// token and the audio (Python's `hear`).
#[derive(Clone, Debug)]
pub struct Hearing {
    pub transcript: String,
    /// The token the texts share; a model of acoustic units has none.
    pub token: Option<String>,
    pub texts: Vec<String>,
    /// The waveform text's record (`speech.teach`'s `audio`), or null.
    pub audio: Json,
    /// A model of acoustic units' units (`""` when nothing was heard), `None`
    /// for every other model.
    pub units: Option<String>,
    pub asr: Json,
}

impl Hearing {
    /// The record's fields, in the order the `heard` event lists them.
    fn pairs(&self) -> Vec<(&'static str, Json)> {
        let opt = |v: &Option<String>| v.clone().map(Json::Str).unwrap_or(Json::Null);
        vec![
            ("transcript", Json::str(&self.transcript)),
            ("token", opt(&self.token)),
            ("texts", Json::strs(self.texts.clone())),
            ("audio", self.audio.clone()),
            ("units", opt(&self.units)),
            ("asr", self.asr.clone()),
        ]
    }
}

/// What an utterance was heard as.
///
/// A model of acoustic units learns the recording as its units
/// ([`radixnet::phonetic::hear_audio`]) - the transcript is kept for the
/// conversation, but nothing made of words is in its alphabet.
pub fn hear(data: Option<&[u8]>, encoding: Encoding, o: &VoiceOptions) -> Result<Hearing, String> {
    if encoding.unit == Unit::Acoustic {
        let words = squash(&o.teach.transcript);
        let units = match data {
            Some(data) if !data.is_empty() => radixnet::phonetic::hear_audio(data)?,
            _ => String::new(),
        };
        let asr = AsrReport {
            backend: if words.is_empty() {
                None
            } else {
                Some("given".to_string())
            },
            model: None,
            language: o.teach.asr.language.clone(),
            seconds: 0.0,
            error: None,
        };
        return Ok(Hearing {
            transcript: words,
            token: None,
            texts: if units.is_empty() {
                Vec::new()
            } else {
                vec![units.clone()]
            },
            audio: Json::Null,
            units: Some(units),
            asr: asr.to_json(),
        });
    }
    let taught = teach(data.unwrap_or(&[]), &o.teach)?;
    Ok(Hearing {
        transcript: taught.transcript.clone(),
        token: Some(taught.token.clone()),
        texts: taught.texts.clone(),
        audio: taught.audio.as_ref().map(|a| a.to_json()).unwrap_or(Json::Null),
        units: None,
        asr: taught.asr.to_json(),
    })
}

/// The encoding a reply is spoken in: the model's own, unless it is words a
/// model of acoustic units cannot say.
pub fn speech_encoding(encoding: Encoding, by: &str) -> Encoding {
    if encoding.unit == Unit::Acoustic && by != "model" {
        Encoding::default()
    } else {
        encoding
    }
}

/// The reply's audio as `audio` events of `chunk` samples: 16-bit PCM,
/// base64, played as they arrive.
pub fn audio_events(pcm: &[u8], rate: u32, chunk: usize) -> Vec<Json> {
    let step = chunk.max(1) * 2;
    pcm.chunks(step)
        .map(|piece| {
            Json::obj([
                ("event", Json::str("audio")),
                ("rate", Json::Int(rate as i64)),
                ("samples", Json::Int((piece.len() / 2) as i64)),
                ("pcm_base64", Json::str(base64::encode(piece))),
            ])
        })
        .collect()
}

/// `{"event": name, **doc}`.
fn event(name: &str, doc: &Json) -> Json {
    let mut pairs = vec![("event".to_string(), Json::str(name))];
    if let Json::Obj(fields) = doc {
        pairs.extend(fields.iter().cloned());
    }
    Json::Obj(pairs)
}

/// `doc.setdefault(key, value)`.
fn set_default(doc: &mut Json, key: &str, value: Json) {
    if let Json::Obj(fields) = doc {
        if !fields.iter().any(|(k, _)| k == key) {
            fields.push((key.to_string(), value));
        }
    }
}

/// One turn of talking with the model: heard, trained, answered, spoken,
/// taught - each an event to `write`.
///
/// `data` is the recording (a WAV; `None` for a typed line, with the
/// transcript in the options the words), `history` the conversation so far,
/// `learn` and `model_reply` the model's parts (see [`Learn`],
/// [`ModelReply`]), `ollama` the client when Ollama may answer.  Fails for a
/// turn that cannot be heard (unreadable audio, nothing said, a token that is
/// no unit) - a bad request, on the server.
#[allow(clippy::too_many_arguments)]
pub fn turn(
    data: Option<&[u8]>,
    encoding: Encoding,
    o: &VoiceOptions,
    history: &[(String, String)],
    learn: Option<&mut Learn>,
    model_reply: Option<&mut ModelReply>,
    ollama: Option<&dyn LlmClient>,
    write: Option<&mut Write>,
) -> Result<Outcome, String> {
    o.validate()?;
    let (mut learn, mut model_reply, mut write) = (learn, model_reply, write);
    let mut emit = |doc: Json| {
        if let Some(w) = write.as_deref_mut() {
            w(doc)
        }
    };
    let started = Instant::now();
    // 1. heard
    let heard = hear(data, encoding, o)?;
    let line = if heard.transcript.is_empty() {
        heard.units.clone().unwrap_or_default()
    } else {
        heard.transcript.clone()
    };
    emit(event("heard", &Json::obj(heard.pairs())));
    // 2. trained
    let mut trained = Json::Null;
    if o.train && !heard.texts.is_empty() {
        if let Some(learn) = learn.as_mut() {
            let began = Instant::now();
            let mut doc = learn(heard.texts.as_slice())?;
            set_default(&mut doc, "texts", Json::Int(heard.texts.len() as i64));
            extend(
                &mut doc,
                vec![("seconds", Json::Num(py_round(began.elapsed().as_secs_f64(), 3)))],
            );
            emit(event("trained", &doc));
            trained = doc;
        }
    }
    // 3. answered
    let mut said: Vec<String> = history.iter().map(|(_, text)| text.clone()).collect();
    if !line.is_empty() {
        said.push(line.clone());
    }
    let said = Heard::new(&said);
    let mut by = "none";
    let mut reply_text = String::new();
    let mut record = Json::Null;
    let mut ollama_error: Option<String> = None;
    let mut model_turn: Option<Turn> = None;
    if !line.is_empty() && matches!(o.answer.as_str(), "model" | "auto") {
        if let Some(model_reply) = model_reply.as_mut() {
            model_turn = model_reply(&line, &said)?;
        }
    }
    if let Some(turn) = &model_turn {
        by = "model";
        reply_text = turn.text.clone();
        record = turn.to_json();
    }
    // Ollama answers for the model, or steps in when it could not answer the person:
    // nothing, or a fresh text that picked up none of the line
    let wants_ollama = o.answer == "ollama" || (o.answer == "auto" && model_turn.as_ref().is_none_or(|t| t.fresh));
    if wants_ollama && !heard.transcript.is_empty() {
        match ollama {
            None => ollama_error = Some("no Ollama to ask".to_string()),
            Some(client) => {
                // Ollama reads the model's lines as words: a model of sounds is heard in the words it spells
                let mut transcript: Vec<(String, String)> = history
                    .iter()
                    .map(|(who, text)| {
                        let text = if who == SPEAKERS.1 {
                            spelled(encoding, text)
                        } else {
                            text.clone()
                        };
                        (who.clone(), text)
                    })
                    .collect();
                transcript.push((SPEAKERS.0.to_string(), heard.transcript.clone()));
                match reply_line(
                    client,
                    &transcript,
                    &o.persona,
                    &o.topic,
                    SPEAKERS,
                    client.model(),
                    o.ollama_temperature,
                ) {
                    Err(err) => ollama_error = Some(err.to_string()),
                    Ok(text) if !text.is_empty() => {
                        by = "ollama";
                        reply_text = text;
                    }
                    Ok(_) => {}
                }
            }
        }
    }
    let voice = speech_encoding(encoding, by);
    let reply_doc = Json::obj([
        ("by", Json::str(by)),
        ("text", Json::str(&reply_text)),
        (
            "spelled",
            Json::str(if reply_text.is_empty() {
                String::new()
            } else {
                spelled(voice, &reply_text)
            }),
        ),
        ("turn", record),
        (
            "ollama_error",
            ollama_error.clone().map(Json::Str).unwrap_or(Json::Null),
        ),
    ]);
    emit(event("reply", &reply_doc));
    // 4. spoken
    let mut spoken_doc = Json::Null;
    let mut pcm: Vec<u8> = Vec::new();
    let mut rate = o.voice_rate.max(1) as u32;
    if o.speak && !reply_text.is_empty() {
        let spoken = say(voice, &[reply_text.clone()], &o.say_options())?;
        rate = spoken.rate;
        for chunk in audio_events(&spoken.pcm, rate, o.chunk.max(1) as usize) {
            emit(chunk);
        }
        spoken_doc = spoken.to_json();
        emit(event("spoken", &spoken_doc));
        pcm = spoken.pcm;
    }
    // 5. taught
    let mut taught = Json::Null;
    if by == "ollama" && o.learn_reply && !reply_text.is_empty() && encoding.unit != Unit::Acoustic {
        if let Some(learn) = learn.as_mut() {
            let mut doc = learn(&[reply_text.clone()])?;
            set_default(&mut doc, "texts", Json::Int(1));
            emit(event("taught", &doc));
            taught = doc;
        }
    }
    let mut spoken_pairs: Vec<Json> = history
        .iter()
        .map(|(speaker, text)| Json::strs([speaker.clone(), text.clone()]))
        .collect();
    if !line.is_empty() {
        spoken_pairs.push(Json::strs([SPEAKERS.0.to_string(), line.clone()]));
    }
    if !reply_text.is_empty() {
        spoken_pairs.push(Json::strs([SPEAKERS.1.to_string(), reply_text.clone()]));
    }
    let opt = |v: &Option<String>| v.clone().map(Json::Str).unwrap_or(Json::Null);
    let document = Json::obj([
        ("transcript", Json::str(&heard.transcript)),
        ("line", Json::str(&line)),
        ("token", opt(&heard.token)),
        ("texts", Json::strs(heard.texts.clone())),
        ("audio", heard.audio.clone()),
        ("units", opt(&heard.units)),
        ("asr", heard.asr.clone()),
        ("trained", trained),
        ("reply", reply_doc),
        ("by", Json::str(by)),
        ("spoken", spoken_doc),
        ("taught", taught),
        ("history", Json::Arr(spoken_pairs)),
        ("encoding", Json::str(encoding.to_string())),
        ("answer", Json::str(&o.answer)),
        ("seconds", Json::Num(py_round(started.elapsed().as_secs_f64(), 3))),
    ]);
    Ok(Outcome { document, pcm, rate })
}

/// What the Voice tab has to work with: the speakers, the answer modes, the
/// voice and the transcription (Python's `describe`).
pub fn describe(encoding: Encoding, ollama: Option<&OllamaClient>) -> Json {
    let speech = speech::describe();
    Json::obj([
        ("speakers", Json::strs([SPEAKERS.0, SPEAKERS.1])),
        ("answers", Json::strs(ANSWERS)),
        ("vocoders", Json::strs(crate::voice::VOCODERS)),
        ("encoding", Json::str(encoding.to_string())),
        ("decoder", Json::str(decoder_name(encoding))),
        (
            "vocoder",
            if encoding.unit == Unit::Acoustic {
                // a vocoder file that is not the codebook's: the tab says so, the turn fails
                match crate::voice::vocoder_name(encoding, "auto") {
                    Ok(Some(name)) => Json::str(name),
                    Ok(None) => Json::Null,
                    Err(e) => Json::str(format!("error: {e}")),
                }
            } else {
                Json::Null
            },
        ),
        ("acoustic", Json::Bool(encoding.unit == Unit::Acoustic)),
        ("default_rate", Json::Int(DEFAULT_RATE)),
        ("chunk", Json::Int(CHUNK_SAMPLES as i64)),
        (
            "transcription",
            Json::obj([
                ("backends", speech.at("backends").clone()),
                ("auto", speech.at("auto").clone()),
                ("faster_whisper", speech.at("faster_whisper").clone()),
                ("whisper", speech.at("whisper").clone()),
            ]),
        ),
        (
            "ollama",
            match ollama {
                None => Json::Null,
                Some(client) => Json::obj([("url", Json::str(&client.url)), ("model", Json::str(&client.model))]),
            },
        ),
    ])
}

/// The options of one reply of the model's, as the conversation route builds
/// them: `beam` 0 is the default width, and a repeat is thought about before
/// the reply backs up, as it is in every conversation here.
fn reply_options<'a, 'v: 'a>(
    o: &'a VoiceOptions,
    heard: &'a Heard,
    index: usize,
    veto: Option<&'a mut Veto<'v>>,
) -> ReplyOptions<'a, 'v, 'static> {
    ReplyOptions {
        heard,
        index,
        speaker: SPEAKERS.1,
        mode: &o.mode,
        max_length: o.max_length.max(0) as usize,
        context: o.context.max(0) as usize,
        temperature: o.temperature,
        k: o.k.max(1) as usize,
        beam: o.beam.unwrap_or(0).max(0) as usize,
        step_penalty: o.step_penalty,
        avoid_repeats: o.avoid_repeats,
        avoid_word_repeats: o.avoid_word_repeats,
        explore: o.explore.max(0) as usize,
        learn: o.learn,
        veto,
        think: true,
        think_depth: THINK_DEPTH,
        stream: None,
        trace: None,
    }
}

// -- the routes -------------------------------------------------------------------------------

/// What one voice turn is made of on the server: the recording (if any), the
/// options, the history, the Ollama to ask, and how the guard stands
/// (Python's `_voice_request`).
struct VoiceRequest {
    data: Option<Vec<u8>>,
    options: VoiceOptions,
    history: Vec<(String, String)>,
    ollama: Option<OllamaClient>,
    guard: bool,
    provenance: Option<bool>,
}

/// Whether an option was left out: absent, null or `""` (Python's `number`
/// reads all three as the default).
fn left_out(form: &Form, name: &str) -> bool {
    match form.option(name) {
        None | Some(Json::Null) => true,
        Some(Json::Str(text)) => text.is_empty(),
        Some(_) => false,
    }
}

/// A whole number, `default` when left out; anything else must be one.
fn int_option(form: &Form, name: &str, default: i64) -> Result<i64, ApiError> {
    if left_out(form, name) {
        return Ok(default);
    }
    form.exact_int(name, default)
        .map_err(|_| ApiError::bad_request(format!("'{name}' must be a number")))
}

/// A number, `default` when left out; anything else must be one.
fn float_option(form: &Form, name: &str, default: f64) -> Result<f64, ApiError> {
    if left_out(form, name) {
        return Ok(default);
    }
    form.exact_float(name, default)
        .map_err(|_| ApiError::bad_request(format!("'{name}' must be a number")))
}

/// A whole number that may be left out.
fn maybe_int(form: &Form, name: &str) -> Result<Option<i64>, ApiError> {
    if left_out(form, name) {
        return Ok(None);
    }
    int_option(form, name, 0).map(Some)
}

/// One voice turn's settings from a JSON body or a multipart / raw request's
/// query string (Python's `_voice_options`).
pub(crate) fn voice_options(form: &Form) -> Result<VoiceOptions, ApiError> {
    let options = VoiceOptions {
        teach: TeachOptions {
            transcript: form.text("transcript", ""),
            asr: asr::AsrOptions {
                backend: form.text("backend", "auto"),
                text: String::new(),
                language: form.maybe_text("language"),
                model: form.maybe_text("asr_model"),
                url: form.maybe_text("asr_url"),
            },
            rate: int_option(form, "rate", DEFAULT_RATE)?,
            codec: form.text("codec", "auto"),
            normalise: form.flag("normalise", false),
            waveform: form.flag("waveform", true),
            pair: form.flag("pair", false),
            token: None,
            unique: form.flag("unique", true),
        },
        train: form.flag("train", true),
        epochs: int_option(form, "epochs", 2)?,
        lr: float_option(form, "lr", 0.5)?,
        batch_size: int_option(form, "batch_size", 8)?,
        answer: form.text("answer", "auto").trim().to_lowercase(),
        learn_reply: form.flag("learn_reply", true),
        persona: form.text("persona", ""),
        topic: form.text("topic", ""),
        mode: form.text("mode", "beam"),
        max_length: int_option(form, "max_length", 60)?,
        context: int_option(form, "context", 12)?,
        k: int_option(form, "k", 5)?,
        beam: maybe_int(form, "beam")?,
        step_penalty: float_option(form, "step_penalty", 0.0)?,
        temperature: float_option(form, "temperature", 1.0)?,
        seed: maybe_int(form, "seed")?,
        avoid_repeats: form.flag("avoid_repeats", true),
        avoid_word_repeats: form.flag("avoid_word_repeats", true),
        explore: int_option(form, "explore", EXPLORE as i64)?,
        learn: form.flag("learn", true),
        ollama_temperature: float_option(form, "ollama_temperature", 0.8)?,
        speak: form.flag("speak", true),
        voice_rate: int_option(form, "voice_rate", phonetok::synth::RATE as i64)?,
        pitch: float_option(form, "pitch", 120.0)?,
        tempo: float_option(form, "tempo", 1.0)?,
        gain: float_option(form, "gain", 0.5)?,
        polish: int_option(form, "polish", 0)?,
        vocoder: form.text("vocoder", "auto"),
        chunk: CHUNK_SAMPLES as i64,
    };
    options.validate().map_err(ApiError::bad_request)?;
    Ok(options)
}

/// What one voice turn is made of: the recording (if any), the options, the
/// history, the Ollama to ask.
fn voice_request(svc: &Arc<Service>, r: &Request) -> Result<VoiceRequest, ApiError> {
    let form = Form::read(r, Some("recording"))?;
    let data = if form.has_files() {
        Some(form.bytes("recording")?.1)
    } else {
        None
    };
    let options = voice_options(&form)?;
    if data.is_none() && options.teach.transcript.trim_matches(is_python_space).is_empty() {
        return Err(ApiError::bad_request(
            "nothing was said: send a recording (multipart, a raw body or JSON {name, content_base64}), \
             a transcript, or both",
        ));
    }
    let history = match form.option("history") {
        None | Some(Json::Null) => Vec::new(),
        Some(Json::Arr(items)) => items,
        Some(Json::Str(text)) => {
            if text.trim_matches(is_python_space).is_empty() {
                Vec::new()
            } else {
                match radixnet::json::parse(&text) {
                    Ok(Json::Arr(items)) => items,
                    Ok(_) => {
                        return Err(ApiError::bad_request(
                            "'history' must be a list of [speaker, text] pairs",
                        ))
                    }
                    Err(_) => {
                        return Err(ApiError::bad_request(
                            "'history' must be a JSON list of [speaker, text] pairs",
                        ))
                    }
                }
            }
        }
        Some(_) => {
            return Err(ApiError::bad_request(
                "'history' must be a list of [speaker, text] pairs",
            ))
        }
    };
    let history = history_pairs(&history).map_err(ApiError::bad_request)?;
    let mut ollama = None;
    if matches!(options.answer.as_str(), "auto" | "ollama") {
        let timeout = if left_out(&form, "timeout") {
            None
        } else {
            let seconds = form
                .exact_float("timeout", 0.0)
                .map_err(|err| ApiError::bad_request(format!("'timeout' must be a number: {err}")))?;
            crate::llm::seconds(seconds)
        };
        ollama = Some(svc.llm.ollama(
            form.maybe_text("url").as_deref(),
            form.maybe_text("ollama_model").as_deref(),
            timeout,
        )?);
    }
    if options.train || (options.learn_reply && matches!(options.answer.as_str(), "auto" | "ollama")) {
        svc.ensure_idle()?;
    }
    let provenance = if left_out(&form, "provenance") {
        None
    } else {
        Some(form.flag("provenance", false))
    };
    Ok(VoiceRequest {
        data,
        options,
        history,
        ollama,
        guard: form.flag("guard", true),
        provenance,
    })
}

/// One turn on the server (Python's `ModelService.voice_turn`).
///
/// Training and the model's own reply hold the model's lock for exactly as
/// long as they run (a job running elsewhere refused them at the door, 409);
/// Ollama is asked outside it, as the chat loop does, so a slow answer never
/// holds the server.  The guard vetoes the model's reply as it does every
/// other answer ([`Service::guard`]).
fn server_turn(svc: &Arc<Service>, req: &VoiceRequest, write: Option<&mut Write>) -> Result<Outcome, ApiError> {
    let o = &req.options;
    let encoding = svc.with_model(|m| m.encoding());
    let (epochs, lr, batch_size) = (o.epochs.max(0) as usize, o.lr, o.batch_size.max(1) as usize);
    let index = 1 + req.history.len();
    let mut rng = o.seed.map(Mt19937::new);
    // a failure of the guard's own (the negative network could not be opened) keeps its status
    let mut failure: Option<ApiError> = None;
    let mut learn = |texts: &[String]| -> Result<Json, String> {
        svc.with_model(|m| radixnet::kinds::train_at(m, texts, epochs, lr, batch_size))?;
        Ok(Json::obj([
            ("texts", Json::Int(texts.len() as i64)),
            ("epochs", Json::Int(epochs as i64)),
        ]))
    };
    let mut model_reply = |line: &str, heard: &Heard| -> Result<Option<Turn>, String> {
        if req.guard {
            let guarded = svc.guard(req.provenance, |pair| -> Result<Option<Turn>, String> {
                let config = pair.config.clone();
                let negative = &mut *pair.negative;
                let mut judged: Vec<(String, bool)> = Vec::new();
                let mut veto = |positive: &mut Model, text: &str| -> bool {
                    // a candidate offered again after a shorter context is judged once
                    if let Some((_, refused)) = judged.iter().find(|(t, _)| t == text) {
                        return *refused;
                    }
                    let mut judge = Filter {
                        positive,
                        negative: &mut *negative,
                        config: config.clone(),
                    };
                    let refused = judge.judge(text).decision == "reject";
                    judged.push((text.to_string(), refused));
                    refused
                };
                reply(
                    pair.positive,
                    None,
                    line,
                    reply_options(o, heard, index, Some(&mut veto as &mut Veto)),
                    &mut rng,
                )
            });
            match guarded {
                Ok(Some(result)) => return result,
                Ok(None) => {} // nothing to guard with: the model answers unguarded
                Err(err) => {
                    let message = err.message.clone();
                    failure = Some(err);
                    return Err(message);
                }
            }
        }
        svc.with_model(|m| reply(m, None, line, reply_options(o, heard, index, None), &mut rng))
    };
    let outcome = turn(
        req.data.as_deref(),
        encoding,
        o,
        &req.history,
        Some(&mut learn),
        Some(&mut model_reply),
        req.ollama.as_ref().map(|c| c as &dyn LlmClient),
        write,
    );
    match outcome {
        Ok(outcome) => Ok(outcome),
        Err(message) => Err(failure.take().unwrap_or_else(|| ApiError::bad_request(message))),
    }
}

/// `GET /api/voice`: what the Voice tab has to work with.
fn r_voice(svc: &Arc<Service>, _r: &Request) -> Answer {
    let encoding = svc.with_model(|m| m.encoding());
    let ollama = svc.llm.ollama(None, None, None).ok();
    Ok(describe(encoding, ollama.as_ref()))
}

/// `POST /api/voice/turn`: one turn of talking with the model by voice,
/// answered whole: the document and the reply's WAV.
fn r_turn(svc: &Arc<Service>, r: &Request) -> Answer {
    let req = voice_request(svc, r)?;
    let outcome = server_turn(svc, &req, None)?;
    let wav = if outcome.pcm.is_empty() {
        Json::Null
    } else {
        Json::str(base64::encode(&outcome.wav()))
    };
    let Outcome { mut document, rate, .. } = outcome;
    extend(
        &mut document,
        vec![("rate", Json::Int(rate as i64)), ("wav_base64", wav)],
    );
    Ok(document)
}

/// `POST /api/voice/turn/stream`: the same turn as it happens, every step one
/// JSON line, the reply's audio in chunks played as they arrive, then `done`
/// with the whole document (without `wav_base64`: the audio was the events).
fn r_turn_stream(svc: &Arc<Service>, r: &Request, sink: &mut Sink) -> Result<(), ApiError> {
    let req = voice_request(svc, r)?;
    let outcome = {
        let mut write = |event: Json| sink.send(&event);
        server_turn(svc, &req, Some(&mut write))?
    };
    let mut done = event("done", &outcome.document);
    extend(&mut done, vec![("rate", Json::Int(outcome.rate as i64))]);
    sink.send(&done);
    Ok(())
}

/// `GET /api/voice`, `POST /api/voice/turn` and `POST /api/voice/turn/stream`.
pub fn routes(server: &mut Server<Service>) {
    server.route("GET", "/api/voice", r_voice);
    server.route("POST", "/api/voice/turn", r_turn);
    server.event_route("POST", "/api/voice/turn/stream", r_turn_stream);
}

// -- the command line -------------------------------------------------------------------------

/// A conversation file: one line per utterance, `Speaker: text` (a bare line
/// is the person's) - Python's `_history_lines`.
fn history_lines(text: &str) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    for line in text.lines() {
        let line = line.trim_matches(is_python_space);
        if line.is_empty() {
            continue;
        }
        match line.split_once(':') {
            Some((speaker, said)) if !speaker.is_empty() && !speaker.trim_matches(is_python_space).contains(' ') => {
                pairs.push((
                    speaker.trim_matches(is_python_space).to_string(),
                    said.trim_matches(is_python_space).to_string(),
                ));
            }
            _ => pairs.push((SPEAKERS.0.to_string(), line.to_string())),
        }
    }
    pairs
}

/// Where the reply's speech goes - a player (`--play`), stdout (`--raw`) or a
/// WAV (`--out`, `reply.wav` by default) - and the sink's name.
fn deliver(ctx: &Ctx, pcm: &[u8], rate: u32) -> Result<String, String> {
    use std::io::Write as _;

    let a = &ctx.args;
    if a.on("play") {
        let Some(player) = phonetok::synth::find_player() else {
            return Err("no player found (aplay, paplay, ffplay, play or afplay); use --out FILE or --raw".to_string());
        };
        let mut child = std::process::Command::new(&player[0])
            .args(&player[1..])
            .stdin(std::process::Stdio::piped())
            .spawn()
            .map_err(|err| format!("{}: {err}", player[0]))?;
        {
            let stdin = child.stdin.as_mut().ok_or("no stdin")?;
            stdin
                .write_all(&phonetok::synth::wav_header(rate, None))
                .and_then(|_| stdin.write_all(pcm))
                .and_then(|_| stdin.flush())
                .map_err(|err| format!("{}: {err}", player[0]))?;
        }
        child.wait().ok();
        return Ok(player[0].clone());
    }
    if a.on("raw") {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        out.write_all(pcm)
            .and_then(|_| out.flush())
            .map_err(|err| format!("stdout: {err}"))?;
        return Ok("stdout".to_string());
    }
    let path = a.str("out", "reply.wav");
    std::fs::write(&path, phonetok::synth::wav_bytes(pcm, rate))
        .map_err(|err| format!("cannot write {path}: {err}"))?;
    Ok(path)
}

/// `radixnet speech talk [FILE] [--seconds N] [--text ...]`: one turn of
/// talking with the model by voice - a recording (a file, or `--seconds`
/// from the microphone) is heard, learned, answered and spoken back, the same
/// turn the Voice tab makes for every utterance (Python's `cmd_speech_talk`).
/// Prints the turn's document, unless `--raw` sent the reply to stdout.
pub fn talk_cli(ctx: &Ctx) -> Result<(), String> {
    let a = &ctx.args;
    let seconds = a.float("seconds", 0.0)?;
    let (data, source): (Option<Vec<u8>>, String) = if let Some(path) = a.rest.get(1) {
        (Some(speech::read_audio_file(path)?), path.clone())
    } else if seconds > 0.0 {
        let rate = a.int("record-rate", 16_000)?;
        radixnet::log_warn!(LOG, "recording {}s from the microphone - speak now", py_repr(seconds));
        let data = asr::record(seconds, rate, a.get("recorder"))?;
        radixnet::log_warn!(LOG, "recording finished");
        (Some(data), format!("microphone ({}s)", py_repr(seconds)))
    } else {
        (None, "typed".to_string())
    };
    let mut teach = speech::teach_options(ctx)?;
    teach.token = None;
    if data.is_none() && teach.transcript.trim_matches(is_python_space).is_empty() {
        return Err("nothing was said: give a recording FILE, --seconds N to record one, or --text".to_string());
    }
    let history = match a.get("history") {
        Some(path) => {
            let text = std::fs::read_to_string(path).map_err(|err| format!("cannot read {path}: {err}"))?;
            tidy_pairs(history_lines(&text))
        }
        None => Vec::new(),
    };
    let defaults = VoiceOptions::default();
    let options = VoiceOptions {
        teach,
        train: !a.on("no-train"),
        epochs: a.int("epochs", 2)?,
        lr: a.float("lr", 0.5)?,
        batch_size: a.int("batch-size", 8)?,
        answer: a.str("answer", "auto"),
        learn_reply: !a.on("no-learn-reply"),
        persona: a.str("persona", ""),
        topic: a.str("topic", ""),
        mode: a.str("mode", "beam"),
        max_length: a.int("max-length", 60)?,
        context: a.int("context", 12)?,
        k: a.int("k", 5)?,
        beam: if a.get("beam").is_some() {
            Some(a.int("beam", 0)?)
        } else {
            None
        },
        temperature: a.float("temperature", 1.0)?,
        seed: a.get("seed").map(|_| ctx.seed),
        explore: a.int("explore", EXPLORE as i64)?,
        learn: !a.on("no-learn"),
        speak: !a.on("no-speak"),
        voice_rate: a.int("voice-rate", phonetok::synth::RATE as i64)?,
        pitch: a.float("pitch", 120.0)?,
        tempo: a.float("tempo", 1.0)?,
        gain: a.float("gain", 0.5)?,
        polish: a.int("polish", 0)?,
        vocoder: a.str("vocoder", "auto"),
        ..defaults
    };
    options.validate()?;
    let ollama = if matches!(options.answer.as_str(), "auto" | "ollama") {
        Some(OllamaClient::new(
            &a.str("url", ""),
            &a.str("ollama-model", ""),
            crate::llm::timeout_flag(ctx)?,
        )?)
    } else {
        None
    };
    let model = std::cell::RefCell::new(ctx.open(false)?);
    let encoding = model.borrow().encoding();
    let mut guard = ctx.open_guard()?;
    let guarding = guard.is_some();
    let mut judged: Vec<(String, bool)> = Vec::new();
    let mut rng = options.seed.map(Mt19937::new);
    let index = 1 + history.len();
    let learned = std::cell::Cell::new(0usize);
    let (epochs, lr, batch_size) = (
        options.epochs.max(0) as usize,
        options.lr,
        options.batch_size.max(1) as usize,
    );
    let mut learn = |texts: &[String]| -> Result<Json, String> {
        let records = radixnet::kinds::train_at(&mut model.borrow_mut(), texts, epochs, lr, batch_size)?;
        learned.set(learned.get() + texts.len());
        Ok(Json::obj([
            ("texts", Json::Int(texts.len() as i64)),
            ("epochs", Json::Int(records.len() as i64)),
            ("loss", records.last().map(|r| Json::Num(r.loss)).unwrap_or(Json::Null)),
        ]))
    };
    let o = &options;
    let mut model_reply = |line: &str, heard: &Heard| -> Result<Option<Turn>, String> {
        let mut veto = |positive: &mut Model, text: &str| -> bool {
            let Some((negative, config)) = guard.as_mut() else {
                return false;
            };
            if let Some((_, refused)) = judged.iter().find(|(t, _)| t == text) {
                return *refused;
            }
            let mut pair = Filter {
                positive,
                negative,
                config: config.clone(),
            };
            let refused = pair.judge(text).decision == "reject";
            judged.push((text.to_string(), refused));
            refused
        };
        let veto: Option<&mut Veto> = if guarding { Some(&mut veto) } else { None };
        let mut m = model.borrow_mut();
        reply(&mut m, None, line, reply_options(o, heard, index, veto), &mut rng)
    };
    let outcome = turn(
        data.as_deref(),
        encoding,
        o,
        &history,
        Some(&mut learn),
        Some(&mut model_reply),
        ollama.as_ref().map(|c| c as &dyn LlmClient),
        None,
    )?;
    let Outcome {
        mut document,
        pcm,
        rate,
    } = outcome;
    let saved = if learned.get() > 0 {
        let out = a.str("model-out", &ctx.model_path);
        speech::save_model(&mut model.borrow_mut(), &out)?
    } else {
        Json::Null
    };
    extend(
        &mut document,
        vec![("source", Json::str(source)), ("saved", saved), ("sink", Json::Null)],
    );
    if !pcm.is_empty() {
        let sink = deliver(ctx, &pcm, rate)?;
        extend(&mut document, vec![("sink", Json::str(sink))]);
    }
    if let Some(error) = document.at("reply").at("ollama_error").as_str() {
        radixnet::log_warn!(LOG, "note: Ollama could not answer ({error})");
    }
    if !a.on("raw") {
        ctx.emit(document);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cell::{Cell, RefCell};
    use std::time::Duration;

    use super::*;
    use crate::speech::wav_bytes;

    /// A 440 Hz tone as a WAV, `seconds` long.
    fn recording(seconds: f64) -> Vec<u8> {
        let rate = 16_000i64;
        let n = (rate as f64 * seconds) as usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / rate as f64).sin()) as f32)
            .collect();
        wav_bytes(&samples, rate, 1)
    }

    fn a_turn(text: &str, fresh: bool) -> Turn {
        Turn {
            index: 1,
            speaker: SPEAKERS.1.to_string(),
            text: text.to_string(),
            context: if fresh {
                String::new()
            } else {
                text.split_whitespace().next().unwrap_or("").to_string()
            },
            reply: text.to_string(),
            cost: 1.0,
            probability: 0.5,
            reached_end: true,
            fresh,
            ..Default::default()
        }
    }

    fn options(transcript: &str, answer: &str) -> VoiceOptions {
        let mut o = VoiceOptions::default();
        o.teach.transcript = transcript.to_string();
        o.answer = answer.to_string();
        o
    }

    fn kinds(events: &[Json]) -> Vec<String> {
        events
            .iter()
            .map(|e| e.at("event").as_str().unwrap_or("").to_string())
            .collect()
    }

    fn pairs(history: &[(&str, &str)]) -> Vec<(String, String)> {
        history.iter().map(|(s, t)| (s.to_string(), t.to_string())).collect()
    }

    fn strings(texts: &[&str]) -> Vec<String> {
        texts.iter().map(|t| t.to_string()).collect()
    }

    fn ollama_says(path: &str, _body: &Json) -> (u16, String) {
        match path {
            "/api/generate" => (200, r#"{"response": "Hi there, friend."}"#.to_string()),
            _ => (404, "{}".to_string()),
        }
    }

    #[test]
    fn heard_trained_answered_and_spoken() {
        let mut events: Vec<Json> = Vec::new();
        let learned: RefCell<Vec<Vec<String>>> = RefCell::new(Vec::new());
        let asked: RefCell<Option<(String, Heard)>> = RefCell::new(None);
        let mut learn = |texts: &[String]| -> Result<Json, String> {
            learned.borrow_mut().push(texts.to_vec());
            Ok(Json::obj([
                ("texts", Json::Int(texts.len() as i64)),
                ("epochs", Json::Int(1)),
            ]))
        };
        let mut model_reply = |line: &str, heard: &Heard| -> Result<Option<Turn>, String> {
            *asked.borrow_mut() = Some((line.to_string(), heard.clone()));
            Ok(Some(a_turn("sat on the mat", false)))
        };
        let mut write = |event: Json| events.push(event);
        let mut o = options("the cat sat", "model");
        o.epochs = 1;
        let history = pairs(&[("You", "hello"), ("Model", "hi there")]);
        let out = turn(
            Some(&recording(0.2)),
            Encoding::default(),
            &o,
            &history,
            Some(&mut learn),
            Some(&mut model_reply),
            None,
            Some(&mut write),
        )
        .unwrap();
        let doc = &out.document;
        let kinds = kinds(&events);
        assert_eq!(&kinds[..3], ["heard", "trained", "reply"]);
        assert_eq!(kinds.last().map(String::as_str), Some("spoken"));
        assert!(kinds.len() > 4 && kinds[3..kinds.len() - 1].iter().all(|k| k == "audio"));
        // heard: the transcript and the waveform behind one token, both learned before the reply
        assert_eq!(doc.at("transcript").as_str(), Some("the cat sat"));
        assert!(doc.at("token").as_str().unwrap().starts_with("<speech:"));
        let texts = doc.at("texts").to_strings();
        assert_eq!(texts.len(), 2);
        assert!(texts[0].ends_with(" the cat sat") && texts[1].contains("aud:mu:"));
        assert_eq!(*learned.borrow(), vec![texts.clone()]);
        assert_eq!(doc.at("trained").at("texts").as_i64(), Some(2));
        assert_eq!(doc.at("audio").at("codec").as_str(), Some("mu"));
        // answered: the line, with the conversation so far heard
        let (line, heard) = asked.borrow().clone().unwrap();
        assert_eq!(line, "the cat sat");
        assert!(heard.duplicate("hi there", ""));
        assert!(heard.duplicate("the cat sat", ""));
        assert_eq!(
            (
                doc.at("by").as_str(),
                doc.at("reply").at("text").as_str(),
                doc.at("reply").at("spelled").as_str()
            ),
            (Some("model"), Some("sat on the mat"), Some("sat on the mat"))
        );
        assert_eq!(doc.at("reply").at("turn").at("text").as_str(), Some("sat on the mat"));
        assert!(doc.at("reply").at("ollama_error").is_null());
        let expected = radixnet::json::parse(
            r#"[["You", "hello"], ["Model", "hi there"], ["You", "the cat sat"], ["Model", "sat on the mat"]]"#,
        )
        .unwrap();
        assert_eq!(doc.at("history"), &expected);
        // spoken: the audio events are the reply's speech, chunk by chunk, exactly what the decoder says
        let chunks: Vec<&Json> = events
            .iter()
            .filter(|e| e.at("event").as_str() == Some("audio"))
            .collect();
        let pcm: Vec<u8> = chunks
            .iter()
            .flat_map(|e| base64::decode(e.at("pcm_base64").as_str().unwrap()).unwrap())
            .collect();
        assert_eq!(pcm, out.pcm);
        assert_eq!(
            pcm,
            say(
                Encoding::default(),
                &strings(&["sat on the mat"]),
                &SayOptions::default()
            )
            .unwrap()
            .pcm
        );
        assert!(
            chunks
                .iter()
                .all(|e| e.at("samples").as_i64().unwrap() <= CHUNK_SAMPLES as i64
                    && e.at("rate").as_i64() == Some(16000))
        );
        assert_eq!(
            chunks.iter().map(|e| e.at("samples").as_i64().unwrap()).sum::<i64>(),
            doc.at("spoken").at("samples").as_i64().unwrap()
        );
        assert_eq!(
            (doc.at("spoken").at("decoder").as_str(), out.rate),
            (Some("voice"), 16000)
        );
        assert!(out.wav().starts_with(b"RIFF"));
        assert_eq!(out.wav().len(), 44 + pcm.len());
        assert!(doc.at("taught").is_null());
        assert_eq!(doc.at("encoding").as_str(), Some("char:3:1"));
    }

    #[test]
    fn typed_text_nothing_to_say_and_nobody_asked() {
        let mut events: Vec<Json> = Vec::new();
        let (learned, replied) = (Cell::new(0), Cell::new(0));
        let mut learn = |texts: &[String]| -> Result<Json, String> {
            learned.set(learned.get() + 1);
            Ok(Json::obj([("texts", Json::Int(texts.len() as i64))]))
        };
        let mut model_reply = |_: &str, _: &Heard| -> Result<Option<Turn>, String> {
            replied.set(replied.get() + 1);
            Ok(None)
        };
        let mut write = |event: Json| events.push(event);
        let out = turn(
            None,
            Encoding::default(),
            &options("hello there", "auto"),
            &[],
            Some(&mut learn),
            Some(&mut model_reply),
            None,
            Some(&mut write),
        )
        .unwrap();
        let doc = &out.document;
        assert_eq!(doc.at("texts").as_array().len(), 1); // the transcript alone, behind its token
        assert!(doc.at("audio").is_null());
        assert_eq!(
            (
                doc.at("by").as_str(),
                doc.at("reply").at("text").as_str(),
                doc.at("spoken").is_null(),
                out.pcm.len()
            ),
            (Some("none"), Some(""), true, 0)
        );
        assert_eq!(kinds(&events), ["heard", "trained", "reply"]);
        assert_eq!(
            doc.at("history"),
            &radixnet::json::parse(r#"[["You", "hello there"]]"#).unwrap()
        );
        assert_eq!((learned.get(), replied.get()), (1, 1));
        // nobody asked: learned, not answered
        let out = turn(
            None,
            Encoding::default(),
            &options("hello", "none"),
            &[],
            Some(&mut learn),
            Some(&mut model_reply),
            None,
            None,
        )
        .unwrap();
        assert_eq!((out.document.at("by").as_str(), replied.get()), (Some("none"), 1));
        // not learned when told not to
        let mut o = options("hello", "auto");
        o.train = false;
        let out = turn(
            None,
            Encoding::default(),
            &o,
            &[],
            Some(&mut learn),
            Some(&mut model_reply),
            None,
            None,
        )
        .unwrap();
        assert!(out.document.at("trained").is_null());
        assert_eq!(learned.get(), 2);
        // nothing said at all, a nobody, a history that is no pair
        assert!(turn(
            None,
            Encoding::default(),
            &VoiceOptions::default(),
            &[],
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(turn(
            None,
            Encoding::default(),
            &options("hi", "nobody"),
            &[],
            None,
            None,
            None,
            None
        )
        .is_err());
        assert!(history_pairs(&[Json::str("not a pair")]).is_err());
    }

    #[test]
    fn ollama_answers_for_the_model_and_the_model_is_taught() {
        let fake = crate::ollama::tests::fake(ollama_says);
        let client = OllamaClient::new(&fake.url, "fake:latest", None).unwrap();
        let learned: RefCell<Vec<Vec<String>>> = RefCell::new(Vec::new());
        let mut learn = |texts: &[String]| -> Result<Json, String> {
            learned.borrow_mut().push(texts.to_vec());
            Ok(Json::obj([("texts", Json::Int(texts.len() as i64))]))
        };
        let mut never = |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("never", false))) };
        let mut events: Vec<Json> = Vec::new();
        let mut write = |event: Json| events.push(event);
        let mut o = options("how are you", "ollama");
        o.persona = "a cat".to_string();
        let out = turn(
            None,
            Encoding::default(),
            &o,
            &pairs(&[("You", "hello"), ("Model", "hi")]),
            Some(&mut learn),
            Some(&mut never),
            Some(&client),
            Some(&mut write),
        )
        .unwrap();
        let doc = &out.document;
        assert_eq!(doc.at("by").as_str(), Some("ollama"));
        let text = doc.at("reply").at("text").as_str().unwrap().to_string();
        assert!(!text.is_empty());
        assert_eq!(doc.at("reply").at("spelled").as_str(), Some(text.as_str()));
        assert!(doc.at("reply").at("turn").is_null());
        // what was said, then what Ollama said
        assert_eq!(
            *learned.borrow(),
            vec![doc.at("texts").to_strings(), vec![text.clone()]]
        );
        assert_eq!(doc.at("taught").at("texts").as_i64(), Some(1));
        let told: Vec<String> = kinds(&events).into_iter().filter(|k| k != "audio").collect();
        assert_eq!(told, ["heard", "trained", "reply", "spoken", "taught"]);
        assert!(out.pcm.len() > 1000);
        {
            let seen = fake.seen.lock().unwrap();
            let (path, body) = seen.last().unwrap();
            assert_eq!(
                (path.as_str(), body.at("model").as_str()),
                ("/api/generate", Some("fake:latest"))
            );
            let system = body.at("system").as_str().unwrap();
            assert!(system.contains("voice of a very small language model") && system.contains("You are a cat"));
            let prompt = body.at("prompt").as_str().unwrap();
            assert!(
                prompt.contains("You: how are you") && prompt.contains("Model: hi"),
                "{prompt}"
            );
        }
        // auto: the model first; Ollama when it has nothing to say, or only a fresh text that picked up
        // none of the line
        let asked = || fake.seen.lock().unwrap().len();
        let before = asked();
        let mut answers =
            |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("the cat sat", false))) };
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "auto"),
            &[],
            Some(&mut learn),
            Some(&mut answers),
            Some(&client),
            None,
        )
        .unwrap();
        assert_eq!((out.document.at("by").as_str(), asked()), (Some("model"), before));
        let mut silent = |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(None) };
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "auto"),
            &[],
            Some(&mut learn),
            Some(&mut silent),
            Some(&client),
            None,
        )
        .unwrap();
        assert_eq!((out.document.at("by").as_str(), asked()), (Some("ollama"), before + 1));
        let mut fresh = |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("a bird", true))) };
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "auto"),
            &[],
            Some(&mut learn),
            Some(&mut fresh),
            Some(&client),
            None,
        )
        .unwrap();
        assert_eq!((out.document.at("by").as_str(), asked()), (Some("ollama"), before + 2));
        // not taught when told not to
        let mut o = options("the cat", "ollama");
        o.learn_reply = false;
        let out = turn(
            None,
            Encoding::default(),
            &o,
            &[],
            Some(&mut learn),
            None,
            Some(&client),
            None,
        )
        .unwrap();
        assert_eq!(
            (out.document.at("by").as_str(), out.document.at("taught").is_null()),
            (Some("ollama"), true)
        );
    }

    #[test]
    fn an_unreachable_ollama_is_recorded_not_raised() {
        let down = OllamaClient::new("http://127.0.0.1:1", "fake:latest", Some(Duration::from_secs(1))).unwrap();
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "ollama"),
            &[],
            None,
            None,
            Some(&down),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                out.document.at("by").as_str(),
                out.document.at("reply").at("text").as_str()
            ),
            (Some("none"), Some(""))
        );
        let error = out
            .document
            .at("reply")
            .at("ollama_error")
            .as_str()
            .unwrap()
            .to_string();
        assert!(error.contains("cannot reach Ollama"), "{error}");
        // in auto the model's fresh text stands when Ollama is down, and the error is on record
        let mut fresh = |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("a bird", true))) };
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "auto"),
            &[],
            None,
            Some(&mut fresh),
            Some(&down),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                out.document.at("by").as_str(),
                out.document.at("reply").at("text").as_str()
            ),
            (Some("model"), Some("a bird"))
        );
        assert!(out
            .document
            .at("reply")
            .at("ollama_error")
            .as_str()
            .unwrap()
            .contains("cannot reach Ollama"));
        let out = turn(
            None,
            Encoding::default(),
            &options("the cat", "auto"),
            &[],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            out.document.at("reply").at("ollama_error").as_str(),
            Some("no Ollama to ask")
        );
    }

    #[test]
    fn a_model_of_sounds_and_one_of_acoustic_units() {
        // sounds: the model's reply is phones, spoken as they are and spelled for the record
        let phones = Encoding::new(Unit::Phones, 3, 1).unwrap();
        let mut says_dog =
            |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("DH AH0 # D AO1 G", false))) };
        let out = turn(
            None,
            phones,
            &options("the cat", "model"),
            &[],
            None,
            Some(&mut says_dog),
            None,
            None,
        )
        .unwrap();
        assert_eq!(
            (
                out.document.at("reply").at("spelled").as_str(),
                out.document.at("spoken").at("decoder").as_str()
            ),
            (Some("the dog"), Some("voice"))
        );
        assert_eq!(
            out.pcm,
            say(phones, &strings(&["DH AH0 # D AO1 G"]), &SayOptions::default())
                .unwrap()
                .pcm
        );
        // acoustic units: the recording is heard as units, the one text of sound the model can learn
        let acoustic = Encoding::new(Unit::Acoustic, 3, 1).unwrap();
        let heard = hear(Some(&recording(0.3)), acoustic, &options("the cat", "auto")).unwrap();
        assert_eq!(
            (heard.transcript.as_str(), heard.token.as_deref(), heard.texts.len()),
            ("the cat", None, 1)
        );
        assert!(heard.texts[0].split_whitespace().all(|u| u.starts_with('q')));
        assert_eq!(heard.units.as_deref(), Some(heard.texts[0].as_str()));
        let learned: RefCell<Vec<Vec<String>>> = RefCell::new(Vec::new());
        let mut learn = |texts: &[String]| -> Result<Json, String> {
            learned.borrow_mut().push(texts.to_vec());
            Ok(Json::obj([("texts", Json::Int(texts.len() as i64))]))
        };
        let mut says_units =
            |_: &str, _: &Heard| -> Result<Option<Turn>, String> { Ok(Some(a_turn("q2 q28 q55 q5", false))) };
        let out = turn(
            Some(&recording(0.3)),
            acoustic,
            &options("the cat", "model"),
            &[],
            Some(&mut learn),
            Some(&mut says_units),
            None,
            None,
        )
        .unwrap();
        assert_eq!(*learned.borrow(), vec![out.document.at("texts").to_strings()]);
        assert_eq!(
            (
                out.document.at("by").as_str(),
                out.document.at("spoken").at("decoder").as_str(),
                out.rate
            ),
            (Some("model"), Some("vocoder"), 16000)
        );
        // a reply Ollama wrote is words: spoken through the voice, and not taught to a model of units
        let fake = crate::ollama::tests::fake(ollama_says);
        let client = OllamaClient::new(&fake.url, "fake:latest", None).unwrap();
        let out = turn(
            None,
            acoustic,
            &options("the cat", "ollama"),
            &[],
            Some(&mut learn),
            None,
            Some(&client),
            None,
        )
        .unwrap();
        assert_eq!(
            (
                out.document.at("by").as_str(),
                out.document.at("spoken").at("decoder").as_str(),
                out.document.at("taught").is_null()
            ),
            (Some("ollama"), Some("voice"), true)
        );
        assert!(out.document.at("texts").as_array().is_empty()); // nothing of sound to learn from a typed line
        assert_eq!(learned.borrow().len(), 1);
        // without a recording an acoustic model learns nothing, and without a transcript nobody can answer
        let out = turn(None, acoustic, &options("hi", "none"), &[], None, None, None, None).unwrap();
        assert!(out.document.at("texts").as_array().is_empty() && out.document.at("trained").is_null());
    }

    #[test]
    fn the_small_parts() {
        let history = vec![
            Json::strs(["A", " x  y "]),
            Json::obj([("speaker", Json::str("")), ("text", Json::str("z"))]),
            Json::strs(["B", "  "]),
        ];
        assert_eq!(history_pairs(&history).unwrap(), pairs(&[("A", "x y"), ("You", "z")]));
        for bad in [
            vec![Json::strs(["x"])],
            vec![Json::Arr(vec![Json::str("A"), Json::Int(1)])],
            vec![Json::Int(1)],
        ] {
            assert!(history_pairs(&bad).is_err(), "{bad:?}");
        }
        let (acoustic, phones) = (
            Encoding::new(Unit::Acoustic, 3, 1).unwrap(),
            Encoding::new(Unit::Phones, 3, 1).unwrap(),
        );
        assert_eq!(speech_encoding(acoustic, "ollama"), Encoding::default());
        assert_eq!(speech_encoding(acoustic, "model"), acoustic);
        assert_eq!(speech_encoding(phones, "ollama"), phones);
        let chunks = audio_events(&vec![0u8; 2 * 20000], 16000, 8000);
        assert_eq!(
            chunks
                .iter()
                .map(|c| c.at("samples").as_i64().unwrap())
                .collect::<Vec<_>>(),
            [8000, 8000, 4000]
        );
        assert_eq!(
            base64::decode(chunks[0].at("pcm_base64").as_str().unwrap())
                .unwrap()
                .len(),
            16000
        );
        let bad: Vec<VoiceOptions> = vec![
            VoiceOptions {
                answer: "x".to_string(),
                ..Default::default()
            },
            VoiceOptions {
                mode: "dijkstra".to_string(),
                ..Default::default()
            },
            VoiceOptions {
                epochs: -1,
                ..Default::default()
            },
            VoiceOptions {
                pitch: 0.0,
                ..Default::default()
            },
            VoiceOptions {
                k: 0,
                ..Default::default()
            },
            VoiceOptions {
                chunk: 0,
                ..Default::default()
            },
        ];
        for o in &bad {
            assert!(o.validate().is_err(), "{o:?}");
        }
        VoiceOptions::default().validate().unwrap();
        assert_eq!(
            VoiceOptions {
                answer: "x".to_string(),
                ..Default::default()
            }
            .validate()
            .unwrap_err(),
            "'answer' must be one of auto, model, ollama, none, got 'x'"
        );
        let client = OllamaClient::new("http://127.0.0.1:1", "m", None).unwrap();
        let info = describe(phones, Some(&client));
        assert_eq!(info.at("speakers").to_strings(), ["You", "Model"]);
        assert_eq!(info.at("answers").to_strings(), ANSWERS);
        assert_eq!(
            (info.at("decoder").as_str(), info.at("acoustic").as_bool()),
            (Some("voice"), Some(false))
        );
        assert_eq!(
            info.at("ollama"),
            &Json::obj([("url", Json::str("http://127.0.0.1:1")), ("model", Json::str("m"))])
        );
        assert!(info.at("transcription").get("backends").is_some());
        assert!(describe(acoustic, None).at("ollama").is_null());
        assert_eq!(
            history_lines("You: hello\nModel: hi there\n\nthe dog sat\nnot a speaker: x\n"),
            pairs(&[
                ("You", "hello"),
                ("Model", "hi there"),
                ("You", "the dog sat"),
                ("You", "not a speaker: x")
            ])
        );
    }
}
