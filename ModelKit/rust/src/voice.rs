//! Speech from a walk: the model is heard as it traverses its graph (the port
//! of `radixnet/voice.py`).  A model whose symbols are sounds emits phones as
//! it walks; a model of words or letters emits text.  [`Speaker`] takes
//! either, one step at a time, turns it into the tokens of the phonetic
//! tokenizer and feeds the formant synthesizer, which hands back 16-bit PCM as
//! soon as a word can be committed.  The utterance is closed by the final
//! sentinel: when the walk steps onto END, [`Speaker::end`] flushes what is
//! pending with the closing intonation.  One walk, one utterance.

use phonetok::acoustic::Vocoder;
use phonetok::synth::{Synthesizer, VoiceSettings, RATE};

use radixnet::encoding::{Encoding, Unit};
use radixnet::graph::{END, FIRST};
use radixnet::json::Json;
use radixnet::model::Model;
use radixnet::mt19937::Mt19937;
use radixnet::search::{SamplingFilter, Traversal};

/// How a model is spoken.
#[derive(Clone, Debug)]
pub struct SpeakOptions {
    pub prefix: String,
    pub count: usize,
    /// Units per walk at most; `None` is no cap.
    pub max_length: Option<usize>,
    pub temperature: f64,
    /// A private RNG; `None` uses the model's own.
    pub seed: Option<i64>,
    pub rate: u32,
    pub pitch: f64,
    pub tempo: f64,
    pub gain: f64,
}

/// Turns what a model emits into speech, step by step.
pub struct Speaker {
    enc: Encoding,
    synth: Synthesizer,
    /// Acoustic units are spoken through their codebook's vocoder instead.
    vocoder: Option<Vocoder>,
    /// The sample rate of the PCM: the synthesizer's, or the codebook's for acoustic units.
    pub rate: u32,
    letters: String,
    spoken_words: usize,
    /// Every token that reached the voice, for the record.
    pub tokens: Vec<String>,
}

impl Speaker {
    pub fn new(enc: Encoding, rate: u32, pitch: f64, tempo: f64, gain: f64) -> Result<Speaker, String> {
        let voice = VoiceSettings {
            pitch,
            tempo,
            gain,
            ..VoiceSettings::default()
        };
        if enc.unit == Unit::Acoustic {
            let tok = radixnet::phonetic::acoustic_tokenizer()?;
            // the vocoder's gain is a multiplier on the level the units were learned at, so the
            // voice's default of half scale is the codebook's own level
            let vocoder = Vocoder::new(tok.book.clone(), gain * 2.0, pitch)?;
            return Ok(Speaker {
                enc,
                synth: Synthesizer::new(rate, voice),
                vocoder: Some(vocoder),
                rate: tok.book.analysis.rate,
                letters: String::new(),
                spoken_words: 0,
                tokens: Vec::new(),
            });
        }
        radixnet::phonetic::tokenizer(Unit::Phones)?; // the words of a word or letter model are read through it
        Ok(Speaker {
            enc,
            synth: Synthesizer::new(rate, voice),
            vocoder: None,
            rate,
            letters: String::new(),
            spoken_words: 0,
            tokens: Vec::new(),
        })
    }

    /// One emitted piece (the units a step added); the PCM that is ready comes back.
    /// A token that is not a unit of the codebook (which a model over its units never emits) is skipped.
    pub fn feed(&mut self, piece: &str) -> Vec<u8> {
        if let Some(voc) = self.vocoder.as_mut() {
            let units: Vec<String> = piece.split_whitespace().map(String::from).collect();
            let mut out = Vec::new();
            for unit in &units {
                if let Ok(chunk) = voc.feed(unit) {
                    out.extend(chunk);
                }
            }
            self.tokens.extend(units);
            return out;
        }
        if self.enc.unit.phonetic() {
            let tokens: Vec<String> = radixnet::phonetic::text(self.enc.unit, piece)
                .split_whitespace()
                .map(String::from)
                .collect();
            return self.feed_tokens(&tokens);
        }
        if self.enc.unit == Unit::Words {
            let words: Vec<String> = piece.split_whitespace().map(String::from).collect();
            return self.feed_words(&words);
        }
        // letters: a word ends at whitespace; punctuation ends it too and becomes a pause
        let mut out = Vec::new();
        for c in piece.chars() {
            if c.is_whitespace() {
                out.extend(self.flush_letters());
                continue;
            }
            self.letters.push(c);
            if ".,;:!?".contains(c) {
                out.extend(self.flush_letters());
            }
        }
        out
    }

    fn flush_letters(&mut self) -> Vec<u8> {
        if self.letters.is_empty() {
            return Vec::new();
        }
        let word = std::mem::take(&mut self.letters);
        self.feed_words(&[word])
    }

    fn feed_words(&mut self, words: &[String]) -> Vec<u8> {
        let mut out = Vec::new();
        let Ok(bridge) = radixnet::phonetic::tokenizer(Unit::Phones) else {
            return out;
        };
        for w in words {
            let tokens = {
                let mut tok = bridge.lock().unwrap_or_else(|e| e.into_inner());
                tok.tokens(w)
            };
            if tokens.is_empty() {
                continue;
            }
            if self.spoken_words > 0 && !matches!(tokens[0].as_str(), "," | "." | "?") {
                out.extend(self.feed_tokens(&["#".to_string()]));
            }
            self.spoken_words += 1;
            out.extend(self.feed_tokens(&tokens));
        }
        out
    }

    fn feed_tokens(&mut self, tokens: &[String]) -> Vec<u8> {
        let mut out = Vec::new();
        for t in tokens {
            self.tokens.push(t.clone());
            out.extend(self.synth.feed(t));
        }
        out
    }

    /// The final sentinel: the utterance is closed, and what was pending is spoken.
    pub fn end(&mut self) -> Vec<u8> {
        if let Some(voc) = self.vocoder.as_mut() {
            self.tokens.push("</s>".to_string());
            return voc.end();
        }
        // a letter model's last word, so the sentinel is recorded after it
        let mut out = self.flush_letters();
        self.tokens.push("</s>".to_string());
        out.extend(self.synth.end());
        self.spoken_words = 0;
        out
    }
}

/// Walks the model `count` times from `prefix` and speaks each walk as it
/// goes: `emit` gets every chunk of PCM the moment it is made, `said` is
/// told what each walk said once it has ended - the text in the model's
/// units and the words it spells.
pub fn speak_walks(
    model: &mut Model,
    o: &SpeakOptions,
    emit: &mut dyn FnMut(&[u8]),
    said: &mut dyn FnMut(usize, &str, &str),
) -> Result<(), String> {
    let enc = model.g.enc;
    let mut rng = o.seed.map(Mt19937::new);
    for i in 0..o.count {
        let mut speaker = Speaker::new(enc, o.rate, o.pitch, o.tempo, o.gain)?;
        let mut parts: Vec<String> = Vec::new();
        let (node, offset, lead) = model.prefix_start(&o.prefix);
        let mut opening: Vec<String> = Vec::new();
        if !o.prefix.is_empty() {
            opening.push(o.prefix.clone());
        }
        if !lead.is_empty() {
            opening.push(lead);
        }
        if node >= FIRST {
            // a compressed node's label runs on past the located gram: the walk's first emission
            let label = model.g.label(node).to_string();
            opening.push(enc.slice(&label, offset + enc.n, usize::MAX));
        }
        for piece in opening {
            if !piece.is_empty() {
                parts.push(piece.clone());
                let chunk = speaker.feed(&piece);
                if !chunk.is_empty() {
                    emit(&chunk);
                }
            }
        }
        // from START nothing precedes the first node stepped onto, so it is said whole;
        // every node after it - and every node after a located prefix - adds what lies
        // past the overlap (the rule `decode_path` decodes a path by)
        let overlap = enc.overlap();
        let mut whole_first = node < FIRST;
        let mut listener = |c: usize, label: &str| {
            if c >= FIRST {
                let cut = if whole_first { 0 } else { overlap };
                whole_first = false;
                let piece = enc.slice(label, cut, usize::MAX);
                if !piece.is_empty() {
                    parts.push(piece.clone());
                    let chunk = speaker.feed(&piece);
                    if !chunk.is_empty() {
                        emit(&chunk);
                    }
                }
            } else if c == END {
                let chunk = speaker.end();
                if !chunk.is_empty() {
                    emit(&chunk);
                }
            }
        };
        let walk = model.g.sample_walk_listening(
            node,
            offset,
            o.max_length,
            o.temperature,
            rng.as_mut(),
            None,
            None,
            Traversal::Reward,
            SamplingFilter::default(),
            Some(&mut listener),
        )?;
        if !walk.reached_end {
            // cut off by the length: the sentinel is ours to send
            let chunk = speaker.end();
            if !chunk.is_empty() {
                emit(&chunk);
            }
        }
        let refs: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        let text = enc.join(&refs);
        let spelled = enc.spell(&text);
        said(i, &text, &spelled);
    }
    Ok(())
}

// -- the output decoder --------------------------------------------------------------

/// How the output decoder speaks: the voice `speak_walks` walks with, and for
/// acoustic units how many Griffin-Lim iterations polish each whole utterance
/// (0: the streaming vocoder's output as it is).
#[derive(Clone, Debug)]
pub struct SayOptions {
    pub rate: u32,
    pub pitch: f64,
    pub tempo: f64,
    pub gain: f64,
    pub polish: usize,
}

impl Default for SayOptions {
    fn default() -> Self {
        SayOptions {
            rate: RATE,
            pitch: 120.0,
            tempo: 1.0,
            gain: 0.5,
            polish: 0,
        }
    }
}

/// One text of a model spoken: what it said, what that spells, and how much
/// audio it made.
#[derive(Clone, Debug, PartialEq)]
pub struct Utterance {
    /// The text as the model wrote it, in its units.
    pub text: String,
    /// The words a text of sounds spells; the text itself for every other unit.
    pub spelled: String,
    /// Every token that reached the voice, the closing sentinel last.
    pub tokens: Vec<String>,
    pub samples: usize,
    pub seconds: f64,
}

impl Utterance {
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("text", Json::str(&self.text)),
            ("spelled", Json::str(&self.spelled)),
            ("tokens", Json::strs(self.tokens.clone())),
            ("samples", Json::Int(self.samples as i64)),
            ("seconds", Json::Num(self.seconds)),
        ])
    }
}

/// The speech of one or more texts: 16-bit mono PCM, and what each utterance was.
#[derive(Clone, Debug)]
pub struct Spoken {
    pub pcm: Vec<u8>,
    pub rate: u32,
    /// The encoding the texts were read in.
    pub encoding: String,
    /// `"voice"`: the formant synthesizer; `"vocoder"`: the acoustic codebook's vocoder.
    pub decoder: &'static str,
    pub utterances: Vec<Utterance>,
}

impl Spoken {
    pub fn samples(&self) -> usize {
        self.pcm.len() / 2
    }

    pub fn seconds(&self) -> f64 {
        if self.rate == 0 {
            0.0
        } else {
            self.samples() as f64 / self.rate as f64
        }
    }

    /// The speech as a WAV file.
    pub fn wav(&self) -> Vec<u8> {
        phonetok::synth::wav_bytes(&self.pcm, self.rate)
    }

    /// The record without the audio: what the CLI prints and the API returns beside `wav_base64`.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("rate", Json::Int(self.rate as i64)),
            ("samples", Json::Int(self.samples() as i64)),
            ("seconds", Json::Num(self.seconds())),
            ("encoding", Json::str(&self.encoding)),
            ("decoder", Json::str(self.decoder)),
            ("count", Json::Int(self.utterances.len() as i64)),
            (
                "utterances",
                Json::Arr(self.utterances.iter().map(Utterance::to_json).collect()),
            ),
        ])
    }
}

/// What speaks a model's texts: the codebook's vocoder for acoustic units, the
/// formant voice for the rest.
pub fn decoder_name(enc: Encoding) -> &'static str {
    if enc.unit == Unit::Acoustic {
        "vocoder"
    } else {
        "voice"
    }
}

/// The words a text spells: through the tokenizer for a model of sounds (so
/// `"the cat"` given to a phone model spells `"the cat"` too), the text itself
/// for every other unit.
pub fn spelled(enc: Encoding, text: &str) -> String {
    if !enc.unit.phonetic() {
        return text.to_string();
    }
    enc.spell(&radixnet::phonetic::text(enc.unit, text))
}

/// A text of acoustic units may hold nothing but units of the codebook, as the
/// Python vocoder insists (the `Speaker` skips such a token, which a model
/// over its units never emits; a text given from outside can hold anything).
fn check_units(enc: Encoding, text: &str) -> Result<(), String> {
    if enc.unit != Unit::Acoustic {
        return Ok(());
    }
    let tok = radixnet::phonetic::acoustic_tokenizer()?;
    match text.split_whitespace().find(|u| !tok.is_unit(u)) {
        Some(u) => Err(format!("not a unit of this codebook: {u:?}")),
        None => Ok(()),
    }
}

/// The output decoder in its streaming form: every text - a prediction, a
/// sample, a turn - is fed whole to a [`Speaker`] for the encoding and closed
/// by the END sentinel, exactly as a walk is, so a text of sounds is spoken
/// directly, a text of acoustic units through the codebook's vocoder, and a
/// text of words or letters is read through the tokenizer word by word.
/// `emit` gets every chunk of PCM as it is made, `said` each utterance once it
/// has been spoken.
pub fn speak_texts(
    enc: Encoding,
    texts: &[String],
    o: &SayOptions,
    emit: &mut dyn FnMut(&[u8]),
    said: &mut dyn FnMut(usize, Utterance),
) -> Result<(), String> {
    for (i, text) in texts.iter().enumerate() {
        check_units(enc, text)?;
        let mut speaker = Speaker::new(enc, o.rate, o.pitch, o.tempo, o.gain)?;
        let mut made = 0usize;
        for chunk in [speaker.feed(text), speaker.end()] {
            if !chunk.is_empty() {
                made += chunk.len();
                emit(&chunk);
            }
        }
        said(
            i,
            Utterance {
                text: text.clone(),
                spelled: spelled(enc, text),
                tokens: speaker.tokens.clone(),
                samples: made / 2,
                seconds: made as f64 / 2.0 / speaker.rate as f64,
            },
        );
    }
    Ok(())
}

/// The output decoder: texts in a model's units become speech, one utterance
/// each, through the same voice [`speak_walks`] walks with and closed by
/// the same rule, the END sentinel.  `o.polish` applies to acoustic units only:
/// that many Griffin-Lim iterations over each whole utterance once it is
/// known, which the streaming vocoder cannot do.
pub fn say(enc: Encoding, texts: &[String], o: &SayOptions) -> Result<Spoken, String> {
    let rate = radixnet::phonetic::output_rate(enc, o.rate)?;
    let mut spoken = Spoken {
        pcm: Vec::new(),
        rate,
        encoding: enc.to_string(),
        decoder: decoder_name(enc),
        utterances: Vec::new(),
    };
    if spoken.decoder == "vocoder" && o.polish > 0 {
        // the whole utterance is known, so it can be polished: the vocoder's gain
        // convention is the Speaker's (the voice's half scale is the codebook's own level)
        let tok = radixnet::phonetic::acoustic_tokenizer()?;
        for text in texts {
            check_units(enc, text)?;
            let units: Vec<String> = text.split_whitespace().map(String::from).collect();
            let pcm = phonetok::acoustic::synthesize(&units, &tok.book, o.polish, o.gain * 2.0, o.pitch)?;
            let mut tokens = units;
            tokens.push("</s>".to_string());
            spoken.utterances.push(Utterance {
                text: text.clone(),
                spelled: text.clone(),
                tokens,
                samples: pcm.len() / 2,
                seconds: pcm.len() as f64 / 2.0 / rate as f64,
            });
            spoken.pcm.extend(pcm);
        }
        return Ok(spoken);
    }
    let mut pcm: Vec<u8> = Vec::new();
    let mut utterances: Vec<Utterance> = Vec::new();
    speak_texts(
        enc,
        texts,
        o,
        &mut |chunk: &[u8]| pcm.extend_from_slice(chunk),
        &mut |_, u| utterances.push(u),
    )?;
    spoken.pcm = pcm;
    spoken.utterances = utterances;
    Ok(spoken)
}

#[cfg(test)]
mod tests {
    use super::*;
    use radixnet::encoding::parse_encoding;
    use radixnet::graph::GraphOptions;
    use radixnet::model::{GenerateOptions, TrainOptions};

    fn spoken_model(spec: &str) -> Model {
        let opts = GraphOptions {
            encoding: parse_encoding(spec).unwrap(),
            ..GraphOptions::default()
        };
        let mut model = Model::new(1, opts).unwrap();
        model.workers = 1;
        model.g.workers = 1;
        let texts: Vec<String> = [
            "the cat sat on the mat",
            "the cat sat on the floor",
            "the dog sat on the mat",
            "a bird in the hand",
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
        model
    }

    /// The acoustic unit (D-082): a recording is heard as a text of learned
    /// units, a model learns from such texts alone, and is heard back through
    /// the vocoder.
    #[test]
    fn acoustic_units_are_heard_and_spoken() {
        use phonetok::synth::wav_bytes;
        use radixnet::phonetic::{hear_audio, is_audio_file, output_rate};

        let enc = parse_encoding("acoustic:3:1").unwrap();
        assert_eq!((enc.unit, enc.n, enc.stride), (Unit::Acoustic, 3, 1));
        assert_eq!(parse_encoding("units:2:2").unwrap().unit, Unit::Acoustic);
        assert!(parse_encoding("rune:3").is_err());
        assert!(!Unit::Acoustic.phonetic() && Unit::Acoustic.tokens());
        assert_eq!(
            (Unit::Acoustic.name(), Unit::Acoustic.units_name()),
            ("acoustic", "units")
        );
        assert_eq!(enc.units("q1  q2\nq3").len(), 3);
        assert_eq!(
            enc.encode("q1 q2 q3 q4"),
            vec!["q1 q2 q3".to_string(), "q2 q3 q4".to_string()]
        );
        assert_eq!(enc.join(&["q1 q2", "q3"]), "q1 q2 q3");
        assert!(enc.has_unit_prefix("q1 q2 q3", "q1 q2") && !enc.has_unit_prefix("q1 q22 q3", "q1 q2"));
        assert_eq!(enc.spell("q1 q2"), "q1 q2");
        assert!(is_audio_file("x.WAV") && !is_audio_file("x.txt"));
        let mut texts = Vec::new();
        for tokens in [
            "DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # M AE1 T",
            "DH AH0 # K AE1 T # S AE1 T # AA1 N # DH AH0 # F L AO1 R",
            "DH AH0 # D AO1 G # S AE1 T # AA1 N # DH AH0 # M AE1 T",
        ] {
            let tokens: Vec<String> = tokens.split_whitespace().map(String::from).collect();
            let pcm = Synthesizer::new(16000, VoiceSettings::default()).speak(&tokens);
            let text = hear_audio(&wav_bytes(&pcm, 16000)).unwrap();
            let units: Vec<&str> = text.split_whitespace().collect();
            assert!(units.len() > 10, "{text}");
            assert!(units.iter().all(|u| u.starts_with('q')));
            assert!(units.windows(2).all(|w| w[0] != w[1]), "a run survived");
            texts.push(text);
        }
        let opts = GraphOptions {
            encoding: enc,
            ..GraphOptions::default()
        };
        let mut model = Model::new(1, opts).unwrap();
        model.workers = 1;
        model.g.workers = 1;
        model
            .train(
                &texts,
                &TrainOptions {
                    epochs: 2,
                    ..Default::default()
                },
            )
            .unwrap();
        assert!(model.g.num_trigrams() > 10);
        let opts = SpeakOptions {
            prefix: String::new(),
            count: 2,
            max_length: Some(30),
            temperature: 1.0,
            seed: Some(1),
            rate: 8000,
            pitch: 120.0,
            tempo: 1.0,
            gain: 0.5,
        };
        let mut said = Vec::new();
        let mut pcm = 0usize;
        speak_walks(
            &mut model,
            &opts,
            &mut |chunk: &[u8]| pcm += chunk.len(),
            &mut |_, text, spelled| {
                assert_eq!(text, spelled);
                said.push(text.to_string())
            },
        )
        .unwrap();
        assert_eq!(said.len(), 2);
        assert!(pcm > 16000, "{pcm} bytes");
        assert!(said[0].split_whitespace().all(|u| u.starts_with('q')), "{}", said[0]);
        let mut speaker = Speaker::new(enc, 8000, 120.0, 1.0, 0.5).unwrap();
        assert_eq!(speaker.rate, 16000);
        assert!(!speaker.feed("q2 q28").is_empty());
        assert!(!speaker.end().is_empty());
        assert_eq!(speaker.tokens, vec!["q2", "q28", "</s>"]);
        assert_eq!(output_rate(enc, 8000).unwrap(), 16000);
        assert_eq!(output_rate(Encoding::default(), 8000).unwrap(), 8000);
    }

    /// The output decoder: hearing a walk as it walks and saying its text
    /// afterwards are the same audio, byte for byte; acoustic units are spoken
    /// through the vocoder, polished or not, and refused when one is not a unit.
    #[test]
    fn say_is_the_walk_heard_after_the_fact() {
        for spec in ["phone:3:1", "char:3:1", "word:2:1"] {
            let mut model = spoken_model(spec);
            let enc = model.g.enc;
            let opts = SpeakOptions {
                prefix: "the cat".to_string(),
                count: 2,
                max_length: Some(30),
                temperature: 1.0,
                seed: Some(1),
                rate: 16000,
                pitch: 120.0,
                tempo: 1.0,
                gain: 0.5,
            };
            let mut said: Vec<String> = Vec::new();
            let mut live: Vec<u8> = Vec::new();
            speak_walks(
                &mut model,
                &opts,
                &mut |chunk: &[u8]| live.extend_from_slice(chunk),
                &mut |_, text, _| said.push(text.to_string()),
            )
            .unwrap();
            let spoken = say(enc, &said, &SayOptions::default()).unwrap();
            assert_eq!(spoken.pcm, live, "{spec}");
            assert_eq!(
                (
                    spoken.utterances.len(),
                    spoken.decoder,
                    spoken.rate,
                    spoken.encoding.as_str()
                ),
                (2, "voice", 16000, spec)
            );
            for u in &spoken.utterances {
                assert_eq!(u.tokens.last().map(String::as_str), Some("</s>"), "{spec}: {u:?}");
                assert_eq!(u.seconds, u.samples as f64 / 16000.0);
            }
            assert_eq!(
                spoken.samples(),
                spoken.utterances.iter().map(|u| u.samples).sum::<usize>()
            );
            assert_eq!(spoken.seconds(), spoken.samples() as f64 / 16000.0);
            let wav = spoken.wav();
            assert_eq!((wav.len(), &wav[..4]), (44 + spoken.pcm.len(), b"RIFF".as_slice()));
            let doc = spoken.to_json();
            assert_eq!(doc.at("count").as_i64(), Some(2));
            assert_eq!(doc.at("decoder").as_str(), Some("voice"));
            assert_eq!(doc.at("utterances").as_array().len(), 2);
        }
        // what a text spells: a text of sounds through the tokenizer, anything else as it is
        let phones = parse_encoding("phone:3:1").unwrap();
        assert_eq!(spelled(phones, "DH AH0 # K AE1 T"), "the cat");
        assert_eq!(spelled(phones, "the cat"), "the cat");
        assert_eq!(spelled(Encoding::default(), "DH AH0"), "DH AH0");
        // acoustic units: the codebook's vocoder, at its rate
        let acoustic = parse_encoding("acoustic:3:1").unwrap();
        let units = vec!["q2 q28 q55 q5".to_string(), "q1 q2".to_string()];
        let spoken = say(acoustic, &units, &SayOptions::default()).unwrap();
        assert_eq!(
            (spoken.decoder, spoken.rate, spoken.utterances.len()),
            ("vocoder", 16000, 2)
        );
        assert!(spoken.samples() > 0);
        assert_eq!(spoken.utterances[1].tokens, vec!["q1", "q2", "</s>"]);
        assert_eq!(spoken.utterances[1].spelled, "q1 q2");
        let polish = SayOptions {
            polish: 4,
            ..SayOptions::default()
        };
        let polished = say(acoustic, &units, &polish).unwrap();
        assert_eq!(polished.samples(), spoken.samples());
        assert_ne!(polished.pcm, spoken.pcm);
        let nope = vec!["q2 nope".to_string()];
        for opts in [SayOptions::default(), polish] {
            let err = say(acoustic, &nope, &opts).expect_err("a token that is not a unit was spoken");
            assert!(err.contains("not a unit"), "{err}");
        }
    }

    /// What is spoken is what the walk says: from START (the first node whole)
    /// and from a prefix, the utterance is the text a seeded sample of the same
    /// walk generates.
    #[test]
    fn what_is_spoken_is_what_the_walk_says() {
        for spec in ["phone:3:1", "char:3:1", "word:2:1"] {
            let mut model = spoken_model(spec);
            let enc = model.g.enc;
            for prefix in ["", "the cat"] {
                for seed in 1..=3 {
                    let opts = SpeakOptions {
                        prefix: prefix.to_string(),
                        count: 2,
                        max_length: Some(40),
                        temperature: 1.0,
                        seed: Some(seed),
                        rate: 16000,
                        pitch: 120.0,
                        tempo: 1.0,
                        gain: 0.5,
                    };
                    let mut said: Vec<String> = Vec::new();
                    let mut pcm = 0usize;
                    speak_walks(
                        &mut model,
                        &opts,
                        &mut |chunk: &[u8]| pcm += chunk.len(),
                        &mut |_, text, _| said.push(text.to_string()),
                    )
                    .unwrap();
                    let walks = model
                        .generate(&GenerateOptions {
                            max_length: 40,
                            mode: "sample".to_string(),
                            temperature: 1.0,
                            count: 2,
                            seed: Some(seed),
                            prefix: prefix.to_string(),
                            step_penalty: 0.0,
                            beam: 0,
                            traversal: "reward".to_string(),
                            penalty_scale: 1.0,
                            merit_scale: 1.0,
                            top_k: 0,
                            top_p: 1.0,
                            min_p: 0.0,
                            diversity: 0.0,
                        })
                        .unwrap();
                    assert_eq!((said.len(), walks.len()), (2, 2), "{spec} {prefix:?} {seed}");
                    assert!(pcm > 0, "{spec} {prefix:?} {seed}: nothing spoken");
                    for (spoken, walk) in said.iter().zip(&walks) {
                        if prefix.is_empty() {
                            assert_eq!(enc.truncate(spoken, 40), walk.text, "{spec} seed {seed}");
                        } else {
                            assert!(
                                spoken.starts_with(&walk.text),
                                "{spec} {prefix:?} {seed}: {spoken:?} vs {:?}",
                                walk.text
                            );
                        }
                    }
                }
            }
        }
    }
}
