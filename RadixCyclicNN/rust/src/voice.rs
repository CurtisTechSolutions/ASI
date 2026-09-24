//! Speech from a walk: the model is heard as it traverses its graph (the port
//! of `radixnet/voice.py`).  A model whose symbols are sounds emits phones as
//! it walks; a model of words or letters emits text.  [`Speaker`] takes
//! either, one step at a time, turns it into the tokens of the phonetic
//! tokenizer and feeds the formant synthesizer, which hands back 16-bit PCM as
//! soon as a word can be committed.  The utterance is closed by the final
//! sentinel: when the walk steps onto END, [`Speaker::end`] flushes what is
//! pending with the closing intonation.  One walk, one utterance.

use phonetok::synth::{Synthesizer, VoiceSettings};

use crate::encoding::{Encoding, Unit};
use crate::graph::{END, FIRST};
use crate::model::Model;
use crate::mt19937::Mt19937;
use crate::search::{SamplingFilter, Traversal};

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
    letters: String,
    spoken_words: usize,
    /// Every token that reached the synthesizer, for the record.
    pub tokens: Vec<String>,
}

impl Speaker {
    pub fn new(enc: Encoding, rate: u32, pitch: f64, tempo: f64, gain: f64) -> Result<Speaker, String> {
        crate::phonetic::tokenizer(Unit::Phones)?; // the words of a word or letter model are read through it
        let voice = VoiceSettings { pitch, tempo, gain, ..VoiceSettings::default() };
        Ok(Speaker { enc, synth: Synthesizer::new(rate, voice), letters: String::new(), spoken_words: 0, tokens: Vec::new() })
    }

    /// One emitted piece (the units a step added); the PCM that is ready comes back.
    pub fn feed(&mut self, piece: &str) -> Vec<u8> {
        if self.enc.unit.phonetic() {
            let tokens: Vec<String> = crate::phonetic::text(self.enc.unit, piece).split_whitespace().map(String::from).collect();
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
        let Ok(bridge) = crate::phonetic::tokenizer(Unit::Phones) else { return out };
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
        let mut out = self.flush_letters();
        self.tokens.push("</s>".to_string());
        out.extend(self.synth.end());
        self.spoken_words = 0;
        out
    }
}

impl Model {
    /// Walks the model `count` times from `prefix` and speaks each walk as it
    /// goes: `emit` gets every chunk of PCM the moment it is made, `said` is
    /// told what each walk said once it has ended - the text in the model's
    /// units and the words it spells.
    pub fn speak_walks(
        &mut self,
        o: &SpeakOptions,
        emit: &mut dyn FnMut(&[u8]),
        said: &mut dyn FnMut(usize, &str, &str),
    ) -> Result<(), String> {
        let enc = self.g.enc;
        let mut rng = o.seed.map(Mt19937::new);
        for i in 0..o.count {
            let mut speaker = Speaker::new(enc, o.rate, o.pitch, o.tempo, o.gain)?;
            let mut parts: Vec<String> = Vec::new();
            let (node, offset, lead) = self.prefix_start(&o.prefix);
            let mut opening: Vec<String> = Vec::new();
            if !o.prefix.is_empty() {
                opening.push(o.prefix.clone());
            }
            if !lead.is_empty() {
                opening.push(lead);
            }
            if node >= FIRST {
                // a compressed node's label runs on past the located gram: the walk's first emission
                let label = self.g.label(node).to_string();
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
            let walk = self.g.sample_walk_listening(
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::parse_encoding;
    use crate::graph::GraphOptions;
    use crate::model::{GenerateOptions, TrainOptions};

    fn spoken_model(spec: &str) -> Model {
        let opts = GraphOptions { encoding: parse_encoding(spec).unwrap(), ..GraphOptions::default() };
        let mut model = Model::new(1, opts).unwrap();
        model.workers = 1;
        model.g.workers = 1;
        let texts: Vec<String> =
            ["the cat sat on the mat", "the cat sat on the floor", "the dog sat on the mat", "a bird in the hand"]
                .iter()
                .map(|s| s.to_string())
                .collect();
        model.train(&texts, &TrainOptions { epochs: 2, ..Default::default() }).unwrap();
        model
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
                    model
                        .speak_walks(&opts, &mut |chunk: &[u8]| pcm += chunk.len(), &mut |_, text, _| {
                            said.push(text.to_string())
                        })
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
                            assert!(spoken.starts_with(&walk.text), "{spec} {prefix:?} {seed}: {spoken:?} vs {:?}", walk.text);
                        }
                    }
                }
            }
        }
    }
}
