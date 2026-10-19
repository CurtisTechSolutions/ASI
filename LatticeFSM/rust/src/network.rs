//! Networks of machines - lattices connected like the layers of a neural network.
//!
//! A machine reads a string and walks its matrix; a layer makes that a block that also *writes* one: by default
//! the state each move lands in, one symbol per move, so the next block reads the walk the previous one took.
//! The symbols a layer writes are its states' letters ([`SYMBOLS`]: `a` for state 0, `b` for 1, ..., then `A`-`Z`
//! and `0`-`9`), and the next machine is built over exactly that alphabet.
//!
//! Blocks compose, as in Keras: a [`Spec::Layer`] is one machine (`Emit::States` writes every state the walk
//! lands in, `Emit::Final` only the last); a [`Spec::Sequential`] runs blocks one after another; a
//! [`Spec::Parallel`] runs blocks side by side on the same input, merges what they write (each branch's letters
//! shifted past the branches before it) and, as the network's last block, votes on their verdicts.
//! [`Spec::Encoder`] reads English as the sounds it is made of (the repository's phonetic tokenizer,
//! `phonetic.rs`), as a network's first block; [`Spec::Decoder`] is a layer whose states are sounds, so what it
//! writes is phonemes, spelled back as English - with both, `"enc>13>dec"` translates English to English.  Blocks nest,
//! so any series-parallel graph is a network.  [`Network`] is the model: `compile`, `fit`, `evaluate`, `predict`,
//! `summary`, `save` and [`load_network`]; `"13>(13|13)>13"` writes one as a string ([`parse_architecture`]).
//!
//! Training is the single machine's, spread over the layers: every machine walks, the network's verdict is the
//! last block's, it is rewarded when right and punished when wrong, and the credit travels back through the
//! blocks - the last in full, each block before it `layer_discount` times the one after, every branch of a
//! parallel block the block's share (or, with `Credit::Own`, the output layers each by their own verdict).  A
//! network of one layer is exactly the machine it holds.  `latticefsm/network.py` is the same model; the two read
//! and write the same `latticefsm-network` files.

use std::collections::HashMap;
use std::fs;
use std::io::{Read, Write};

use phonetok::Tokenizer;

use crate::gzip;
use crate::json::{parse, Json};
use crate::languages::{random_string, Language};
use crate::machine::{Machine, Settings};
use crate::phonetic::{inventory, tokenizer};
use crate::rng::Rng;

/// The letters blocks write: state `i` is the `i`-th.  A block writes at most 62 distinct symbols.
pub const SYMBOLS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
/// How much of the credit each block further back from the output receives.
pub const LAYER_DISCOUNT: f64 = 0.8;

const FORMAT: &str = "latticefsm-network";
const FORMAT_VERSION: f64 = 1.0;

/// The `i`-th letter of [`SYMBOLS`].
pub fn symbol(i: usize) -> char {
    SYMBOLS.as_bytes()[i] as char
}

/// The first `n` letters of [`SYMBOLS`]: what a block with `n` distinct outputs writes.
pub fn state_symbols(n: usize) -> Result<Vec<String>, String> {
    if !(1..=SYMBOLS.len()).contains(&n) {
        return Err(format!(
            "a block writes 1 to {} distinct symbols, not {n}",
            SYMBOLS.len()
        ));
    }
    Ok(SYMBOLS[..n].chars().map(|c| c.to_string()).collect())
}

macro_rules! choice {
    ($name:ident { $($variant:ident => $text:literal),+ $(,)? }) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub enum $name { $($variant),+ }
        impl $name {
            pub fn parse(text: &str) -> Result<$name, String> {
                match text {
                    $($text => Ok($name::$variant),)+
                    other => Err(format!(concat!(stringify!($name), " must be one of {:?}, got {:?}"),
                                         [$($text),+], other)),
                }
            }
            pub fn as_str(self) -> &'static str {
                match self { $($name::$variant => $text),+ }
            }
        }
    };
}

choice!(Emit { States => "states", Final => "final" });
choice!(Merge { Concat => "concat", Interleave => "interleave" });
choice!(Vote { Majority => "majority", Any => "any", All => "all" });
choice!(Credit { Shared => "shared", Own => "own" });

/// A network before it is built: what each block is, not yet over which alphabet.
#[derive(Clone, Debug)]
pub enum Spec {
    Layer {
        states: usize,
        emit: Emit,
        name: Option<String>,
        accepting: Vec<usize>,
        settings: Settings,
        /// The machine's seed; `None` is the network's seed plus the layer's index, depth first.
        seed: Option<u64>,
    },
    Sequential {
        blocks: Vec<Spec>,
        name: Option<String>,
    },
    Parallel {
        branches: Vec<Spec>,
        merge: Merge,
        vote: Vote,
        name: Option<String>,
    },
    /// What the network reads, turned into symbols ([`Source`]): only a network's first block.
    Encoder {
        source: Source,
        name: Option<String>,
    },
    /// English out: a layer whose states are the sounds, starting from `#`, skipping nothing, never rearranged.
    Decoder {
        stress: bool,
        name: Option<String>,
        accepting: Vec<usize>,
        settings: Settings,
        seed: Option<u64>,
    },
}

impl Spec {
    /// A layer of `states` states writing every state it lands in, with the machine's default settings.
    pub fn layer(states: usize) -> Spec {
        Spec::Layer {
            states,
            emit: Emit::States,
            name: None,
            accepting: Vec::new(),
            settings: Settings::default(),
            seed: None,
        }
    }
}

/// One machine as a block.  A decoder is a layer with `labels`: its states' sounds.
pub struct Layer {
    pub name: String,
    pub states: usize,
    pub emit: Emit,
    pub machine: Machine,
    /// A decoder's sounds, state by state; `None` for a layer, which writes [`SYMBOLS`].
    pub labels: Option<Vec<String>>,
    pub stress: bool,
    last_accepted: bool,
    /// The latest pass's moves: `(source, symbol, target)`.
    last_moves: Vec<(usize, usize, usize)>,
}

impl Layer {
    fn written(&self) -> Vec<String> {
        match &self.labels {
            Some(l) => l.clone(),
            None => state_symbols(self.states).expect("checked when built"),
        }
    }

    pub fn is_decoder(&self) -> bool {
        self.labels.is_some()
    }

    /// Teach the latest pass its target sounds (padded with `#` to its length, or cut to it): every sound written
    /// wrong is punished, then the target's path is taught edge by edge from `#`.  Returns (right, of).
    fn teach_targets(&mut self, targets: &[usize], reward: f64, punish: f64) -> (usize, usize) {
        let m = &mut self.machine;
        let n = self.last_moves.len();
        let mut want: Vec<usize> = targets.to_vec();
        want.resize(want.len().max(n), 0);
        want.truncate(n);
        let mut right = 0;
        for (&(s, a, t), &w) in self.last_moves.iter().zip(&want) {
            if t == w {
                right += 1;
            } else {
                let i = m.lattice.offset(s, a, t).expect("a move the machine made");
                let (clock, life, widen, narrow) = (m.clock, m.life, m.reward_widening, m.punish_narrowing);
                m.lattice.edges[i].credit(-punish, clock, life, widen, narrow);
            }
        }
        let mut prev = 0;
        for (&(_, a, _), &w) in self.last_moves.iter().zip(&want) {
            m.teach(prev, a, w, reward).expect("a decoder's states and symbols");
            prev = w;
        }
        (right, n)
    }
}

/// What an encoder reads (`media.rs`, `phonetic.rs`; the Python port's encoder classes).
#[derive(Clone, Debug, PartialEq)]
pub enum Source {
    /// English as the sounds it is made of, then `pad` boundaries `#`.
    Phonetic { stress: bool, pad: usize },
    /// Text, character by character.
    Text,
    /// A WAV file as acoustic units.
    Audio,
    /// An image as a `grid x grid` of `levels` brightness levels.
    Image { grid: usize, levels: usize, order: String },
    /// Frames as their differences.
    Diff { grid: usize, order: String, frames: usize },
    /// A video: keyframes, differences and audio units.
    Video {
        grid: usize,
        levels: usize,
        order: String,
        keyframe: usize,
        frames: usize,
        audio: bool,
        fps: usize,
    },
}

impl Source {
    /// Its name in the architecture string and the file.
    pub fn kind(&self) -> &'static str {
        match self {
            Source::Phonetic { .. } => "phonetic",
            Source::Text => "text",
            Source::Audio => "audio",
            Source::Image { .. } => "image",
            Source::Diff { .. } => "diff",
            Source::Video { .. } => "video",
        }
    }

    /// The defaults of a kind.
    pub fn of(kind: &str) -> Result<Source, String> {
        Ok(match kind {
            "phonetic" | "enc" => Source::Phonetic { stress: false, pad: 2 },
            "text" => Source::Text,
            "audio" => Source::Audio,
            "image" => Source::Image {
                grid: 8,
                levels: 4,
                order: "rows".into(),
            },
            "diff" => Source::Diff {
                grid: 8,
                order: "rows".into(),
                frames: 16,
            },
            "video" => Source::Video {
                grid: 8,
                levels: 4,
                order: "rows".into(),
                keyframe: 0,
                frames: 16,
                audio: true,
                fps: 8,
            },
            other => {
                return Err(format!(
                    "no encoder {other:?}: phonetic, text, audio, image, diff or video"
                ))
            }
        })
    }

    /// Every symbol it can write.
    pub fn labels(&self) -> Vec<String> {
        use crate::media::{level_symbols, unit_names, DIFFS, TEXT_SYMBOLS};
        match self {
            Source::Phonetic { stress, .. } => inventory(*stress),
            Source::Text => TEXT_SYMBOLS.iter().map(|s| s.to_string()).collect(),
            Source::Audio => unit_names(),
            Source::Image { levels, .. } => level_symbols(*levels),
            Source::Diff { .. } => std::iter::once("|").chain(DIFFS).map(String::from).collect(),
            Source::Video { levels, audio, .. } => {
                let mut v = vec!["|".to_string()];
                v.extend(level_symbols(*levels));
                v.extend(DIFFS.iter().map(|s| s.to_string()));
                if *audio {
                    v.extend(unit_names());
                }
                v
            }
        }
    }

    /// The phonetic encoder's stress setting; `None` for the others.
    pub fn stress(&self) -> Option<bool> {
        match self {
            Source::Phonetic { stress, .. } => Some(*stress),
            _ => None,
        }
    }

    fn describe(&self, n: usize) -> String {
        match self {
            Source::Phonetic { .. } => format!("English -> {n} sounds"),
            Source::Text => format!("text -> {n} characters"),
            Source::Audio => format!("audio -> {n} units"),
            Source::Image { grid, levels, .. } => format!("image {grid}x{grid} -> {levels} levels"),
            Source::Diff { grid, .. } => format!("frames {grid}x{grid} -> {} changes", n - 1),
            Source::Video { grid, audio, .. } => {
                format!(
                    "video {grid}x{grid}{} -> {n} symbols",
                    if *audio { " + audio" } else { "" }
                )
            }
        }
    }

    fn to_json(&self, j: Json) -> Json {
        match self {
            Source::Phonetic { stress, pad } => j.with("stress", (*stress).into()).with("pad", (*pad).into()),
            Source::Text | Source::Audio => j,
            Source::Image { grid, levels, order } => j
                .with("grid", (*grid).into())
                .with("levels", (*levels).into())
                .with("order", order.as_str().into()),
            Source::Diff { grid, order, frames } => j
                .with("grid", (*grid).into())
                .with("order", order.as_str().into())
                .with("frames", (*frames).into()),
            Source::Video {
                grid,
                levels,
                order,
                keyframe,
                frames,
                audio,
                fps,
            } => j
                .with("grid", (*grid).into())
                .with("levels", (*levels).into())
                .with("order", order.as_str().into())
                .with("keyframe", (*keyframe).into())
                .with("frames", (*frames).into())
                .with("audio", (*audio).into())
                .with("fps", (*fps).into()),
        }
    }

    fn from_json(v: &Json) -> Result<Source, String> {
        let n = |k: &str, d: f64| v.num(k, d) as usize;
        let order = v.str_or("order", "rows").to_string();
        Ok(match v.str_or("source", "phonetic") {
            "phonetic" => Source::Phonetic {
                stress: v.bool_or("stress", false),
                pad: n("pad", 2.0),
            },
            "text" => Source::Text,
            "audio" => Source::Audio,
            "image" => Source::Image {
                grid: n("grid", 8.0),
                levels: n("levels", 4.0),
                order,
            },
            "diff" => Source::Diff {
                grid: n("grid", 8.0),
                order,
                frames: n("frames", 16.0),
            },
            "video" => Source::Video {
                grid: n("grid", 8.0),
                levels: n("levels", 4.0),
                order,
                keyframe: n("keyframe", 0.0),
                frames: n("frames", 16.0),
                audio: v.bool_or("audio", true),
                fps: n("fps", 8.0),
            },
            other => return Err(format!("unknown encoder source {other:?}")),
        })
    }

    /// One input's symbols, before they are checked against the labels (the phonetic one is done by the network,
    /// which holds the tokenizer).
    fn symbols_of(&self, x: &str) -> Result<Vec<String>, String> {
        use crate::media;
        match self {
            Source::Phonetic { .. } => Err("the network reads English itself".to_string()),
            Source::Text => Ok(media::text_symbols(x)),
            Source::Audio => media::audio_units(x),
            Source::Image { grid, levels, order } => {
                media::image_symbols(&media::cells(&media::read_image(x)?, *grid), *levels, order)
            }
            Source::Diff { grid, order, frames } => media::diff_stream(x, *grid, order, *frames),
            Source::Video {
                grid,
                levels,
                order,
                keyframe,
                frames,
                audio,
                fps,
            } => media::video_symbols(x, *grid, *levels, order, *keyframe, *frames, *audio, *fps),
        }
    }

    fn caches(&self) -> bool {
        !matches!(self, Source::Phonetic { .. } | Source::Text)
    }
}

/// What the network reads, turned into symbols: its first block, with no machine and no credit.
pub struct Encoder {
    pub name: String,
    pub source: Source,
    pub labels: Vec<String>,
    cache: crate::media::Cache,
}

impl Encoder {
    pub fn new(source: Source, name: String) -> Result<Encoder, String> {
        if let Source::Image { grid, levels, order }
        | Source::Video {
            grid, levels, order, ..
        } = &source
        {
            if !(1..=64).contains(grid) || !(2..=16).contains(levels) {
                return Err("grid is 1 to 64 and levels 2 to 16".to_string());
            }
            crate::media::order(*grid, order)?;
        }
        let labels = source.labels();
        Ok(Encoder {
            name,
            source,
            labels,
            cache: Default::default(),
        })
    }

    /// The phonetic encoder's stress setting; `None` for the others.
    pub fn stress(&self) -> Option<bool> {
        self.source.stress()
    }

    /// One input (a text, or a file's path) as this encoder's symbols - the media read once per file and kept.
    fn encode_media(&mut self, x: &str) -> Result<Vec<usize>, String> {
        if self.source.caches() {
            if let Some(hit) = self.cache.get(x) {
                return Ok(hit);
            }
        }
        let symbols = self.source.symbols_of(x)?;
        let out: Vec<usize> = symbols
            .iter()
            .filter_map(|t| self.labels.iter().position(|l| l == t))
            .collect();
        if self.source.caches() {
            self.cache.put(x, &out);
        }
        Ok(out)
    }
}

/// A built block.
pub enum Block {
    Encoder(Encoder),
    Layer(Box<Layer>),
    Sequential {
        name: String,
        blocks: Vec<Block>,
    },
    Parallel {
        name: String,
        merge: Merge,
        vote: Vote,
        branches: Vec<Block>,
        /// Where each branch's letters start in what the block writes.
        offsets: Vec<usize>,
    },
}

/// One layer's part of a pass: what it read, the states it walked, what it wrote (in its own letters or sounds),
/// and - for an encoder or a decoder - those sounds in English.
#[derive(Clone, Debug)]
pub struct Step {
    pub layer: String,
    pub read: String,
    pub states: Vec<usize>,
    pub wrote: String,
    pub accepted: bool,
    pub english: Option<String>,
    stress: Option<bool>,
    sounds: Vec<String>,
}

/// Symbols written out: run together when each is one character, else separated by spaces.
fn text_of(symbols: &[usize], names: &[String]) -> String {
    let parts: Vec<&str> = symbols.iter().map(|&s| names[s].as_str()).collect();
    let sep = if parts.iter().all(|p| p.chars().count() == 1) {
        ""
    } else {
        " "
    };
    parts.join(sep)
}

fn name_of(given: &Option<String>, kind: &str, names: &mut HashMap<String, usize>) -> String {
    match given {
        Some(n) if !n.is_empty() => n.clone(),
        _ => {
            let c = names.entry(kind.to_string()).or_insert(0);
            *c += 1;
            format!("{kind}_{c}")
        }
    }
}

fn wire(name: &str, widths: &[usize]) -> Result<Vec<usize>, String> {
    let total: usize = widths.iter().sum();
    if total > SYMBOLS.len() {
        return Err(format!(
            "parallel block {name}: its branches write {total} distinct symbols, more than {}",
            SYMBOLS.len()
        ));
    }
    let mut offsets = Vec::new();
    let mut at = 0;
    for w in widths {
        offsets.push(at);
        at += w;
    }
    Ok(offsets)
}

impl Block {
    fn build(
        spec: &Spec,
        alphabet: &[String],
        seed: u64,
        counter: &mut u64,
        names: &mut HashMap<String, usize>,
    ) -> Result<(Block, Vec<String>), String> {
        match spec {
            Spec::Encoder { source, name } => {
                let name = name_of(name, "encoder", names);
                let e = Encoder::new(source.clone(), name)?;
                let labels = e.labels.clone();
                Ok((Block::Encoder(e), labels))
            }
            Spec::Decoder {
                stress,
                name,
                accepting,
                settings,
                seed: own,
            } => {
                let name = name_of(name, "decoder", names);
                let labels = inventory(*stress);
                let mut settings = settings.clone();
                settings.seed = own.unwrap_or(seed + *counter);
                *counter += 1;
                settings.skip = false;
                settings.rearrange_every = 0;
                settings.focus = None;
                settings.learn_focus = false;
                settings.start = 0;
                let mut machine = Machine::new(labels.len(), alphabet, accepting, settings)?;
                machine.greedy_first = true;
                let layer = Layer {
                    name,
                    states: labels.len(),
                    emit: Emit::States,
                    machine,
                    labels: Some(labels.clone()),
                    stress: *stress,
                    last_accepted: false,
                    last_moves: Vec::new(),
                };
                Ok((Block::Layer(Box::new(layer)), labels))
            }
            Spec::Layer {
                states,
                emit,
                name,
                accepting,
                settings,
                seed: own,
            } => {
                let name = name_of(name, "lattice", names);
                if !(1..=SYMBOLS.len()).contains(states) {
                    return Err(format!("a layer has 1 to {} states, not {states}", SYMBOLS.len()));
                }
                let mut settings = settings.clone();
                settings.seed = own.unwrap_or(seed + *counter);
                *counter += 1;
                let mut machine = Machine::new(*states, alphabet, accepting, settings)?;
                machine.greedy_first = true;
                let layer = Layer {
                    name,
                    states: *states,
                    emit: *emit,
                    machine,
                    labels: None,
                    stress: false,
                    last_accepted: false,
                    last_moves: Vec::new(),
                };
                Ok((Block::Layer(Box::new(layer)), state_symbols(*states)?))
            }
            Spec::Sequential { blocks, name } => {
                if blocks.is_empty() {
                    return Err("a sequential block needs at least one block".to_string());
                }
                let name = name_of(name, "sequential", names);
                let mut alphabet = alphabet.to_vec();
                let mut built = Vec::new();
                for b in blocks {
                    let (block, written) = Block::build(b, &alphabet, seed, counter, names)?;
                    built.push(block);
                    alphabet = written;
                }
                Ok((Block::Sequential { name, blocks: built }, alphabet))
            }
            Spec::Parallel {
                branches,
                merge,
                vote,
                name,
            } => {
                if branches.len() < 2 {
                    return Err("a parallel block needs at least two branches".to_string());
                }
                let name = name_of(name, "parallel", names);
                let mut built = Vec::new();
                let mut widths = Vec::new();
                for b in branches {
                    let (block, written) = Block::build(b, alphabet, seed, counter, names)?;
                    built.push(block);
                    widths.push(written.len());
                }
                let offsets = wire(&name, &widths)?;
                let total: usize = widths.iter().sum();
                Ok((
                    Block::Parallel {
                        name,
                        merge: *merge,
                        vote: *vote,
                        branches: built,
                        offsets,
                    },
                    state_symbols(total)?,
                ))
            }
        }
    }

    /// How many distinct symbols the block writes.
    pub fn width(&self) -> usize {
        self.written().len()
    }

    /// The symbols the block writes.
    pub fn written(&self) -> Vec<String> {
        match self {
            Block::Encoder(e) => e.labels.clone(),
            Block::Layer(l) => l.written(),
            Block::Sequential { blocks, .. } => blocks.last().map_or(Vec::new(), Block::written),
            Block::Parallel { branches, .. } => {
                state_symbols(branches.iter().map(Block::width).sum()).unwrap_or_default()
            }
        }
    }

    pub fn name(&self) -> &str {
        match self {
            Block::Encoder(e) => &e.name,
            Block::Layer(l) => &l.name,
            Block::Sequential { name, .. } | Block::Parallel { name, .. } => name,
        }
    }

    fn forward(
        &mut self,
        symbols: &[usize],
        quiet: bool,
        temperature: Option<f64>,
        trace: &mut Option<&mut Vec<Step>>,
    ) -> (Vec<usize>, bool) {
        match self {
            Block::Encoder(e) => {
                if let Some(steps) = trace.as_deref_mut() {
                    steps.push(Step {
                        layer: e.name.clone(),
                        read: String::new(),
                        states: Vec::new(),
                        wrote: text_of(symbols, &e.labels),
                        accepted: false,
                        english: None,
                        stress: e.stress(),
                        sounds: symbols.iter().map(|&i| e.labels[i].clone()).collect(),
                    });
                }
                (symbols.to_vec(), false)
            }
            Block::Layer(l) => {
                let run = l.machine.run(symbols, None, temperature, quiet);
                let out: Vec<usize> = match l.emit {
                    Emit::States => run.transitions.iter().map(|t| t.target).collect(),
                    Emit::Final => vec![run.final_state],
                };
                l.last_accepted = run.accepted;
                l.last_moves = run
                    .transitions
                    .iter()
                    .zip(symbols)
                    .map(|(t, &a)| (t.source, a, t.target))
                    .collect();
                if let Some(steps) = trace.as_deref_mut() {
                    steps.push(Step {
                        layer: l.name.clone(),
                        read: text_of(symbols, l.machine.alphabet()),
                        states: run.states(),
                        wrote: text_of(&out, &l.written()),
                        accepted: run.accepted,
                        english: None,
                        stress: l.labels.as_ref().map(|_| l.stress),
                        sounds: match &l.labels {
                            Some(labels) => out.iter().map(|&i| labels[i].clone()).collect(),
                            None => Vec::new(),
                        },
                    });
                }
                (out, run.accepted)
            }
            Block::Sequential { blocks, .. } => {
                let mut symbols = symbols.to_vec();
                let mut accepted = false;
                for b in blocks.iter_mut() {
                    let (out, a) = b.forward(&symbols, quiet, temperature, trace);
                    symbols = out;
                    accepted = a;
                }
                (symbols, accepted)
            }
            Block::Parallel {
                merge,
                vote,
                branches,
                offsets,
                ..
            } => {
                let tracing = trace.is_some();
                let mut jobs: Vec<_> = branches
                    .iter_mut()
                    .zip(offsets.iter())
                    .map(|(b, &off)| (b, off, Vec::new(), false, Vec::new()))
                    .collect();
                // Thread startup is only worthwhile for substantial walks.
                let workers = if symbols.len() < 128 { 1 } else { 0 };
                crate::parallel::parallel_fill(&mut jobs, workers, |_, (b, off, out, accepted, steps)| {
                    let mut local_trace = tracing.then_some(&mut *steps);
                    let (written, yes) = b.forward(symbols, quiet, temperature, &mut local_trace);
                    *out = written.into_iter().map(|s| s + *off).collect();
                    *accepted = yes;
                });
                let mut outs = Vec::with_capacity(jobs.len());
                let mut yes = 0;
                for (_, _, out, accepted, steps) in jobs {
                    outs.push(out);
                    yes += accepted as usize;
                    if let Some(trace) = trace.as_deref_mut() {
                        trace.extend(steps);
                    }
                }
                let merged = match merge {
                    Merge::Concat => outs.concat(),
                    Merge::Interleave => {
                        let longest = outs.iter().map(Vec::len).max().unwrap_or(0);
                        (0..longest)
                            .flat_map(|i| outs.iter().filter_map(move |o| o.get(i).copied()))
                            .collect()
                    }
                };
                let n = branches.len();
                let accepted = match vote {
                    Vote::Majority => yes * 2 > n,
                    Vote::Any => yes > 0,
                    Vote::All => yes == n,
                };
                (merged, accepted)
            }
        }
    }

    fn credit(&mut self, amount: f64, layer_discount: f64, own: Option<(bool, f64, f64)>) -> usize {
        match self {
            Block::Encoder(_) => 0,
            Block::Layer(l) => {
                let amount = match own {
                    Some((label, reward, punish)) => {
                        if l.last_accepted == label {
                            reward
                        } else {
                            -punish
                        }
                    }
                    None => amount,
                };
                l.machine.credit(amount)
            }
            Block::Sequential { blocks, .. } => {
                let (mut amount, mut own, mut n) = (amount, own, 0);
                for b in blocks.iter_mut().rev() {
                    n += b.credit(amount, layer_discount, own);
                    amount *= layer_discount;
                    own = None;
                }
                n
            }
            Block::Parallel { branches, .. } => {
                branches.iter_mut().map(|b| b.credit(amount, layer_discount, own)).sum()
            }
        }
    }

    fn layers<'a>(&'a self, out: &mut Vec<&'a Layer>) {
        match self {
            Block::Encoder(_) => {}
            Block::Layer(l) => out.push(l),
            Block::Sequential { blocks, .. } => blocks.iter().for_each(|b| b.layers(out)),
            Block::Parallel { branches, .. } => branches.iter().for_each(|b| b.layers(out)),
        }
    }

    fn output_layers_mut<'a>(&'a mut self, out: &mut Vec<&'a mut Layer>) {
        match self {
            Block::Encoder(_) => {}
            Block::Layer(l) => out.push(l),
            Block::Sequential { blocks, .. } => {
                if let Some(b) = blocks.last_mut() {
                    b.output_layers_mut(out)
                }
            }
            Block::Parallel { branches, .. } => branches.iter_mut().for_each(|b| b.output_layers_mut(out)),
        }
    }

    fn output_layers<'a>(&'a self, out: &mut Vec<&'a Layer>) {
        match self {
            Block::Encoder(_) => {}
            Block::Layer(l) => out.push(l),
            Block::Sequential { blocks, .. } => {
                if let Some(b) = blocks.last() {
                    b.output_layers(out)
                }
            }
            Block::Parallel { branches, .. } => branches.iter().for_each(|b| b.output_layers(out)),
        }
    }

    pub fn to_json(&self) -> Json {
        match self {
            Block::Encoder(e) => e.source.to_json(
                Json::object()
                    .with("type", "encoder".into())
                    .with("source", e.source.kind().into())
                    .with("name", e.name.as_str().into()),
            ),
            Block::Layer(l) => {
                let mut j = Json::object()
                    .with("type", if l.is_decoder() { "decoder" } else { "layer" }.into())
                    .with("name", l.name.as_str().into())
                    .with("states", l.states.into())
                    .with("emit", l.emit.as_str().into())
                    .with("machine", l.machine.to_json());
                if let Some(labels) = &l.labels {
                    j = j.with("stress", l.stress.into()).with("labels", Json::strings(labels));
                }
                j
            }
            Block::Sequential { name, blocks } => Json::object()
                .with("type", "sequential".into())
                .with("name", name.as_str().into())
                .with("blocks", Json::Array(blocks.iter().map(Block::to_json).collect())),
            Block::Parallel {
                name,
                merge,
                vote,
                branches,
                ..
            } => Json::object()
                .with("type", "parallel".into())
                .with("name", name.as_str().into())
                .with("merge", merge.as_str().into())
                .with("vote", vote.as_str().into())
                .with("branches", Json::Array(branches.iter().map(Block::to_json).collect())),
        }
    }

    pub fn from_json(v: &Json) -> Result<Block, String> {
        let name = v.str_or("name", "").to_string();
        match v.str_or("type", "") {
            "encoder" => Ok(Block::Encoder(Encoder::new(Source::from_json(v)?, name)?)),
            kind @ ("layer" | "decoder") => {
                let states = v.num("states", 0.0) as usize;
                let mut machine = Machine::from_json(v.get("machine").ok_or("a layer without its machine")?)?;
                machine.greedy_first = true;
                if machine.n_states() != states {
                    return Err(format!(
                        "layer {name}: {states} states, but its machine has {}",
                        machine.n_states()
                    ));
                }
                let labels = if kind == "decoder" {
                    let labels: Vec<String> = v
                        .get("labels")
                        .and_then(Json::as_array)
                        .ok_or("a decoder without its labels")?
                        .iter()
                        .map(|s| s.as_str().map(String::from).ok_or("labels are strings"))
                        .collect::<Result<_, _>>()?;
                    if labels.len() != states {
                        return Err(format!("decoder {name}: {states} states, but {} labels", labels.len()));
                    }
                    Some(labels)
                } else {
                    None
                };
                Ok(Block::Layer(Box::new(Layer {
                    name,
                    states,
                    emit: Emit::parse(v.str_or("emit", "states"))?,
                    machine,
                    labels,
                    stress: v.bool_or("stress", false),
                    last_accepted: false,
                    last_moves: Vec::new(),
                })))
            }
            "sequential" => {
                let blocks = v
                    .get("blocks")
                    .and_then(Json::as_array)
                    .ok_or("a sequential block without blocks")?
                    .iter()
                    .map(Block::from_json)
                    .collect::<Result<Vec<_>, _>>()?;
                if blocks.is_empty() {
                    return Err("a sequential block needs at least one block".to_string());
                }
                Ok(Block::Sequential { name, blocks })
            }
            "parallel" => {
                let branches = v
                    .get("branches")
                    .and_then(Json::as_array)
                    .ok_or("a parallel block without branches")?
                    .iter()
                    .map(Block::from_json)
                    .collect::<Result<Vec<_>, _>>()?;
                if branches.len() < 2 {
                    return Err("a parallel block needs at least two branches".to_string());
                }
                let widths: Vec<usize> = branches.iter().map(Block::width).collect();
                let offsets = wire(&name, &widths)?;
                Ok(Block::Parallel {
                    name,
                    merge: Merge::parse(v.str_or("merge", "concat"))?,
                    vote: Vote::parse(v.str_or("vote", "majority"))?,
                    branches,
                    offsets,
                })
            }
            other => Err(format!("unknown block type {other:?}")),
        }
    }
}

/// `"13>(13|13)>13"`: `>` one block after another, `(a|b|...)` side by side, a number a layer of that many states
/// (`13f`: writing only its final state); `enc` an encoder and `dec` a decoder.  `settings` go to every layer's
/// machine.
pub fn parse_architecture(text: &str, merge: Merge, vote: Vote, settings: &Settings) -> Result<Spec, String> {
    let src: Vec<char> = text.chars().filter(|c| *c != ' ').collect();
    let mut pos = 0usize;
    let fail = |msg: &str, pos: usize| Err(format!("architecture {text:?}: {msg} at position {pos}"));

    fn seq(
        src: &[char],
        pos: &mut usize,
        merge: Merge,
        vote: Vote,
        settings: &Settings,
        text: &str,
    ) -> Result<Spec, String> {
        let mut blocks = vec![atom(src, pos, merge, vote, settings, text)?];
        while src.get(*pos) == Some(&'>') {
            *pos += 1;
            blocks.push(atom(src, pos, merge, vote, settings, text)?);
        }
        Ok(if blocks.len() == 1 {
            blocks.pop().expect("one")
        } else {
            Spec::Sequential { blocks, name: None }
        })
    }

    fn atom(
        src: &[char],
        pos: &mut usize,
        merge: Merge,
        vote: Vote,
        settings: &Settings,
        text: &str,
    ) -> Result<Spec, String> {
        if src.get(*pos) == Some(&'(') {
            *pos += 1;
            let mut alts = vec![seq(src, pos, merge, vote, settings, text)?];
            while src.get(*pos) == Some(&'|') {
                *pos += 1;
                alts.push(seq(src, pos, merge, vote, settings, text)?);
            }
            if src.get(*pos) != Some(&')') {
                return Err(format!("architecture {text:?}: expected ')' at position {pos}"));
            }
            *pos += 1;
            return Ok(if alts.len() == 1 {
                alts.pop().expect("one")
            } else {
                Spec::Parallel {
                    branches: alts,
                    merge,
                    vote,
                    name: None,
                }
            });
        }
        for word in ["enc", "dec", "text", "audio", "image", "diff", "video"] {
            let w: Vec<char> = word.chars().collect();
            if src.len() >= *pos + w.len() && src[*pos..*pos + w.len()] == w[..] {
                *pos += w.len();
                return Ok(if word != "dec" {
                    Spec::Encoder {
                        source: Source::of(word)?,
                        name: None,
                    }
                } else {
                    Spec::Decoder {
                        stress: false,
                        name: None,
                        accepting: Vec::new(),
                        settings: settings.clone(),
                        seed: None,
                    }
                });
            }
        }
        let start = *pos;
        while src.get(*pos).is_some_and(|c| c.is_ascii_digit()) {
            *pos += 1;
        }
        if start == *pos {
            return Err(format!(
                "architecture {text:?}: expected a number of states, an encoder (enc text audio image diff video), dec or \
                 '(' at position {pos}"
            ));
        }
        let states: usize = src[start..*pos]
            .iter()
            .collect::<String>()
            .parse()
            .map_err(|_| format!("architecture {text:?}: a number too large at position {start}"))?;
        let mut emit = Emit::States;
        if src.get(*pos) == Some(&'f') {
            *pos += 1;
            emit = Emit::Final;
        }
        Ok(Spec::Layer {
            states,
            emit,
            name: None,
            accepting: Vec::new(),
            settings: settings.clone(),
            seed: None,
        })
    }

    let spec = seq(&src, &mut pos, merge, vote, settings, text)?;
    if pos != src.len() {
        return fail(&format!("unexpected {:?}", src[pos]), pos);
    }
    Ok(spec)
}

/// What `fit_pairs` measured, epoch by epoch.
#[derive(Clone, Debug, Default)]
pub struct PairHistory {
    pub sounds: Vec<f64>,
    pub exact: Vec<f64>,
    pub val_exact: Vec<f64>,
}

/// A network of machines: built over an input alphabet, compiled, fitted, evaluated, saved.  Its quiet greedy passes
/// are deterministic: a tie goes to the lowest state, and nothing is drawn (`Machine::greedy_first`).
pub struct Network {
    pub name: String,
    pub architecture: String,
    pub alphabet: Vec<String>,
    pub seed: u64,
    pub root: Block,
    pub layer_discount: f64,
    pub reward: f64,
    pub punish: f64,
    pub credit: Credit,
    pub epochs: u64,
    pub examples: u64,
    /// The English words of the pairs it was taught, sorted: the tokenizer hears them before it spells anything, so a
    /// word it was taught is spelled back as itself (its memory of the words it sounded out) in every process.
    pub words: Vec<String>,
    tokenizers: HashMap<bool, Tokenizer>,
    primed: std::collections::HashSet<(bool, usize)>,
}

fn count_encoders(b: &Block) -> usize {
    match b {
        Block::Encoder(_) => 1,
        Block::Layer(_) => 0,
        Block::Sequential { blocks, .. } => blocks.iter().map(count_encoders).sum(),
        Block::Parallel { branches, .. } => branches.iter().map(count_encoders).sum(),
    }
}

impl Network {
    /// Build `spec` over `alphabet`.  Each layer's machine is seeded `seed` plus its index, depth first, unless its
    /// spec says otherwise.
    pub fn new(spec: Spec, alphabet: &[String], seed: u64) -> Result<Network, String> {
        if alphabet.is_empty() || (1..alphabet.len()).any(|i| alphabet[..i].contains(&alphabet[i])) {
            return Err("the input alphabet is one or more distinct symbols".to_string());
        }
        let spec = match spec {
            s @ Spec::Sequential { .. } => s,
            other => Spec::Sequential {
                blocks: vec![other],
                name: None,
            },
        };
        let (root, _) = Block::build(&spec, alphabet, seed, &mut 0, &mut HashMap::new())?;
        Network::check(&root)?;
        let alphabet = match &root {
            Block::Sequential { blocks, .. } => match blocks.first() {
                Some(Block::Encoder(e)) => e.labels.clone(),
                _ => alphabet.to_vec(),
            },
            _ => alphabet.to_vec(),
        };
        Ok(Network {
            tokenizers: HashMap::new(),
            primed: Default::default(),
            words: Vec::new(),
            name: "network".to_string(),
            architecture: String::new(),
            alphabet,
            seed,
            root,
            layer_discount: LAYER_DISCOUNT,
            reward: 1.0,
            punish: 1.0,
            credit: Credit::Shared,
            epochs: 0,
            examples: 0,
        })
    }

    fn check(root: &Block) -> Result<(), String> {
        let first_is_encoder =
            matches!(root, Block::Sequential { blocks, .. } if matches!(blocks.first(), Some(Block::Encoder(_))));
        let n = count_encoders(root);
        if n > 1 || (n == 1 && !first_is_encoder) {
            return Err("an encoder can only be the network's first block".to_string());
        }
        let mut layers = Vec::new();
        root.layers(&mut layers);
        if layers.is_empty() {
            return Err("a network needs at least one layer".to_string());
        }
        Ok(())
    }

    /// The encoder, when the network's first block is one: the network then reads English.
    pub fn encoder(&self) -> Option<&Encoder> {
        match &self.root {
            Block::Sequential { blocks, .. } => match blocks.first() {
                Some(Block::Encoder(e)) => Some(e),
                _ => None,
            },
            _ => None,
        }
    }

    /// The decoder, when the network's last block is one: the network then writes English.
    pub fn decoder(&self) -> Option<&Layer> {
        match &self.root {
            Block::Sequential { blocks, .. } => match blocks.last() {
                Some(Block::Layer(l)) if l.is_decoder() => Some(l),
                _ => None,
            },
            _ => None,
        }
    }

    fn encoder_mut(&mut self) -> Option<&mut Encoder> {
        match &mut self.root {
            Block::Sequential { blocks, .. } => match blocks.first_mut() {
                Some(Block::Encoder(e)) => Some(e),
                _ => None,
            },
            _ => None,
        }
    }

    fn decoder_mut(&mut self) -> Option<&mut Layer> {
        match &mut self.root {
            Block::Sequential { blocks, .. } => match blocks.last_mut() {
                Some(Block::Layer(l)) if l.is_decoder() => Some(l),
                _ => None,
            },
            _ => None,
        }
    }

    fn tokenizer(&mut self, stress: bool) -> &mut Tokenizer {
        self.tokenizers.entry(stress).or_insert_with(|| tokenizer(stress))
    }

    /// Sounds spelled back as English.
    pub fn english(&mut self, sounds: &[String], stress: bool) -> String {
        self.prime();
        self.tokenizer(stress).decode(sounds)
    }

    fn prime(&mut self) {
        let stresses: Vec<bool> = [
            self.encoder().and_then(Encoder::stress),
            self.decoder().map(|d| d.stress),
        ]
        .into_iter()
        .flatten()
        .collect();
        for stress in stresses {
            let key = (stress, self.words.len());
            if !self.words.is_empty() && !self.primed.contains(&key) {
                let all = self.words.join(" ");
                self.tokenizer(stress).tokens(&all);
                self.primed.insert(key);
            }
        }
    }

    /// A network from an architecture string ([`parse_architecture`]).
    pub fn from_architecture(
        text: &str,
        alphabet: &[String],
        seed: u64,
        merge: Merge,
        vote: Vote,
        settings: &Settings,
    ) -> Result<Network, String> {
        let spec = parse_architecture(text, merge, vote, settings)?;
        let mut net = Network::new(spec, alphabet, seed)?;
        net.architecture = text.replace(' ', "");
        Ok(net)
    }

    /// Set the output layers' accepting states (when given), how much of the credit each block further back
    /// receives, how much a right verdict is rewarded and a wrong one punished, and whose verdict the output layers
    /// are credited by.
    pub fn compile(
        &mut self,
        accepting: Option<&[usize]>,
        layer_discount: f64,
        reward: f64,
        punish: f64,
        credit: Credit,
    ) -> Result<(), String> {
        if !(0.0..=1.0).contains(&layer_discount) {
            return Err(format!("layer_discount must be in [0, 1], got {layer_discount}"));
        }
        if let Some(acc) = accepting {
            let mut outs = Vec::new();
            self.root.output_layers_mut(&mut outs);
            for layer in outs {
                let n = layer.machine.n_states();
                if let Some(&s) = acc.iter().find(|&&s| s >= n) {
                    return Err(format!(
                        "accepting state {s} is not one of {}'s 0..{}",
                        layer.name,
                        n - 1
                    ));
                }
                for (i, st) in layer.machine.lattice.states.iter_mut().enumerate() {
                    st.accepting = acc.contains(&i);
                }
            }
        }
        self.layer_discount = layer_discount;
        self.reward = reward.abs();
        self.punish = punish.abs();
        self.credit = credit;
        Ok(())
    }

    /// The input's symbols: with an encoder, the text's sounds (those it has, then its `pad` boundaries); else one
    /// per character, whitespace skipped.
    pub fn tokenize(&mut self, text: &str) -> Result<Vec<usize>, String> {
        if let Some(e) = self.encoder() {
            let (source, labels) = (e.source.clone(), e.labels.clone());
            let Source::Phonetic { stress, pad } = source else {
                return self.encoder_mut().expect("there is one").encode_media(text);
            };
            let tokens = self.tokenizer(stress).tokens(text);
            let mut out: Vec<usize> = tokens
                .iter()
                .filter_map(|t| labels.iter().position(|l| l == t))
                .collect();
            out.extend(std::iter::repeat_n(0, pad));
            return Ok(out);
        }
        text.chars()
            .filter(|c| !c.is_whitespace())
            .map(|c| {
                let s = c.to_string();
                self.alphabet
                    .iter()
                    .position(|a| *a == s)
                    .ok_or_else(|| format!("{s:?} is not in the network's alphabet {}", self.alphabet.concat()))
            })
            .collect()
    }

    /// Read `symbols` through every block: what the last block wrote, and the network's verdict.
    pub fn forward(&mut self, symbols: &[usize], quiet: bool, temperature: Option<f64>) -> (Vec<usize>, bool) {
        self.root.forward(symbols, quiet, temperature, &mut None)
    }

    /// The greedy quiet pass, layer by layer: each layer's step, what the network wrote, and its verdict.
    pub fn trace(&mut self, text: &str) -> Result<(Vec<Step>, String, bool), String> {
        self.prime();
        let symbols = self.tokenize(text)?;
        let mut steps = Vec::new();
        let (out, accepted) = self.root.forward(&symbols, true, Some(0.0), &mut Some(&mut steps));
        if self.encoder().is_some() {
            if let Some(first) = steps.first_mut() {
                first.read = text.to_string();
            }
        }
        for step in steps.iter_mut() {
            if let Some(stress) = step.stress {
                let sounds = step.sounds.clone();
                step.english = Some(self.english(&sounds, stress));
            }
        }
        let written = self.root.written();
        let output = match self.decoder().map(|d| d.stress) {
            Some(stress) => {
                let sounds: Vec<String> = out.iter().map(|&i| written[i].clone()).collect();
                self.english(&sounds, stress)
            }
            None => text_of(&out, &written),
        };
        Ok((steps, output, accepted))
    }

    // ---- English to English --------------------------------------------------------------------------------------

    fn check_translates(&self) -> Result<(), String> {
        if self.encoder().is_none() || self.decoder().is_none() {
            return Err("translating needs an encoder first and a decoder last (e.g. \"enc>13>dec\")".to_string());
        }
        Ok(())
    }

    /// The sounds the network writes for `text` (greedy, quiet), the trailing boundaries dropped.
    pub fn sounds(&mut self, text: &str) -> Result<Vec<String>, String> {
        self.check_translates()?;
        self.prime();
        let symbols = self.tokenize(text)?;
        let (out, _) = self.forward(&symbols, true, Some(0.0));
        let labels = self.decoder().expect("checked").written();
        let mut sounds: Vec<String> = out.iter().map(|&i| labels[i].clone()).collect();
        while sounds.last().is_some_and(|s| s == "#") {
            sounds.pop();
        }
        Ok(sounds)
    }

    /// English in, English out: `text` read as sounds, walked, and what the decoder writes spelled back.
    pub fn translate(&mut self, text: &str) -> Result<String, String> {
        let sounds = self.sounds(text)?;
        let stress = self.decoder().expect("checked").stress;
        Ok(self.english(&sounds, stress))
    }

    /// The target's sounds as the decoder's states (sounds it has no state for dropped).
    fn targets(&mut self, text: &str) -> Vec<usize> {
        let (stress, labels) = {
            let d = self.decoder().expect("checked");
            (d.stress, d.written())
        };
        let tokens = self.tokenizer(stress).tokens(text);
        tokens
            .iter()
            .filter_map(|t| labels.iter().position(|l| l == t))
            .collect()
    }

    pub fn target_sounds(&mut self, text: &str) -> Vec<String> {
        let labels = self.decoder().expect("checked").written();
        let mut out: Vec<String> = self.targets(text).into_iter().map(|i| labels[i].clone()).collect();
        while out.last().is_some_and(|s| s == "#") {
            out.pop();
        }
        out
    }

    /// One English pair: every machine walks `x`; the decoder is taught `y`'s sounds step by step, and the blocks
    /// before it are rewarded by the share it got right (rewarded only, as in the Python port).  Returns (sounds
    /// right, sounds written).
    pub fn teach_pair(&mut self, x: &str, y: &str) -> Result<(usize, usize), String> {
        self.check_translates()?;
        let mut grew = false;
        for w in format!("{x} {y}").split_whitespace() {
            let w = w.to_lowercase();
            if !self.words.contains(&w) {
                self.words.push(w);
                grew = true;
            }
        }
        if grew {
            self.words.sort();
        }
        let symbols = self.tokenize(x)?;
        self.forward(&symbols, false, None);
        let targets = self.targets(y);
        let (reward, punish, ld) = (self.reward, self.punish, self.layer_discount);
        let (right, n) = self
            .decoder_mut()
            .expect("checked")
            .teach_targets(&targets, reward, punish);
        if n > 0 {
            let mut amount = reward * (right as f64 / n as f64) * ld;
            if let Block::Sequential { blocks, .. } = &mut self.root {
                let k = blocks.len();
                for b in blocks[..k - 1].iter_mut().rev() {
                    b.credit(amount, ld, None);
                    amount *= ld;
                }
            }
        }
        self.examples += 1;
        Ok((right, n))
    }

    /// The share of pairs whose translation has exactly the target's sounds.  Nothing moves.
    pub fn evaluate_pairs(&mut self, pairs: &[(String, String)]) -> Result<f64, String> {
        if pairs.is_empty() {
            return Ok(0.0);
        }
        let mut right = 0;
        for (x, y) in pairs {
            if self.sounds(x)? == self.target_sounds(y) {
                right += 1;
            }
        }
        Ok(right as f64 / pairs.len() as f64)
    }

    /// Teach every pair in order, `epochs` times.  Per epoch: the share of sounds right while taught, the share of
    /// pairs exactly right after it on `pairs`, and on `validation` when given.
    pub fn fit_pairs(
        &mut self,
        pairs: &[(String, String)],
        epochs: usize,
        validation: Option<&[(String, String)]>,
    ) -> Result<PairHistory, String> {
        let mut h = PairHistory::default();
        for _ in 0..epochs {
            let (mut right, mut total) = (0, 0);
            for (x, y) in pairs {
                let (r, n) = self.teach_pair(x, y)?;
                right += r;
                total += n;
            }
            self.epochs += 1;
            h.sounds.push(if total > 0 { right as f64 / total as f64 } else { 0.0 });
            h.exact.push(self.evaluate_pairs(pairs)?);
            if let Some(v) = validation {
                h.val_exact.push(self.evaluate_pairs(v)?);
            }
        }
        Ok(h)
    }

    /// The verdict on each string: the greedy quiet pass.  Nothing moves.
    pub fn predict(&mut self, texts: &[String]) -> Result<Vec<bool>, String> {
        texts
            .iter()
            .map(|t| {
                let s = self.tokenize(t)?;
                Ok(self.forward(&s, true, Some(0.0)).1)
            })
            .collect()
    }

    /// The share of `(string, verdict)` examples the greedy quiet pass gets right.  Nothing moves.
    pub fn evaluate(&mut self, examples: &[(String, bool)]) -> Result<f64, String> {
        if examples.is_empty() {
            return Ok(0.0);
        }
        let mut right = 0usize;
        for (text, label) in examples {
            let s = self.tokenize(text)?;
            if self.forward(&s, true, Some(0.0)).1 == *label {
                right += 1;
            }
        }
        Ok(right as f64 / examples.len() as f64)
    }

    /// One example: every machine walks it, and the network is rewarded if its verdict is `label` and punished if
    /// not.  Returns whether it was right.
    pub fn teach(&mut self, text: &str, label: bool) -> Result<bool, String> {
        let s = self.tokenize(text)?;
        let (_, accepted) = self.forward(&s, false, None);
        let right = accepted == label;
        let own = (self.credit == Credit::Own).then_some((label, self.reward, self.punish));
        let amount = if right { self.reward } else { -self.punish };
        self.root.credit(amount, self.layer_discount, own);
        self.examples += 1;
        Ok(right)
    }

    /// Teach every example in order, `epochs` times.  Returns, per epoch, the share classified right while taught.
    pub fn fit(&mut self, examples: &[(String, bool)], epochs: usize) -> Result<Vec<f64>, String> {
        let mut history = Vec::new();
        for _ in 0..epochs {
            let mut right = 0usize;
            for (text, label) in examples {
                right += self.teach(text, *label)? as usize;
            }
            self.epochs += 1;
            history.push(if examples.is_empty() {
                0.0
            } else {
                right as f64 / examples.len() as f64
            });
        }
        Ok(history)
    }

    /// Every layer, depth first.
    pub fn layers(&self) -> Vec<&Layer> {
        let mut out = Vec::new();
        self.root.layers(&mut out);
        out
    }

    /// How many distinct symbols the network writes.
    pub fn output_width(&self) -> usize {
        self.root.width()
    }

    /// The blocks, nested, with each layer's matrix and what it writes; the totals.
    pub fn summary(&self) -> String {
        let mut rows: Vec<[String; 5]> = Vec::new();
        fn walk(b: &Block, depth: usize, rows: &mut Vec<[String; 5]>) {
            let pad = "  ".repeat(depth);
            match b {
                Block::Encoder(e) => {
                    rows.push([
                        format!("{pad}{}", e.name),
                        e.source.describe(e.labels.len()),
                        if e.stress().is_some() { "sounds" } else { "symbols" }.into(),
                        "".into(),
                        "".into(),
                    ]);
                }
                Block::Layer(l) => {
                    let (s, a, t) = l.machine.lattice.shape();
                    rows.push([
                        format!("{pad}{}", l.name),
                        format!("{s} x {a} x {t}"),
                        if l.is_decoder() { "sounds" } else { l.emit.as_str() }.to_string(),
                        format!("{}", s * a * t),
                        format!("{}", l.machine.lattice.touched().count()),
                    ]);
                }
                Block::Sequential { name, blocks } => {
                    rows.push([
                        format!("{pad}{name}"),
                        "sequential".into(),
                        "".into(),
                        "".into(),
                        "".into(),
                    ]);
                    blocks.iter().for_each(|c| walk(c, depth + 1, rows));
                }
                Block::Parallel {
                    name,
                    merge,
                    vote,
                    branches,
                    ..
                } => {
                    rows.push([
                        format!("{pad}{name}"),
                        format!("parallel: {}, {}", merge.as_str(), vote.as_str()),
                        "".into(),
                        "".into(),
                        "".into(),
                    ]);
                    branches.iter().for_each(|c| walk(c, depth + 1, rows));
                }
            }
        }
        if let Block::Sequential { blocks, .. } = &self.root {
            blocks.iter().for_each(|c| walk(c, 0, &mut rows));
        }
        let head = ["block", "matrix", "emits", "edges", "touched"].map(String::from);
        let widths: Vec<usize> = (0..5)
            .map(|i| {
                rows.iter()
                    .chain(std::iter::once(&head))
                    .map(|r| r[i].chars().count())
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let fmt = |r: &[String; 5]| {
            let cells: Vec<String> = (0..5)
                .map(|i| {
                    if i < 3 {
                        format!("{:<w$}", r[i], w = widths[i])
                    } else {
                        format!("{:>w$}", r[i], w = widths[i])
                    }
                })
                .collect();
            cells.join("  ").trim_end().to_string()
        };
        let layers = self.layers();
        let mut outs = Vec::new();
        self.root.output_layers(&mut outs);
        let mut accepting: Vec<usize> = outs.iter().flat_map(|l| l.machine.accepting()).collect();
        accepting.sort_unstable();
        accepting.dedup();
        let arch = if self.architecture.is_empty() {
            String::new()
        } else {
            format!(" ({})", self.architecture)
        };
        let mut lines = vec![
            format!(
                "network \"{}\"{arch}: reads {}, writes {}",
                self.name,
                match self.encoder() {
                    Some(e) if e.stress().is_some() => "English".to_string(),
                    Some(e) => e.source.kind().to_string(),
                    None => self.alphabet.concat(),
                },
                if self.decoder().is_some() {
                    "English".to_string()
                } else {
                    format!("{} symbols", self.output_width())
                }
            ),
            fmt(&head),
            fmt(&widths
                .iter()
                .map(|&w| "-".repeat(w))
                .collect::<Vec<_>>()
                .try_into()
                .expect("five")),
        ];
        lines.extend(rows.iter().map(fmt));
        let edges: usize = layers.iter().map(|l| l.machine.lattice.edges.len()).sum();
        let touched: usize = layers.iter().map(|l| l.machine.lattice.touched().count()).sum();
        lines.push(format!(
            "{} layers, {edges} edges, {touched} touched; output accepting {accepting:?}; {} credit, layer discount \
             {}, reward {}, punish {}; {} examples taught over {} epochs",
            layers.len(),
            self.credit.as_str(),
            g(self.layer_discount),
            g(self.reward),
            g(self.punish),
            self.examples,
            self.epochs
        ));
        lines.join("\n")
    }

    pub fn to_json(&self) -> Json {
        Json::object()
            .with("format", FORMAT.into())
            .with("version", FORMAT_VERSION.into())
            .with("name", self.name.as_str().into())
            .with("architecture", self.architecture.as_str().into())
            .with("alphabet", Json::strings(&self.alphabet))
            .with("seed", (self.seed as f64).into())
            .with("layer_discount", self.layer_discount.into())
            .with("reward", self.reward.into())
            .with("punish", self.punish.into())
            .with("credit", self.credit.as_str().into())
            .with("words", Json::strings(&self.words))
            .with("epochs", (self.epochs as f64).into())
            .with("examples", (self.examples as f64).into())
            .with("root", self.root.to_json())
    }

    pub fn from_json(v: &Json) -> Result<Network, String> {
        if v.str_or("format", "") != FORMAT {
            return Err(format!("not a {FORMAT} file"));
        }
        let alphabet: Vec<String> = v
            .get("alphabet")
            .and_then(Json::as_array)
            .ok_or("no alphabet")?
            .iter()
            .map(|s| s.as_str().map(String::from).ok_or("the alphabet is strings"))
            .collect::<Result<_, _>>()?;
        let root = match Block::from_json(v.get("root").ok_or("no root")?)? {
            s @ Block::Sequential { .. } => s,
            other => Block::Sequential {
                name: "sequential_1".to_string(),
                blocks: vec![other],
            },
        };
        Network::check(&root)?;
        let net = Network {
            tokenizers: HashMap::new(),
            primed: Default::default(),
            words: {
                let mut w: Vec<String> = v
                    .get("words")
                    .and_then(Json::as_array)
                    .map(|a| a.iter().filter_map(Json::as_str).map(String::from).collect())
                    .unwrap_or_default();
                w.sort();
                w
            },
            name: v.str_or("name", "network").to_string(),
            architecture: v.str_or("architecture", "").to_string(),
            alphabet,
            seed: v.num("seed", 1.0) as u64,
            root,
            layer_discount: v.num("layer_discount", LAYER_DISCOUNT),
            reward: v.num("reward", 1.0),
            punish: v.num("punish", 1.0),
            credit: Credit::parse(v.str_or("credit", "shared"))?,
            epochs: v.num("epochs", 0.0) as u64,
            examples: v.num("examples", 0.0) as u64,
        };
        if net.encoder().is_none() && net.layers()[0].machine.alphabet() != net.alphabet.as_slice() {
            return Err("the first layer's alphabet is not the network's".to_string());
        }
        Ok(net)
    }

    /// Write the network to `path` (`.json`, or `.json.gz` to gzip it).
    pub fn save(&self, path: &str) -> Result<(), String> {
        let text = self.to_json().dump();
        let bytes = if path.ends_with(".gz") {
            gzip::compress(text.as_bytes())
        } else {
            text.into_bytes()
        };
        let mut f = fs::File::create(path).map_err(|e| format!("{path}: {e}"))?;
        f.write_all(&bytes).map_err(|e| format!("{path}: {e}"))
    }
}

/// A number the way Python's `{:g}` writes it, for the values a summary shows.
fn g(x: f64) -> String {
    if x == x.trunc() && x.abs() < 1e15 {
        format!("{}", x as i64)
    } else {
        format!("{x}")
    }
}

/// Read a network from `path` (`.json` or `.json.gz`).
pub fn load_network(path: &str) -> Result<Network, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .and_then(|mut f| f.read_to_end(&mut bytes))
        .map_err(|e| format!("{path}: {e}"))?;
    let bytes = if path.ends_with(".gz") || bytes.starts_with(&[0x1f, 0x8b]) {
        gzip::decompress(&bytes)?
    } else {
        bytes
    };
    let text = String::from_utf8(bytes).map_err(|_| format!("{path}: not UTF-8"))?;
    Network::from_json(&parse(&text)?)
}

/// Teach `episodes` random strings of a language, drawn exactly as `experiment::teach_language` draws them, so a
/// one-layer network and the machine it holds see the same strings.  Returns how many were classified right.
pub fn teach_network(net: &mut Network, lang: &Language, episodes: usize, rng: &mut Rng, max_length: usize) -> usize {
    let mut right = 0;
    for _ in 0..episodes {
        let s = random_string(rng, max_length);
        right += net.teach(&s, (lang.member)(&s)).expect("over the alphabet") as usize;
    }
    right
}
