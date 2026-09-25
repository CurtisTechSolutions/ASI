//! The Rust port's command line: the same commands as the Python package's
//! (`phonetok tokenize "The cat sat."`, explain, ipa, decode, pronounce, rhymes,
//! affinity, coin, blend, lexicon, eval, say, and the acoustic units' hear,
//! learn, replay and codebook), the same output.

use std::io::{BufRead, Write};
use std::process::ExitCode;

use phonetok::acoustic::{learn, load_wav, resample, AcousticTokenizer, Analysis, Codebook};
use phonetok::json::Json;
use phonetok::lexicon::{read_entries, Lexicon, ENV_LEXICON};
use phonetok::mt::{Mt, Rng};
use phonetok::phones::{strip_stress, to_ipa};
use phonetok::rules::letter_to_sound;
use phonetok::syllables::{rhymes, syllabify};
use phonetok::synth::{find_player, wav_bytes, wav_header, Synthesizer, VoiceSettings, RATE};
use phonetok::tokenizer::{Level, Tokenizer};

const USAGE: &str = "usage: phonetok <command> [options] [args]

commands:
  tokenize TEXT...        the tokens, one line (--ids for ids)
  explain TEXT...         every word: phones, source, syllables, IPA
  ipa TEXT...             the IPA
  decode TOKENS...        sounds (or ids) back to words
  pronounce WORD...       one word, every pronunciation the lexicon has
  rhymes WORD [WORD]      rhymes in the lexicon, or whether two words rhyme
  affinity A B            how well two sounds go together
  coin                    new pronounceable words (--count, --seed, --syllables)
  blend A B               a portmanteau
  lexicon [WORD...]       what the lexicon holds and where it came from
  eval                    the rules against the full dictionary (PHONETOK_LEXICON)
  say TEXT...             speak: the formant synthesizer to a WAV (--out), a player (--play) or stdout (--raw);
                          tokens on stdin when no text is given (--rate, --pitch, --tempo, --gain)
  hear WAV...             the acoustic units of recordings: sounds learned, nothing written down (--frames)
  learn WAV...            learn a codebook of acoustic units from recordings (--units, --seed, --iterations,
                          --note, --out FILE)
  replay UNIT...          units spoken back through the vocoder (--out, --play, --raw, --polish, --pitch, --gain);
                          units on stdin when none are given
  codebook [FILE]         what a codebook holds (the bundled one when no file is named)

options (after the command):
  --level L  --no-stress  --no-boundaries  --no-pauses  --core  --lexicon FILE  --json
  --codebook FILE         the codebook of acoustic units (hear, replay; the bundled one otherwise)";

struct Options {
    level: Level,
    no_stress: bool,
    no_boundaries: bool,
    no_pauses: bool,
    core: bool,
    lexicon: Option<String>,
    json: bool,
    ids: bool,
    limit: Option<usize>,
    count: usize,
    seed: Option<u64>,
    syllables: usize,
    out: String,
    play: bool,
    raw: bool,
    rate: u32,
    pitch: f64,
    tempo: f64,
    gain: f64,
    codebook: Option<String>,
    frames: bool,
    units: usize,
    iterations: usize,
    note: String,
    polish: usize,
    out_set: bool,
    gain_set: bool,
    positional: Vec<String>,
}

fn parse_args(args: &[String]) -> Result<Options, String> {
    let mut o = Options {
        level: Level::Phoneme,
        no_stress: false,
        no_boundaries: false,
        no_pauses: false,
        core: false,
        lexicon: None,
        json: false,
        ids: false,
        limit: None,
        count: 5,
        seed: None,
        syllables: 0,
        out: "say.wav".to_string(),
        play: false,
        raw: false,
        rate: RATE,
        pitch: 120.0,
        tempo: 1.0,
        gain: 0.5,
        codebook: None,
        frames: false,
        units: 64,
        iterations: 50,
        note: String::new(),
        polish: 0,
        out_set: false,
        gain_set: false,
        positional: Vec::new(),
    };
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let (name, inline) = match a.split_once('=') {
            Some((n, v)) if n.starts_with("--") => (n.to_string(), Some(v.to_string())),
            _ => (a.clone(), None),
        };
        let value = |i: &mut usize| -> Result<String, String> {
            if let Some(v) = &inline {
                return Ok(v.clone());
            }
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match name.as_str() {
            "--level" => o.level = Level::parse(&value(&mut i)?)?,
            "--no-stress" => o.no_stress = true,
            "--no-boundaries" => o.no_boundaries = true,
            "--no-pauses" => o.no_pauses = true,
            "--core" => o.core = true,
            "--lexicon" => o.lexicon = Some(value(&mut i)?),
            "--json" => o.json = true,
            "--ids" => o.ids = true,
            "--limit" => {
                o.limit = Some(
                    value(&mut i)?
                        .parse()
                        .map_err(|_| "limit must be a number")?,
                )
            }
            "--count" => {
                o.count = value(&mut i)?
                    .parse()
                    .map_err(|_| "count must be a number")?
            }
            "--seed" => {
                o.seed = Some(
                    value(&mut i)?
                        .parse()
                        .map_err(|_| "seed must be a number")?,
                )
            }
            "--syllables" => {
                o.syllables = value(&mut i)?
                    .parse()
                    .map_err(|_| "syllables must be a number")?
            }
            "--out" => {
                o.out = value(&mut i)?;
                o.out_set = true;
            }
            "--codebook" => o.codebook = Some(value(&mut i)?),
            "--frames" => o.frames = true,
            "--units" => {
                o.units = value(&mut i)?
                    .parse()
                    .map_err(|_| "units must be a number")?
            }
            "--iterations" => {
                o.iterations = value(&mut i)?
                    .parse()
                    .map_err(|_| "iterations must be a number")?
            }
            "--note" => o.note = value(&mut i)?,
            "--polish" => {
                o.polish = value(&mut i)?
                    .parse()
                    .map_err(|_| "polish must be a number")?
            }
            "--play" => o.play = true,
            "--raw" => o.raw = true,
            "--rate" => {
                o.rate = value(&mut i)?
                    .parse()
                    .map_err(|_| "rate must be a number")?
            }
            "--pitch" => {
                o.pitch = value(&mut i)?
                    .parse()
                    .map_err(|_| "pitch must be a number")?
            }
            "--tempo" => {
                o.tempo = value(&mut i)?
                    .parse()
                    .map_err(|_| "tempo must be a number")?
            }
            "--gain" => {
                o.gain = value(&mut i)?
                    .parse()
                    .map_err(|_| "gain must be a number")?;
                o.gain_set = true;
            }
            _ if name.starts_with("--") => return Err(format!("unknown option {name}")),
            _ => o.positional.push(a.clone()),
        }
        i += 1;
    }
    Ok(o)
}

fn tokenizer(o: &Options) -> Result<Tokenizer, String> {
    let lexicon = if let Some(path) = &o.lexicon {
        let mut lex = Lexicon::core();
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        lex.extend(read_entries(&text));
        lex.sources.push(format!("file {path}"));
        lex
    } else if o.core {
        Lexicon::core()
    } else {
        Lexicon::portable()?
    };
    let mut tok = Tokenizer::new(o.level, lexicon);
    tok.stress = !o.no_stress;
    tok.boundaries = !o.no_boundaries;
    tok.pauses = !o.no_pauses;
    Ok(tok)
}

fn emit(o: &Options, doc: Json, lines: Vec<String>) {
    if o.json {
        println!("{}", doc.to_pretty());
    } else {
        for l in lines {
            println!("{l}");
        }
    }
}

fn edit_distance(a: &[String], b: &[String]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i];
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur.push((prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost));
        }
        prev = cur;
    }
    prev[b.len()]
}

struct SystemRng(u64);

impl Rng for SystemRng {
    fn float64(&mut self) -> f64 {
        // a small xorshift for unseeded coining; seeded coining uses the Mersenne Twister
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn rand_below(&mut self, n: usize) -> usize {
        (self.float64() * n as f64) as usize % n
    }
}

fn run(args: &[String]) -> Result<i32, String> {
    let Some(command) = args.first() else {
        eprintln!("{USAGE}");
        return Ok(2);
    };
    let o = parse_args(&args[1..])?;
    let mut tok = tokenizer(&o)?;
    let text = o.positional.join(" ");
    match command.as_str() {
        "tokenize" => {
            let tokens = tok.tokenize(&text);
            let ids = tok.encode(&text, true);
            let texts: Vec<String> = tokens.iter().map(|t| t.text.clone()).collect();
            let kinds: Vec<String> = tokens.iter().map(|t| t.kind.to_string()).collect();
            let line = if o.ids {
                ids.iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            } else {
                texts.join(" ")
            };
            let doc = Json::obj(vec![
                ("text", Json::str(&text)),
                ("level", Json::str(tok.level.as_str())),
                ("tokens", Json::strs(&texts)),
                (
                    "ids",
                    Json::Arr(ids.iter().map(|&i| Json::Num(i as f64)).collect()),
                ),
                ("kinds", Json::strs(&kinds)),
                ("vocab_size", Json::Num(tok.vocab.len() as f64)),
            ]);
            emit(&o, doc, vec![line]);
        }
        "explain" => {
            let rows = tok.explain(&text);
            let mut lines = vec![format!(
                "{:<16} {:<11} {:<32} {:<32} ipa",
                "word", "how", "phones", "syllables"
            )];
            let mut docs = Vec::new();
            for r in &rows {
                lines.push(format!(
                    "{:<16} {:<11} {:<32} {:<32} {}",
                    r.word,
                    r.how,
                    r.phones.join(" "),
                    r.syllables.join(" "),
                    r.ipa
                ));
                docs.push(Json::obj(vec![
                    ("word", Json::str(&r.word)),
                    ("phones", Json::strs(&r.phones)),
                    ("how", Json::str(r.how)),
                    ("syllables", Json::strs(&r.syllables)),
                    ("ipa", Json::str(&r.ipa)),
                    ("score", r.score.map(Json::Num).unwrap_or(Json::Null)),
                ]));
            }
            emit(
                &o,
                Json::obj(vec![
                    ("words", Json::Arr(docs)),
                    ("tokenizer", Json::str(&tok.describe())),
                ]),
                lines,
            );
        }
        "ipa" => {
            let ipa = tok.ipa(&text);
            emit(
                &o,
                Json::obj(vec![("text", Json::str(&text)), ("ipa", Json::str(&ipa))]),
                vec![ipa],
            );
        }
        "decode" => {
            let tokens: Vec<String> = text.split_whitespace().map(|s| s.to_string()).collect();
            let words = if !tokens.is_empty() && tokens.iter().all(|t| t.parse::<usize>().is_ok()) {
                let ids: Vec<usize> = tokens.iter().map(|t| t.parse().unwrap()).collect();
                tok.decode_ids(&ids)
            } else {
                tok.decode(&tokens)
            };
            emit(
                &o,
                Json::obj(vec![
                    ("tokens", Json::strs(&tokens)),
                    ("text", Json::str(&words)),
                ]),
                vec![words],
            );
        }
        "pronounce" => {
            let mut lines = Vec::new();
            let mut docs = Vec::new();
            for word in &o.positional {
                let (phones, how) = tok.transcriber.explain(word);
                let mut variants = tok.lexicon().pronunciations(word);
                if variants.is_empty() {
                    variants = vec![phones.clone()];
                }
                let syl: Vec<String> = syllabify(&phones, true).iter().map(|s| s.text()).collect();
                let ipa = tok.ipa(word);
                lines.push(format!(
                    "{}: {}  [{}]  /{}/  {}",
                    word,
                    phones.join(" "),
                    how,
                    ipa,
                    syl.join(" | ")
                ));
                let pad = " ".repeat(word.chars().count());
                for v in &variants[1..] {
                    lines.push(format!("{pad}  {}  [lexicon, alternative]", v.join(" ")));
                }
                if how != "rules" {
                    lines.push(format!(
                        "{pad}  {}  [the rules alone would say]",
                        letter_to_sound(word).join(" ")
                    ));
                }
                docs.push(Json::obj(vec![
                    ("word", Json::str(word)),
                    ("how", Json::str(how)),
                    ("phones", Json::strs(&phones)),
                    (
                        "variants",
                        Json::Arr(variants.iter().map(|v| Json::strs(v)).collect()),
                    ),
                    ("syllables", Json::strs(&syl)),
                    ("ipa", Json::str(&ipa)),
                    ("rules", Json::strs(&letter_to_sound(word))),
                ]));
            }
            emit(&o, Json::obj(vec![("words", Json::Arr(docs))]), lines);
        }
        "rhymes" => {
            if o.positional.len() >= 2 {
                let (a, b) = (o.positional[0].clone(), o.positional[1].clone());
                let yes = tok.rhymes_with(&a, &b);
                let line = format!("{a} / {b}: {}", if yes { "rhyme" } else { "do not rhyme" });
                emit(
                    &o,
                    Json::obj(vec![
                        ("a", Json::str(&a)),
                        ("b", Json::str(&b)),
                        ("rhymes", Json::Bool(yes)),
                    ]),
                    vec![line],
                );
                return Ok(0);
            }
            let Some(word) = o.positional.first().cloned() else {
                eprintln!("{USAGE}");
                return Ok(2);
            };
            let phones = tok.pronounce(&word);
            let mut found: Vec<String> = tok
                .lexicon()
                .items()
                .into_iter()
                .filter(|(w, p)| *w != word.to_lowercase() && rhymes(&phones, p))
                .map(|(w, _)| w)
                .collect();
            found.sort();
            found.dedup();
            let limit = o.limit.unwrap_or(50);
            if limit > 0 && found.len() > limit {
                found.truncate(limit);
            }
            let line = if found.is_empty() {
                "(none in the lexicon)".to_string()
            } else {
                found.join(" ")
            };
            emit(
                &o,
                Json::obj(vec![
                    ("word", Json::str(&word)),
                    ("phones", Json::strs(&phones)),
                    ("rhymes", Json::strs(&found)),
                ]),
                vec![line],
            );
        }
        "affinity" => {
            if o.positional.len() < 2 {
                eprintln!("{USAGE}");
                return Ok(2);
            }
            let (a, b) = (
                o.positional[0].to_uppercase(),
                o.positional[1].to_uppercase(),
            );
            let value = tok.affinity(&a, &b);
            let p = tok.phonotactics().prob(&a, &b);
            let verdict = if value > 1.0 {
                "go together"
            } else if value < -1.0 {
                "avoid each other"
            } else {
                "meet by chance"
            };
            let line = format!("{a} -> {b}: {value:+.2} bits (P = {p:.3}): they {verdict}");
            emit(
                &o,
                Json::obj(vec![
                    ("a", Json::str(&a)),
                    ("b", Json::str(&b)),
                    ("affinity_bits", Json::Num(value)),
                    ("probability", Json::Num(p)),
                    ("verdict", Json::str(verdict)),
                ]),
                vec![line],
            );
        }
        "coin" => {
            let mut mt;
            let mut sys;
            let rng: &mut dyn Rng = match o.seed {
                Some(seed) => {
                    mt = Mt::new(seed);
                    &mut mt
                }
                None => {
                    sys = SystemRng(std::process::id() as u64 | 1);
                    &mut sys
                }
            };
            let avoid = tok.lexicon().items();
            let mut lines = Vec::new();
            let mut docs = Vec::new();
            for _ in 0..o.count {
                let phones = tok.phonotactics().build(rng, o.syllables, 1, 3, &avoid, 50);
                let spelling = tok.transcriber.spell(&phones);
                lines.push(format!(
                    "{:<14} {:<24} /{}/",
                    spelling,
                    phones.join(" "),
                    to_ipa(&phones, true)
                ));
                docs.push(Json::obj(vec![
                    ("phones", Json::strs(&phones)),
                    ("spelling", Json::str(&spelling)),
                    ("ipa", Json::str(&to_ipa(&phones, true))),
                ]));
            }
            emit(
                &o,
                Json::obj(vec![
                    ("words", Json::Arr(docs)),
                    (
                        "seed",
                        o.seed.map(|s| Json::Num(s as f64)).unwrap_or(Json::Null),
                    ),
                ]),
                lines,
            );
        }
        "blend" => {
            if o.positional.len() < 2 {
                eprintln!("{USAGE}");
                return Ok(2);
            }
            let (a, b) = (o.positional[0].clone(), o.positional[1].clone());
            let (pa, pb) = (tok.pronounce(&a), tok.pronounce(&b));
            let (phones, _, _) = tok.phonotactics().blend(&pa, &pb)?;
            let spelling = tok.transcriber.spell(&phones);
            let line = format!(
                "{a} + {b} = {spelling}  ({}, /{}/)",
                phones.join(" "),
                to_ipa(&phones, true)
            );
            emit(
                &o,
                Json::obj(vec![
                    ("a", Json::str(&a)),
                    ("b", Json::str(&b)),
                    ("phones", Json::strs(&phones)),
                    ("spelling", Json::str(&spelling)),
                    ("ipa", Json::str(&to_ipa(&phones, true))),
                ]),
                vec![line],
            );
        }
        "lexicon" => {
            let full = match std::env::var(ENV_LEXICON) {
                Ok(p) if !p.is_empty() => format!("file {p}"),
                _ => "none found (set PHONETOK_LEXICON)".to_string(),
            };
            let described = tok.phonotactics().describe();
            let lex = tok.lexicon();
            let mut lines = vec![
                format!("{} words from {}", lex.len(), lex.sources.join(", ")),
                format!("full dictionary: {full}"),
                format!("phonotactics: {described}"),
            ];
            let mut lookups = Vec::new();
            for w in &o.positional {
                let prons = lex.pronunciations(w);
                if prons.is_empty() {
                    lines.push(format!("{w}: (not in the lexicon)"));
                } else {
                    lines.push(format!(
                        "{w}: {}",
                        prons
                            .iter()
                            .map(|p| p.join(" "))
                            .collect::<Vec<_>>()
                            .join(" | ")
                    ));
                }
                lookups.push((
                    w.clone(),
                    Json::Arr(prons.iter().map(|p| Json::strs(p)).collect()),
                ));
            }
            let mut doc = vec![
                ("words", Json::Num(lex.len() as f64)),
                ("sources", Json::strs(&lex.sources)),
                ("full_dictionary", Json::str(&full)),
                ("phonotactics", Json::str(&described)),
            ];
            if !o.positional.is_empty() {
                doc.push(("lookups", Json::Obj(lookups)));
            }
            emit(&o, Json::obj(doc), lines);
        }
        "say" => {
            let voice = VoiceSettings {
                pitch: o.pitch,
                tempo: o.tempo,
                gain: o.gain,
                ..VoiceSettings::default()
            };
            let mut synth = Synthesizer::new(o.rate, voice);
            let tokens: Vec<String> = if o.positional.is_empty() {
                std::io::stdin()
                    .lock()
                    .lines()
                    .map_while(Result::ok)
                    .flat_map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
                    .collect()
            } else {
                tok.tokens(&text)
            };
            if o.play {
                let Some(player) = find_player() else {
                    eprintln!("phonetok: no player found (aplay, paplay, ffplay, play or afplay)");
                    return Ok(2);
                };
                let mut child = std::process::Command::new(&player[0])
                    .args(&player[1..])
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("{}: {e}", player[0]))?;
                let mut total = 0usize;
                {
                    let stdin = child.stdin.as_mut().ok_or("no stdin")?;
                    stdin
                        .write_all(&wav_header(o.rate, None))
                        .map_err(|e| e.to_string())?;
                    for t in &tokens {
                        let chunk = synth.feed(t);
                        stdin.write_all(&chunk).map_err(|e| e.to_string())?;
                        stdin.flush().ok();
                        total += chunk.len();
                    }
                    let chunk = synth.end();
                    stdin.write_all(&chunk).map_err(|e| e.to_string())?;
                    total += chunk.len();
                }
                child.wait().ok();
                let seconds = total as f64 / 2.0 / o.rate as f64;
                emit(
                    &o,
                    Json::obj(vec![
                        ("seconds", Json::Num(seconds)),
                        ("player", Json::str(&player[0])),
                    ]),
                    vec![format!("{seconds:.2} s spoken through {}", player[0])],
                );
            } else if o.raw {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                for t in &tokens {
                    out.write_all(&synth.feed(t)).map_err(|e| e.to_string())?;
                    out.flush().ok();
                }
                out.write_all(&synth.end()).map_err(|e| e.to_string())?;
                out.flush().ok();
            } else {
                let pcm = synth.speak(&tokens);
                std::fs::write(&o.out, wav_bytes(&pcm, o.rate))
                    .map_err(|e| format!("{}: {e}", o.out))?;
                let seconds = pcm.len() as f64 / 2.0 / o.rate as f64;
                emit(
                    &o,
                    Json::obj(vec![
                        ("path", Json::str(&o.out)),
                        ("seconds", Json::Num(seconds)),
                        ("rate", Json::Num(o.rate as f64)),
                        ("bytes", Json::Num((pcm.len() + 44) as f64)),
                    ]),
                    vec![format!(
                        "{seconds:.2} s of speech written to {} ({} Hz)",
                        o.out, o.rate
                    )],
                );
            }
        }
        "hear" => {
            let tok = AcousticTokenizer::new(load_book(&o)?, !o.frames)?;
            let mut docs = Vec::new();
            let mut lines = Vec::new();
            for path in &o.positional {
                let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
                let h = tok.listen_wav(&data).map_err(|e| format!("{path}: {e}"))?;
                docs.push(Json::obj(vec![
                    ("path", Json::str(path)),
                    ("units", Json::strs(&h.units)),
                    ("frames", Json::Num(h.frames as f64)),
                    ("seconds", Json::Num(h.seconds)),
                    (
                        "codes",
                        Json::Arr(h.codes.iter().map(|&c| Json::Num(c as f64)).collect()),
                    ),
                ]));
                lines.push(h.text());
            }
            emit(&o, Json::Arr(docs), lines);
        }
        "learn" => {
            let a = Analysis::default();
            let mut recordings = Vec::new();
            let mut seconds = 0.0;
            for path in &o.positional {
                let (samples, rate) = load_wav(path)?;
                let samples =
                    resample(&samples, rate, a.rate).map_err(|e| format!("{path}: {e}"))?;
                seconds += samples.len() as f64 / a.rate as f64;
                recordings.push(samples);
            }
            let note = if o.note.is_empty() {
                format!(
                    "learned from {} recording(s), {seconds:.0} s",
                    o.positional.len()
                )
            } else {
                o.note.clone()
            };
            let seed = o.seed.unwrap_or(1) as i64;
            let book = learn(&recordings, o.units, seed, o.iterations, &a, &note)?;
            let out = if o.out_set {
                o.out.clone()
            } else {
                "codebook.tsv".to_string()
            };
            book.dump(&out)?;
            let total: usize = book.counts.iter().sum();
            emit(
                &o,
                Json::obj(vec![
                    ("path", Json::str(&out)),
                    ("units", Json::Num(book.k() as f64)),
                    ("frames", Json::Num(total as f64)),
                    ("seconds", Json::Num(seconds)),
                    ("inertia", Json::Num(book.inertia)),
                    ("seed", Json::Num(seed as f64)),
                    ("note", Json::str(&note)),
                ]),
                vec![format!(
                    "{} units learned from {total} frames ({seconds:.1} s of audio), inertia {:.1}: written to {out}",
                    book.k(),
                    book.inertia
                )],
            );
        }
        "replay" => {
            let tok = AcousticTokenizer::new(load_book(&o)?, true)?;
            let units: Vec<String> = if o.positional.is_empty() {
                std::io::stdin()
                    .lock()
                    .lines()
                    .map_while(Result::ok)
                    .flat_map(|l| l.split_whitespace().map(String::from).collect::<Vec<_>>())
                    .collect()
            } else {
                tok.units_of(&text)?
            };
            let rate = tok.book.analysis.rate;
            let gain = if o.gain_set { o.gain } else { 1.0 };
            if o.play {
                let Some(player) = find_player() else {
                    eprintln!("phonetok: no player found (aplay, paplay, ffplay, play or afplay)");
                    return Ok(2);
                };
                let mut child = std::process::Command::new(&player[0])
                    .args(&player[1..])
                    .stdin(std::process::Stdio::piped())
                    .spawn()
                    .map_err(|e| format!("{}: {e}", player[0]))?;
                let mut total = 0usize;
                {
                    let stdin = child.stdin.as_mut().ok_or("no stdin")?;
                    stdin
                        .write_all(&wav_header(rate, None))
                        .map_err(|e| e.to_string())?;
                    let mut voc = tok.vocoder(gain, o.pitch)?;
                    for u in &units {
                        let chunk = voc.feed(u)?;
                        stdin.write_all(&chunk).map_err(|e| e.to_string())?;
                        stdin.flush().ok();
                        total += chunk.len();
                    }
                    let chunk = voc.end();
                    stdin.write_all(&chunk).map_err(|e| e.to_string())?;
                    total += chunk.len();
                }
                child.wait().ok();
                let seconds = total as f64 / 2.0 / rate as f64;
                emit(
                    &o,
                    Json::obj(vec![
                        ("seconds", Json::Num(seconds)),
                        ("player", Json::str(&player[0])),
                    ]),
                    vec![format!("{seconds:.2} s replayed through {}", player[0])],
                );
            } else if o.raw {
                let stdout = std::io::stdout();
                let mut out = stdout.lock();
                let mut voc = tok.vocoder(gain, o.pitch)?;
                for u in &units {
                    out.write_all(&voc.feed(u)?).map_err(|e| e.to_string())?;
                    out.flush().ok();
                }
                out.write_all(&voc.end()).map_err(|e| e.to_string())?;
                out.flush().ok();
            } else {
                let pcm = tok.synthesize(&units, o.polish, gain, o.pitch)?;
                let out = if o.out_set {
                    o.out.clone()
                } else {
                    "replay.wav".to_string()
                };
                std::fs::write(&out, wav_bytes(&pcm, rate)).map_err(|e| format!("{out}: {e}"))?;
                let seconds = pcm.len() as f64 / 2.0 / rate as f64;
                emit(
                    &o,
                    Json::obj(vec![
                        ("path", Json::str(&out)),
                        ("seconds", Json::Num(seconds)),
                        ("rate", Json::Num(rate as f64)),
                        ("bytes", Json::Num((pcm.len() + 44) as f64)),
                        ("polish", Json::Num(o.polish as f64)),
                    ]),
                    vec![format!(
                        "{seconds:.2} s of speech written to {out} ({rate} Hz)"
                    )],
                );
            }
        }
        "codebook" => {
            let (book, path) = match o.positional.first() {
                Some(p) => (Codebook::load(p)?, p.clone()),
                None => (Codebook::bundled()?, "(bundled)".to_string()),
            };
            let a = book.analysis;
            let total: usize = book.counts.iter().sum();
            let mut lines = vec![
                format!(
                    "{} units over {} log-mel bands ({}-{} Hz), {} ms frames every {} ms at {} Hz: {path}",
                    book.k(),
                    a.bands,
                    a.fmin,
                    a.fmax,
                    a.frame * 1000 / a.rate as usize,
                    a.hop * 1000 / a.rate as usize,
                    a.rate
                ),
                format!(
                    "learned from {total} frames ({:.1} s), seed {}, inertia {:.1}",
                    (total * a.hop) as f64 / a.rate as f64,
                    book.seed,
                    book.inertia
                ),
            ];
            if !book.note.is_empty() {
                lines.push(book.note.clone());
            }
            lines.push(String::new());
            lines.push(format!(
                "{:<6} {:>7} {:>6} {:>5}",
                "unit", "frames", "share", "run"
            ));
            for i in 0..book.k() {
                let share = if total > 0 {
                    book.counts[i] as f64 / total as f64 * 100.0
                } else {
                    0.0
                };
                lines.push(format!(
                    "{:<6} {:>7} {:>5.1}% {:>5.1}",
                    book.name(i),
                    book.counts[i],
                    share,
                    book.runs[i]
                ));
            }
            emit(
                &o,
                Json::obj(vec![
                    ("path", Json::str(&path)),
                    ("units", Json::Num(book.k() as f64)),
                    ("frames", Json::Num(total as f64)),
                    ("seed", Json::Num(book.seed as f64)),
                    ("inertia", Json::Num(book.inertia)),
                    ("note", Json::str(&book.note)),
                    (
                        "analysis",
                        Json::obj(vec![
                            ("rate", Json::Num(a.rate as f64)),
                            ("frame", Json::Num(a.frame as f64)),
                            ("hop", Json::Num(a.hop as f64)),
                            ("fft", Json::Num(a.fft as f64)),
                            ("bands", Json::Num(a.bands as f64)),
                            ("fmin", Json::Num(a.fmin)),
                            ("fmax", Json::Num(a.fmax)),
                            ("preemphasis", Json::Num(a.preemphasis)),
                            ("normalize", Json::Bool(a.normalize)),
                        ]),
                    ),
                    (
                        "counts",
                        Json::Arr(book.counts.iter().map(|&c| Json::Num(c as f64)).collect()),
                    ),
                    ("runs", Json::nums(&book.runs)),
                ]),
                lines,
            );
        }
        "eval" => {
            let Ok(path) = std::env::var(ENV_LEXICON) else {
                eprintln!("no full dictionary found: set PHONETOK_LEXICON");
                return Ok(2);
            };
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
            let limit = o.limit.unwrap_or(20000);
            let mut seen = std::collections::HashSet::new();
            let (mut words, mut right, mut right_stress, mut errors, mut total) =
                (0usize, 0usize, 0usize, 0usize, 0usize);
            for (word, gold) in read_entries(&text) {
                if !word.bytes().all(|b| b.is_ascii_lowercase()) || !seen.insert(word.clone()) {
                    continue;
                }
                if limit > 0 && words >= limit {
                    break;
                }
                let got = letter_to_sound(&word);
                let (g0, s0) = (strip_stress(&gold), strip_stress(&got));
                words += 1;
                right += usize::from(g0 == s0);
                right_stress += usize::from(gold == got);
                errors += edit_distance(&g0, &s0);
                total += g0.len();
            }
            let w = words.max(1) as f64;
            let line = format!(
                "{words} words of file {path}: {:.1}% right, {:.1}% with stress, phone error rate {:.1}%",
                100.0 * right as f64 / w, 100.0 * right_stress as f64 / w, 100.0 * errors as f64 / total.max(1) as f64
            );
            emit(
                &o,
                Json::obj(vec![
                    ("dictionary", Json::str(&format!("file {path}"))),
                    ("words", Json::Num(words as f64)),
                    ("word_accuracy", Json::Num(right as f64 / w)),
                    (
                        "word_accuracy_with_stress",
                        Json::Num(right_stress as f64 / w),
                    ),
                    (
                        "phone_error_rate",
                        Json::Num(errors as f64 / total.max(1) as f64),
                    ),
                ]),
                vec![line],
            );
        }
        _ => {
            eprintln!("{USAGE}");
            return Ok(2);
        }
    }
    Ok(0)
}

/// The codebook `--codebook` names, or the bundled one.
fn load_book(o: &Options) -> Result<Codebook, String> {
    match &o.codebook {
        Some(path) => Codebook::load(path),
        None => Codebook::bundled(),
    }
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(code) => ExitCode::from(code as u8),
        Err(e) => {
            eprintln!("phonetok: {e}");
            ExitCode::from(1)
        }
    }
}
