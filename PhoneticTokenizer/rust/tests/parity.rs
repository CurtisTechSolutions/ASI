//! The cross-port contract: ../../tests/parity.json is what the Python package
//! says about a battery of texts, and this port must say the same.

use phonetok::json::{self, Json};
use phonetok::lexicon::Lexicon;
use phonetok::mt::Mt;
use phonetok::phones::{features, symbols};
use phonetok::phonotactics::Phonotactics;
use phonetok::rules::letter_to_sound;
use phonetok::syllables::syllabify;
use phonetok::tokenizer::{Level, Tokenizer};
use phonetok::{respell, Transcriber};

fn fixture() -> Json {
    let path = format!("{}/../tests/parity.json", env!("CARGO_MANIFEST_DIR"));
    json::parse(&std::fs::read_to_string(&path).expect("the parity fixture")).expect("valid JSON")
}

fn split(s: &str) -> Vec<String> {
    s.split_whitespace().map(|x| x.to_string()).collect()
}

#[test]
fn alphabet() {
    assert_eq!(fixture().get("symbols").strings(), *symbols());
}

#[test]
fn levels() {
    let doc = fixture();
    for level in Level::ALL {
        let want = doc.get("levels").get(level.as_str());
        let mut tok = Tokenizer::new(level, Lexicon::core());
        let mut texts = Vec::new();
        for (i, row) in want.get("rows").as_arr().iter().enumerate() {
            let text = row.get("text").as_str().unwrap().to_string();
            texts.push(text.clone());
            let toks = tok.tokenize(&text);
            let got: Vec<String> = toks.iter().map(|t| t.text.clone()).collect();
            assert_eq!(
                got,
                row.get("tokens").strings(),
                "{} row {i}: tokens",
                level.as_str()
            );
            let kinds: Vec<String> = toks.iter().map(|t| t.kind.to_string()).collect();
            assert_eq!(
                kinds,
                row.get("kinds").strings(),
                "{} row {i}: kinds",
                level.as_str()
            );
            let words: Vec<i64> = toks.iter().map(|t| t.word).collect();
            assert_eq!(
                words,
                row.get("words").ints(),
                "{} row {i}: words",
                level.as_str()
            );
            let ids: Vec<i64> = tok.encode(&text, true).iter().map(|&i| i as i64).collect();
            assert_eq!(
                ids,
                row.get("ids").ints(),
                "{} row {i}: ids",
                level.as_str()
            );
            let tokens = tok.tokens(&text);
            assert_eq!(
                tok.decode(&tokens),
                row.get("decoded").as_str().unwrap(),
                "{} row {i}: decoded",
                level.as_str()
            );
            let form = tok.text(&text);
            assert_eq!(
                tok.text(&form),
                form,
                "{} row {i}: idempotent",
                level.as_str()
            );
        }
        assert_eq!(
            tok.vocab.tokens,
            want.get("vocab").strings(),
            "{}: vocabulary",
            level.as_str()
        );
        let mut bare = Tokenizer::new(level, Lexicon::core());
        bare.stress = false;
        bare.boundaries = false;
        bare.pauses = false;
        let want_bare = want.get("bare").strings();
        for (i, text) in texts.iter().enumerate() {
            assert_eq!(bare.text(text), want_bare[i], "{} bare {i}", level.as_str());
        }
    }
}

#[test]
fn words() {
    let doc = fixture();
    let mut tok = Tokenizer::new(Level::Phoneme, Lexicon::core());
    let rows = doc.get("levels").get("phoneme").get("rows");
    for (i, row) in rows.as_arr().iter().enumerate() {
        let text = row.get("text").as_str().unwrap();
        let got = tok.explain(text);
        let want = &doc.get("explain").as_arr()[i];
        assert_eq!(got.len(), want.as_arr().len(), "explain {i}: rows");
        for (k, w) in want.as_arr().iter().enumerate() {
            let g = &got[k];
            assert_eq!(g.word, w.get("word").as_str().unwrap(), "explain {i}/{k}");
            assert_eq!(
                g.phones,
                w.get("phones").strings(),
                "explain {i}/{k}: {}",
                g.word
            );
            assert_eq!(
                g.how,
                w.get("how").as_str().unwrap(),
                "explain {i}/{k}: {}",
                g.word
            );
            assert_eq!(
                g.syllables,
                w.get("syllables").strings(),
                "explain {i}/{k}: {}",
                g.word
            );
            assert_eq!(
                g.ipa,
                w.get("ipa").as_str().unwrap(),
                "explain {i}/{k}: {}",
                g.word
            );
        }
        assert_eq!(
            tok.ipa(text),
            doc.get("ipa").get(text).as_str().unwrap(),
            "ipa {i}"
        );
    }
    for (word, want) in doc.get("syllables").as_obj() {
        let phones = tok.pronounce(word);
        let got: Vec<String> = syllabify(&phones, true).iter().map(|s| s.text()).collect();
        assert_eq!(got, want.strings(), "syllables {word}");
    }
    for (word, want) in doc.get("rules").as_obj() {
        assert_eq!(letter_to_sound(word), want.strings(), "rules {word}");
    }
    for (phones, want) in doc.get("respell").as_obj() {
        assert_eq!(
            respell(&split(phones)),
            want.as_str().unwrap(),
            "respell {phones}"
        );
    }
    for (phone, want) in doc.get("features").as_obj() {
        let got = features(phone).unwrap();
        for (i, w) in want.as_arr().iter().enumerate() {
            assert!(
                (got[i] - w.as_f64().unwrap()).abs() < 1e-12,
                "features {phone}[{i}]"
            );
        }
    }
    let mut tr = Transcriber::new(Lexicon::core());
    for w in ["two", "two", "right", "zebra"] {
        tr.word(w);
    }
    for (phones, want) in doc.get("transcriber_spell").as_obj() {
        assert_eq!(
            tr.spell(&split(phones)),
            want.as_str().unwrap(),
            "spell {phones}"
        );
    }
}

#[test]
fn phonotactics() {
    let doc = fixture();
    let lex = Lexicon::core();
    let p = Phonotactics::from_lexicon(&lex, false);
    for (pair, want) in doc.get("affinity").as_obj() {
        let ab = split(pair);
        let w = want.as_arr();
        assert!(
            (p.affinity(&ab[0], &ab[1]) - w[0].as_f64().unwrap()).abs() < 1e-9,
            "affinity {pair}"
        );
        assert!(
            (p.prob(&ab[0], &ab[1]) - w[1].as_f64().unwrap()).abs() < 1e-12,
            "prob {pair}"
        );
    }
    for (phones, want) in doc.get("score").as_obj() {
        assert!(
            (p.score(&split(phones)) - want.as_f64().unwrap()).abs() < 1e-9,
            "score {phones}"
        );
    }
    for (a, want) in doc.get("next").as_obj() {
        let got = p.next(a, 5);
        let w = want.as_arr();
        assert_eq!(got.len(), w.len(), "next {a}");
        for (i, item) in w.iter().enumerate() {
            let pair = item.as_arr();
            assert_eq!(got[i].0, pair[0].as_str().unwrap(), "next {a}[{i}]");
            assert!(
                (got[i].1 - pair[1].as_f64().unwrap()).abs() < 1e-12,
                "next {a}[{i}]"
            );
        }
    }
    let avoid = lex.items();
    let mut tok = Tokenizer::new(Level::Phoneme, Lexicon::core());
    for (key, want) in doc.get("coin").as_obj() {
        let got = if let Some(n) = key.strip_prefix("seed 5 syllables ") {
            p.build(&mut Mt::new(5), n.parse().unwrap(), 1, 3, &[], 50)
        } else {
            p.build(&mut Mt::new(key.parse().unwrap()), 0, 1, 3, &avoid, 50)
        };
        assert_eq!(got, want.strings(), "coin {key}");
    }
    for (pair, want) in doc.get("blend").as_obj() {
        let ab = split(pair);
        let (pa, pb) = (tok.pronounce(&ab[0]), tok.pronounce(&ab[1]));
        let (phones, kept, dropped) = p.blend(&pa, &pb).unwrap();
        assert_eq!(phones, want.get("phones").strings(), "blend {pair}");
        assert_eq!(
            kept as i64,
            want.get("kept").as_f64().unwrap() as i64,
            "blend {pair}: kept"
        );
        assert_eq!(
            dropped as i64,
            want.get("dropped").as_f64().unwrap() as i64,
            "blend {pair}: dropped"
        );
        assert_eq!(
            tok.blend(&ab[0], &ab[1]).unwrap(),
            want.get("spelling").as_str().unwrap(),
            "blend {pair}: spelling"
        );
    }
}

#[test]
fn mersenne_twister_is_pythons() {
    use phonetok::mt::Rng;
    assert_eq!(Mt::new(1).float64(), 0.13436424411240122);
    assert_eq!(Mt::new(0).float64(), 0.8444218515250481);
}

#[test]
fn save_and_load() {
    let dir = std::env::temp_dir().join(format!("phonetok-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tok.json");
    let mut tok = Tokenizer::new(Level::Syllable, Lexicon::core());
    tok.encode("the zebra sat", true);
    tok.save(path.to_str().unwrap()).unwrap();
    let mut back = Tokenizer::load(path.to_str().unwrap(), Lexicon::core()).unwrap();
    assert_eq!(back.vocab.tokens, tok.vocab.tokens);
    let ids = back.encode("zebra", false);
    assert_eq!(back.decode_ids(&ids), "zebra");
    let mut other = Tokenizer::new(Level::Phoneme, Lexicon::core());
    assert!(other.load_json(&tok.to_json()).is_err());
    std::fs::remove_dir_all(&dir).ok();
}

fn nums(j: &Json) -> Vec<f64> {
    j.as_arr().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn close_enough(what: &str, got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tolerance * w.abs().max(1.0),
            "{what}: value {i} is {g}, want {w}"
        );
    }
}

/// The acoustic units: the utterances Python synthesized are heard as the same
/// units, learn the same codebook, and are spoken back as the same samples.
#[test]
fn acoustic() {
    use phonetok::acoustic::{
        frames_of, learn, pcm_samples, synthesize, AcousticTokenizer, Analysis, Codebook,
    };
    use phonetok::synth::{Synthesizer, VoiceSettings};

    let doc = fixture();
    let doc = doc.get("acoustic");
    let voice = doc.get("voice");
    let mut clips = Vec::new();
    for (i, tokens) in doc.get("tokens").as_arr().iter().enumerate() {
        let settings = VoiceSettings {
            pitch: voice.get("pitch").as_f64().unwrap(),
            tempo: voice.get("tempo").as_f64().unwrap(),
            gain: voice.get("gain").as_f64().unwrap(),
            ..VoiceSettings::default()
        };
        let clip = pcm_samples(&Synthesizer::new(16000, settings).speak(&tokens.strings()));
        assert_eq!(
            clip.len() as f64,
            doc.get("samples").as_arr()[i].as_f64().unwrap(),
            "clip {i}"
        );
        clips.push(clip);
    }
    let book = Codebook::bundled().unwrap();
    let frames = frames_of(&clips[0], &book.analysis).unwrap();
    assert_eq!(
        frames.len() as f64,
        doc.get("frames").get("count").as_f64().unwrap()
    );
    let rows = doc.get("frames").get("rows");
    for (key, row) in [
        ("0", &frames[0]),
        ("50", &frames[50]),
        ("last", frames.last().unwrap()),
    ] {
        close_enough(&format!("frame {key}"), row, &nums(rows.get(key)), 1e-9);
    }
    let tok = AcousticTokenizer::new(book.clone(), true).unwrap();
    let heard = tok.listen(&clips[0], 0).unwrap();
    assert_eq!(heard.units, doc.get("units").strings());
    let codes: Vec<f64> = heard.codes.iter().map(|&c| c as f64).collect();
    assert_eq!(codes, nums(doc.get("codes")));
    let want = doc.get("learn");
    let small = learn(
        &clips,
        want.get("k").as_f64().unwrap() as usize,
        want.get("seed").as_f64().unwrap() as i64,
        want.get("iterations").as_f64().unwrap() as usize,
        &Analysis::default(),
        "",
    )
    .unwrap();
    for (c, w) in want.get("centroids").as_arr().iter().enumerate() {
        close_enough(
            &format!("centroid {c}"),
            &small.centroids[c],
            &nums(w),
            1e-9,
        );
    }
    let counts: Vec<f64> = small.counts.iter().map(|&c| c as f64).collect();
    assert_eq!(counts, nums(want.get("counts")));
    assert_eq!(small.runs, nums(want.get("runs")));
    close_enough("mean", &small.mean, &nums(want.get("mean")), 1e-9);
    close_enough(
        "inertia",
        &[small.inertia],
        &[want.get("inertia").as_f64().unwrap()],
        1e-9,
    );
    for key in ["replay", "polished"] {
        let spec = doc.get(key);
        let polish = spec.get("polish").as_f64().unwrap() as usize;
        let pcm = synthesize(&spec.get("units").strings(), &book, polish, 1.0, 120.0).unwrap();
        let values: Vec<i64> = pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as i64)
            .collect();
        assert_eq!(
            values.len() as f64,
            spec.get("samples").as_f64().unwrap(),
            "{key}"
        );
        let at = spec.get("at").as_f64().unwrap() as usize;
        for (i, w) in nums(spec.get("values")).iter().enumerate() {
            let got = values[at + i];
            assert!(
                (got - *w as i64).abs() <= 2,
                "{key}: sample {} is {got}, want {w}",
                at + i
            );
        }
    }
}
