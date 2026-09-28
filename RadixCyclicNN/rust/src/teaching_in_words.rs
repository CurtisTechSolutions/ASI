//! The teaching loops of a model of sounds: every loop that shows an LLM what
//! the model wrote shows it the words its sounds spell
//! ([`crate::phonetic::reader_text`]), while the model is rewarded, punished
//! and blamed for the sounds it said.  Python's
//! `tests/test_teaching_in_words.py`, loop for loop.

use std::sync::Mutex;

use crate::chat::{Chat, ChatConfig};
use crate::encoding::{parse_encoding, Encoding};
use crate::json::Json;
use crate::llm::{LlmClient, LlmError, LlmOptions};
use crate::model::{GenerateOptions, Model};
use crate::negative::NegativeOptions;
use crate::review::{
    correct_texts, faults_from_corrections, review_texts, teach_corrections, CHAT_SPEAKERS, DEFAULT_BATCH,
};
use crate::tutor::{faults_from_lessons, Networks, TutorConfig, TutorTrainer};
use crate::voicechat::{turn, VoiceOptions};
use crate::{GraphOptions, TrainOptions};

const CORPUS: [&str; 5] = [
    "the cat sat on the mat",
    "the cat sat on the floor",
    "the dog sat on the mat",
    "the dog ran to the park",
    "a bird sang in the tree",
];

/// Whether a text holds what English never does: a stressed vowel (`AE1`), a
/// word boundary (`#`) or a syllable (`K.AE1`).
fn holds_sounds(text: &str) -> bool {
    text.split_whitespace().any(|word| {
        let bytes = word.as_bytes();
        let stressed = (2..=3).contains(&bytes.len())
            && bytes[..bytes.len() - 1].iter().all(u8::is_ascii_uppercase)
            && matches!(bytes[bytes.len() - 1], b'0'..=b'2');
        let syllable = word.contains('.')
            && word
                .split('.')
                .all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit()));
        word == "#" || stressed || syllable
    })
}

/// A scripted LLM: every request is answered from a script, and every prompt recorded.
struct Teacher {
    prompts: Mutex<Vec<String>>,
}

impl Teacher {
    fn new() -> Teacher {
        Teacher {
            prompts: Mutex::new(Vec::new()),
        }
    }

    fn fix(text: &str) -> String {
        match text.rfind(' ') {
            Some(at) => format!("{} mat", &text[..at]),
            None => text.to_string(),
        }
    }

    fn words_only(&self) {
        let prompts = self.prompts.lock().unwrap();
        assert!(!prompts.is_empty(), "the LLM was never asked");
        for prompt in prompts.iter() {
            assert!(!holds_sounds(prompt), "the LLM was shown sounds:\n{prompt}");
        }
    }

    fn last(&self) -> String {
        self.prompts.lock().unwrap().last().cloned().unwrap_or_default()
    }
}

/// A prompt's `[i] text` lines.
fn indexed(prompt: &str) -> Vec<(i64, String)> {
    prompt
        .lines()
        .filter_map(|line| {
            let end = line.find(']')?;
            let index = line.strip_prefix('[')?.get(..end - 1)?.parse().ok()?;
            Some((index, line.get(end + 2..).unwrap_or("").to_string()))
        })
        .collect()
}

impl LlmClient for Teacher {
    fn provider(&self) -> &'static str {
        "ollama"
    }
    fn url(&self) -> &str {
        "http://scripted"
    }
    fn model(&self) -> &str {
        "scripted"
    }
    fn models(&self) -> Result<Vec<Json>, LlmError> {
        Ok(Vec::new())
    }
    fn generate(&self, prompt: &str, _o: &LlmOptions) -> Result<String, LlmError> {
        self.prompts.lock().unwrap().push(prompt.to_string());
        let each = |make: &dyn Fn(i64, &str) -> Json| -> Json {
            Json::Arr(indexed(prompt).iter().map(|(i, text)| make(*i, text)).collect())
        };
        let answer = if prompt.contains("exercises now") {
            Json::obj([(
                "exercises",
                Json::Arr(vec![
                    Json::obj([
                        ("prefix", Json::str("The cat")),
                        ("answer", Json::str("The cat sat on the mat.")),
                        ("focus", Json::str("past tense")),
                    ]),
                    Json::obj([
                        ("prefix", Json::str("The dog")),
                        ("answer", Json::str("The dog sat on the mat.")),
                        ("focus", Json::str("past tense")),
                    ]),
                ]),
            )])
        } else if prompt.contains("Mark these") {
            Json::obj([(
                "grades",
                each(&|i, _| {
                    Json::obj([
                        ("index", Json::Int(i)),
                        ("grammar", Json::Int(3)),
                        ("spelling", Json::Int(5)),
                        ("fluency", Json::Int(3)),
                        ("error", Json::str("tense")),
                        ("correction", Json::str("The cat sits on the mat.")),
                        ("comment", Json::str("Use the present tense.")),
                    ])
                }),
            )])
        } else if prompt.contains("Explain these") {
            Json::obj([(
                "mistakes",
                each(&|i, _| {
                    Json::obj([
                        ("index", Json::Int(i)),
                        ("why", Json::str("the tense is wrong")),
                        (
                            "again",
                            Json::Arr(vec![Json::obj([
                                ("wrong", Json::str("the dog sit")),
                                ("right", Json::str("the dog sits")),
                            ])]),
                        ),
                    ])
                }),
            )])
        } else if prompt.contains("Review these") {
            let item = |i: i64, _: &str| {
                Json::obj([
                    ("index", Json::Int(i)),
                    ("rating", Json::Int(2)),
                    ("critique", Json::str("it repeats itself")),
                ])
            };
            Json::obj([("reviews", each(&item))])
        } else if prompt.contains("Correct these") {
            Json::obj([(
                "corrections",
                each(&|i, text| {
                    Json::obj([
                        ("index", Json::Int(i)),
                        ("correction", Json::str(Teacher::fix(text))),
                        ("reason", Json::str("vocabulary")),
                        ("note", Json::str("the last word")),
                    ])
                }),
            )])
        } else if prompt.contains("The conversation:") {
            let item = |i: i64, _: &str| {
                Json::obj([
                    ("index", Json::Int(i)),
                    ("rating", Json::Int(2)),
                    ("critique", Json::str("it does not follow on")),
                ])
            };
            Json::obj([
                ("reviews", each(&item)),
                (
                    "overall",
                    Json::obj([("rating", Json::Int(3)), ("critique", Json::str("short"))]),
                ),
            ])
        } else {
            // a line of a conversation: the partner's, or Ollama's for the Voice tab
            return Ok("the dog sat on the mat".to_string());
        };
        Ok(answer.render(0))
    }
    fn chat(&self, messages: &[Json], o: &LlmOptions) -> Result<String, LlmError> {
        let said: Vec<&str> = messages.iter().filter_map(|m| m.at("content").as_str()).collect();
        self.generate(&said.join("\n"), o)
    }
}

fn syllables() -> Encoding {
    parse_encoding("syllable:2:1").unwrap()
}

fn sounds_model() -> Model {
    let opts = GraphOptions {
        encoding: syllables(),
        ..GraphOptions::default()
    };
    let mut model = Model::new(1, opts).unwrap();
    model.workers = 1;
    model.g.workers = 1;
    let corpus: Vec<String> = CORPUS.iter().map(|s| s.to_string()).collect();
    model
        .train(
            &corpus,
            &TrainOptions {
                epochs: 2,
                ..Default::default()
            },
        )
        .unwrap();
    model
}

fn sounds_negative(model: &Model) -> Model {
    Model::new_negative(
        1,
        &NegativeOptions {
            encoding: model.encoding(),
            ..NegativeOptions::default()
        },
    )
    .unwrap()
}

#[test]
fn the_tutor_marks_words_and_teaches_the_sounds() {
    let mut model = sounds_model();
    let mut negative = sounds_negative(&model);
    let teacher = Teacher::new();
    let mut trainer = TutorTrainer::new(
        &teacher,
        &teacher,
        TutorConfig {
            topic: "animals".to_string(),
            exercises: 2,
            attempts: 1,
            variants: 1,
            ..Default::default()
        },
    )
    .unwrap();
    let (record, lessons) = trainer
        .run_round(
            &mut Networks::Owned {
                model: &mut model,
                negative: Some(&mut negative),
            },
            1,
            1,
        )
        .unwrap();
    teacher.words_only();
    assert_eq!(lessons.len(), 2);
    for lesson in &lessons {
        // the teacher's prefix as it wrote it, then the words the sounds spell
        assert!(
            lesson.sentence.starts_with(&lesson.exercise.cue()),
            "{}",
            lesson.sentence
        );
        assert!(!holds_sounds(&lesson.sentence), "{}", lesson.sentence);
        assert_eq!(lesson.prefix(), lesson.exercise.cue());
        // ... and what the model said, in its own units
        assert!(lesson.said.starts_with("DH.AH0 # "), "{}", lesson.said);
        assert_eq!(lesson.own(), lesson.said);
        assert_eq!(lesson.to_json().at("said").as_str(), Some(lesson.said.as_str()));
    }
    let all: Vec<&_> = lessons.iter().collect();
    let corrections = trainer.corrections_of(&all, true);
    assert_eq!(
        corrections.iter().map(|c| c.wrong.as_str()).collect::<Vec<_>>(),
        lessons.iter().map(|l| l.said.as_str()).collect::<Vec<_>>()
    );
    let (faults, cleared) = faults_from_lessons(&lessons, 6.0, "tutor");
    for fault in faults.iter().filter(|f| f.source == "tutor") {
        assert!(lessons.iter().any(|l| l.said == fault.text), "blamed {:?}", fault.text);
        assert_eq!(fault.correction, "The cat sits on the mat.");
    }
    assert!(cleared.contains(&"The cat sits on the mat.".to_string()));
    assert!(record.at("corrections").as_i64().unwrap() > 0);
    assert!(record.at("edits").as_i64().unwrap() > 0);
    assert!(record.at("negative_blamed").as_i64().unwrap() > 0);
}

#[test]
fn the_reviewer_and_the_editor_read_words() {
    let mut model = sounds_model();
    let enc = model.encoding();
    let sounds: Vec<String> = model
        .generate(&GenerateOptions {
            count: 3,
            mode: "beam".to_string(),
            prefix: "the".to_string(),
            max_length: 20,
            ..Default::default()
        })
        .unwrap()
        .into_iter()
        .map(|r| r.text)
        .collect();
    let teacher = Teacher::new();
    let reviews = review_texts(&teacher, &sounds, "", "", 6.0, DEFAULT_BATCH, enc).unwrap();
    for (review, said) in reviews.iter().zip(&sounds) {
        assert_eq!(review.text, *said);
        assert_eq!(review.spelled.as_deref(), Some(enc.spell(said).as_str()));
    }
    let corrections = correct_texts(&teacher, &sounds, "", "", DEFAULT_BATCH, enc).unwrap();
    teacher.words_only();
    for (entry, said) in corrections.iter().zip(&sounds) {
        let words = enc.spell(said);
        assert_eq!(
            (entry.text.as_str(), entry.spelled.as_deref()),
            (said.as_str(), Some(words.as_str()))
        );
        assert_eq!(entry.correction.as_deref(), Some(Teacher::fix(&words).as_str()));
        for change in &entry.changes {
            assert!(
                !holds_sounds(&format!("{} {}", change.wrong, change.right)),
                "{change:?}"
            );
        }
    }
    let (faults, _) = faults_from_corrections(&corrections, 1.0, "correction");
    assert!(!faults.is_empty() && faults.iter().all(|f| sounds.contains(&f.text)));
    let mut negative = sounds_negative(&model);
    let report = teach_corrections(
        &mut negative,
        &corrections,
        1.0,
        true,
        "correction",
        &Default::default(),
    )
    .unwrap();
    assert!(
        report.edges > 0,
        "the correction, read as sounds, lined up with the sounds said"
    );
    // the default encoding reads every text as it is
    let plain = review_texts(
        &Teacher::new(),
        &["the cat sat".to_string()],
        "",
        "",
        6.0,
        20,
        Encoding::default(),
    )
    .unwrap();
    assert_eq!(plain[0].spelled, None);
}

#[test]
fn the_chat_partner_and_the_judge_hear_words() {
    let mut model = sounds_model();
    let mut negative = sounds_negative(&model);
    let enc = model.encoding();
    let teacher = Teacher::new();
    let mut chat = Chat::new(
        &teacher,
        &teacher,
        ChatConfig {
            turns: 2,
            opening: "the cat".to_string(),
            guard: false,
            ..Default::default()
        },
    )
    .unwrap();
    let record = chat
        .run_conversation(&mut Networks::Owned {
            model: &mut model,
            negative: Some(&mut negative),
        })
        .unwrap();
    teacher.words_only();
    let mut said = 0;
    for line in record.at("transcript").as_array() {
        let text = line.at("text").as_str().unwrap();
        if line.at("speaker").as_str() == Some(CHAT_SPEAKERS[1]) {
            said += 1;
            assert_eq!(line.at("spelled").as_str(), Some(enc.spell(text).as_str()));
        } else {
            assert!(line.get("spelled").is_none(), "{}", line.render(0));
        }
    }
    assert!(said > 0);
}

#[test]
fn the_voice_tab_s_ollama_hears_words() {
    let teacher = Teacher::new();
    let mut o = VoiceOptions {
        answer: "ollama".to_string(),
        train: false,
        speak: false,
        ..Default::default()
    };
    o.teach.transcript = "how are you".to_string();
    let history = vec![
        ("You".to_string(), "hello".to_string()),
        ("Model".to_string(), "DH.AH0 # K.AE1.T # S.AE1.T".to_string()),
    ];
    let out = turn(None, syllables(), &o, &history, None, None, Some(&teacher), None).unwrap();
    teacher.words_only();
    assert!(teacher.last().contains("Model: the cat sat"), "{}", teacher.last());
    assert_eq!(out.document.at("by").as_str(), Some("ollama"));
}
