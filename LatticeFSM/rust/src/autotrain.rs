//! Train the N-gram model automatically: from a corpus, and from a local language model through Ollama.
//!
//! The same loop as `latticefsm/autotrain.py`, asking the same questions with the same seeds: round 0 trains on the
//! corpus (every tenth line held out) and measures; each round after that the LLM writes sentences about the
//! round's topics, continues the model's own contexts, and judges texts the model wrote - approved texts' edges
//! rewarded, rejected ones punished - and the model is measured again, saved if asked, stopped once the held-out
//! perplexity has not improved for `patience` rounds.  Ollama's HTTP API is spoken over a plain socket (no
//! dependencies): `OLLAMA_HOST` or `http://127.0.0.1:11434`, the model `LATTICEFSM_OLLAMA_MODEL` or `llama3.2`.

use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use crate::json::{parse, Json};
use crate::ngram::{corpus_sequences, detokenize, split_lines, NGram, Sequence};
use crate::rng::Rng;

pub const DEFAULT_URL: &str = "http://127.0.0.1:11434";
pub const DEFAULT_MODEL: &str = "llama3.2";
pub const CORPUS_PROMPT: &str = "Write {n} different, natural English sentences about {topic}. Write one sentence per \
                                 line, with no numbering, no bullet points and nothing else.";
pub const CONTINUE_PROMPT: &str = "Continue the following text with the next few words only. Reply with the \
                                   continuation alone, without repeating the text and without quotes.\n\nText: {text}";
pub const JUDGE_PROMPT: &str = "Here are {n} numbered texts written by a small language model. For each one, decide \
                                whether it is fluent, sensible English. Reply only with JSON, a list like [{\"id\": 1, \
                                \"ok\": true}].\n\n{texts}";

/// Words of five letters or more too common to be a topic.
pub const STOPWORDS: &str = "about above after again against along among another around because before being below \
                             between both cannot could doing down during each every first further having however \
                             itself might never other others ought since should still their theirs there these thing \
                             things those though through under until upon where whether which while whose within \
                             without would yours always often rather quite";

/// The Ollama address normalised: a scheme added when missing, `0.0.0.0` read as this machine.
pub fn normalise_url(url: Option<&str>) -> String {
    let mut u = url
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .unwrap_or(DEFAULT_URL)
        .to_string();
    if !u.contains("://") {
        u = format!("http://{u}");
    }
    u.trim_end_matches('/').replace("://0.0.0.0", "://127.0.0.1")
}

/// A minimal client for Ollama's HTTP API (`/api/tags`, `/api/generate`).
pub struct Ollama {
    pub url: String,
    pub model: String,
    pub timeout: Duration,
}

impl Ollama {
    pub fn new(url: Option<&str>, model: Option<&str>) -> Ollama {
        let env_url = std::env::var("OLLAMA_HOST").ok();
        let env_model = std::env::var("LATTICEFSM_OLLAMA_MODEL").ok();
        Ollama {
            url: normalise_url(url.or(env_url.as_deref())),
            model: model
                .map(String::from)
                .or(env_model.filter(|m| !m.is_empty()))
                .unwrap_or_else(|| DEFAULT_MODEL.to_string()),
            timeout: Duration::from_secs(120),
        }
    }

    fn unreachable(&self, e: impl std::fmt::Display) -> String {
        format!(
            "Ollama at {} is not answering ({e}); start it with `ollama serve` and pull a model (`ollama pull {}`)",
            self.url, self.model
        )
    }

    fn request(&self, method: &str, path: &str, body: Option<&Json>) -> Result<Json, String> {
        let rest = self.url.split_once("://").map(|(_, r)| r).unwrap_or(&self.url);
        let host = rest.split('/').next().unwrap_or(rest);
        let addr = if host.contains(':') {
            host.to_string()
        } else {
            format!("{host}:80")
        };
        let sock = addr
            .to_socket_addrs()
            .map_err(|e| self.unreachable(e))?
            .next()
            .ok_or_else(|| self.unreachable("no address"))?;
        let mut stream = TcpStream::connect_timeout(&sock, Duration::from_secs(5)).map_err(|e| self.unreachable(e))?;
        stream.set_read_timeout(Some(self.timeout)).ok();
        let payload = body.map(Json::dump).unwrap_or_default();
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nAccept: application/json\r\n\
             Connection: close\r\nContent-Length: {}\r\n\r\n",
            payload.len()
        );
        stream.write_all(head.as_bytes()).map_err(|e| self.unreachable(e))?;
        stream.write_all(payload.as_bytes()).map_err(|e| self.unreachable(e))?;
        let mut raw = Vec::new();
        stream.read_to_end(&mut raw).map_err(|e| self.unreachable(e))?;
        let split = raw
            .windows(4)
            .position(|w| w == b"\r\n\r\n")
            .ok_or("Ollama: no response head")?;
        let head = String::from_utf8_lossy(&raw[..split]).to_string();
        let mut body = raw[split + 4..].to_vec();
        let status: u16 = head
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse().ok())
            .ok_or("Ollama: no status")?;
        if head.to_lowercase().contains("transfer-encoding: chunked") {
            body = dechunk(&body)?;
        }
        let text = String::from_utf8_lossy(&body).to_string();
        if status >= 400 {
            return Err(format!(
                "Ollama {path}: HTTP {status} {}",
                text.chars().take(300).collect::<String>()
            ));
        }
        parse(&text).map_err(|e| format!("Ollama {path}: not JSON ({e})"))
    }

    /// The models pulled (`GET /api/tags`).
    pub fn models(&self) -> Result<Vec<String>, String> {
        let v = self.request("GET", "/api/tags", None)?;
        let list = v
            .get("models")
            .and_then(Json::as_array)
            .ok_or("unexpected /api/tags response")?;
        Ok(list.iter().map(|m| m.str_or("name", "").to_string()).collect())
    }

    pub fn available(&self) -> bool {
        self.models().is_ok()
    }

    /// One completion (`POST /api/generate`, not streamed).
    pub fn generate(
        &self,
        prompt: &str,
        temperature: f64,
        num_predict: Option<u32>,
        seed: Option<u64>,
    ) -> Result<String, String> {
        let mut options = Json::object().with("temperature", temperature.into());
        if let Some(n) = num_predict {
            options = options.with("num_predict", (n as f64).into());
        }
        if let Some(s) = seed {
            options = options.with("seed", (s as f64).into());
        }
        let body = Json::object()
            .with("model", self.model.as_str().into())
            .with("prompt", prompt.into())
            .with("stream", false.into())
            .with("options", options);
        let v = self.request("POST", "/api/generate", Some(&body))?;
        let text = v
            .get("response")
            .and_then(Json::as_str)
            .ok_or("unexpected /api/generate response (no 'response' text)")?;
        Ok(strip_thinking(text))
    }
}

fn dechunk(body: &[u8]) -> Result<Vec<u8>, String> {
    let mut out = Vec::new();
    let mut at = 0;
    loop {
        let end = body[at..].windows(2).position(|w| w == b"\r\n").ok_or("a bad chunk")? + at;
        let size_text = String::from_utf8_lossy(&body[at..end]);
        let size = usize::from_str_radix(size_text.split(';').next().unwrap_or("").trim(), 16)
            .map_err(|_| "a bad chunk size")?;
        at = end + 2;
        if size == 0 {
            return Ok(out);
        }
        out.extend_from_slice(body.get(at..at + size).ok_or("a chunk runs past the end")?);
        at += size + 2;
    }
}

fn strip_thinking(text: &str) -> String {
    let mut s = text.to_string();
    while let Some(a) = s.find("<think>") {
        match s[a..].find("</think>") {
            Some(b) => s.replace_range(a..a + b + "</think>".len(), ""),
            None => break,
        }
    }
    s.trim().to_string()
}

const QUOTES: [char; 4] = ['"', '\'', '“', '”'];

/// A bullet or number at the start of a line, as `^\s*(?:[-*•]+|\(?\d+[.):]|\d+\s*-)\s*` matches it: the rest.
fn strip_bullet(line: &str) -> &str {
    let s = line.trim_start();
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let at = |i: usize| chars.get(i).map(|&(_, c)| c);
    let byte = |i: usize| chars.get(i).map(|&(b, _)| b).unwrap_or(s.len());
    let mut end = None;
    // [-*•]+
    let mut i = 0;
    while at(i).is_some_and(|c| c == '-' || c == '*' || c == '•') {
        i += 1;
    }
    if i > 0 {
        end = Some(i);
    } else {
        // \(?\d+[.):]
        let mut j = 0;
        if at(j) == Some('(') {
            j += 1;
        }
        let d0 = j;
        while at(j).is_some_and(|c| c.is_ascii_digit()) {
            j += 1;
        }
        if j > d0 && at(j).is_some_and(|c| ".):".contains(c)) {
            end = Some(j + 1);
        } else {
            // \d+\s*-
            let mut k = 0;
            while at(k).is_some_and(|c| c.is_ascii_digit()) {
                k += 1;
            }
            if k > 0 {
                let mut m = k;
                while at(m).is_some_and(char::is_whitespace) {
                    m += 1;
                }
                if at(m) == Some('-') {
                    end = Some(m + 1);
                }
            }
        }
    }
    match end {
        Some(e) => s[byte(e)..].trim_start(),
        None => line,
    }
}

/// An LLM's lines: numbering, bullets and wrapping quotes stripped; empty lines and preambles (ending in `:`)
/// dropped.
pub fn parse_lines(text: &str, limit: Option<usize>) -> Vec<String> {
    let mut out = Vec::new();
    for raw in crate::ngram::split_text_lines(text) {
        let line = strip_bullet(&raw).trim().trim_matches(&QUOTES[..]).trim().to_string();
        if line.is_empty() || line.ends_with(':') {
            continue;
        }
        out.push(line);
        if limit.is_some_and(|l| out.len() >= l) {
            break;
        }
    }
    out
}

/// The judge's verdicts on `n` texts: the JSON list it was asked for, else lines like `1: yes`.
pub fn parse_judgements(text: &str, n: usize) -> Vec<Option<bool>> {
    let mut verdicts = vec![None; n];
    if let (Some(a), Some(b)) = (text.find('['), text.rfind(']')) {
        if b > a {
            if let Ok(items) = parse(&text[a..=b]) {
                if let Some(list) = items.as_array() {
                    for item in list {
                        if let (Some(id), Some(Json::Bool(ok))) =
                            (item.get("id").and_then(Json::as_f64), item.get("ok"))
                        {
                            let i = id as i64 - 1;
                            if i >= 0 && (i as usize) < n {
                                verdicts[i as usize] = Some(*ok);
                            }
                        }
                    }
                }
                return verdicts;
            }
        }
    }
    let chars: Vec<char> = text.chars().collect();
    let lower: Vec<char> = text.to_lowercase().chars().collect();
    let words = ["yes", "no", "true", "false", "ok", "fail"];
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_ascii_digit() {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < chars.len() && chars[j].is_ascii_digit() {
            j += 1;
        }
        let num: usize = chars[i..j].iter().collect::<String>().parse().unwrap_or(0);
        let mut k = j;
        while k < chars.len() && chars[k].is_whitespace() {
            k += 1;
        }
        let mut matched = None;
        if k < chars.len() && ":.)-".contains(chars[k]) {
            k += 1;
            while k < chars.len() && chars[k].is_whitespace() {
                k += 1;
            }
            for w in words {
                let wc: Vec<char> = w.chars().collect();
                if lower.len() == chars.len() && k + wc.len() <= lower.len() && lower[k..k + wc.len()] == wc[..] {
                    matched = Some((w, k + wc.len()));
                    break;
                }
            }
        }
        match matched {
            Some((w, end)) => {
                if num >= 1 && num <= n {
                    verdicts[num - 1] = Some(matches!(w, "yes" | "true" | "ok"));
                }
                i = end;
            }
            None => i += 1,
        }
    }
    verdicts
}

fn topics(model: &NGram, count: usize) -> Vec<String> {
    let stop: Vec<&str> = STOPWORDS.split_whitespace().collect();
    let mut seen: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
    for row in model.grams.values() {
        for (&w, &c) in row {
            *seen.entry(w).or_insert(0) += c;
        }
    }
    let mut words: Vec<u32> = seen
        .keys()
        .copied()
        .filter(|&w| {
            let t = &model.vocab[w as usize];
            !t.is_empty() && t.chars().all(char::is_alphabetic) && t.chars().count() >= 5 && !stop.contains(&t.as_str())
        })
        .collect();
    words.sort_by(|a, b| seen[b].cmp(&seen[a]).then(a.cmp(b)));
    let out: Vec<String> = words
        .iter()
        .take(count)
        .map(|&w| model.vocab[w as usize].clone())
        .collect();
    if out.is_empty() {
        vec!["everyday life".to_string()]
    } else {
        out
    }
}

fn prefixes(model: &NGram, lines: &[Sequence], count: usize, words: usize, offset: usize) -> Vec<String> {
    let n = lines.len();
    let mut out = Vec::new();
    if n == 0 {
        return out;
    }
    let step = (n / count.max(1)).max(1);
    let take = if model.unit == "base64" { 4 * words } else { words };
    for j in 0..count {
        let mut toks = lines[(offset + j * step) % n].clone();
        toks.truncate(take);
        if !toks.is_empty() {
            out.push(detokenize(&toks, &model.unit));
        }
    }
    out
}

/// The settings of an autotrain run.
pub struct Options {
    pub data: Vec<String>,
    pub held_out: usize,
    pub topics: Vec<String>,
    pub rounds: usize,
    pub lines: usize,
    pub continuations: usize,
    pub judge: usize,
    pub prefix_words: usize,
    pub topics_per_round: usize,
    pub patience: usize,
    pub keep: Option<String>,
    pub save: Option<String>,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Options {
        Options {
            data: vec!["data/corpus".to_string()],
            held_out: 10,
            topics: Vec::new(),
            rounds: 3,
            lines: 20,
            continuations: 20,
            judge: 10,
            prefix_words: 4,
            topics_per_round: 2,
            patience: 2,
            keep: None,
            save: None,
            seed: 1,
        }
    }
}

fn fmt_opt(v: Option<f64>, digits: usize) -> String {
    v.map(|x| format!("{x:.digits$}")).unwrap_or_else(|| "-".to_string())
}

/// Train on the corpus, then up to `rounds` rounds from Ollama (when given); every round measured on the held-out
/// corpus lines and recorded in `model.history`.  `log` hears one line a round.
pub fn autotrain(
    model: &mut NGram,
    o: &Options,
    ollama: Option<&Ollama>,
    log: &mut dyn FnMut(&str),
) -> Result<Vec<Json>, String> {
    let (mut train, mut test) = (Vec::new(), Vec::new());
    for (path, lines) in corpus_sequences(&o.data, &model.unit.clone(), model.lower)? {
        let (tr, te) = if o.held_out > 0 {
            split_lines(&lines, o.held_out)
        } else {
            (lines, Vec::new())
        };
        model.train_sequences(&tr, Some(&path));
        train.extend(tr);
        test.extend(te);
    }
    let mut records: Vec<Json> = Vec::new();
    let mut record = |model: &mut NGram,
                      r: usize,
                      added: Vec<(&str, usize)>,
                      log: &mut dyn FnMut(&str)|
     -> Result<Option<f64>, String> {
        let ev = if test.is_empty() {
            Default::default()
        } else {
            model.evaluate(&test, 2000)
        };
        let mut rec = Json::object()
            .with("round", r.into())
            .with("tokens", (model.clock as f64).into())
            .with("vocab", (model.vocab.len() - 3).into())
            .with("ngrams", model.n_grams().into())
            .with("credited_edges", model.credited_edges().into())
            .with("perplexity", ev.perplexity.map(Json::from).unwrap_or(Json::Null))
            .with("bits", ev.bits.map(Json::from).unwrap_or(Json::Null))
            .with("accuracy", ev.accuracy.map(Json::from).unwrap_or(Json::Null))
            .with("top5", ev.top5.map(Json::from).unwrap_or(Json::Null))
            .with("held_out", test.len().into());
        let mut line = format!(
            "round {r}: {} tokens, {} words, {} n-grams; held out: perplexity {}, next-token accuracy {}",
            model.clock,
            model.vocab.len() - 3,
            model.n_grams(),
            fmt_opt(ev.perplexity, 2),
            fmt_opt(ev.accuracy, 3)
        );
        for (k, v) in added {
            rec = rec.with(k, v.into());
            line.push_str(&format!(", {k} {v}"));
        }
        log(&line);
        model.history.push(rec.clone());
        records.push(rec);
        if let Some(path) = &o.save {
            model.save(path)?;
        }
        Ok(ev.perplexity)
    };
    let mut best = record(model, 0, vec![("corpus_lines", train.len())], log)?;
    let Some(ollama) = ollama else { return Ok(records) };
    if o.rounds == 0 {
        return Ok(records);
    }
    let subjects = if o.topics.is_empty() {
        topics(model, 8)
    } else {
        o.topics.clone()
    };
    let mut stale = 0;
    let mut rng = Rng::new(o.seed);
    for r in 1..=o.rounds {
        let mut written: Vec<String> = Vec::new();
        for j in 0..o.topics_per_round {
            let topic = &subjects[((r - 1) * o.topics_per_round + j) % subjects.len()];
            let prompt = CORPUS_PROMPT
                .replace("{n}", &o.lines.to_string())
                .replace("{topic}", topic);
            let reply = ollama.generate(&prompt, 0.8, None, Some(o.seed * 1000 + r as u64 * 10 + j as u64))?;
            let got = parse_lines(&reply, Some(o.lines));
            model.train_lines(&got, Some(&format!("ollama:{}:corpus:{topic}", ollama.model)));
            written.extend(got);
        }
        let mut continued: Vec<String> = Vec::new();
        for (j, prefix) in prefixes(model, &train, o.continuations, o.prefix_words, r * 7919)
            .iter()
            .enumerate()
        {
            let prompt = CONTINUE_PROMPT.replace("{text}", prefix);
            let reply = ollama.generate(&prompt, 0.7, Some(32), Some(o.seed * 1000 + r as u64 * 10 + j as u64))?;
            if let Some(rest) = parse_lines(&reply, Some(1)).first() {
                continued.push(format!("{prefix} {rest}"));
            }
        }
        model.train_lines(&continued, Some(&format!("ollama:{}:continuations", ollama.model)));
        let (mut judged, mut approved) = (0usize, 0usize);
        if o.judge > 0 {
            let texts: Vec<String> = prefixes(model, &train, o.judge, 2, r * 104729)
                .iter()
                .map(|p| model.complete(p, 24, 0.0, &mut rng))
                .filter(|t| !t.trim().is_empty())
                .collect();
            if !texts.is_empty() {
                let numbered: Vec<String> = texts
                    .iter()
                    .enumerate()
                    .map(|(i, t)| format!("{}. {t}", i + 1))
                    .collect();
                let prompt = JUDGE_PROMPT
                    .replace("{n}", &texts.len().to_string())
                    .replace("{texts}", &numbered.join("\n"));
                let reply = ollama.generate(&prompt, 0.0, None, Some(o.seed * 1000 + r as u64))?;
                for (text, ok) in texts.iter().zip(parse_judgements(&reply, texts.len())) {
                    if let Some(ok) = ok {
                        model.credit_text(text, if ok { 1.0 } else { -1.0 });
                        judged += 1;
                        approved += ok as usize;
                    }
                }
            }
        }
        if let Some(keep) = &o.keep {
            if !written.is_empty() || !continued.is_empty() {
                if let Some(dir) = std::path::Path::new(keep).parent() {
                    if !dir.as_os_str().is_empty() {
                        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                    }
                }
                let mut f = OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(keep)
                    .map_err(|e| format!("{keep}: {e}"))?;
                for line in written.iter().chain(continued.iter()) {
                    writeln!(f, "{line}").map_err(|e| e.to_string())?;
                }
            }
        }
        let ppl = record(
            model,
            r,
            vec![
                ("ollama_lines", written.len()),
                ("continuations", continued.len()),
                ("judged", judged),
                ("approved", approved),
            ],
            log,
        )?;
        if let (Some(b), Some(p)) = (best, ppl) {
            if p < b - 1e-9 {
                best = Some(p);
                stale = 0;
            } else {
                stale += 1;
                if stale >= o.patience {
                    log(&format!(
                        "held-out perplexity has not improved for {} rounds: stopping",
                        o.patience
                    ));
                    break;
                }
            }
        }
    }
    Ok(records)
}
