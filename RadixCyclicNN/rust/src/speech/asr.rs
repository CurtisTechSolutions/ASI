//! Speech to text, and the programs speech needs from the machine: the
//! transcription backends, `ffmpeg` for audio that is not a WAV, and the
//! microphone recorders (`radixnet/speech.py`'s second half).
//!
//! Python has four transcription backends.  Two of them - `faster-whisper`
//! and `openai-whisper` - are Python packages running a model in-process, and
//! nothing here can load one, so they are named and refused with the reason.
//! The other two are what a crate with no dependencies *can* do, and so they
//! are here in full:
//!
//! * `given` - the words come with the audio: the browser's dictation,
//!   `--text`, the API's `transcript`.  Always available.
//! * `server` - any OpenAI-compatible `/v1/audio/transcriptions` endpoint
//!   (whisper.cpp's server, Speaches, ...), spoken to over [`crate::fetch`]
//!   with the WAV as a multipart upload; `auto` picks it when
//!   `$RADIXNET_ASR_URL` names one.
//!
//! `ffmpeg` and the recorders are child processes, as they are in Python:
//! [`std::process`] is the standard library, and shelling out to the tool that
//! decodes every audio format is what keeps an MP3 decoder out of the crate.

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::json::Json;
use crate::multipart::base64::is_python_space;
use crate::negative::python_repr;

/// The transcription backends, in the order `--backend` lists them.
pub const ASR_BACKENDS: &[&str] = &["auto", "given", "faster-whisper", "whisper", "server"];

/// The microphone recorders `speech listen` tries, in order.
pub const RECORDERS: &[&str] = &["arecord", "rec", "sox", "ffmpeg"];

/// `$RADIXNET_WHISPER_MODEL`: the local Whisper size (reported; the backends are Python's).
pub fn default_whisper_model() -> String {
    std::env::var("RADIXNET_WHISPER_MODEL").unwrap_or_else(|_| "base".to_string())
}

/// `$RADIXNET_ASR_URL`: the transcription server `auto` falls back to.
pub fn default_asr_url() -> String {
    std::env::var("RADIXNET_ASR_URL").unwrap_or_default()
}

/// `$RADIXNET_ASR_MODEL`: the model name sent to the transcription server.
pub fn default_asr_model() -> String {
    std::env::var("RADIXNET_ASR_MODEL")
        .ok()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(|| "whisper-1".to_string())
}

/// `$RADIXNET_ASR_TIMEOUT`: seconds one transcription may take (300).
fn asr_timeout() -> Duration {
    let seconds = std::env::var("RADIXNET_ASR_TIMEOUT")
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|s| *s > 0.0)
        .unwrap_or(300.0);
    Duration::from_secs_f64(seconds)
}

/// What one transcription came back with (`transcribe()`'s dict).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Transcript {
    pub transcript: String,
    pub backend: String,
    pub model: Option<String>,
    pub language: Option<String>,
    pub seconds: f64,
}

impl Transcript {
    /// `{"transcript", "model", "language", "backend", "seconds"}`.
    pub fn to_json(&self) -> Json {
        let opt = |v: &Option<String>| v.clone().map(Json::Str).unwrap_or(Json::Null);
        Json::obj([
            ("transcript", Json::str(self.transcript.clone())),
            ("model", opt(&self.model)),
            ("language", opt(&self.language)),
            ("backend", Json::str(self.backend.clone())),
            ("seconds", Json::Num(super::py_round(self.seconds, 3))),
        ])
    }
}

/// How one transcription is asked for.
#[derive(Clone, Debug, Default)]
pub struct AsrOptions {
    /// `auto`, `given`, `faster-whisper`, `whisper` or `server`.
    pub backend: String,
    /// A transcript the caller already has; it wins under `auto`.
    pub text: String,
    pub language: Option<String>,
    /// The Whisper size or the server's model name.
    pub model: Option<String>,
    /// The transcription server (default `$RADIXNET_ASR_URL`).
    pub url: Option<String>,
}

/// A backend, resolved.
enum Backend {
    Given(String),
    Server {
        url: String,
        model: String,
    },
    /// A Python package: named, so the refusal can say which.
    Python(&'static str),
}

impl Backend {
    fn name(&self) -> &'static str {
        match self {
            Backend::Given(_) => "given",
            Backend::Server { .. } => "server",
            Backend::Python(name) => name,
        }
    }
}

/// The backend `backend` names; `auto` picks the first usable one
/// (`get_transcriber`): a given transcript, then a configured server.
fn resolve(o: &AsrOptions) -> Result<Backend, String> {
    let raw = if o.backend.trim().is_empty() {
        "auto"
    } else {
        o.backend.as_str()
    };
    let key = raw.trim().to_lowercase().replace('_', "-");
    let url = o.url.clone().unwrap_or_else(default_asr_url).trim().to_string();
    let model = o
        .model
        .clone()
        .filter(|m| !m.is_empty())
        .unwrap_or_else(default_asr_model);
    match key.as_str() {
        "given" => Ok(Backend::Given(o.text.clone())),
        "server" => Ok(Backend::Server { url, model }),
        "faster-whisper" => Ok(Backend::Python("faster-whisper")),
        "whisper" => Ok(Backend::Python("whisper")),
        "auto" => {
            if !o.text.trim_matches(is_python_space).is_empty() {
                Ok(Backend::Given(o.text.clone()))
            } else if !url.is_empty() {
                Ok(Backend::Server { url, model })
            } else {
                Err(
                    "no transcription backend: pass the text you spoke, or set $RADIXNET_ASR_URL to an \
                     OpenAI-compatible /v1/audio/transcriptions endpoint (the local Whisper backends, \
                     faster-whisper and openai-whisper, are Python packages: use the Python server for them)"
                        .to_string(),
                )
            }
        }
        _ => Err(format!(
            "unknown transcription backend {}; expected one of: {}",
            python_repr(raw),
            ASR_BACKENDS.join(", ")
        )),
    }
}

/// Audio bytes -> the words (`speech.transcribe`).
///
/// Audio that is not a WAV is converted with ffmpeg first (16 kHz), because
/// that is what every backend reads.
pub fn transcribe(data: &[u8], o: &AsrOptions) -> Result<Transcript, String> {
    let backend = resolve(o)?;
    let started = Instant::now();
    let (transcript, model) = match &backend {
        Backend::Given(text) => {
            let words = text.trim_matches(is_python_space);
            if words.is_empty() {
                return Err("no transcript was given (pass the text, or install a transcription backend)".to_string());
            }
            (words.to_string(), None)
        }
        Backend::Python(name) => {
            let install = if *name == "whisper" {
                "openai-whisper"
            } else {
                "faster-whisper"
            };
            return Err(format!(
                "{name} runs Whisper in-process ({install} is a Python package): transcribe with the Python \
                 server, point --asr-url at a transcription server, or send the words with the audio"
            ));
        }
        Backend::Server { url, model } => {
            if data.is_empty() {
                return Err("no audio to transcribe".to_string());
            }
            let wav = if data.starts_with(b"RIFF") {
                data.to_vec()
            } else {
                convert_with_ffmpeg(data, Some(16_000))?
            };
            (
                server_transcribe(url, model, &wav, o.language.as_deref())?,
                Some(model.clone()),
            )
        }
    };
    Ok(Transcript {
        transcript,
        backend: backend.name().to_string(),
        model,
        language: o.language.clone(),
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// A `multipart/form-data` body of plain fields and one file (`speech._multipart`).
fn multipart_body(fields: &[(&str, &str)], file_field: &str, filename: &str, data: &[u8]) -> (Vec<u8>, String) {
    let head = &data[..data.len().min(4096)];
    let boundary = format!("----radixnet-{}", crate::blake2b::hexdigest(head, 8));
    let mut body = Vec::with_capacity(data.len() + 512);
    for (name, value) in fields {
        body.extend_from_slice(
            format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n").as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{filename}\"\r\n\
             Content-Type: audio/wav\r\n\r\n"
        )
        .as_bytes(),
    );
    body.extend_from_slice(data);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    (body, format!("multipart/form-data; boundary={boundary}"))
}

/// One POST to an OpenAI-compatible `/v1/audio/transcriptions` endpoint
/// (`ServerTranscriber.transcribe`); the key is `$RADIXNET_ASR_KEY`, else
/// `$OPENAI_API_KEY`.
fn server_transcribe(url: &str, model: &str, wav: &[u8], language: Option<&str>) -> Result<String, String> {
    if url.is_empty() {
        return Err("no transcription server configured (set $RADIXNET_ASR_URL or pass --asr-url)".to_string());
    }
    let mut fields = vec![("model", model)];
    if let Some(lang) = language.filter(|l| !l.is_empty()) {
        fields.push(("language", lang));
    }
    let (body, content_type) = multipart_body(&fields, "file", "speech.wav", wav);
    let mut request = crate::fetch::Request::new("POST", url)
        .header("Content-Type", &content_type)
        .timeout(asr_timeout())
        .follow(3);
    request.body = body;
    let key = std::env::var("RADIXNET_ASR_KEY")
        .ok()
        .filter(|k| !k.is_empty())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok().filter(|k| !k.is_empty()));
    if let Some(key) = key {
        request = request.header("Authorization", &format!("Bearer {key}"));
    }
    let response = request
        .send()
        .map_err(|err| format!("cannot reach the transcription server at {url}: {err}"))?;
    let raw = response.text();
    if !response.ok() {
        let detail: String = raw.trim().chars().take(200).collect();
        return Err(format!(
            "the transcription server answered {}: {detail}",
            response.status
        ));
    }
    let doc = crate::json::parse(&raw).unwrap_or_else(|_| Json::obj([("text", Json::str(raw.clone()))]));
    match doc.get("text") {
        Some(Json::Str(text)) => Ok(text.trim_matches(is_python_space).to_string()),
        _ => Err("the transcription server did not answer with a 'text' field".to_string()),
    }
}

// -- programs on the machine ------------------------------------------------------------------

/// A program on `PATH`, as `shutil.which` finds it.
pub fn which(name: &str) -> Option<String> {
    if name.contains('/') {
        return is_executable(std::path::Path::new(name)).then(|| name.to_string());
    }
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|candidate| is_executable(candidate))
        .map(|p| p.to_string_lossy().into_owned())
}

fn is_executable(path: &std::path::Path) -> bool {
    let Ok(meta) = std::fs::metadata(path) else {
        return false;
    };
    if !meta.is_file() {
        return false;
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        meta.permissions().mode() & 0o111 != 0
    }
    #[cfg(not(unix))]
    {
        true
    }
}

/// `ffmpeg` on `PATH`; `$RADIXNET_FFMPEG` overrides it (`speech.ffmpeg_path`).
pub fn ffmpeg_path() -> Option<String> {
    match std::env::var("RADIXNET_FFMPEG") {
        Ok(configured) if !configured.is_empty() => {
            if std::path::Path::new(&configured).is_file() {
                Some(configured)
            } else {
                which(&configured)
            }
        }
        _ => which("ffmpeg"),
    }
}

/// Runs a program with `input` on its stdin; `(succeeded, stdout, stderr)`.
///
/// Stdin is written from its own thread and both outputs are drained from
/// theirs, so a program that fills a pipe before reading all of its input
/// cannot deadlock the call; past `timeout` it is killed.
fn run(command: &[String], input: &[u8], timeout: Duration) -> Result<(bool, Vec<u8>, Vec<u8>), String> {
    let mut child = Command::new(&command[0])
        .args(&command[1..])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| format!("{} could not be run: {err}", command[0]))?;
    let mut stdin = child.stdin.take();
    let feed = input.to_vec();
    let writer = std::thread::spawn(move || {
        if let Some(pipe) = stdin.as_mut() {
            let _ = pipe.write_all(&feed); // a program that stops reading early is not an error here
        }
        drop(stdin);
    });
    let drain = |pipe: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut out = Vec::new();
            if let Some(mut pipe) = pipe {
                let _ = pipe.read_to_end(&mut out);
            }
            out
        })
    };
    let stdout = drain(child.stdout.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let stderr = drain(child.stderr.take().map(|p| Box::new(p) as Box<dyn Read + Send>));
    let started = Instant::now();
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() > timeout => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(format!("{} did not finish within {}s", command[0], timeout.as_secs()));
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(10)),
            Err(err) => return Err(format!("{} could not be run: {err}", command[0])),
        }
    };
    let _ = writer.join();
    let out = stdout.join().unwrap_or_default();
    let err = stderr.join().unwrap_or_default();
    Ok((status.success(), out, err))
}

/// The last line of a program's stderr, for an error message.
fn last_line(stderr: &[u8], fallback: &str) -> String {
    let text = String::from_utf8_lossy(stderr);
    text.trim()
        .lines()
        .last()
        .map(str::to_string)
        .unwrap_or_else(|| fallback.to_string())
}

/// The audio format the first bytes suggest - only to make an error helpful.
fn looks_like(data: &[u8]) -> &'static str {
    const MAGIC: &[(&[u8], &str)] = &[
        (b"ID3", "MP3"),
        (b"\xff\xfb", "MP3"),
        (b"\xff\xf3", "MP3"),
        (b"\xff\xf2", "MP3"),
        (b"OggS", "Ogg / Opus"),
        (b"fLaC", "FLAC"),
        (b"\x1a\x45\xdf\xa3", "WebM / Matroska"),
        (b"FORM", "AIFF"),
    ];
    if data.len() >= 8 && &data[4..8] == b"ftyp" {
        return "MP4 / M4A";
    }
    MAGIC
        .iter()
        .find(|(magic, _)| data.starts_with(magic))
        .map(|(_, name)| *name)
        .unwrap_or("an unknown format")
}

/// Anything ffmpeg reads (MP3, Opus, WebM, M4A, FLAC, ...) -> a 16-bit PCM WAV
/// (`speech.convert_with_ffmpeg`).
pub fn convert_with_ffmpeg(data: &[u8], rate: Option<i64>) -> Result<Vec<u8>, String> {
    let Some(binary) = ffmpeg_path() else {
        return Err(format!(
            "the audio is {}; install ffmpeg to read it (or send a WAV file)",
            looks_like(data)
        ));
    };
    let mut command: Vec<String> = [
        binary.as_str(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
        "-vn",
        "-map",
        "a:0",
        "-f",
        "wav",
        "-acodec",
        "pcm_s16le",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if let Some(rate) = rate {
        command.extend(["-ar".to_string(), rate.to_string()]);
    }
    command.push("pipe:1".to_string());
    let (ok, out, err) = run(&command, data, Duration::from_secs(600))?;
    if !ok || out.is_empty() {
        return Err(format!(
            "ffmpeg could not decode the audio: {}",
            last_line(&err, "no output")
        ));
    }
    Ok(out)
}

/// The command a recorder records with, or `None` where it cannot record.
fn recorder_command(binary: &str, name: &str, seconds: f64, rate: i64, path: &str) -> Option<Vec<String>> {
    let s = |x: &str| x.to_string();
    let secs = crate::json::py_repr(seconds);
    match name {
        "arecord" => Some(vec![
            s(binary),
            s("-q"),
            s("-f"),
            s("S16_LE"),
            s("-c"),
            s("1"),
            s("-r"),
            rate.to_string(),
            s("-d"),
            (seconds.ceil() as i64).to_string(),
            s(path),
        ]),
        "rec" | "sox" => {
            let mut head = vec![s(binary), s("-q")];
            if name == "sox" {
                head.push(s("-d"));
            }
            head.extend([
                s("-c"),
                s("1"),
                s("-r"),
                rate.to_string(),
                s("-b"),
                s("16"),
                s(path),
                s("trim"),
                s("0"),
                secs,
            ]);
            Some(head)
        }
        "ffmpeg" => {
            let (format, device) = match std::env::consts::OS {
                "linux" => ("alsa", "default"),
                "macos" => ("avfoundation", ":0"),
                _ => return None, // Windows needs a dshow device name nobody can guess
            };
            Some(vec![
                s(binary),
                s("-hide_banner"),
                s("-loglevel"),
                s("error"),
                s("-y"),
                s("-f"),
                s(format),
                s("-i"),
                s(device),
                s("-t"),
                secs,
                s("-ac"),
                s("1"),
                s("-ar"),
                rate.to_string(),
                s(path),
            ])
        }
        _ => None,
    }
}

/// The binary a recorder name runs.
fn recorder_binary(name: &str) -> Option<String> {
    if name == "ffmpeg" {
        ffmpeg_path()
    } else {
        which(name)
    }
}

/// The recorders found on `PATH` (`speech.recorders`).
pub fn recorders() -> Vec<String> {
    RECORDERS
        .iter()
        .filter(|name| {
            recorder_binary(name).is_some_and(|b| recorder_command(&b, name, 1.0, 16_000, "out.wav").is_some())
        })
        .map(|name| name.to_string())
        .collect()
}

/// Records `seconds` of mono audio from the default input device with the
/// first recorder installed (or the one named); returns the WAV bytes
/// (`speech.record`).
pub fn record(seconds: f64, rate: i64, recorder: Option<&str>) -> Result<Vec<u8>, String> {
    if seconds <= 0.0 {
        return Err(format!(
            "the recording length must be positive, got {}",
            crate::json::py_repr(seconds)
        ));
    }
    if let Some(name) = recorder {
        if !RECORDERS.contains(&name) {
            return Err(format!(
                "unknown recorder {}; expected one of: {}",
                python_repr(name),
                RECORDERS.join(", ")
            ));
        }
    }
    let names: Vec<&str> = match recorder {
        Some(name) => vec![name],
        None => RECORDERS.to_vec(),
    };
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let path = std::env::temp_dir().join(format!("radixnet-record-{}-{stamp}.wav", std::process::id()));
    let path_text = path.to_string_lossy().into_owned();
    let result = (|| {
        for name in names {
            let Some(binary) = recorder_binary(name) else { continue };
            let Some(command) = recorder_command(&binary, name, seconds, rate, &path_text) else {
                continue;
            };
            let (ok, _, err) = run(&command, &[], Duration::from_secs_f64(seconds + 60.0))?;
            let written = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            if !ok || written == 0 {
                return Err(format!(
                    "{name} could not record: {}",
                    last_line(&err, "no audio was written")
                ));
            }
            return std::fs::read(&path).map_err(|err| format!("cannot read the recording: {err}"));
        }
        Err(
            "no microphone recorder found: install alsa-utils (arecord), sox (rec) or ffmpeg, or record in \
             the browser's Speech tab"
                .to_string(),
        )
    })();
    let _ = std::fs::remove_file(&path);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_given_transcript_needs_nothing_installed() {
        let o = AsrOptions {
            backend: "auto".into(),
            text: "  the cat sat  ".into(),
            language: Some("en".into()),
            ..Default::default()
        };
        let t = transcribe(b"", &o).unwrap();
        assert_eq!(t.transcript, "the cat sat");
        assert_eq!(t.backend, "given");
        assert_eq!(t.model, None);
        let doc = t.to_json();
        assert_eq!(doc.at("language").as_str(), Some("en"));
        let given = AsrOptions {
            backend: "given".into(),
            ..Default::default()
        };
        assert!(transcribe(b"RIFF", &given)
            .unwrap_err()
            .contains("no transcript was given"));
    }

    #[test]
    fn the_python_backends_and_unknown_names_are_refused() {
        for backend in ["whisper", "faster_whisper"] {
            let o = AsrOptions {
                backend: backend.into(),
                ..Default::default()
            };
            assert!(transcribe(b"RIFF", &o).unwrap_err().contains("Python"), "{backend}");
        }
        let o = AsrOptions {
            backend: "telepathy".into(),
            ..Default::default()
        };
        assert!(transcribe(b"RIFF", &o)
            .unwrap_err()
            .contains("unknown transcription backend"));
        let server = AsrOptions {
            backend: "server".into(),
            url: Some(String::new()),
            ..Default::default()
        };
        assert!(transcribe(b"RIFF", &server)
            .unwrap_err()
            .contains("no transcription server"));
    }

    #[test]
    fn a_transcription_server_is_spoken_to_over_http() {
        // a one-shot OpenAI-compatible endpoint on a real port
        let listener = std::net::TcpListener::bind(("127.0.0.1", 0)).unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut seen = Vec::new();
            let mut buf = [0u8; 4096];
            // read until the closing multipart delimiter arrives
            while !String::from_utf8_lossy(&seen).contains("--\r\n") {
                let n = stream.read(&mut buf).unwrap();
                if n == 0 {
                    break;
                }
                seen.extend_from_slice(&buf[..n]);
            }
            let body = r#"{"text": " hello there "}"#;
            let answer = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            );
            stream.write_all(answer.as_bytes()).unwrap();
            String::from_utf8_lossy(&seen).into_owned()
        });
        let o = AsrOptions {
            backend: "auto".into(),
            url: Some(format!("http://127.0.0.1:{port}/v1/audio/transcriptions")),
            model: Some("tiny".into()),
            language: Some("en".into()),
            ..Default::default()
        };
        let t = transcribe(b"RIFF....WAVEfmt ", &o).unwrap();
        assert_eq!((t.transcript.as_str(), t.backend.as_str()), ("hello there", "server"));
        assert_eq!(t.model.as_deref(), Some("tiny"));
        let request = server.join().unwrap();
        assert!(request.starts_with("POST /v1/audio/transcriptions"));
        assert!(request.contains("name=\"model\"\r\n\r\ntiny\r\n"));
        assert!(request.contains("name=\"language\"\r\n\r\nen\r\n"));
        assert!(request.contains("filename=\"speech.wav\""));
    }

    #[test]
    fn a_format_is_named_by_its_first_bytes() {
        assert_eq!(looks_like(b"ID3\x03"), "MP3");
        assert_eq!(looks_like(b"\0\0\0\x20ftypM4A "), "MP4 / M4A");
        assert_eq!(looks_like(b"OggS"), "Ogg / Opus");
        assert_eq!(looks_like(b"what"), "an unknown format");
    }

    #[test]
    fn the_recorder_commands_are_pythons() {
        let cmd = recorder_command("/usr/bin/arecord", "arecord", 2.5, 16000, "o.wav").unwrap();
        assert_eq!(cmd.join(" "), "/usr/bin/arecord -q -f S16_LE -c 1 -r 16000 -d 3 o.wav");
        let cmd = recorder_command("sox", "sox", 5.0, 8000, "o.wav").unwrap();
        assert_eq!(cmd.join(" "), "sox -q -d -c 1 -r 8000 -b 16 o.wav trim 0 5.0");
        assert!(record(0.0, 16000, None).is_err());
        assert!(record(1.0, 16000, Some("gramophone")).is_err());
    }
}
