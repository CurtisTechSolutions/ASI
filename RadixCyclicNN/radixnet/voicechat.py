"""Talking with the model by voice: every utterance is heard, learned, answered and spoken (D-089).

The Voice tab listens all the time.  Each thing the person says reaches here
as one **turn**: the recording, and what the browser heard in it.  A turn is

1. **heard** - :func:`radixnet.speech.teach` makes the utterance into the texts
   the model learns, the transcript and the waveform behind one unique token
   (D-035); a model of acoustic units hears the recording as its units instead
   (D-082), the one text made of sound it can learn;
2. **trained** - the model is trained on those texts at once, so the reply
   already knows them;
3. **answered** - the model replies as it does in every conversation here
   (:func:`radixnet.dialogue.reply`: the end of the line picked up and
   continued), or Ollama answers for it (``answer = "ollama"``), or Ollama
   steps in only when the model has nothing to say to the person (``"auto"``:
   no reply, or a fresh text that picked up none of the line);
4. **spoken** - the reply goes through the output decoder (D-088) and its
   audio comes back in chunks the browser plays as they arrive;
5. **taught** - a reply Ollama wrote is taught to the model, so that next time
   it may answer by itself.

:func:`turn` does all of it and reports every step to ``write`` as it happens
(the JSON Lines of ``POST /api/voice/turn/stream``).  The model's own parts -
training and replying - come in as callables, so the service takes its locks
around exactly those and Ollama is asked outside them.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from .dialogue import EXPLORE, Heard, Turn
from .encoding import ACOUSTIC, Encoding, hear_audio
from .llm import LLMError
from .speech import DEFAULT_RATE, teach
from .voice import RATE, VOCODERS, say, spelled

SPEAKERS = ("You", "Model")
"""Who speaks: the person, and the model (or Ollama on its behalf)."""

ANSWERS = ("auto", "model", "ollama", "none")
"""Who answers: the model with Ollama stepping in when it has nothing to say, the model alone, Ollama alone, nobody."""

CHUNK_SAMPLES = 8000
"""Samples per ``audio`` event: half a second at 16 kHz, so playback starts long before the reply has ended."""

Learn = Callable[[list[str]], dict]
"""Trains the model on texts and says what it did (``{"texts", "epochs", ...}``)."""

ModelReply = Callable[[str, Heard], "Turn | None"]
"""The model's own answer to a line, given what the conversation has heard."""

Write = Callable[[dict], Any]
"""Where the events of a turn go, one at a time, as they happen."""

Transcript = list[tuple[str, str]]


@dataclass
class VoiceOptions:
    """One turn's settings: how it is heard, learned, answered and spoken."""

    transcript: str = ""
    """What the browser heard; it wins over the server's own transcription."""
    backend: str = "auto"
    language: str | None = None
    asr_model: str | None = None
    asr_url: str | None = None
    rate: int = DEFAULT_RATE
    """The waveform text's sample rate (:func:`radixnet.speech.teach`)."""
    codec: str = "auto"
    normalise: bool = False
    waveform: bool = True
    pair: bool = False
    unique: bool = True
    train: bool = True
    epochs: int = 2
    lr: float = 0.5
    batch_size: int = 8
    answer: str = "auto"
    learn_reply: bool = True
    """Teach the model a reply Ollama wrote for it."""
    persona: str = ""
    topic: str = ""
    mode: str = "beam"
    max_length: int = 60
    context: int = 12
    k: int = 5
    beam: int | None = None
    step_penalty: float = 0.0
    temperature: float = 1.0
    seed: int | None = None
    avoid_repeats: bool = True
    avoid_word_repeats: bool = True
    explore: int = EXPLORE
    learn: bool = True
    """What a rethink finds out is taught to the graph, as in every conversation here."""
    ollama_temperature: float = 0.8
    speak: bool = True
    voice_rate: int = RATE
    pitch: float = 120.0
    tempo: float = 1.0
    gain: float = 0.5
    polish: int = 0
    vocoder: str = "auto"
    """How acoustic units are spoken (:data:`radixnet.voice.VOCODERS`): the codebook's neural vocoder when it has
    one, that or nothing, or its own centroid vocoder."""
    chunk: int = CHUNK_SAMPLES

    def validate(self) -> None:
        if self.answer not in ANSWERS:
            raise ValueError(f"'answer' must be one of {', '.join(ANSWERS)}, got {self.answer!r}")
        if self.mode not in ("beam", "sample"):
            raise ValueError(f"'mode' must be beam or sample, got {self.mode!r}")
        if self.epochs < 0 or self.batch_size < 1 or self.lr < 0:
            raise ValueError("'epochs' and 'lr' must be at least 0 and 'batch_size' at least 1")
        if self.pitch <= 0 or self.tempo <= 0 or self.gain < 0 or self.voice_rate < 1:
            raise ValueError("'pitch' and 'tempo' must be above 0, 'gain' at least 0 and 'voice_rate' at least 1")
        if self.chunk < 1 or self.polish < 0 or self.max_length < 0 or self.context < 0 or self.k < 1:
            raise ValueError("'chunk' and 'k' must be at least 1; 'polish', 'max_length' and 'context' at least 0")
        if self.vocoder not in VOCODERS:
            raise ValueError(f"'vocoder' must be one of {', '.join(VOCODERS)}")


@dataclass
class Outcome:
    """What one turn came to: the document the API answers with, and the reply's audio."""

    document: dict
    pcm: bytes = b""
    rate: int = RATE

    def wav(self) -> bytes:
        from .encoding import phonetok_module

        return phonetok_module("synth", "speaking").wav_bytes(self.pcm, self.rate)


def history_pairs(history: Any) -> Transcript:
    """A conversation so far as ``(speaker, text)`` pairs, from pairs or ``{"speaker", "text"}`` records."""
    pairs: Transcript = []
    for i, item in enumerate(history or ()):
        if isinstance(item, dict):
            speaker, text = item.get("speaker", ""), item.get("text", "")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            speaker, text = item
        else:
            raise ValueError(f"history[{i}] must be [speaker, text] or {{speaker, text}}")
        if not isinstance(speaker, str) or not isinstance(text, str):
            raise ValueError(f"history[{i}]: the speaker and the text must be strings")
        if text.strip():
            pairs.append((speaker.strip() or SPEAKERS[0], " ".join(text.split())))
    return pairs


def hear(data: bytes | None, encoding: Encoding, o: VoiceOptions) -> dict:
    """What an utterance was heard as: the transcript, the texts to learn, the token and the audio.

    A model of acoustic units learns the recording as its units (:func:`radixnet.encoding.hear_audio`) -
    the transcript is kept for the conversation, but nothing made of words is in its alphabet.
    """
    if encoding.unit == ACOUSTIC:
        words = " ".join(str(o.transcript or "").split())
        units = hear_audio(data) if data else ""
        return {
            "transcript": words, "token": None, "texts": [units] if units else [], "audio": None, "units": units,
            "asr": {"backend": "given" if words else None, "model": None, "language": o.language, "seconds": 0.0,
                    "error": None},
        }
    heard = teach(
        data, transcript=o.transcript, backend=o.backend, language=o.language, asr_model=o.asr_model,
        asr_url=o.asr_url, rate=o.rate, codec=o.codec, normalise=o.normalise, waveform=o.waveform, pair=o.pair,
        unique=o.unique,
    )
    return {
        "transcript": heard["transcript"], "token": heard["token"], "texts": list(heard["texts"]),
        "audio": heard["audio"], "units": None, "asr": heard["asr"],
    }


def speech_encoding(encoding: Encoding, by: str) -> Encoding:
    """The encoding a reply is spoken in: the model's own, unless it is words a model of acoustic units cannot say."""
    if encoding.unit == ACOUSTIC and by != "model":
        return Encoding()
    return encoding


def audio_events(pcm: bytes, rate: int, chunk: int = CHUNK_SAMPLES) -> Iterable[dict]:
    """The reply's audio as ``audio`` events of ``chunk`` samples: 16-bit PCM, base64, played as they arrive."""
    step = max(1, chunk) * 2
    for i in range(0, len(pcm), step):
        piece = pcm[i:i + step]
        yield {"event": "audio", "rate": rate, "samples": len(piece) // 2,
               "pcm_base64": base64.b64encode(piece).decode("ascii")}


def turn(
    data: bytes | None,
    *,
    encoding: Encoding,
    options: VoiceOptions | None = None,
    history: Any = (),
    learn: Learn | None = None,
    model_reply: ModelReply | None = None,
    ollama: Any = None,
    write: Write | None = None,
) -> Outcome:
    """One turn of talking with the model: heard, trained, answered, spoken, taught - each an event to ``write``.

    ``data`` is the recording (a WAV; ``None`` for a typed line, with
    ``options.transcript`` the words), ``history`` the conversation so far,
    ``learn`` and ``model_reply`` the model's parts (see :data:`Learn`,
    :data:`ModelReply`), ``ollama`` an :class:`radixnet.ollama.OllamaClient`
    when Ollama may answer.  Raises :class:`ValueError` (a
    :class:`radixnet.speech.SpeechError` is one) for a turn that cannot be
    heard.
    """
    o = options or VoiceOptions()
    o.validate()
    from .ollama import reply_line

    emit: Write = write if write is not None else (lambda event: None)
    transcript = history_pairs(history)
    started = time.monotonic()
    # 1. heard
    heard = hear(data, encoding, o)
    line = heard["transcript"] or heard["units"] or ""
    emit({"event": "heard", **{k: heard[k] for k in ("transcript", "token", "texts", "audio", "units", "asr")}})
    # 2. trained
    trained: dict | None = None
    if o.train and heard["texts"] and learn is not None:
        began = time.monotonic()
        trained = dict(learn(list(heard["texts"])))
        trained.setdefault("texts", len(heard["texts"]))
        trained["seconds"] = round(time.monotonic() - began, 3)
        emit({"event": "trained", **trained})
    # 3. answered
    said = Heard([text for _, text in transcript] + ([line] if line else []))
    by, reply_text, record, ollama_error = "none", "", None, None
    model_turn: Turn | None = None
    if line and o.answer in ("model", "auto") and model_reply is not None:
        model_turn = model_reply(line, said)
    if model_turn is not None:
        by, reply_text, record = "model", model_turn.text, model_turn.to_dict()
    # Ollama answers for the model, or steps in when it could not answer the person: nothing, or a fresh
    # text that picked up none of the line
    wants_ollama = o.answer == "ollama" or (o.answer == "auto" and (model_turn is None or model_turn.fresh))
    if wants_ollama and heard["transcript"]:
        if ollama is None:
            ollama_error = "no Ollama to ask"
        else:
            try:
                text = reply_line(
                    ollama, transcript + [(SPEAKERS[0], heard["transcript"])], persona=o.persona, topic=o.topic,
                    speakers=SPEAKERS, temperature=o.ollama_temperature,
                )
            except LLMError as exc:
                ollama_error, text = str(exc), ""
            if text:
                by, reply_text = "ollama", text
    voice = speech_encoding(encoding, by)
    reply = {
        "by": by, "text": reply_text, "spelled": spelled(voice, reply_text) if reply_text else "", "turn": record,
        "ollama_error": ollama_error,
    }
    emit({"event": "reply", **reply})
    # 4. spoken
    spoken_doc: dict | None = None
    pcm = b""
    rate = o.voice_rate
    if o.speak and reply_text:
        spoken = say(voice, [reply_text], rate=o.voice_rate, pitch=o.pitch, tempo=o.tempo, gain=o.gain, polish=o.polish,
                     vocoder=o.vocoder)
        pcm, rate = spoken.pcm, spoken.rate
        for event in audio_events(pcm, rate, o.chunk):
            emit(event)
        spoken_doc = spoken.to_dict()
        emit({"event": "spoken", **spoken_doc})
    # 5. taught
    taught: dict | None = None
    if by == "ollama" and o.learn_reply and reply_text and learn is not None and encoding.unit != ACOUSTIC:
        taught = dict(learn([reply_text]))
        taught.setdefault("texts", 1)
        emit({"event": "taught", **taught})
    spoken_pairs: Transcript = list(transcript)
    if line:
        spoken_pairs.append((SPEAKERS[0], line))
    if reply_text:
        spoken_pairs.append((SPEAKERS[1], reply_text))
    document = {
        "transcript": heard["transcript"],
        "line": line,
        "token": heard["token"],
        "texts": heard["texts"],
        "audio": heard["audio"],
        "units": heard["units"],
        "asr": heard["asr"],
        "trained": trained,
        "reply": reply,
        "by": by,
        "spoken": spoken_doc,
        "taught": taught,
        "history": [list(pair) for pair in spoken_pairs],
        "encoding": str(encoding),
        "answer": o.answer,
        "seconds": round(time.monotonic() - started, 3),
    }
    return Outcome(document, pcm, rate)


def describe(encoding: Encoding, ollama: Any = None) -> dict:
    """What the Voice tab has to work with: the speakers, the answer modes, the voice and the transcription."""
    from .speech import describe as describe_speech
    from .voice import decoder_name, vocoder_name

    speech = describe_speech()
    vocoder = None
    if encoding.unit == ACOUSTIC:
        try:
            vocoder = vocoder_name(encoding)
        except ValueError as exc:  # a vocoder file that is not the codebook's: the tab says so, the turn raises
            vocoder = f"error: {exc}"
    return {
        "speakers": list(SPEAKERS),
        "answers": list(ANSWERS),
        "vocoders": list(VOCODERS),
        "encoding": str(encoding),
        "decoder": decoder_name(encoding),
        "vocoder": vocoder,
        "acoustic": encoding.unit == ACOUSTIC,
        "default_rate": DEFAULT_RATE,
        "chunk": CHUNK_SAMPLES,
        "transcription": {"backends": speech.get("backends"), "auto": speech.get("auto"),
                          "faster_whisper": speech.get("faster_whisper"), "whisper": speech.get("whisper")},
        "ollama": None if ollama is None else {"url": ollama.url, "model": ollama.model},
    }
