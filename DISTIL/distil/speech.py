"""Turning speech into a task string.

Every question this system answers arrives as text, and typing is a poor way to
state a problem you are in the middle of having. Two routes exist, and which one
runs decides where the audio goes -- so the choice is made explicit rather than
hidden behind a microphone icon.

**In the browser.** The Web Speech API is already there, costs nothing and needs
no model. In Chrome it is *not* local: the audio is sent to Google for
recognition. For a tool whose entire premise is that it runs on your machine with
no key and no network, that is a real contradiction, so the frontend says so at
the point of use instead of in a footnote.

**Here.** If a Whisper binary is installed, `transcribe` shells out to it and the
audio never leaves the machine. Nothing is vendored and nothing is downloaded:
this is zero-dependency like the rest of the package, which means it works when
the tool is already there and says so plainly when it is not.

Ollama is deliberately not in that list. It serves language and embedding models
and does not transcribe audio, so pointing this at an Ollama server would fail at
runtime with a confusing error rather than at startup with a clear one.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

#: Recognised binaries, in the order they are tried. Each entry is the command
#: name and a function building its argv for a wav path. whisper.cpp names its
#: binary `whisper-cli` now and `main` historically; `main` is deliberately NOT
#: probed, because a program called `main` on PATH is as likely to be anything.
BACKENDS = [
    ("whisper-cli", lambda exe, wav, model: [exe, "-m", model, "-f", str(wav), "-nt", "-otxt"]),
    ("whisper-cpp", lambda exe, wav, model: [exe, "-m", model, "-f", str(wav), "-nt", "-otxt"]),
    ("whisper", lambda exe, wav, model: [exe, str(wav), "--model", model or "base",
                                         "--output_format", "txt", "--fp16", "False"]),
]

MAX_SECONDS = 120.0
MAX_BYTES = 25 * 1024 * 1024


class SpeechError(RuntimeError):
    """Transcription could not be done here. The message is shown to the user."""


def model_path() -> str:
    """The ggml model whisper.cpp needs, from $DISTIL_WHISPER_MODEL."""
    return os.environ.get("DISTIL_WHISPER_MODEL", "")


def available() -> dict:
    """What this machine can do, for the frontend to decide what to offer.

    Called on every page load, so it probes PATH and does not start anything.
    """
    found = None
    for name, _ in BACKENDS:
        exe = shutil.which(name)
        if exe:
            found = {"backend": name, "path": exe}
            break
    model = model_path()
    ready = bool(found) and (found["backend"] == "whisper" or bool(model))
    return {
        "local": ready,
        "backend": found["backend"] if found else None,
        "model": model or None,
        "why": _why(found, model),
    }


def _why(found, model) -> str:
    if not found:
        return ("no whisper binary on PATH -- install whisper.cpp or openai-whisper "
                "to transcribe locally; until then the browser does it, and in "
                "Chrome that means the audio goes to Google")
    if found["backend"] != "whisper" and not model:
        return (f"{found['backend']} found, but it needs a model: set "
                f"DISTIL_WHISPER_MODEL to a ggml-*.bin path")
    return f"local transcription via {found['backend']}"


def check_wav(raw: bytes) -> float:
    """Confirm this is a wav we can hand to whisper, and how long it is.

    Whisper is given a file path, and handing an unvalidated upload straight to a
    subprocess is how a transcription endpoint becomes a file-format parser with
    someone else's bugs. This reads the header with the standard library first and
    refuses anything that is not plausibly short speech.
    """
    if not raw:
        raise SpeechError("no audio was sent")
    if len(raw) > MAX_BYTES:
        raise SpeechError(f"audio is {len(raw) // 1024}kB; the limit is {MAX_BYTES // 1024}kB")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
        fh.write(raw)
        probe = Path(fh.name)
    try:
        with wave.open(str(probe), "rb") as w:
            frames, rate = w.getnframes(), w.getframerate()
        if not rate:
            raise SpeechError("the audio declares a sample rate of zero")
        seconds = frames / float(rate)
        if seconds > MAX_SECONDS:
            raise SpeechError(f"audio is {seconds:.0f}s; the limit is {MAX_SECONDS:.0f}s")
        return seconds
    except wave.Error as exc:
        raise SpeechError(f"not a readable wav file: {exc}") from exc
    finally:
        probe.unlink(missing_ok=True)


def transcribe(raw: bytes, timeout: float = 120.0) -> dict:
    """Wav bytes to text, locally, or a SpeechError saying why not."""
    state = available()
    if not state["local"]:
        raise SpeechError(state["why"])
    seconds = check_wav(raw)
    exe = shutil.which(state["backend"])
    build = dict((n, f) for n, f in BACKENDS)[state["backend"]]

    work = Path(tempfile.mkdtemp(prefix="distil-speech-"))
    wav = work / "speech.wav"
    wav.write_bytes(raw)
    try:
        proc = subprocess.run(build(exe, wav, model_path()), capture_output=True,
                              text=True, timeout=timeout, cwd=work)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            raise SpeechError(f"{state['backend']} failed: {detail[-1] if detail else 'no output'}")
        text = _collect(work, proc.stdout)
        if not text:
            raise SpeechError("nothing was transcribed -- was anything said?")
        return {"text": text, "seconds": round(seconds, 2), "backend": state["backend"]}
    except subprocess.TimeoutExpired as exc:
        raise SpeechError(f"{state['backend']} did not finish within {timeout:.0f}s") from exc
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _collect(work: Path, stdout: str) -> str:
    """Whisper writes a .txt beside the input; some builds only print.

    Preferring the file matters because the binaries print progress and model
    banners to stdout, and a transcript assembled from that carries them.
    """
    for txt in sorted(work.glob("*.txt")):
        found = txt.read_text(errors="replace").strip()
        if found:
            return " ".join(found.split())
    for line in reversed((stdout or "").strip().splitlines()):
        clean = line.strip()
        if clean and not clean.startswith("["):
            return " ".join(clean.split())
    return ""
