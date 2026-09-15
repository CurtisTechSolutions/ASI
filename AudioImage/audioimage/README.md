# audioimage

The Python package: a waveform encoded into a picture, and decoded back out of
it. Time across, frequency up, loudness as ink — black is 1.0, white is 0.0.

**Standard library only.** The FFT, the PNG encoder, the WAV parser and the font
are all in this directory. numpy is picked up automatically when it happens to be
installed and is about 3.7× faster, but nothing requires it.

`../README.md` is the user-facing guide and `../DESIGN.md` the specification;
this file says which module does what.

## The modules

### The plane, and the round trip

| module | what it is |
|---|---|
| `dsp.py` | the signal processing, written from scratch — Hann window, real FFT, STFT and its exact inverse, Griffin-Lim phase reconstruction |
| `dsp_numpy.py` | the same transforms vectorised with numpy. Held to agreeing with `dsp.py` to 1e-9, value for value, by the test suite |
| `backend.py` | which of the two runs. `AUDIOIMAGE_BACKEND=python` forces the hand-written path |
| `spectrogram.py` | the plane itself: time across, frequency up, loudness as ink — and the mapping back |
| `colormap.py` | intensity ↔ pixel: how loud becomes how dark, and how to read it back out of a false-colour picture |
| `codec.py` | the two operations the tool exists for — audio in, picture out, audio back |

### Files on disk

| module | what it is |
|---|---|
| `wav.py` | WAV in and out, as plain floating-point samples |
| `png.py` | PNG in and out, written from scratch on `zlib` and `struct` |
| `font.py` | a 5×7 bitmap font, so a picture can label its own axes |
| `render.py` | pictures for *looking at*, as opposed to pictures for decoding — labelled charts, waveform plots |

### Making sound, and learning from it

| module | what it is |
|---|---|
| `synth.py` | making sound to look at: tones, sweeps, noise, chords and little tunes |
| `qlearn.py` | a Q-learning network over the plane — one node per hertz, reading one vertical strip at a time |

### Interfaces

| module | what it is |
|---|---|
| `cli.py` | `python -m audioimage <command>` / `audioimage` — every command, with `--json` for machine-readable output |
| `__main__.py` | dispatches `python -m audioimage` to the CLI |
| `server.py` | a small stdlib HTTP API, and the static files of the browser front end |
| `frontend/` | the browser page the server serves (see its own README) |
| `__init__.py` | the public surface of the package |

## Using it as a library

```python
from audioimage import Audio, EncodeConfig, decode, encode, read_wav

audio = read_wav("voice.wav")
picture = encode(audio)          # -> a PNG image of the spectrum
back = decode(picture).audio     # -> a waveform again
```

`encode_file` / `decode_file` are the same two operations over paths.

`../README.md` § *Using it as a library* has the full surface, including the
report on what survives a round trip.

## Tests

Every module here has a matching `test_*.py` in `../tests/`.

```bash
cd .. && make test            # the suite
cd .. && make test-python     # the same, forced onto the standard-library backend
```
