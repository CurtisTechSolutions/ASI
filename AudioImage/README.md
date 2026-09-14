# AudioImage

**A sound, drawn. Time across, frequency up, loudness as ink — and then read back
out again as sound.**

A waveform is a line that wiggles far too fast to look at. This tool turns it
into a rectangle you can actually see: every column is a moment, every row is a
frequency, and how dark a point is says how much of that frequency was sounding
at that moment. **Black is 1.0, white is 0.0.**

The picture is not an illustration of the sound. It *is* the sound, written
down differently — `decode` turns it back into a WAV.

```
                  frequency
                      ^
                 8kHz |                                    ....
                      |                            ........
                      |                    ........
                 1kHz |            ........                       a rising tone
                      |    ........
                    0 +------------------------------------------> time
```

---

## Table of contents

- [The idea](#the-idea)
- [Quick start](#quick-start)
- [What survives the round trip](#what-survives-the-round-trip)
- [The commands](#the-commands)
- [The picture format](#the-picture-format)
- [Choosing the settings](#choosing-the-settings)
- [In the browser](#in-the-browser)
- [Using it as a library](#using-it-as-a-library)
- [How it works](#how-it-works)
- [Dependencies and speed](#dependencies-and-speed)
- [Tests](#tests)
- [Limits](#limits)

---

## The idea

Sound arrives as one number per instant — air pressure, 22050 times a second.
That is the wrong shape to look at and the wrong shape to reason about. Chop it
into short overlapping windows, take the Fourier transform of each one, and it
becomes a *plane*:

| axis | meaning | in the picture |
|------|---------|----------------|
| **X** | time | one column per analysis window, left to right |
| **Y** | frequency | one row per frequency, low at the bottom |
| **intensity** | amplitude | **black = 1.0 (as loud as the clip gets), white = 0.0 (silence)** |

That plane is written as a 16-bit greyscale PNG. Nothing else goes in the
file — every pixel is data — and the settings that produced it ride along in a
text chunk, so the picture knows how to become sound again.

## Quick start

No installation and no dependencies; Python 3.11 or newer is enough.

```bash
cd AudioImage

# make something to look at
python3 -m audioimage tone -o tune.wav --kind melody --notes "C4 E4 G4 C5 G4 E4 C4"

# the sound as a picture
python3 -m audioimage encode tune.wav -o tune.png

# the picture as sound again
python3 -m audioimage decode tune.png -o back.wav

# both at once, with a report on what survived
python3 -m audioimage roundtrip tune.wav
```

Or the whole tour in one step:

```bash
make demo
```

`encode` writes the bare plane — the thing that decodes. To get something
*readable*, with a labelled frequency axis, a time axis and a legend, ask for a
view:

```bash
python3 -m audioimage view tune.wav -o view.png --freq-scale mel --view-colormap magma
```

That is where the notes of the melody become obvious: a staircase of bright
blocks, each with its overtones stacked above it.

## What survives the round trip

The magnitudes survive almost perfectly. The **phase** is the question, because
a grey picture has one number per point and phase is the second one.

By default it is not stored, and decoding rebuilds it with Griffin-Lim.
Measured on 2 second clips at 22050 Hz, `n_fft` 1024:

| clip | passes | spectral error | log distance | decode |
|------|-------:|---------------:|-------------:|-------:|
| melody | 16 | 11.0 % | 1.07 dB | 0.16 s |
| melody | 64 | 4.1 % | 0.60 dB | 0.40 s |
| melody | 128 | 1.3 % | 0.47 dB | 0.73 s |
| chord | 64 | 6.0 % | 0.89 dB | 0.33 s |
| sweep | 64 | 4.8 % | 0.43 dB | 0.33 s |
| white noise | 64 | 8.3 % | 1.47 dB | 0.36 s |

Two things are worth knowing about those numbers.

**The picture itself is nearly lossless.** Take the phase question out and
measure only what the pixels hold: a 16-bit picture stores the magnitudes to a
relative error of **2.2 × 10⁻⁴**, and an 8-bit one to 7.8 × 10⁻³. That is the
cost of the encoding. Everything else in the table is the cost of *guessing the
phase back*.

**Waveform SNR is a lie here, and the tool says so.** `roundtrip` prints it
anyway — it usually comes out around −2 dB — because the rebuilt waveform is a
different waveform that happens to have the same spectrum. Judge the decode by
the spectral error, or by listening to it; sample-by-sample comparison measures
the phase you deliberately threw away.

Noise is the hardest case, and for a good reason: noise is *nothing but* phase
relationships, so there is least to recover.

### Or store the phase instead

`--phase rgb` writes the angle into the blue channel while the red and green
channels keep the level, so the picture still reads as a spectrogram:

```
R = G = magnitude     the picture stays grey, and stays readable
B     = phase         only where there is energy to have a phase
```

Decoding then needs no guessing at all — the spectrum goes straight back
through the inverse transform:

| clip | | spectral error | waveform SNR | decode | size |
|------|-|---------------:|-------------:|-------:|-----:|
| melody | grey, Griffin-Lim | 2.51 % | −2.8 dB | 0.45 s | 19.4 KiB |
| melody | `--phase rgb` | **0.46 %** | **+44.1 dB** | **0.08 s** | **18.6 KiB** |
| melody | `--phase rgb --depth 16` | **0.10 %** | **+58.2 dB** | 0.09 s | 42.9 KiB |
| noise | grey, Griffin-Lim | 8.30 % | −3.0 dB | 0.42 s | 166 KiB |
| noise | `--phase rgb --depth 16` | **0.002 %** | **+92.2 dB** | 0.11 s | 485 KiB |
| sweep | grey, Griffin-Lim | 4.84 % | −3.1 dB | 0.38 s | 16.5 KiB |
| sweep | `--phase rgb` | **0.40 %** | **+45.1 dB** | 0.09 s | 13.9 KiB |

The waveform SNR is the number that changes character. Without the phase it is
*negative* — the rebuilt waveform is further from the original than silence is,
even with the spectrum 97 % right. With it, the samples genuinely come back.

At 8 bits the file does not grow, and on two of those clips it shrinks, because
the masking pays for itself: phase is noise, noise does not compress, so it is
written only where the magnitude is within `--phase-floor` dB (60 by default) of
the peak — 5.7 % of the pixels on the melody. Everywhere else the blue channel
repeats the grey, which is also what keeps the picture legible: unmasked phase
is confetti across the whole frame.

Noise is the case that does grow, and there is no trick for it — noise *is*
phase, so its picture has to carry phase everywhere.

Storing the phase needs the exactly-invertible layout, and says so rather than
quietly doing something worse: `gray`, a `linear` frequency axis from 0 Hz, and
one row per bin. An angle cannot be interpolated — halfway between +3.1 and
−3.1 radians is not 0, it is a whole turn from the truth.

It is **not** the default. The grey picture is the format as described at the
top of this file, and `--phase rgb` is the switch for when the sound matters
more than the picture.

### Or make it exact

`--lossless` adds a fourth channel holding a correction: the encoder decodes
its own picture, subtracts the result from the source, and writes the
difference into alpha. Decoding adds it back.

```bash
audioimage roundtrip voice.wav --lossless
cmp voice.wav voice-decoded.wav        # no output: the files are identical
```

Not "close", not "+92 dB" — **byte-for-byte identical**, on a melody, on a
sweep, and on white noise, with a test for each. The picture stays grey and
readable; the correction is invisible in the alpha channel.

| | source WAV | lossless picture |
|---|---:|---:|
| melody | 90.5 KiB | **80.3 KiB** |
| sweep | 44 KiB | 53.8 KiB |
| noise | 88 KiB | 292 KiB |

The correction costs whatever it costs to store, which is why it is paired
with a stored phase — the closer the picture already is, the less there is to
correct. The defaults (`--phase-floor 60`, the `db` scale) turn out to be the
smallest combination: raising the floor shrinks the correction but grows the
phase plane faster, because phase is noise and noise does not compress.

Exactness is defined against a bit depth — `--lossless-bits`, 16 by default,
which is what a WAV holds. It is the *samples* that come back exactly.

## In the browser

```bash
python3 -m audioimage serve          # then open http://127.0.0.1:8080/
```

One page, no build step, no npm — it is three files of plain HTML, CSS and
JavaScript served by :mod:`http.server`, and every button is one of the
commands above.

What the page adds is the part that only makes sense live:

- **a playhead that runs across the plane** while the sound plays, so you can
  see where in the picture you are hearing;
- **click anywhere on the plane to seek** there;
- **a readout** of the time, frequency, intensity and dB under the cursor;
- **a brush** — paint straight onto the plane and decode it to hear what you
  drew, which is the metadata-free path in the section above made interactive;
- **playback of the original and the decoded clip side by side**, on the same
  picture, so the difference is audible rather than described.

Anything the browser can decode (MP3, M4A, OGG, FLAC) is converted to mono WAV
in the page before it is sent, so the server only has to understand one format.
The microphone works too.

The server listens on 127.0.0.1 by default. Putting it on a public interface
hands anyone who can reach it a way to spend your CPU on Griffin-Lim, so
`--host` has to be given deliberately.

## The commands

```
audioimage tone       synthesise a test clip
audioimage encode     a WAV -> a picture of its spectrum
audioimage decode     a picture -> a WAV
audioimage roundtrip  both, and measure what survived
audioimage view       a labelled chart of a clip or a picture
audioimage waveform   the samples themselves
audioimage info       describe a WAV or a picture
audioimage serve      open the whole tool in a browser
```

Global options work before or after the command: `--json` (one JSON document on
stdout, everything else to stderr), `--backend {auto,python,numpy}`, `-q`, `-v`.
Exit codes are `0`, `1` for a handled error (one line on stderr, never a
traceback), `130` for an interrupt.

```bash
# a two-second sweep, drawn on a mel axis at 8 bits
audioimage encode sweep.wav -o sweep.png --freq-scale mel --height 256 --depth 8

# decode something that was never encoded by this tool
audioimage decode drawing.png -o drawing.wav --rate 22050 --iters 96

# machine-readable
audioimage --json roundtrip voice.wav | jq .metrics
```

Run `audioimage <command> --help` for the full list of options.

## The picture format

A picture written by `encode` is an ordinary PNG: 16-bit greyscale by default,
one column per frame, one row per frequency bin, readable by anything.

The settings live in a `tEXt` chunk under the key `audioimage`, as JSON:

```json
{"v":1,"tool":"audioimage 0.1.0","sample_rate":22050,"samples":55125,
 "n_fft":1024,"hop":256,"window":"hann","center":true,"frames":217,
 "scale":"db","top_db":80.0,"ref":125.9,"freq_scale":"linear",
 "f_min":0.0,"f_max":11025.0,"height":513,"width":217,"origin":"lower",
 "colormap":"gray","depth":16,"peak":0.468,"rms":0.173}
```

`ref` is the magnitude that became black, and `peak`/`rms` are the loudness of
the original, which is how a decode comes out at the volume it went in at.

**A picture without that chunk still decodes.** Draw a spectrogram by hand, take
one from another program, scribble in an image editor — `decode` falls back to
its defaults, infers `n_fft` from the height when it can, and makes a sound out
of it. Writing to an image and listening to the result is a legitimate way to
use this:

```bash
# the letters "ASI" drawn as dark pixels on a white 256x257 plane
audioimage decode drawn.png -o drawn.wav --rate 22050 --iters 60
audioimage view drawn.wav -o check.png --n-fft 512   # the letters are audible
```

## Choosing the settings

| what | option | effect |
|------|--------|--------|
| window length | `--n-fft` | bigger = finer frequencies, blurrier timing. 1024 suits music, 256–512 suits speech |
| overlap | `--hop` | `n_fft / 4` reconstructs exactly; larger hops are faster and rougher |
| amplitude | `--scale` | `db` (default) shows quiet detail the way hearing does; `linear` shows only the loudest parts; `sqrt` is between |
| dynamic range | `--top-db` | how far below the peak reaches white. 80 dB by default |
| frequency axis | `--freq-scale` | `linear` is the **only exactly invertible** one. `log` and `mel` compress the top; `circle` compresses *both* ends and gives the middle the detail |
| bulge | `--freq-bulge` | how hard `circle` bulges: 0 is linear, 1 the full arc (0.7) |
| size | `--height`, `--width` | resample the plane to a fixed rectangle (useful when something downstream wants one shape) |
| precision | `--depth` | 16 bits = 65536 levels of loudness, 8 bits = 256 and a third of the file size |
| phase | `--phase` | `rgb` stores it in the blue channel: no Griffin-Lim, +45 dB on the waveform |
| phase detail | `--phase-floor` | how far under the peak still gets a phase written (60 dB) |
| exactness | `--lossless` | a correction channel: the decode is the original, byte for byte |
| colour | `--colormap` | `gray` is the format. `fire`, `ice`, `viridis`, `magma` are invertible too, through their lookup tables, but lossier |
| phase effort | `--iters` | 16 is rough, 64 is good, 128 is better and twice as slow |

A 2.5 second melody encodes to 21.6 KiB at 16 bits (8.0 KiB at 8 bits) against
105 KiB of WAV — but this is not a compressor, and it should not be used as one.
The size comes from throwing the phase away.

## Using it as a library

```python
from audioimage import (
    Audio, EncodeConfig, PlaneConfig,
    compare, decode, encode, read_wav, write_png, write_wav,
)

audio = read_wav("voice.wav")

picture = encode(audio, EncodeConfig(
    n_fft=1024,
    hop=256,
    plane=PlaneConfig(scale="db", top_db=80, freq_scale="linear"),
))
write_png("voice.png", picture)

result = decode(picture, iterations=64)
write_wav("back.wav", result.audio)

print(compare(audio, result.audio))
# {'spectral_convergence': 0.041, 'log_spectral_distance': 0.60, ...}
```

The layers underneath are usable on their own:

| module | what it holds |
|--------|---------------|
| `audioimage.wav` | RIFF/WAVE in and out: 8/16/24/32-bit PCM and IEEE float |
| `audioimage.png` | a PNG codec: 8/16-bit greyscale and RGB, adaptive filtering, text chunks |
| `audioimage.dsp` | FFT, real FFT, STFT, inverse STFT, Griffin-Lim — written from scratch |
| `audioimage.spectrogram` | the plane mapping and its inverse |
| `audioimage.codec` | the whole chain, and the metadata |
| `audioimage.colormap` | intensity to colour, and back |
| `audioimage.render` | labelled views and waveform plots |
| `audioimage.synth` | tones, sweeps, noise, chords, melodies |
| `audioimage.server` | the HTTP API and the page, on `http.server` |
| `audioimage.font` | a 5x7 bitmap font, so views can label themselves |

## How it works

```
   samples ──stft──> complex spectrum ──abs──> magnitudes ──scale──> plane ──> pixels ──> PNG
                                                                                          │
   samples <─griffin-lim─ magnitudes <──unscale── plane <── pixels <────────────────────────
```

1. **Analyse.** Slide a Hann window over the samples, `n_fft` long, `hop` apart,
   and take the real FFT of each position. The window is periodic and the hop
   divides it evenly, so analysis and synthesis cancel exactly — a spectrum that
   is never touched reconstructs to about 5 × 10⁻¹⁶.
2. **Map.** Each magnitude becomes a number in `[0, 1]`: in dB, `1 + 20·log₁₀(m/ref)/top_db`,
   clamped. 1.0 is black.
3. **Draw.** Row 0 of the image is the top, so with `origin=lower` the lowest
   frequency sits at the bottom, the way a spectrogram is normally drawn.
4. **Undo.** All of the above runs backwards. Intensity 0 decodes to *silence*,
   not to a tone 80 dB down, so the floor of the picture does not become a hiss.
5. **Rebuild the phase.** Griffin-Lim: invert the magnitudes with the current
   phase guess, transform the result straight back, keep the phase that comes
   out, force the magnitudes back to the target, repeat. The over-relaxation of
   "fast Griffin-Lim" is damped by `α/(1+α)` — undamped, an `α` of 0.99 diverges
   and the error climbs.

`DESIGN.md` has the full specification.

## Dependencies and speed

**None.** The whole package is the Python standard library — the FFT, the PNG
encoder, the WAV parser and the font are all in this directory.

numpy, if it happens to be installed, is picked up automatically and used for
the transforms and for reading colours out of a picture. It is about **3.7×**
faster on the test suite (8.4 s against 30.7 s) and around 10× on Griffin-Lim
over long clips. The two
paths are held to agreeing within 1e-9, and the test suite checks that on every
operation, including the same random seed producing the same waveform.

```bash
audioimage info tune.png              # says which backend is in use
AUDIOIMAGE_BACKEND=python make test   # force the standard-library path
```

scikit-learn was tried for the nearest-colour search that inverts a false-colour
picture. It works, but two table entries are often exactly equidistant from an
off-table pixel and a KD-tree breaks those ties by its own internal layout — so
the same picture could decode to two different waveforms depending on whether
the library was installed. That is not acceptable in a codec, so the search
settles ties towards the lowest index and gets its speed from scoring a whole
picture's distinct colours in one vectorised pass instead.

## Tests

```bash
make test                 # 334 tests
make test-python          # the same, on the standard-library backend
```

They cover the transforms against a naive DFT, exact STFT reconstruction,
Griffin-Lim convergence, both backends agreeing value for value, PNG and WAV
round trips at every supported depth, plane invertibility, colour-map inversion
determinism, every CLI command and its error paths, and every HTTP route
including its refusals and a directory-traversal attempt. When Pillow is installed,
two extra tests check that it reads the PNGs this package writes and that this
package reads the PNGs it writes; when it is not, they skip.

## Limits

- **Phase is not stored by default.** A decode sounds like the original without
  sample-matching it. `--phase rgb` stores it and removes this limitation
  entirely, at the cost of an RGB picture that is tinted where the signal is.
- **Not a compressor.** A lossless picture lands near the size of the WAV it
  came from — sometimes under it, well over it for noise — and nowhere near
  FLAC. Smallness was never the point.
- **Mono.** Stereo input is mixed down on the way in.
- **`log` and `mel` axes resample**, so they are approximate in both directions,
  and they cannot carry a phase. `linear` is the exact one. The default height
  is already one row per bin, which is where they lose least; *reducing*
  `--height` costs a warped axis far more than a linear one (a `log` axis goes
  from 6 % error at 513 rows to 31 % at 128).
- **Lossy image formats will ruin a picture.** JPEG smears the spectrum and the
  decode hears the smear. PNG only.

---

Part of the [ASI](../README.md) repository. See the repository [LICENSE](../LICENSE)
— all rights reserved.
