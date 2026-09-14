# AudioImage — design

The specification of the encoding, why each decision was made the way it was,
and what it costs. `README.md` is the guide; this is the reference.

---

## 1. The mapping

A clip of audio is a sequence of real samples `x[n]`, `n = 0 … N-1`, at a rate
of `R` hertz. The encoding turns it into a rectangle `P[r][c]` of numbers in
`[0, 1]`:

| | |
|---|---|
| **column `c`** | the analysis frame starting at sample `c · hop` — time |
| **row `r`** | a frequency — with `origin = lower`, row 0 is the *highest* |
| **`P[r][c]`** | intensity: `1.0` is as loud as the clip gets, `0.0` is silence |

and then into pixels, with the rule the whole format rests on:

```
pixel = round((1 − intensity) · maxval)        # 1.0 → black, 0.0 → white
```

`maxval` is 255 at 8 bits, 65535 at 16. The inverse is
`intensity = 1 − pixel / maxval`, exact at both depths.

## 2. Analysis

### 2.1 The short-time Fourier transform

The signal is padded with `n_fft / 2` zeros at each end (`center = true`, the
default) so that frame `c` is *centred* on sample `c · hop`, and then with
enough zeros at the tail that every input sample falls inside at least one
frame. Each frame is multiplied by a window `w` and transformed:

```
X[c][k] = Σ(i = 0 … n_fft−1)  x[c·hop + i] · w[i] · exp(−2πi·k·i / n_fft)
```

for `k = 0 … n_fft/2`, which is `n_fft/2 + 1` bins. Bin `k` is the frequency
`k · R / n_fft`.

### 2.2 Why the window is periodic

The windows are defined periodically — `cos(2πn / N)`, not `cos(2πn / (N−1))`.
That is what makes overlapping frames sum to a constant, which is what makes
the inverse exact. `dsp.cola_sum` reports the overlap sum for any window and
hop; for Hann at `hop = n_fft/4` it is flat to 1e-12.

### 2.3 The inverse

Weighted overlap-add. Each frame is transformed back, multiplied by the same
window *again*, and added into an accumulator; a parallel accumulator sums the
squared window. Dividing one by the other removes the windowing exactly:

```
y[n] = Σ_c (IFFT(X[c])[n − c·hop] · w[n − c·hop]) / Σ_c w²[n − c·hop]
```

Where the denominator is ~0 the output is set to 0 rather than divided.
Measured: a random signal through `stft` then `istft` returns with a maximum
absolute error of **5.6 × 10⁻¹⁶** for Hann, Hamming and Blackman alike.

### 2.4 The real FFT

A real signal of length `n` is packed into `n/2` complex numbers
(`z[k] = x[2k] + i·x[2k+1]`), transformed with one half-length complex FFT, and
unpacked:

```
E[k] = (Z[k] + conj(Z[n/2 − k])) / 2
O[k] = (Z[k] − conj(Z[n/2 − k])) / 2i
X[k] = E[k] + exp(−2πik/n) · O[k],     X[n/2] = E[0] − O[0]
```

This halves the work against transforming the real signal directly.

**DC and Nyquist.** `X[0]` and `X[n/2]` are real for any real signal. A spectrum
that has come back from a picture does not respect that — its phase is invented,
including on those two bins — so `irfft` takes only their real parts and
discards the impossible imaginary parts, which is what numpy does too. Without
that step the two backends disagree by ~0.25 on Griffin-Lim output, because the
hand-written path folds the impossible component back into the samples instead
of dropping it.

## 3. Amplitude

`ref` is the magnitude that maps to 1.0 — by default the largest magnitude in
the clip, recorded in the metadata so the decode can undo the mapping.

| `scale` | forward | inverse |
|---------|---------|---------|
| `db` (default) | `t = 1 + 20·log₁₀(m/ref) / top_db` | `m = ref · 10^((t−1)·top_db/20)` |
| `linear` | `t = m / ref` | `m = t · ref` |
| `sqrt` | `t = √(m / ref)` | `m = t² · ref` |

All clamped to `[0, 1]`.

**`t = 0` decodes to `m = 0`.** Not to `ref · 10^(−top_db/20)`, which is what the
dB formula alone would give. The floor of the picture is silence; decoding it as
a tone 80 dB down would lay a hiss under the entire clip.

Why `db` is the default: hearing is roughly logarithmic. A harmonic a thousandth
of the peak is clearly audible and, on a linear scale, invisible — the picture
would show the fundamentals and nothing else.

## 4. Frequency

`freq_scale` decides which frequency each row stands for, over `[f_min, f_max]`:

| | rows are evenly spaced in | invertible |
|---|---|---|
| `linear` | hertz | **exactly**, when `height = n_fft/2 + 1` and `f_min = 0` |
| `log` | log hertz (from 1 Hz) | by interpolation |
| `mel` | mels, `2595·log₁₀(1 + f/700)` | by interpolation |

The exact case is detected and short-circuited: one row per bin, no arithmetic
in between. Everything else resamples by linear interpolation in both
directions — forward onto the row frequencies, back onto the bin frequencies —
and a binary search handles the uneven spacing of the log and mel grids.

Measured on a 440 Hz tone, magnitudes out against magnitudes in:

| axis | relative error |
|------|---------------:|
| `linear`, `linear`/`sqrt` amplitude | 3 × 10⁻¹⁷ |
| `linear`, `db` amplitude | 1.7 × 10⁻⁴ *(the leakage floor below −80 dB, clipped)* |
| `mel` | 8.0 × 10⁻² |
| `log` | 8.8 × 10⁻² |

The time axis resamples the same way when `width` is set; left alone it is one
column per frame and untouched.

## 5. The file

An ordinary PNG, written by `audioimage.png`:

- **16-bit greyscale** by default (65536 levels), 8-bit optional, 8-bit RGB for
  the false-colour maps.
- **Adaptive scanline filtering** — each row is written with whichever of the
  five PNG filters leaves it flattest, which is most of what makes a PNG small.
  A 200×200 16-bit gradient goes from 80000 raw bytes to 1753.
- **No interlacing**, on write or read.

### 5.1 Metadata

A `tEXt` chunk keyed `audioimage`, holding compact JSON with sorted keys so the
same input always produces the same file:

| field | meaning |
|-------|---------|
| `v`, `tool` | format version, writer |
| `sample_rate`, `samples` | what to restore on decode |
| `n_fft`, `hop`, `window`, `center` | the analysis |
| `frames` | columns before any resize, so the time axis can be stretched back |
| `scale`, `top_db`, `ref` | the amplitude mapping |
| `freq_scale`, `f_min`, `f_max` | the frequency mapping |
| `height`, `width`, `origin` | the plane's shape and orientation |
| `colormap`, `depth` | how to read the pixels |
| `peak`, `rms` | the loudness of the original |

### 5.2 Pictures from elsewhere

A picture with no chunk, or a damaged one, is not an error: `read_meta` returns
`{}` and the decode falls back to the caller's defaults. When the height is one
more than a power of two it is taken as `n_fft/2 + 1` and `n_fft` is inferred.

This is what makes "draw a picture, hear it" work. A 256×257 plane with the
letters `ASI` stamped on it in black decodes to 1.48 s of audio whose
spectrogram shows the letters.

## 6. Phase

Magnitudes alone do not determine a waveform. Griffin-Lim searches for one whose
spectrum matches:

```
S = the target magnitudes
c = S · exp(i·φ)                     φ random (seeded) or zero
repeat:
    t = STFT(ISTFT(c))               project onto "could be a real signal"
    step = t − α/(1+α) · t_previous  damped over-relaxation
    c = S · step / |step|            project onto "has the right magnitudes"
```

### 6.1 The damping matters

The acceleration is `α/(1+α)`, not `α`. Applied raw, a momentum of 0.99
overshoots so far that the error *rises*. Measured over 32 passes on a 3-tone
signal:

| momentum | undamped | damped |
|---------:|---------:|-------:|
| 0.00 | 0.156 | 0.156 |
| 0.50 | 0.121 | 0.121 |
| 0.90 | 0.262 | **0.075** |
| 0.99 | 0.336 | **0.073** |

Undamped, more momentum is worse. Damped, it behaves the way the literature says
it should.

### 6.2 Reproducibility

The starting phase is drawn from the **standard library's** `random.Random`
in both backends, in the same order, so a seed gives the same waveform whether
or not numpy is installed. Two different random starts land on two different,
equally valid answers, so this is not cosmetic.

### 6.3 Level

The decode is scaled to the RMS recorded at encode time. RMS rather than peak:
by Parseval the energy is preserved when the phase is replaced, but the tallest
sample is not, so peak-matching would mis-scale a correct decode — and would
then make the round trip *look* 0.29 wrong when it is 0.038 wrong.

## 7. Measurement

`compare` levels the two clips by RMS and reports:

- **`spectral_convergence`** — `‖|A| − |B|‖ / ‖|A|‖` over the STFT magnitudes.
  This is the number that matters; it is what the picture stored.
- **`log_spectral_distance`** — RMS difference in dB, floored 80 dB under the
  peak. Without the floor the score is dominated by bins silent in both clips,
  where 1e-12 against 1e-9 counts as 60 dB of error and nothing is audible.
- **`waveform_snr`** — reported to make the point that it is the wrong ruler.

## 8. Backends

`audioimage.dsp` is the standard library alone. `audioimage.dsp_numpy` is the
same maths vectorised, returning the same plain Python types — `.tolist()`, not
`list()`, so numpy scalars never leak into JSON output. `audioimage.backend`
chooses, `$AUDIOIMAGE_BACKEND` overrides, and `tests/test_backends.py` holds
them to 1e-9 on every operation including Griffin-Lim under four combinations
of initialisation and momentum.

Griffin-Lim is implemented twice rather than once over a shared abstraction,
because the loop is the hot path and converting a spectrogram in and out of
numpy on every pass would cost more than the vectorisation saves.

## 9. Colour

`gray` is arithmetic, exact at any depth, and is the only map the codec uses by
default. The others are 256-entry tables interpolated from control points, and
are inverted by finding which entry a pixel matches: a dictionary for an
untouched picture, a nearest-colour search otherwise.

That search must be **deterministic**. Off-table pixels are often exactly
equidistant from two entries — 55 times in 3000 random colours on `viridis` —
so ties always go to the lowest index. A KD-tree (scikit-learn) answers the
query correctly but breaks ties by its own layout, which would make the decode
depend on which libraries are installed; the vectorised path uses
`|a|² − 2a·b + |b|²` with numpy's `argmin`, which returns the first minimum and
so agrees with the scan by construction. Every value involved is a small integer
held exactly in float64, so the arithmetic is exact.

## 10. What is deliberately not here

- **Stereo.** Mixed to mono on the way in. Two planes, or two channels of one
  picture, would work; nothing needs it yet.
- **Compression.** The picture is smaller than WAV as a side effect of
  discarding the phase, not by design.
- **Phase in a second image.** It would make the round trip exact and the
  pictures meaningless — phase looks like noise. The point of the format is that
  the picture is legible.
- **Lossy image formats.** The decode would hear the artefacts.

## 11. Where this goes next

The plane is a fixed-size rectangle of numbers in `[0, 1]`, which is the shape
the rest of this repository's models want. `--height` and `--width` exist for
exactly that. A model that learns to produce these rectangles produces sound,
because `decode` is already written.
