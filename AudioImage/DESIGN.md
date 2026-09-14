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

### 2.1b The window shapes

`hann`, `hamming` and `blackman` are the usual cosine windows. `circle` is a
half circle laid over the frame — `w[i] = √(1 − x²)`, `x` running −1 to +1 —
which comes down to nothing at both ends, so neighbouring frames blend into
each other, while holding full weight across the middle far longer than a
cosine does. It reconstructs exactly like the others, because the synthesis
divides by the overlapped window power rather than assuming any particular
shape sums to one.

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
| `circle` | a circular arc (below) | by interpolation |

The `circle` axis compresses **both** ends rather than just the top. Its rate
of frequency change is `1 − √(1 − x²)` with `x` running −1 to +1 — zero at the
centre, one at the extremes — so integrating it gives an axis that crawls
through the mid range and races through the bottom and the top. `freq_bulge`
blends it back towards linear, because at full strength the centre is *flat*
and concentrates rows there by a factor of thousands; at the default 0.7 the
middle gets about ten times the detail of the ends.

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
| `v`, `tool` | format version (2; a version 1 picture simply has no phase), writer |
| `sample_rate`, `samples` | what to restore on decode |
| `n_fft`, `hop`, `window`, `center` | the analysis |
| `frames` | columns before any resize, so the time axis can be stretched back |
| `scale`, `top_db`, `ref` | the amplitude mapping |
| `freq_scale`, `f_min`, `f_max` | the frequency mapping |
| `height`, `width`, `origin` | the plane's shape and orientation |
| `colormap`, `depth` | how to read the pixels |
| `phase`, `phase_floor` | whether the blue channel is an angle, and down to what level |
| `lossless`, `lossless_bits` | whether the alpha channel is a correction, and to what depth |
| `peak`, `rms` | the loudness of the original |

### 5.2 Pictures from elsewhere

A picture with no chunk, or a damaged one, is not an error: `read_meta` returns
`{}` and the decode falls back to the caller's defaults. When the height is one
more than a power of two it is taken as `n_fft/2 + 1` and `n_fft` is inferred.

This is what makes "draw a picture, hear it" work. A 256×257 plane with the
letters `ASI` stamped on it in black decodes to 1.48 s of audio whose
spectrogram shows the letters.

## 6. Phase

A grey picture has one number per point. The STFT has two — a magnitude and an
angle — so one of them has to go somewhere. There are two answers here.

### 6.0 Storing it: `phase="rgb"`

```
R = G = magnitude    keeps the picture grey, so it still reads as a spectrogram
B     = phase        quantised over a full turn, with wraparound
```

The angle is quantised as `round((phase + pi) / 2pi * steps) mod steps`, and the
modulo is the point: +pi and -pi are the same angle, so the last bucket has to
meet the first rather than clamp against it. At 8 bits that is 1.4 degrees of
error, at 16 bits none worth measuring.

**Masking.** Phase is written only where the intensity is within `phase_floor`
dB of the peak. Below that the blue channel repeats the grey. Two things come
of it, and both matter:

* the file stays small — phase is noise and noise does not compress, so writing
  it everywhere costs 4x; masked to the 5.7% of pixels that have energy, an
  8-bit phase picture is *smaller* than the 16-bit grey one;
* the picture stays legible — unmasked phase speckles the entire frame into
  confetti, which was measured by doing it and looking at the result.

Nothing audible is lost: a point 60 dB down is a thousandth of the amplitude,
and its angle cannot be heard.

**What it requires.** The exactly-invertible layout — `gray`, `linear` from
0 Hz, one row per bin, one column per frame — enforced in
:class:`EncodeConfig`. An angle cannot be interpolated: halfway between +3.1 and
-3.1 radians is not 0 but a whole turn from the truth, so a warped or resized
plane is refused rather than quietly resampled.

**What it buys**, measured against the same clips through the grey format:

=========  ====================  ==================  ==================
clip       grey + Griffin-Lim    rgb, 8-bit          rgb, 16-bit
=========  ====================  ==================  ==================
melody     2.51% / -2.8 dB       0.46% / +44.1 dB    0.10% / +58.2 dB
noise      8.30% / -3.0 dB       0.49% / +43.9 dB    0.002% / +92.2 dB
sweep      4.84% / -3.1 dB       0.40% / +45.1 dB    0.025% / +69.0 dB
=========  ====================  ==================  ==================

(spectral error / waveform SNR).  Decoding also stops being iterative: 0.08 s
against 0.45 s, because there is nothing to iterate.

### 6.0b Making it exact: `lossless`

Storing a 16-bit magnitude and a 16-bit phase gets within about one
least-significant bit of a 16-bit WAV - 92% of samples already identical - and
no amount of extra precision in those two numbers closes the last gap cheaply,
because the error is spread across the quantisation of both.

So the last gap is closed directly rather than approximated away. The encoder
decodes the picture it has just written, using the *same* function the decoder
will use (:func:`_reconstruct` - one shared implementation is what makes the
two sides agree), quantises both the source and its own result to
``lossless_bits``, and writes the difference into the alpha channel, one value
per pixel. Decoding adds it back.

The result is byte-for-byte identity, verified by comparing whole WAV files on
a melody, a sweep, white noise and silence.

Two things are checked rather than assumed: that the picture has at least as
many pixels as the clip has samples (it does by a factor of two at
``hop = n_fft/4``, and the encoder refuses rather than truncating if a large
hop makes it false), and that every correction fits the channel width (it
refuses rather than clamping, since a clamped correction is a silent loss).

The correction is what makes the base quality matter for *size* rather than
quality: the closer the picture already is, the less there is to store.
Measured on a melody, the smallest combination is the default one - a 60 dB
phase floor and the ``db`` scale, at 89% of the source WAV. Raising the floor
shrinks the correction but grows the phase plane faster, because phase is noise
and noise does not compress.

### 6.1 Rebuilding it: `phase="none"`

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

### 6.2 The damping matters

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

### 6.3 Reproducibility

The starting phase is drawn from the **standard library's** `random.Random`
in both backends, in the same order, so a seed gives the same waveform whether
or not numpy is installed. Two different random starts land on two different,
equally valid answers, so this is not cosmetic.

### 6.4 Level

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
- **Phase in a second image.** Not needed: it fits in the blue channel of the
  one picture (section 6.0), masked so the frame stays readable.
- **Lossy image formats.** The decode would hear the artefacts.

## 11. Where this goes next

The plane is a fixed-size rectangle of numbers in `[0, 1]`, which is the shape
the rest of this repository's models want. `--height` and `--width` exist for
exactly that. A model that learns to produce these rectangles produces sound,
because `decode` is already written.
