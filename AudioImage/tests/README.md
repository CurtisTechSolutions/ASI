# tests

The AudioImage test suite. Plain `unittest`, no plugins, no fixtures directory —
everything a test needs it synthesises.

## Contents

| file | covers |
|---|---|
| `__init__.py` | puts the project root on `sys.path` so `audioimage` imports from a checkout |
| `test_dsp.py` | the hand-written transforms — against a naive DFT, exact STFT reconstruction, Griffin-Lim convergence |
| `test_backends.py` | the numpy backend must agree with the hand-written one, value for value (1e-9), on every operation, including the same seed producing the same waveform |
| `test_spectrogram.py` | the plane mapping and its invertibility |
| `test_colormap.py` | intensity ↔ pixel, and the determinism of the nearest-colour search that inverts a false-colour picture |
| `test_codec.py` | the encode/decode round trip |
| `test_wav.py` | WAV round trips at every supported depth |
| `test_png.py` | the hand-written PNG codec, round-tripped at every depth |
| `test_render.py` | the bitmap font and the chart renderer |
| `test_synth.py` | the tone / sweep / noise / chord generators |
| `test_qlearn.py` | the per-hertz Q-learning network |
| `test_cli.py` | every command end to end, and its error paths |
| `test_server.py` | every HTTP route including its refusals and a directory-traversal attempt |

## Running them

```bash
cd ..
make test                 # the whole suite
make test-python          # the same, forced onto the standard-library backend
python3 -m unittest tests.test_codec -v      # one module
```

When Pillow is installed, two extra tests check that it reads the PNGs this
package writes and that this package reads the PNGs Pillow writes. When it is
not, they skip.
