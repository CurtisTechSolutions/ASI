# results

The numbers in `../README.md`, committed beside the code that produced them.

| file | what it is |
|---|---|
| `learning_results.json` | four languages × four machine sizes (3, 4, 6 and the default 13) × five seeds: accuracy before and after 4 000 credited episodes, the curve every 500, and the quiet control |
| `stimulation_results.json` | a fork of widths 4, 1, 0.25: the probabilities by stimulation level; a surge of +3 relaxing over four lives |
| `adaptation_results.json` | two edges from one prototype, one rewarded and one punished on every use: every field at six stages, and the two weighting functions |
| `experiments.log` | the three tables as `make experiments` printed them |
| `python/` | the same three from the Python reference (`make py-experiments`) |

The records in this directory are the Rust crate's. `stimulation` and
`adaptation` draw nothing and are identical across the ports to the last
digit (the parity test checks it); `learning` draws its strings from each
port's own generator, so `python/learning_results.json` is a second sample
of the same experiment. Each JSON holds the experiment's settings and its
rows; `DESIGN.md` §§3–7 say what each experiment exercises. Reproduce with:

```bash
make experiments                    # Rust, a second
make py-experiments                 # Python, about a minute
make experiments LIFE=200
rust/target/release/latticefsm experiment --which learning --episodes 8000
```
