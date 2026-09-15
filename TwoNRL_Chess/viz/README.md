# The page

`index.html` is the whole visualisation: one self-contained file with the run
data embedded in a `<script id="viz-data">` block. Open it straight from a
clone - there is no server, no build step to run first, and no network fetch.

It is one file on purpose. The Artifact sandbox this is published into blocks a
page from fetching its own data at runtime, and a single file is also what makes
it openable from a checkout.

| file | what it is |
|---|---|
| `index.html` | the page, with the data already inside it |
| `data.json` | the same data on its own, as `../export_viz.py` wrote it |
| `build.py` | writes `data.json` into the page's data block, in place |
| `checkpoints/` | the network saved at each phase boundary, which is what the action-space views are scored from |

## Rebuilding

```bash
make viz        # capture the checkpoints, export, and build the page
make viz-data   # just re-export and rebuild, reusing the checkpoints
```

`make viz` runs the `2nrl` arm once with `--checkpoints`, which saves the
network at the four states 2NRL has - `untrained`, `phase1_end`,
`after_invert`, `final`. `after_invert` minus `phase1_end` is the whole claim,
and the two action-space views exist to put them side by side.

Everything outside the data block is left untouched by `build.py`, so
`index.html` stays directly editable.
