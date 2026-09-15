# frontend

The browser page for AudioImage. **One page, no build step, no framework** —
three files, served as static assets by `audioimage/server.py`.

The server does all of the audio work: every button here is one of the commands
the tool already has. What the browser adds is the part that only makes sense
live — the playhead running across the plane while the sound plays, a readout of
what is under the cursor, and a brush, so you can paint on the plane and hear
what you painted.

Audio that is not already a WAV is decoded by the browser and re-encoded here,
so the server only ever has to understand one format.

## Contents

| file | what it is |
|---|---|
| `index.html` | the page: numbered panels — a sound to look at, the settings, the plane, the decode, the measurements |
| `app.js` | all of the behaviour — calls the HTTP API, drives the `<canvas>`, the playhead, the cursor readout and the brush, and converts non-WAV audio with the Web Audio API |
| `style.css` | all of the styling, dark by default |

## Opening it

```bash
cd ../..            # the AudioImage/ directory
make serve          # or: python -m audioimage serve
```

Then open the address it prints. There is nothing to install or build — the
files are served exactly as they are.

`server.py`'s `frontend_dir()` looks here by default; `serve --frontend-dir PATH`
points it somewhere else.
