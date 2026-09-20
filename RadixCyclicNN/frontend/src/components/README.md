# components

The panels of the RadixCyclicNN app and the widgets they share. Plain JSX, no UI
library — `LineChart` and `GraphView` are hand-written SVG and `<canvas>`.

`App.jsx` renders these as tabs. `../README.md` covers the conventions they all
follow; `../../../README.md` documents what each panel actually *does*.

## The panels

| file | tab | what it drives |
|---|---|---|
| `TrainPanel.jsx` | Train | corpora, epochs, the learning-rate schedule, uploads |
| `PredictPanel.jsx` | Predict | a prefix in, the K best and K worst continuations out, with a Like button that rewards the result and a traversal to pick (follow the rewards, or avoid the punishments) |
| `GeneratePanel.jsx` | Generate | free generation, beam or sample, either traversal, with 👍 / 👎 feeding 2NRL |
| `ConversePanel.jsx` | Converse | the model talking to itself. Newest turn on top, stutter detection, and "Explore" to back out of a repetition |
| `ChatPanel.jsx` | Chat | an LLM conversing with the model and marking every reply |
| `ScorePanel.jsx` | Score | what the model makes of a piece of text |
| `TwoNRLPanel.jsx` | 2NRL | the three phases — train on the bad, invert, fine-tune on the good |
| `NegativePanel.jsx` | Negative | the negative network: the failures, why they were failures, and the automatic reviewer loop |
| `EvolvePanel.jsx` | Evolve | the self-upgrading loop |
| `OllamaPanel.jsx` | Ollama | a corpus written from a prompt, and adversarial review |
| `TutorPanel.jsx` | Tutor | the automated English lessons and the report card that plans the next batch |
| `CodeGenPanel.jsx` | Code | the code-generation loop — problems, attempts, sandbox runs, the teacher selector |
| `AgentPanel.jsx` | Agent | tool use: the network browses, an LLM judges, 2NRL follows |
| `ImagesPanel.jsx` | Images | the Stable Diffusion encoder, base64, and what the model remembers of a picture |
| `SpeechPanel.jsx` | Speech | teaching it by talking to it — microphone, dictation, and decoding an `aud:` text back to sound |
| `CheckpointPanel.jsx` | Checkpoints | list, save, restore |
| `GraphView.jsx` | Graph | the graph itself, drawn and navigable |

## The shared widgets

| file | what it is |
|---|---|
| `Fields.jsx` | the labelled form controls — `TextField`, `NumberField`, `SelectField`, `CheckField`, `TextArea`. Numeric values stay strings so a field can be cleared mid-edit; parsing happens on submit |
| `Alert.jsx` | an inline message box (`error` / `ok` / `info`) that renders nothing without a message |
| `JobStatus.jsx` | the state badge, timestamps and error of a background job |
| `LineChart.jsx` | a dependency-free SVG line chart. Non-finite points are dropped; an empty chart renders a placeholder |
| `StatusBar.jsx` | the header readout, polling `/api/status` every 2 s |
| `ModelSelector.jsx` | which of the three models is loaded, and where it came from |
| `UploadPicker.jsx` | choosing and uploading training files, ZIP archives included |
| `RatingsCard.jsx` | 👍 / 👎 on an output, and training on the ratings collected |
| `RecallCard.jsx` | what the model remembers of what it was shown |
| `GuardNotice.jsx` | what the negative network stopped on the way out |
| `TraversalFields.jsx` | the **traversal** the search runs — *what* it looks for, as opposed to the mode, which is how it looks. `reward` follows the model's own distribution, rewards and all; `punishment` takes the rewards out of the score and lets the penalties price every step, so the cheapest path is the least punished one. Used by the Predict and Generate tabs |

## Adding a panel

1. Write `ThingPanel.jsx` here, using `Fields.jsx` for inputs, `Alert.jsx` for
   messages and `useJob` for anything long-running.
2. Add a `thing()` call to `../api.js`.
3. Register the tab in `../App.jsx`.
4. From `../..`: `npm run build`, and **commit `../../dist`** — the servers serve
   the built page, not this directory.
