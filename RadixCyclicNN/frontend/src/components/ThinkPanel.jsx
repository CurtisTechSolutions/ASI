import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { THINK_DEPTH, THINK_LENGTH, THINK_QUESTIONS, thinkRequest } from "../thinking.js";
import { unitName } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import ThoughtView from "./ThoughtView.jsx";

/** Thoughts kept on the page, newest first. */
const KEEP = 20;

/**
 * The model thinks. A thought is the prediction search run from the THINK sentinel instead of START, in the
 * language of the thoughts the model was taught (the Ollama tab's "Thinking from a prompt"). "About" thinks at
 * the node where a text ends, and teaches the model to stop and think there. Along its own path, wherever the
 * model has learned to think, the thought questions itself - a nested thought that must say something new - up
 * to "Depth" deep; when it stops it says what it triggered (../../../DESIGN.md section 36).
 */
export default function ThinkPanel({ status }) {
  const [about, setAbout] = useStoredState("think.about", "");
  const [mode, setMode] = useStoredState("think.mode", "beam");
  const [k, setK] = useStoredState("think.k", "5");
  const [maxLength, setMaxLength] = useStoredState("think.maxLength", String(THINK_LENGTH));
  const [temperature, setTemperature] = useStoredState("think.temperature", "1.0");
  const [seed, setSeed] = useStoredState("think.seed", "");
  const [depth, setDepth] = useStoredState("think.depth", String(THINK_DEPTH));
  const [questions, setQuestions] = useStoredState("think.questions", String(THINK_QUESTIONS));
  const [learn, setLearn] = useStoredState("think.learn", true);
  const [thoughts, setThoughts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function run(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const body = thinkRequest({ about, mode, k, maxLength, temperature, seed, depth, questions, learn });
      const data = await api.think(body);
      if (data && typeof data === "object") setThoughts((prev) => [data, ...prev].slice(0, KEEP));
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const latest = thoughts[0] || null;
  const empty = latest && !latest.text && latest.stopped === "nothing";

  return (
    <>
      <form className="card" onSubmit={run}>
        <h2>Think</h2>
        <p className="muted">
          A thought is the prediction search run from the <b>THINK</b> sentinel instead of START, so it is in the
          language of the thoughts the model was taught. Wherever its own path crosses a node the model has learned
          to stop and think at, the thought <b>questions itself</b>: a thought one level deeper that has to say
          something new. When it stops, it says what it triggered.
        </p>
        <TextField
          label="About"
          hint="optional: think at the node where this text ends"
          value={about}
          onChange={setAbout}
          placeholder="the cat sat"
        />
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            hint="beam: the likeliest"
            options={[
              ["beam", "beam"],
              ["sample", "sample"],
            ]}
          />
          <NumberField label="K" hint="thoughts weighed" value={k} onChange={setK} min={1} step={1} />
          <NumberField
            label="Max length"
            hint={`${unitName(status)} a thought may run to`}
            value={maxLength}
            onChange={setMaxLength}
            min={0}
            step={1}
          />
        </div>
        <div className="row">
          <NumberField
            label="Depth"
            hint="how deep it may question itself (0 = never)"
            value={depth}
            onChange={setDepth}
            min={0}
            step={1}
          />
          <NumberField
            label="Questions"
            hint="one thought may ask itself"
            value={questions}
            onChange={setQuestions}
            min={0}
            step={1}
          />
        </div>
        <div className="row">
          <NumberField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0}
            disabled={mode !== "sample"}
          />
          <NumberField
            label="Seed"
            hint="blank = random"
            value={seed}
            onChange={setSeed}
            step={1}
            placeholder="random"
            disabled={mode !== "sample"}
          />
        </div>
        <CheckField
          label="Learn where it thinks"
          hint="“About” teaches the model to stop and think at that node (this changes the model; the server keeps it until the model is saved)"
          checked={learn}
          onChange={setLearn}
          disabled={loading}
        />
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Thinking…" : "Think"}
          </button>
          <button type="button" className="small" disabled={loading || thoughts.length === 0} onClick={() => setThoughts([])}>
            Clear
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Thoughts</h2>
        {thoughts.length === 0 ? (
          <p className="muted">
            Press Think. A conversation thinks too: a voice that catches itself repeating thinks before it backs up,
            and the Converse tab shows what it thought under the turn.
          </p>
        ) : (
          <>
            {empty ? (
              <p className="muted">
                It has no thoughts to think with yet. Teach it some on the <a href="#ollama">Ollama</a> tab:
                “Thinking from a prompt”, with “Teach the thinking to the network” on - a thinking model&apos;s
                reasoning becomes the network&apos;s own thoughts, and every question it asked itself a place where
                the network stops to think.
              </p>
            ) : null}
            {thoughts.length > 1 ? <p className="muted">Newest first.</p> : null}
            <ol className="thoughts" reversed>
              {thoughts.map((t, i) => (
                <li key={thoughts.length - i}>
                  <ThoughtView thought={t} />
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
    </>
  );
}
