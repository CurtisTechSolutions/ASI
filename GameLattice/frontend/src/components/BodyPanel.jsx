import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { curveSeries } from "../matrix.js";
import { agreementOf, verdictWord, walkLine } from "../trace.js";
import { fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";
import JobProgress from "./JobProgress.jsx";
import LineChart from "./LineChart.jsx";
import NamingTrace from "./NamingTrace.jsx";

/** The arms of E1 as toggles: `[key, label, hint, default]`. */
const ARMS = [
  ["encoder_sets_state", "Encoder sets the state", "off: the lattice continues from its own prediction (the plain control)", true],
  ["forward_lessons", "Forward lessons", "teach the row when its prediction and the next named state disagree", false],
  ["punish", "Punish what is wrong", "off: rewards only", true],
  ["whole_run", "Whole run", "credit the run by its final answer instead of every step", false],
  ["label_set", "Label set", "the training label is every right state, not one", false],
];

const DEFAULT_ARMS = Object.fromEntries(ARMS.map(([key, , , d]) => [key, d]));

/**
 * The Body tab: make a body for a language (the encoder's states, window, gram size, seed, and the arms of E1),
 * train it as a job with the held-out curve drawn as it goes, and read a string - quietly (Classify) or as one
 * learning episode (Run) - with the naming trace, the centrepiece: the stage game naming a state at every
 * character.
 */
export default function BodyPanel({ status, onMoved, read, setRead, step, setStep, goToGame }) {
  const [languages, setLanguages] = useState([]);
  const [language, setLanguage] = useStoredState("body.language", "even-b");
  const [states, setStates] = useStoredState("body.states", "13");
  const [width, setWidth] = useStoredState("body.window", "8");
  const [n, setN] = useStoredState("body.n", "1");
  const [seed, setSeed] = useStoredState("body.seed", "1");
  const [arms, setArms] = useStoredState("body.arms", DEFAULT_ARMS);
  const [episodes, setEpisodes] = useStoredState("body.episodes", "1000");
  const [maxLength, setMaxLength] = useStoredState("body.maxLength", "6");
  const [text, setText] = useStoredState("body.text", "abab");
  const [job, setJob] = useState(null);
  const [curve, setCurve] = useState([]);
  const [trained, setTrained] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [info, setInfo] = useState(null);

  useEffect(() => {
    api
      .languages()
      .then((list) => setLanguages(Array.isArray(list) ? list : []))
      .catch((err) => setError(err.message));
  }, []);

  async function makeNew() {
    setBusy(true);
    try {
      const s = await api.newBody({
        language,
        states: parseInteger(states, 13),
        window: parseInteger(width, 8),
        n: parseInteger(n, 1),
        seed: parseInteger(seed, 1),
        ...arms,
      });
      setCurve([]);
      setTrained(null);
      setRead(null);
      setInfo(`A fresh body for ${s.settings.language}: ${s.shape.join(" × ")}, window ${s.settings.window}, seed ${s.settings.seed}.`);
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function train() {
    setBusy(true);
    setInfo(null);
    try {
      const result = await api.train({ episodes: parseInteger(episodes, 1000), max_length: parseInteger(maxLength, 6) }, (j) => {
        setJob(j);
        if (Array.isArray(j.curve) && j.curve.length) setCurve(j.curve);
      });
      setTrained(result);
      setCurve(Array.isArray(result.history) ? result.history : result.curve || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setJob(null);
      setBusy(false);
      if (onMoved) onMoved();
    }
  }

  async function readText(learn) {
    setBusy(true);
    try {
      const r = learn ? await api.run(text) : await api.classify(text);
      setRead(r);
      setStep(0);
      setError(null);
      if (learn && onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const chosen = languages.find((l) => l.name === language);
  const body = status && status.body && !status.body.busy ? status.body : null;
  const c = read && read.classification;
  const agree = agreementOf(read);

  return (
    <div className="panel">
      <div className="card">
        <h2>A body</h2>
        <p className="note">
          A text sense over <code>a</code> and <code>b</code>, read by the GameRadix encoder, naming the states of a
          sense lattice whose accepting set is the language's; the oracle credits every step. The arms are E1&apos;s.
        </p>
        <SelectField
          label="Language"
          value={language}
          onChange={setLanguage}
          options={(languages.length ? languages : [{ name: language, description: "", accepting: [], min_states: "" }]).map((l) => [
            l.name,
            `${l.name}${l.description ? ` — ${l.description}` : ""}`,
          ])}
          hint={chosen ? `accepts at ${chosen.accepting.join(", ")}; ${chosen.min_states} states suffice` : undefined}
        />
        <div className="row">
          <NumberField label="States" value={states} onChange={setStates} min="2" step="1" hint="S" />
          <NumberField label="Window" value={width} onChange={setWidth} min="1" step="1" hint="W" />
          <NumberField label="Gram size" value={n} onChange={setN} min="1" step="1" hint="n" />
          <NumberField label="Seed" value={seed} onChange={setSeed} min="0" step="1" />
        </div>
        <div className="checks">
          {ARMS.map(([key, label, hint]) => (
            <CheckField key={key} label={label} hint={hint} checked={Boolean(arms[key])} onChange={(v) => setArms({ ...arms, [key]: v })} />
          ))}
        </div>
        <div className="actions">
          <button type="button" disabled={busy} onClick={makeNew}>
            New
          </button>
          <span className="muted">replaces the body the server holds; the held-out set is drawn from the seed</span>
        </div>
      </div>

      <div className="card">
        <h2>Train</h2>
        <p className="note">
          Random strings of the language, each one episode: the lattice steps from the state the encoder named, the
          encoder names the next from the walk window, and every step is credited by the oracle. Held-out accuracy on
          a fixed 300 every tenth of the way: the encoder&apos;s verdict (does the named state accept?) and the
          lattice&apos;s own.
        </p>
        <div className="row">
          <NumberField label="Episodes" value={episodes} onChange={setEpisodes} min="1" step="100" />
          <NumberField label="Max length" value={maxLength} onChange={setMaxLength} min="0" step="1" />
        </div>
        <div className="actions">
          <button type="button" className="primary" disabled={busy} onClick={train}>
            {job ? "Training…" : "Train"}
          </button>
          {body ? (
            <span className="muted">
              {body.episodes} episodes so far
              {body.accuracy ? ` · held out ${fmtNum(body.accuracy.encoder, 3)} / ${fmtNum(body.accuracy.lattice, 3)}` : ""}
            </span>
          ) : null}
        </div>
        <JobProgress job={job} />
        {trained ? (
          <p>
            {trained.stopped ? "Stopped after part of " : ""}
            {trained.episodes} episodes of <b>{trained.language}</b>: encoder {fmtNum(trained.before.encoder, 3)} →{" "}
            <b>{fmtNum(trained.after.encoder, 3)}</b>, lattice {fmtNum(trained.before.lattice, 3)} →{" "}
            <b>{fmtNum(trained.after.lattice, 3)}</b> on {trained.test} held-out strings.
          </p>
        ) : null}
        <LineChart series={curveSeries(curve)} xLabel="episodes" yLabel="held-out accuracy" height={220} emptyText="Train to draw the curve." />
      </div>

      <div className="card wide">
        <h2>Read a string</h2>
        <div className="inline-form">
          <TextField label="Text" value={text} onChange={setText} placeholder="abab" hint="a and b; empty is allowed" />
          <div className="actions">
            <button type="button" disabled={busy} onClick={() => readText(false)} title="quiet: nothing moves">
              Classify
            </button>
            <button type="button" disabled={busy} onClick={() => readText(true)} title="one learning episode on this string">
              Run (learn)
            </button>
          </div>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        <Alert kind="info" message={info} onDismiss={() => setInfo(null)} />
        {c ? (
          <>
            <div className="verdicts">
              <span className={`verdict-tile ${verdictWord(c.encoder, read.member)}`}>
                encoder <b>{c.encoder ? "accept" : "reject"}</b> <small>{verdictWord(c.encoder, read.member)}</small>
              </span>
              <span className={`verdict-tile ${verdictWord(c.lattice, read.member)}`}>
                lattice <b>{c.lattice ? "accept" : "reject"}</b> <small>{verdictWord(c.lattice, read.member)}</small>
              </span>
              <span className="verdict-tile">
                in {status && status.body && status.body.settings ? status.body.settings.language : "the language"}{" "}
                <b>{read.member ? "yes" : "no"}</b>
              </span>
              <span className="verdict-tile" title="mean entropy of the aggregates, and the lattice's mean -log2 p">
                bits <b>{fmtNum(c.encoder_bits, 3)}</b> / {fmtNum(c.lattice_bits, 3)}
              </span>
              <span className="verdict-tile" title="steps whose named state was the lattice's prediction">
                agree <b>{agree.agreed}</b> / {agree.steps}
              </span>
              <span className="verdict-tile" title="steps that would have asked the meta-player">
                triggers <b>{c.triggers}</b>
              </span>
            </div>
            <p className="mono walk">
              {read.learned ? "learned: " : "quiet: "}
              {walkLine(read)}
            </p>
            <h3>The naming trace</h3>
            <p className="note">
              One row a character. The auction sells seats to the window&apos;s nodes (bid = reputation × (best belief
              − abstain), above the reserve; every seat pays the highest losing bid); the seats play the congestion
              game over the {read.states} states; the aggregate&apos;s argmax is the named state - where the lattice
              continues from - beside the lattice&apos;s own prediction (▾). Hover a seat for its numbers; click a row
              for the {" "}
              <button type="button" className="link" onClick={goToGame}>
                Game
              </button>{" "}
              tab&apos;s view of it.
            </p>
            <NamingTrace read={read} selected={step} onSelect={setStep} />
          </>
        ) : (
          <p className="muted">Classify a string to see the stage game name a state at every character.</p>
        )}
      </div>
    </div>
  );
}
