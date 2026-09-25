import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { asArray, fmtInt, jobIsRunning, parseInteger, parseNumber, unitName } from "../util.js";
import { ENCODING_PRESETS, describeEncoding, encodingSpec, replayLine } from "../settings.js";
import Alert from "./Alert.jsx";
import AttentionBandCard from "./AttentionBandCard.jsx";
import DynamicWindowCard from "./DynamicWindowCard.jsx";
import { NumberField, SelectField, TextField } from "./Fields.jsx";

/**
 * The score function of every kind that has one, as {name, label, hint, integer, min}.
 * The server rejects the settings a kind does not know, so the form only offers its own.
 */
const SCORE_FUNCTIONS = {
  count: {
    title: "Dual frequency + rewards",
    blurb: (
      <>
        An edge is weighed by its <b>share of its node's traversals</b> twice: over the whole history (global)
        and inside a <b>sliding window</b> of the last N traversals seen anywhere in the graph (recent) - each
        smoothed by 0.5 and compared against the node's total - plus its rewards:
        <code> global_scale · log(R_all) + window_scale · log(R_recent) + reward_scale · reward</code> (+{" "}
        <code>count_scale · log(1 + traversals)</code>, off by default). Probabilities follow{" "}
        <code>R_all^global · R_recent^window · e^reward</code>; with the default 0.5 / 0.5 that is the geometric
        mean of the two shares, so when history and the window agree the probability <i>is</i> the share.
      </>
    ),
    fields: [
      { name: "global_scale", label: "Global scale", hint: "all-time share" },
      { name: "window_scale", label: "Window scale", hint: "recent share" },
      { name: "reward_scale", label: "Reward scale", hint: "how loudly a reward speaks" },
      { name: "count_scale", label: "Count scale", hint: "log(1 + traversals), 0 = off" },
      { name: "path_scale", label: "Path scale", hint: "how loudly a judged path speaks" },
      { name: "window", label: "Window", hint: "traversals remembered", integer: true, min: 1 },
    ],
  },
  resonant: {
    title: "Phase and resonance",
    blurb: (
      <>
        A walk carries a <b>phase</b> advanced by every trigram - a position clock plus a hash kick - and every
        edge learns the phase at which it fires and how <b>coherently</b>. The score of an edge is
        <code> amp_scale · log(share) + reward_scale · reward + resonance_scale · coherence · cos(phase − mu)</code>,
        so an edge is cheap exactly when the walk arrives in phase with it. <code>kick_scale = 0</code> leaves a
        pure clock; turning it up makes the phase a rolling signature of the whole path, at the price of much
        sparser statistics per phase.
      </>
    ),
    fields: [
      { name: "buckets", label: "Buckets", hint: "positions on the phase ring", integer: true, min: 1 },
      { name: "period", label: "Period", hint: "characters per turn of the clock" },
      { name: "kick_scale", label: "Kick scale", hint: "0 = a pure clock" },
      { name: "resonance_scale", label: "Resonance scale", hint: "how loudly the phase speaks" },
      { name: "amp_scale", label: "Amplitude scale", hint: "the phase-free share" },
      { name: "reward_scale", label: "Reward scale", hint: "how loudly a reward speaks" },
      { name: "concentration", label: "Concentration", hint: "shrinks a thinly seen edge towards no opinion" },
    ],
  },
};

const NO_SCORE_FUNCTION = {
  radix: (
    <>
      The sine model has no score function to set: an edge's score is <code>w · f_p · f_c</code>, the weight
      times the two activations, and all of it is <b>learned</b> - the weights by the one-hop local rule and the
      four sine parameters per node by the same pass. What would be a setting here is a trained number there.
      Its learning rates are on the <b>Train</b> tab.
    </>
  ),
  negative: (
    <>
      The negative network's blame function (<code>share_scale</code>, <code>blame_scale</code>,{" "}
      <code>clear_scale</code>) and the thresholds it judges by live on the <b>Negative</b> tab, beside the
      failures they weigh.
    </>
  ),
};

/** The whole score function as the server reports it back, settable parts and fixed ones alike. */
function describeWeights(weights) {
  if (!weights || typeof weights !== "object") return null;
  const parts = Object.entries(weights)
    .filter(([key]) => key !== "function" && key !== "kind")
    .map(([key, value]) => `${key.replace(/_/g, " ")} ${value}`);
  return parts.length ? parts.join(", ") : null;
}

/** The score function of whichever kind is active, loaded from the server and written back to it. */
function ScoreFunctionCard({ status }) {
  const kind = (status && status.kind) || null;
  const spec = kind ? SCORE_FUNCTIONS[kind] : null;
  const [values, setValues] = useState({});
  const [loaded, setLoaded] = useState(null); // the kind the form was loaded for
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const jobRunning = jobIsRunning(status);

  const load = useCallback(async (which) => {
    setError(null);
    try {
      const info = await api.model();
      const weights = info && info.weights;
      if (!weights) {
        setValues({});
      } else {
        setValues(Object.fromEntries(Object.entries(weights).map(([key, value]) => [key, String(value ?? "")])));
      }
      setLoaded(which);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    if (!spec || loaded === kind) return;
    load(kind);
  }, [spec, kind, loaded, load]);

  async function apply() {
    if (!spec) return;
    setBusy(true);
    setNotice(null);
    setError(null);
    try {
      const body = {};
      for (const field of spec.fields) {
        const raw = values[field.name];
        if (raw === undefined || raw === "") continue;
        body[field.name] = field.integer ? parseInteger(raw, 1) : parseNumber(raw, 0);
      }
      const data = await api.modelWeights(body);
      const now = describeWeights(data && data.weights);
      setNotice(
        now
          ? `Every edge weight was recomputed. The score function is now ${now}.`
          : "Applied; every edge weight was recomputed.",
      );
      await load(kind);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Score function</h2>
      {!kind ? (
        <p className="muted">Waiting for the model…</p>
      ) : !spec ? (
        <p className="muted">{NO_SCORE_FUNCTION[kind] || "This model kind has no configurable score function."}</p>
      ) : (
        <>
          <h3>{spec.title}</h3>
          <p className="muted">{spec.blurb}</p>
          {spec.fields.map((field, i) =>
            i % 3 === 0 ? (
              <div className="row" key={field.name}>
                {spec.fields.slice(i, i + 3).map((f) => (
                  <NumberField
                    key={f.name}
                    label={f.label}
                    hint={f.hint}
                    value={values[f.name] ?? ""}
                    onChange={(value) => setValues((prev) => ({ ...prev, [f.name]: value }))}
                    min={f.min}
                    step={f.integer ? 1 : "any"}
                    disabled={busy || jobRunning}
                  />
                ))}
              </div>
            ) : null,
          )}
          <div className="actions">
            <button type="button" className="primary" disabled={busy || jobRunning} onClick={apply}>
              {busy ? "Applying…" : "Apply score function"}
            </button>
            <button type="button" className="small" disabled={busy} onClick={() => load(kind)}>
              Reload from the model
            </button>
          </div>
          {jobRunning ? <p className="muted">A job is running; the score function can be changed once it finishes.</p> : null}
          {notice ? <p className="muted">{notice}</p> : null}
        </>
      )}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}

/** What the text becomes before the graph ever sees it, and what comes back out of it. */
function EncodingCard({ status }) {
  const [info, setInfo] = useState(null);
  const [text, setText] = useStoredState("network.previewText", "the cat sat on the mat");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  // asked again whenever the model changes: a new model can read text another way
  const modelKey = status ? `${status.kind}|${status.encoding}` : "";

  useEffect(() => {
    let alive = true;
    setPreview(null);
    api
      .encoding()
      .then((data) => {
        if (alive && data && typeof data === "object") setInfo(data);
      })
      .catch((err) => {
        if (alive) setError(err.message);
      });
    return () => {
      alive = false;
    };
  }, [modelKey]);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      setPreview(await api.encodingPreview(text));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const windows = asArray(preview && preview.windows);
  const unknown = new Set(asArray(preview && preview.unknown_windows));
  const path = preview && preview.path;
  const labels = asArray(path && path.labels);
  const word = info && info.unit === "word";
  const units = (count) => (word ? (count === 1 ? "word" : "words") : count === 1 ? "character" : "characters");
  const n = info ? info.ngram ?? info.window : null;

  return (
    <div className="card wide">
      <h2>Encoder / decoder</h2>
      {!info ? (
        <p className="muted">Asking the server what the encoding is…</p>
      ) : (
        <>
          <dl className="kv">
            <dt>encoding</dt>
            <dd>
              <code>{String(info.encoding ?? "")}</code> - {describeEncoding(info.encoding)}
            </dd>
            <dt>unit</dt>
            <dd>{units(2)}</dd>
            <dt>n</dt>
            <dd>
              {fmtInt(n)} {units(n)} per gram
            </dd>
            <dt>stride</dt>
            <dd>
              {fmtInt(info.stride)} ({info.stride === 1 ? "a sliding window" : info.stride === n ? "non-overlapping groups" : "overlapping grams"})
            </dd>
            <dt>overlap</dt>
            <dd>
              {fmtInt(info.overlap)} {units(info.overlap)} between neighbours
            </dd>
            <dt>sentinels</dt>
            <dd>
              <code title="where every text begins">{String(info.start_label)}</code>{" "}
              <code title="where every text ends">{String(info.end_label)}</code>{" "}
              <code title="where the graph has learned a walk goes round: taught by the rethinks, never by a corpus">
                {String(info.back_label)}
              </code>
              {info.think_label ? (
                <>
                  {" "}
                  <code title="where the graph has learned to stop and think, and where its thoughts begin">
                    {String(info.think_label)}
                  </code>
                </>
              ) : null}
            </dd>
          </dl>
          <p className="muted">
            Text goes in as {describeEncoding(info.encoding)} and comes back out of the (possibly merged) node labels
            along a path. Every label in the graph is written in it, so it is fixed for the model&apos;s life and
            saved with it: to read text another way, make a new model above.
          </p>
        </>
      )}
      <TextField
        label="Try a text"
        hint="encode it, decode it back, and walk it through the graph"
        value={text}
        onChange={setText}
        placeholder="the cat sat on the mat"
      />
      <div className="actions">
        <button type="button" className="primary" disabled={busy} onClick={run}>
          {busy ? "Encoding…" : "Encode and decode"}
        </button>
      </div>
      {preview ? (
        <>
          <h3>The encoder</h3>
          <p className="muted">
            {fmtInt(preview.chars)} {units(preview.chars)} become {fmtInt(preview.count)} gram(s).
          </p>
          <p className="chips">
            {windows.map((w, i) => (
              <code key={i} className={unknown.has(w) ? "chip unknown" : "chip"} title={unknown.has(w) ? "never seen by this model" : undefined}>
                {w.replace(/ /g, "␣")}
              </code>
            ))}
            {windows.length === 0 ? <span className="muted">nothing: the text is shorter than one gram.</span> : null}
          </p>
          <h3>The decoder</h3>
          <p className="text-display">{String(preview.decoded || "")}</p>
          <p className="muted">
            {preview.round_trip
              ? "The grams decode back to exactly the text that went in."
              : "The grams do not decode back to the text that went in (a text shorter than one gram encodes to nothing, and a grouping encoding drops the tail that fills no group)."}
          </p>
          <h3>Through the graph</h3>
          {path && path.known ? (
            <>
              <p className="muted">
                {fmtInt(path.nodes)} node(s) between the sentinels, {fmtInt(path.compressed)} of them <b>merged</b> -
                a label longer than one gram is a radix chain the graph compressed into one node. The decoder
                reads the text back off those labels.
              </p>
              <p className="chips">
                {labels.map((label, i) => (
                  <code key={i} className="chip">
                    {label.replace(/ /g, "␣")}
                  </code>
                ))}
              </p>
              <p className="text-display">{String(path.decoded || "")}</p>
            </>
          ) : (
            <p className="muted">{(path && path.reason) || "The model cannot walk this text."}</p>
          )}
        </>
      ) : (
        <p className="muted">Press the button to see the grams a text becomes and the nodes it walks.</p>
      )}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}

/** The model that is loaded: its kind, how it reads text, how big it is, and the buffer it rehearses from. */
function ThisModelCard({ status }) {
  if (!status) {
    return (
      <div className="card">
        <h2>This model</h2>
        <p className="muted">Waiting for the server…</p>
      </div>
    );
  }
  return (
    <div className="card">
      <h2>This model</h2>
      <dl className="kv">
        <dt>kind</dt>
        <dd>{String(status.model_label || status.kind || "–")}</dd>
        <dt>encoding</dt>
        <dd>
          {status.encoding ? (
            <>
              <code>{String(status.encoding)}</code> - {describeEncoding(status.encoding)}
            </>
          ) : (
            "–"
          )}
        </dd>
        <dt>counts in</dt>
        <dd>{unitName(status)}</dd>
        <dt>size</dt>
        <dd>
          {fmtInt(status.nodes)} nodes, {fmtInt(status.edges)} edges
        </dd>
        <dt>trained on</dt>
        <dd>
          {fmtInt(status.trained_texts)} texts over {fmtInt(status.epochs_total)} epochs
        </dd>
        <dt>file</dt>
        <dd>{status.model_path ? <code>{String(status.model_path)}</code> : "none (in memory only)"}</dd>
      </dl>
      <p className="muted">
        {replayLine(status.replay)} Its size is set by a training run (the replay buffer size on the{" "}
        <a href="#train">Train</a> tab); the buffer is saved in the model&apos;s file.
      </p>
    </div>
  );
}

/**
 * A new model in any kind and encoding: POST /api/reset. Two clicks, because it
 * replaces the model of that kind in memory.
 */
function NewModelCard({ status, onStatus }) {
  const kinds = asArray(status && status.kinds).filter((k) => k && typeof k.kind === "string");
  const activeKind = (status && status.kind) || "";
  const [kind, setKind] = useStoredState("model.new.kind", "");
  const [preset, setPreset] = useStoredState("model.new.encoding", "char:3:1");
  const [unit, setUnit] = useStoredState("model.new.unit", "char");
  const [n, setN] = useStoredState("model.new.ngram", "3");
  const [stride, setStride] = useStoredState("model.new.stride", "1");
  const [seed, setSeed] = useStoredState("model.new.seed", "");
  const [armed, setArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const jobRunning = jobIsRunning(status);

  const chosenKind = kinds.some((k) => k.kind === kind) ? kind : activeKind;
  const chosen = kinds.find((k) => k.kind === chosenKind);
  const custom = preset === "custom";
  const encoding = custom ? encodingSpec(unit, n, stride) : { spec: preset };
  const seedValue = seed.trim() === "" ? null : parseInteger(seed, null);
  const seedBad = seed.trim() !== "" && (seedValue === null || String(seedValue) !== seed.trim());
  const problem = encoding.error || (seedBad ? "The seed must be a whole number (blank = the server's)." : null);

  async function create() {
    if (problem) return;
    if (!armed) {
      setArmed(true);
      return;
    }
    setArmed(false);
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const body = { kind: chosenKind, encoding: encoding.spec };
      if (seedValue !== null) body.seed = seedValue;
      await api.reset(body);
      setNotice(`A new ${chosen ? chosen.label || chosenKind : chosenKind} model, reading text as ${describeEncoding(encoding.spec)}.`);
      if (onStatus) {
        const next = await api.status();
        if (next && typeof next === "object") onStatus(next);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>New model</h2>
      <p className="muted">
        A fresh, untrained model of any kind, reading text in any encoding. It replaces the model of that kind in
        memory - anything not saved is lost, though its file is untouched until you save - and a model of another
        kind stays in memory, as switching in the header keeps it.
      </p>
      <div className="row">
        <SelectField
          label="Kind"
          value={chosenKind}
          onChange={(value) => {
            setKind(value);
            setArmed(false);
          }}
          options={kinds.length ? kinds.map((k) => [k.kind, k.label || k.kind]) : [["", "–"]]}
          disabled={busy || jobRunning || kinds.length === 0}
        />
        <SelectField
          label="Encoding"
          hint="how it reads text; fixed for its life"
          value={preset}
          onChange={(value) => {
            setPreset(value);
            setArmed(false);
          }}
          options={ENCODING_PRESETS}
          disabled={busy || jobRunning}
        />
      </div>
      {custom ? (
        <div className="row">
          <SelectField
            label="Unit"
            value={unit}
            onChange={setUnit}
            options={[
              ["char", "characters"],
              ["word", "words"],
            ]}
            disabled={busy || jobRunning}
          />
          <NumberField label="n" hint="units per gram" value={n} onChange={setN} min={1} step={1} disabled={busy || jobRunning} />
          <NumberField
            label="Stride"
            hint="1 slides the window, n cuts groups"
            value={stride}
            onChange={setStride}
            min={1}
            step={1}
            disabled={busy || jobRunning}
          />
        </div>
      ) : null}
      <div className="row">
        <TextField label="Seed" hint="blank = the server's" value={seed} onChange={setSeed} placeholder="the server's" disabled={busy || jobRunning} />
      </div>
      {chosen && chosen.description ? <p className="muted">{String(chosen.description)}</p> : null}
      <Alert message={problem} />
      <div className="actions">
        <button
          type="button"
          className={armed ? "danger" : "primary"}
          disabled={busy || jobRunning || Boolean(problem) || !chosenKind}
          onBlur={() => setArmed(false)}
          onClick={create}
        >
          {busy ? "Creating…" : armed ? `Replace the ${chosen ? chosen.label || chosenKind : chosenKind} model? Click again` : "Create the model"}
        </button>
      </div>
      {jobRunning ? <p className="muted">A job is running; a new model can be made once it finishes.</p> : null}
      <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}

/**
 * Model settings: what belongs to the model and is saved with it, as opposed
 * to the settings of this browser (the Settings tab).
 *
 * **This model** - its kind, its encoding, its size and the replay buffer it
 * rehearses from; a **new model** in any kind and encoding; the **score
 * function** of whichever kind is active; the **attention band** - where inside
 * a gram a correction lands; the **dynamic window** - the ladder of node sizes
 * the graph is halved down and regrown up; and the **encoder / decoder** - what
 * a text becomes before the graph ever sees it, and what comes back out of it.
 */
export default function ModelSettingsPanel({ status, onStatus }) {
  return (
    <>
      <div className="card wide">
        <h2>Model settings</h2>
        <p className="muted">
          What belongs to the model and is saved with it: the encoding it reads text in, how an edge is scored,
          where inside a gram a correction lands (the attention band), the dynamic window its nodes are halved down
          and regrown up through, and the replay buffer it rehearses from. The settings of this browser - the
          traversal, the sampling filters, how a run walks its texts - are on the <a href="#settings">Settings</a> tab.
        </p>
      </div>
      <ThisModelCard status={status} />
      <NewModelCard status={status} onStatus={onStatus} />
      <ScoreFunctionCard status={status} />
      <AttentionBandCard status={status} onStatus={onStatus} />
      <DynamicWindowCard status={status} onStatus={onStatus} />
      <EncodingCard status={status} />
    </>
  );
}
