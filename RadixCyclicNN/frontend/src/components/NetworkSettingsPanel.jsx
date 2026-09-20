import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { useNetworkSettings } from "../hooks/useNetworkSettings.jsx";
import { asArray, fmtInt, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, TextField } from "./Fields.jsx";
import TraversalFields from "./TraversalFields.jsx";

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
function EncodingCard() {
  const [info, setInfo] = useState(null);
  const [text, setText] = useStoredState("network.previewText", "the cat sat on the mat");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let alive = true;
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
  }, []);

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

  return (
    <div className="card wide">
      <h2>Encoder / decoder</h2>
      {!info ? (
        <p className="muted">Asking the server what the encoding is…</p>
      ) : (
        <>
          <dl className="kv">
            <dt>window</dt>
            <dd>{fmtInt(info.window)} characters</dd>
            <dt>stride</dt>
            <dd>{fmtInt(info.stride)}</dd>
            <dt>overlap</dt>
            <dd>{fmtInt(info.overlap)} characters between neighbours</dd>
            <dt>sentinels</dt>
            <dd>
              <code>{String(info.start_label)}</code> <code>{String(info.end_label)}</code>{" "}
              <code>{String(info.back_label)}</code>
            </dd>
            <dt>settable</dt>
            <dd>{info.configurable ? "yes" : "no"}</dd>
          </dl>
          <p className="muted">
            {String(info.note || "")} The window is not a setting but part of the <b>model format</b>: the graph's
            labels, its splits and merges, the saved file and the Go port all assume the same number, so a model
            trained at one window could not be read at another. What <i>is</i> adjustable - the score function
            above, and the traversal a search runs - has its own settings.
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
            {fmtInt(preview.chars)} characters become {fmtInt(preview.count)} overlapping windows.
          </p>
          <p className="chips">
            {windows.map((w, i) => (
              <code key={i} className={unknown.has(w) ? "chip unknown" : "chip"} title={unknown.has(w) ? "never seen by this model" : undefined}>
                {w.replace(/ /g, "␣")}
              </code>
            ))}
            {windows.length === 0 ? <span className="muted">nothing: the text is shorter than one window.</span> : null}
          </p>
          <h3>The decoder</h3>
          <p className="text-display">{String(preview.decoded || "")}</p>
          <p className="muted">
            {preview.round_trip
              ? "The windows decode back to exactly the text that went in."
              : "The windows do not decode back to the text that went in (a text shorter than one window encodes to nothing)."}
          </p>
          <h3>Through the graph</h3>
          {path && path.known ? (
            <>
              <p className="muted">
                {fmtInt(path.nodes)} node(s) between the sentinels, {fmtInt(path.compressed)} of them <b>merged</b> -
                a label longer than the window is a radix chain the graph compressed into one node. The decoder
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
        <p className="muted">Press the button to see the windows a text becomes and the nodes it walks.</p>
      )}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}

/**
 * Network settings: the settings of the network itself, as opposed to the
 * options of one run.
 *
 * The **traversal** every search uses (shared with the Predict and Generate
 * tabs, which show the same control), the **score function** of whichever kind
 * is active, and the **encoder / decoder** - what a text becomes before the
 * graph ever sees it, and what comes back out of it.
 */
export default function NetworkSettingsPanel({ status }) {
  const { traversal, shared, reset } = useNetworkSettings();
  return (
    <>
      <div className="card wide">
        <h2>Network settings</h2>
        <p className="muted">
          The settings of the network itself, as opposed to the options of one run: what every search looks for,
          how an edge is scored, and how text goes in and comes back out. The traversal is remembered in this
          browser; the score function is the model's own and is saved with it.
        </p>
      </div>

      <div className="card">
        <h2>Traversal</h2>
        <TraversalFields />
        {shared ? (
          <div className="actions">
            <button type="button" className="small" disabled={traversal === "reward"} onClick={reset}>
              Back to the default (reward)
            </button>
          </div>
        ) : null}
      </div>

      <ScoreFunctionCard status={status} />
      <EncodingCard />
    </>
  );
}
