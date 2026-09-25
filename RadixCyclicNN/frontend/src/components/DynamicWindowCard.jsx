import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { DEFAULT_FLOOR, DEFAULT_TOP, SIZES, clampSize, describeStep, describeWindow, ladder } from "../window.js";
import { jobIsRunning, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";

const SIZE_OPTIONS = SIZES.map((size) => [String(size), String(size)]);

/** The ladder drawn as its rungs, the one the window stands on lit and the next one marked. */
function Ladder({ sizes, current, next, units }) {
  if (!sizes.length) return <p className="muted">Not a ladder: the top and the floor must be powers of two, the floor no larger than the top.</p>;
  return (
    <div className="ladder" aria-label={`the ladder ${sizes.join(", ")} ${units}`}>
      {sizes.map((size, i) => (
        <span key={size} className={`rung${size === current ? " current" : ""}${size === next ? " next" : ""}`} title={`${size} ${units}`}>
          {i > 0 ? <span className="rung-arrow">→</span> : null}
          <b>{size}</b>
        </span>
      ))}
      <span className="rung-arrow" title="from the floor, back up to the top">↺</span>
    </div>
  );
}

/**
 * The dynamic window (../../SPEC-DynamicWindow.md): a ladder of node sizes, halving from 32 to 4 and back up.
 * Read from GET /api/model/window, set with POST /api/model/window - it belongs to the model and is saved with
 * it - and stepped by hand with POST /api/model/window/step: every step merges what fits the window, halves
 * every node that is longer (both halves keep the node's data, joined by a heavy connection) and moves the
 * window down the ladder. With `auto` on, the model steps by itself at the end of every training epoch.
 */
export default function DynamicWindowCard({ status, onStatus }) {
  const [info, setInfo] = useState(null);
  const [on, setOn] = useState(false);
  const [top, setTop] = useState(String(DEFAULT_TOP));
  const [floor, setFloor] = useState(String(DEFAULT_FLOOR));
  const [size, setSize] = useState(String(DEFAULT_TOP));
  const [auto, setAuto] = useState(true);
  const [steps, setSteps] = useState("1");
  const [lastStep, setLastStep] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const jobRunning = jobIsRunning(status);
  const modelKey = status ? `${status.kind}|${status.encoding}` : "";

  const load = useCallback(async () => {
    setError(null);
    try {
      const data = await api.window();
      const window = data && data.window;
      if (!window || typeof window !== "object") return;
      setInfo(window);
      setOn(Boolean(window.on));
      setTop(String(window.on ? window.top : window.default_top ?? DEFAULT_TOP));
      setFloor(String(window.on ? window.floor : window.default_floor ?? DEFAULT_FLOOR));
      setSize(String(window.on ? window.size : window.default_top ?? DEFAULT_TOP));
      setAuto(window.on ? Boolean(window.auto) : true);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    setLastStep(null);
    load();
  }, [modelKey, load]);

  const units = info ? info.units : "units";
  const sizes = ladder(top, floor);
  const chosen = clampSize(size, top, floor);
  const nextOfChosen = chosen === null ? null : sizes[(sizes.indexOf(chosen) + 1) % sizes.length];

  async function refreshStatus() {
    if (!onStatus) return;
    const next = await api.status();
    if (next && typeof next === "object") onStatus(next);
  }

  async function apply() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const body = on ? { on: true, top: Number(top), floor: Number(floor), size: chosen ?? Number(top), auto } : { on: false };
      const data = await api.setWindow(body);
      setNotice(`The window is now ${describeWindow(data && data.window)}. It is saved with the model when the model is saved.`);
      await load();
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function step() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const data = await api.windowStep({ steps: parseInteger(steps, 1) });
      setLastStep(data && data.step);
      await load();
      await refreshStatus();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card wide">
      <h2>Dynamic window</h2>
      <p className="muted">
        A merged node can hold a whole sentence, and a walk through it has nowhere to branch. The dynamic window is
        a ceiling on that length, sized in the binary number system: it starts at <b>32</b> {units}, then moves to{" "}
        <b>16</b>, then <b>8</b>, then <b>4</b>, and then goes back up to 32 and runs again. One <b>step</b> merges
        what fits the window, halves every node that is longer at its middle gram - both halves keep the node&apos;s
        state, activation and count, joined by a <b>heavy connection</b> - and moves the window down the ladder. It
        steps by itself at the end of every training epoch, or by hand with the button below. Off, compression is
        unbounded and no node is halved.
      </p>
      {!info ? (
        <p className="muted">Asking the server about the window…</p>
      ) : (
        <p>
          Now: <b>{describeWindow(info)}</b>
          <br />
          <span className="muted">
            {info.nodes} real node{info.nodes === 1 ? "" : "s"}, the longest {info.longest} {units}
            {info.on ? `, ${info.longer} longer than the window` : ""}.{" "}
            {info.heavy !== null && info.heavy !== undefined
              ? `The bridge between two halves weighs ${info.heavy}.`
              : "The bridge between two halves carries the node's whole count."}
          </span>
        </p>
      )}
      <div className="row">
        <CheckField label="Window on" hint="off = compression is unbounded, no node is halved" checked={on} onChange={setOn} disabled={busy || jobRunning} />
        <SelectField label="Top" hint="where the ladder starts" value={top} onChange={setTop} options={SIZE_OPTIONS} disabled={busy || jobRunning || !on} />
        <SelectField label="Floor" hint="where it turns back up" value={floor} onChange={setFloor} options={SIZE_OPTIONS} disabled={busy || jobRunning || !on} />
        <SelectField
          label="Standing at"
          hint="the window's size now"
          value={chosen === null ? top : String(chosen)}
          onChange={setSize}
          options={sizes.length ? sizes.map((s) => [String(s), String(s)]) : SIZE_OPTIONS}
          disabled={busy || jobRunning || !on}
        />
        <CheckField label="Automatic" hint="step at the end of every training epoch" checked={auto} onChange={setAuto} disabled={busy || jobRunning || !on} />
      </div>
      {on ? <Ladder sizes={sizes} current={chosen} next={nextOfChosen} units={units} /> : null}
      <div className="actions">
        <button type="button" className="primary" disabled={busy || jobRunning || (on && !sizes.length)} onClick={apply}>
          {busy ? "Applying…" : on ? `Apply: window on, ${sizes.join(" → ") || "?"} at ${chosen ?? "?"}` : "Apply: window off"}
        </button>
        <button type="button" className="small" disabled={busy} onClick={load}>
          Reload from the model
        </button>
      </div>
      {jobRunning ? <p className="muted">A job is running; the window can be changed once it finishes.</p> : null}
      <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />

      <h3>Step by hand</h3>
      <p className="muted">
        One step at the size the window stands at: merge what fits, halve what is longer, move the window. The graph
        changes; the model is saved with the next save.
      </p>
      <div className="row">
        <NumberField label="Steps" hint="one rung per step; four walk the whole ladder" value={steps} onChange={setSteps} min={1} step={1} disabled={busy || jobRunning} />
      </div>
      <div className="actions">
        <button type="button" disabled={busy || jobRunning || !(info && info.on)} onClick={step}>
          {busy ? "Working…" : info && info.on ? `Step ×${parseInteger(steps, 1)} (at ${info.size} ${units})` : "Step (switch the window on first)"}
        </button>
      </div>
      {lastStep ? <p>{describeStep(lastStep)}</p> : null}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}
