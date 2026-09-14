import { useState } from "react";
import { api } from "../api.js";
import { asArray, jobIsRunning } from "../util.js";

const ORIGIN_TEXT = {
  memory: "restored from memory (unsaved work kept)",
  file: "loaded from its file",
  new: "a fresh model",
  active: "already active",
};

/**
 * The model-kind selector at the top of the page: RadixNet (sine activation,
 * learned weights) or the count / reward model (edge weight = log(1 +
 * traversals) + rewards, top-K / bottom-K prediction). Switching keeps the
 * previous model in memory on the server; every tab then talks to the
 * selected one.
 */
export default function ModelSelector({ status, onStatus }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);

  const kinds = asArray(status && status.kinds).filter((k) => k && typeof k.kind === "string");
  const kind = status && typeof status.kind === "string" ? status.kind : "";
  const current = kinds.find((k) => k.kind === kind);
  const running = jobIsRunning(status);

  async function choose(next) {
    if (!next || next === kind) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const data = await api.selectModel(next);
      if (data && typeof data === "object") {
        const stats = data.stats && typeof data.stats === "object" ? data.stats : {};
        if (onStatus && status) onStatus({ ...status, ...stats, kind: data.kind, model_label: data.label, model_path: data.model_path });
        setNotice(`${data.label || data.kind}: ${ORIGIN_TEXT[data.origin] || data.origin || "selected"}`);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="model-selector" aria-live="polite">
      <label>
        <span>Model</span>
        <select
          value={kind}
          disabled={!status || kinds.length === 0 || running || busy}
          onChange={(e) => choose(e.target.value)}
          title={running ? "Wait for the running job before switching models" : "Select the model / algorithm"}
        >
          {kinds.length === 0 ? <option value="">{status ? "–" : "offline"}</option> : null}
          {kinds.map((k) => (
            <option key={k.kind} value={k.kind}>
              {k.label || k.kind}
            </option>
          ))}
        </select>
      </label>
      {current && current.description ? <span className="muted description">{current.description}</span> : null}
      {notice ? <span className="muted notice">{notice}</span> : null}
      {error ? <span className="err">{error}</span> : null}
    </div>
  );
}
