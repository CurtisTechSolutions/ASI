import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { asArray, fmtBytes, fmtInt, fmtNum, fmtTime, jobIsRunning, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, TextField } from "./Fields.jsx";

/** Short summary of a checkpoint's metrics record. */
function metricsSummary(metrics) {
  if (!metrics || typeof metrics !== "object") return "–";
  if (typeof metrics.loss === "number") return `loss ${fmtNum(metrics.loss, 4)}`;
  if (typeof metrics.gap === "number") return `gap ${fmtNum(metrics.gap, 3)}`;
  const keys = Object.keys(metrics);
  return keys.length ? keys.slice(0, 3).join(", ") : "–";
}

/** Checkpoint list / save / restore plus model save, load, reset and compress. */
export default function CheckpointPanel({ status }) {
  const [checkpoints, setCheckpoints] = useState([]);
  const [latest, setLatest] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [pending, setPending] = useState(null);
  const [tag, setTag] = useState("manual");
  const [savePath, setSavePath] = useState("");
  const [loadPath, setLoadPath] = useState("");
  const [seed, setSeed] = useState("0");

  const busy = pending !== null;
  const locked = busy || jobIsRunning(status);
  const modelPath = status && status.model_path ? String(status.model_path) : "";
  const checkpointDir = status && status.checkpoint_dir ? String(status.checkpoint_dir) : "";

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.checkpoints();
      setCheckpoints(asArray(data && data.checkpoints));
      setLatest(data && data.latest && typeof data.latest === "object" ? data.latest : null);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function run(name, action, describe) {
    setPending(name);
    setError(null);
    setNotice(null);
    try {
      const data = await action();
      setNotice(describe(data && typeof data === "object" ? data : {}));
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setPending(null);
    }
  }

  const statsText = (d) =>
    `${fmtInt(d.nodes)} nodes, ${fmtInt(d.edges)} edges, compression ${fmtNum(d.compression_ratio, 2)}×`;

  function handleSaveCheckpoint(event) {
    event.preventDefault();
    const cleanTag = tag.trim() || "manual";
    run(
      "ckpt-save",
      () => api.checkpointSave({ tag: cleanTag }),
      (d) => `Saved checkpoint ${d.name ?? cleanTag} (${fmtBytes(d.bytes)}).`,
    );
  }

  function handleRestore(name) {
    run(
      `restore:${name}`,
      () => api.checkpointRestore({ name }),
      (d) => `Restored ${name}: ${statsText(d)}.`,
    );
  }

  function handleSaveModel(event) {
    event.preventDefault();
    const path = savePath.trim();
    run(
      "save",
      () => api.save(path ? { path } : {}),
      (d) => `Model saved to ${d.path || path || modelPath || "model"} (${fmtBytes(d.bytes)}).`,
    );
  }

  function handleLoadModel(event) {
    event.preventDefault();
    const path = loadPath.trim();
    if (!path) {
      setError("Enter the path of a model file to load.");
      return;
    }
    run(
      "load",
      () => api.load({ path }),
      (d) => `Loaded ${path}: ${statsText(d)}.`,
    );
  }

  function handleReset(event) {
    event.preventDefault();
    const seedValue = parseInteger(seed, 0);
    if (!window.confirm(`Replace the current model with a fresh one (seed ${seedValue})? Unsaved training is lost.`))
      return;
    run(
      "reset",
      () => api.reset({ seed: seedValue }),
      (d) => `Model reset with seed ${seedValue}: ${statsText(d)}.`,
    );
  }

  function handleCompress() {
    run(
      "compress",
      () => api.compress(),
      (d) => `Compressed: ${fmtInt(d.merges)} merges, now ${statsText(d)}.`,
    );
  }

  return (
    <>
      <div className="card">
        <h2>Model</h2>
        <form onSubmit={handleSaveModel}>
          <TextField
            label="Save model to"
            hint={modelPath ? `default ${modelPath}` : "server path"}
            value={savePath}
            onChange={setSavePath}
            placeholder={modelPath || "model.json"}
            disabled={locked}
          />
          <div className="actions">
            <button type="submit" className="primary" disabled={locked}>
              {pending === "save" ? "Saving…" : "Save model"}
            </button>
          </div>
        </form>
        <form onSubmit={handleLoadModel}>
          <TextField
            label="Load model from"
            hint="server path, .json or .json.gz"
            value={loadPath}
            onChange={setLoadPath}
            placeholder={modelPath || "model.json"}
            disabled={locked}
          />
          <div className="actions">
            <button type="submit" disabled={locked}>
              {pending === "load" ? "Loading…" : "Load model"}
            </button>
          </div>
        </form>
        <form onSubmit={handleReset}>
          <NumberField label="Reset with seed" value={seed} onChange={setSeed} step={1} disabled={locked} />
          <div className="actions">
            <button type="submit" className="danger" disabled={locked}>
              {pending === "reset" ? "Resetting…" : "Reset model"}
            </button>
            <button type="button" onClick={handleCompress} disabled={locked}>
              {pending === "compress" ? "Compressing…" : "Compress graph now"}
            </button>
          </div>
        </form>
        {jobIsRunning(status) ? <p className="muted">Model actions are locked while a job is running.</p> : null}
        <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <div className="card">
        <div className="toolbar">
          <h2>Checkpoints</h2>
          <form className="inline-form" onSubmit={handleSaveCheckpoint}>
            <TextField label="Tag" value={tag} onChange={setTag} placeholder="manual" disabled={locked} />
            <button type="submit" className="primary" disabled={locked}>
              {pending === "ckpt-save" ? "Saving…" : "Save checkpoint"}
            </button>
            <button type="button" onClick={refresh} disabled={loading}>
              {loading ? "Loading…" : "Refresh"}
            </button>
          </form>
        </div>
        {checkpointDir ? <p className="muted">Directory: {checkpointDir}</p> : null}
        {checkpoints.length === 0 ? (
          <p className="muted">{loading ? "Loading…" : "No checkpoints yet."}</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>name</th>
                  <th>step</th>
                  <th>tag</th>
                  <th>saved</th>
                  <th>size</th>
                  <th>metrics</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {checkpoints.map((c, i) => {
                  const name = c && c.name !== undefined ? String(c.name) : "";
                  const isLatest = latest && latest.name !== undefined && latest.name === (c && c.name);
                  return (
                    <tr key={name || i}>
                      <td className="text">
                        {name || "–"}
                        {isLatest ? (
                          <span className="badge done" style={{ marginLeft: 6 }}>
                            latest
                          </span>
                        ) : null}
                      </td>
                      <td>{fmtInt(c && c.step)}</td>
                      <td>{c && c.tag ? String(c.tag) : "–"}</td>
                      <td>{fmtTime(c && c.saved_at)}</td>
                      <td>{fmtBytes(c && c.bytes)}</td>
                      <td>{metricsSummary(c && c.metrics)}</td>
                      <td>
                        <button
                          type="button"
                          className="small"
                          disabled={locked || !name}
                          onClick={() => handleRestore(name)}
                        >
                          {pending === `restore:${name}` ? "Restoring…" : "Restore"}
                        </button>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
