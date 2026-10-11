import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { progressText, uploadInChunks, uploadName } from "../upload.js";
import { asArray, cleanName, fmtAgo, fmtBytes, fmtTime } from "../util.js";
import Alert from "./Alert.jsx";
import { SelectField, TextField } from "./Fields.jsx";

/** The four models the server holds, and how each is saved to and loaded from a path on the server. */
const MODELS = [
  { id: "chain", label: "the chain", path: "chain.json.gz", save: (p) => api.lmSave("chain", p), load: (p) => api.lmLoad("chain", p) },
  { id: "ngram", label: "the N-gram model", path: "ngram.json.gz", save: (p) => api.lmSave("ngram", p), load: (p) => api.lmLoad("ngram", p) },
  { id: "machine", label: "the machine", path: "machine.json.gz", save: api.save, load: api.load },
  { id: "network", label: "the network", path: "network.json.gz", save: api.networkSave, load: api.networkLoad },
];

/** One model's row: its path on the server, Save and Load. */
function ModelRow({ model, busy, onAct }) {
  // its own setting: the Machine and Network tabs keep their own paths, and two fields must not share one
  const [path, setPath] = useStoredState(`checkpoints.${model.id}Path`, model.path);
  return (
    <tr>
      <td>{model.label}</td>
      <td className="text">
        <input className="path-input" value={path} aria-label={`${model.label}'s file`} onChange={(e) => setPath(e.target.value)} />
      </td>
      <td>
        <button
          type="button"
          className="small"
          disabled={busy || !path.trim()}
          onClick={() => onAct(`saved ${model.label}`, async () => `to ${(await model.save(path.trim())).saved ?? path}`)}
        >
          Save
        </button>{" "}
        <button
          type="button"
          className="small"
          disabled={busy || !path.trim()}
          onClick={() =>
            onAct(`loaded ${model.label}`, async () => {
              await model.load(path.trim());
              return `from ${path.trim()}`;
            })
          }
        >
          Load
        </button>
      </td>
    </tr>
  );
}

/**
 * Saving and restoring. The models card saves and loads each of the server's models - the chain, the N-gram
 * model, the machine, the network - to a path on the server. The checkpoints card keeps named snapshots in
 * the server's checkpoint directory (GET /api/checkpoints): save one of any model with a tag, restore one
 * (the model of its kind is replaced), download it as a file, delete it (two clicks), or upload a checkpoint
 * file from this computer (in pieces of at most 8 MiB, like any upload).
 */
export default function CheckpointPanel({ lm, onMoved }) {
  const [list, setList] = useState([]);
  const [dir, setDir] = useState(null);
  const [latest, setLatest] = useState(null);
  const [model, setModel] = useStoredState("checkpoints.model", lm || "chain");
  const [tag, setTag] = useStoredState("checkpoints.tag", "manual");
  const [armed, setArmed] = useState(null); // the checkpoint a first Delete click armed
  const [progress, setProgress] = useState(null);
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.checkpoints();
      setList(asArray(data && data.checkpoints));
      setDir(data && data.dir ? String(data.dir) : null);
      setLatest(data && data.latest ? (typeof data.latest === "object" ? data.latest.name : String(data.latest)) : null);
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

  async function act(what, fn) {
    setBusy(true);
    setNotice(null);
    try {
      const detail = await fn();
      setNotice(`${what}${detail ? ` ${detail}` : ""}`);
      setError(null);
      if (onMoved) onMoved();
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function uploadFile(file) {
    if (!file) return;
    const name = cleanName(uploadName(file));
    setProgress({ name, sent: 0, total: file.size });
    await act(`uploaded ${name}`, async () => {
      const r = await uploadInChunks(file, (b64, append) => api.checkpointUpload(name, b64, append), {
        onProgress: (sent, total) => setProgress({ name, sent, total }),
      });
      return r && r.name && r.name !== name ? `as ${r.name}` : "";
    });
    setProgress(null);
  }

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Models</h2>
        <p className="note">Each model the server holds, saved to or loaded from a file on the server (.json.gz).</p>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>model</th>
                <th>file on the server</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => (
                <ModelRow key={m.id} model={m} busy={busy} onAct={act} />
              ))}
            </tbody>
          </table>
        </div>
        <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <div className="card wide">
        <div className="toolbar">
          <h2>Checkpoints</h2>
          <div className="inline-form">
            <SelectField label="Model" value={model} onChange={setModel} options={MODELS.map((m) => [m.id, m.id])} />
            <TextField label="Tag" value={tag} onChange={setTag} placeholder="manual" />
            <button
              type="button"
              className="primary"
              disabled={busy}
              onClick={() => act("saved the checkpoint", async () => (await api.checkpointSave(model, tag.trim() || "manual")).name)}
            >
              Save checkpoint
            </button>
            <button type="button" disabled={busy} onClick={() => fileRef.current && fileRef.current.click()}>
              Upload…
            </button>
            <button type="button" disabled={loading} onClick={refresh}>
              {loading ? "Loading…" : "Refresh"}
            </button>
          </div>
        </div>
        <input
          ref={fileRef}
          type="file"
          hidden
          onChange={(e) => {
            uploadFile(e.target.files && e.target.files[0]);
            e.target.value = "";
          }}
        />
        {dir ? <p className="muted">Kept on the server in {dir}.</p> : null}
        {progress ? (
          <p className="muted">
            uploading {progress.name}: {progressText(progress.sent, progress.total)} of {fmtBytes(progress.total)}
          </p>
        ) : null}
        {list.length === 0 ? (
          <p className="muted">{loading ? "Loading…" : "No checkpoints yet."}</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>name</th>
                  <th>model</th>
                  <th>tag</th>
                  <th>saved</th>
                  <th>size</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {list.map((c) => (
                  <tr key={c.name}>
                    <td className="text">
                      {c.name}
                      {latest && latest === c.name ? <span className="badge done tight">latest</span> : null}
                    </td>
                    <td>{c.model || "–"}</td>
                    <td>{c.tag || "–"}</td>
                    <td title={fmtTime(c.saved_at)}>{fmtAgo(c.saved_at)}</td>
                    <td>{fmtBytes(c.bytes)}</td>
                    <td className="row-actions">
                      <button type="button" className="small" disabled={busy} onClick={() => act(`restored ${c.name}`, async () => {
                        const r = await api.checkpointRestore(c.name);
                        return r && r.model ? `(${r.model})` : "";
                      })}>
                        Restore
                      </button>{" "}
                      <a className="button small" href={api.checkpointDownloadUrl(c.name)} download={c.name}>
                        Download
                      </a>{" "}
                      <button
                        type="button"
                        className={`small${armed === c.name ? " danger" : ""}`}
                        disabled={busy}
                        onBlur={() => setArmed(null)}
                        onClick={() => {
                          if (armed !== c.name) {
                            setArmed(c.name);
                            return;
                          }
                          setArmed(null);
                          act(`deleted ${c.name}`, async () => {
                            await api.checkpointDelete(c.name);
                            return "";
                          });
                        }}
                      >
                        {armed === c.name ? "Delete? again" : "Delete"}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
