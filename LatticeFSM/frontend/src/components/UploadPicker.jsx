import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { progressText, uploadInChunks, uploadName } from "../upload.js";
import { asArray, fmtBytes, fmtTime } from "../util.js";
import Alert from "./Alert.jsx";

/** Every file under a dropped folder entry, each given its path inside the drop as `relativePath`. */
async function filesOfEntry(entry) {
  if (!entry) return [];
  if (entry.isFile) {
    const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
    try {
      Object.defineProperty(file, "relativePath", { value: String(entry.fullPath || file.name).replace(/^\/+/, "") });
    } catch {
      // a browser that will not let it be set uploads the file under its bare name
    }
    return [file];
  }
  if (!entry.isDirectory) return [];
  const reader = entry.createReader();
  const children = [];
  // a directory reader hands its entries over in batches until it hands over none
  for (;;) {
    const batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject));
    if (!batch.length) break;
    children.push(...batch);
  }
  const nested = await Promise.all(children.map(filesOfEntry));
  return nested.flat();
}

/** The files of a drop: folders opened (where the browser lets them be), plain files as they are. */
async function droppedFiles(dataTransfer) {
  const items = Array.from((dataTransfer && dataTransfer.items) || []);
  const entries = items.map((item) => (item.webkitGetAsEntry ? item.webkitGetAsEntry() : null));
  if (entries.length && entries.every(Boolean)) {
    const nested = await Promise.all(entries.map(filesOfEntry));
    return nested.flat();
  }
  return Array.from((dataTransfer && dataTransfer.files) || []);
}

/**
 * Files kept in the server's upload directory (GET /api/uploads), with a checkbox per file so a panel can
 * train on a selection of them.
 *
 * Any kind of file goes: text, a ZIP or tar archive, gzip, HTML, Word documents, images - the server opens
 * archives and documents when they are used. Pick files, pick a whole folder (every file is kept under its
 * path inside it), or drop either onto the box. Files go up one at a time, each cut into pieces of at most
 * 8 MiB (`upload.js`), with how far each has got; what was uploaded is selected at once.
 *
 * Props: selected (upload names), onChange(names), disabled, title, hint.
 */
export default function UploadPicker({ selected, onChange, disabled = false, title = "Uploaded files", hint }) {
  const [uploads, setUploads] = useState([]);
  const [dir, setDir] = useState(null);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState([]); // [{name, sent, total, done, error}]
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const filesRef = useRef(null);
  const folderRef = useRef(null);
  const loaded = useRef(false); // the list has come back at least once
  const chosen = asArray(selected);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.uploads();
      setUploads(asArray(data && data.uploads));
      loaded.current = true;
      setDir(data && data.dir ? String(data.dir) : null);
      setError(null);
    } catch (err) {
      setUploads([]);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // forget selections whose file has gone (deleted here or elsewhere)
  useEffect(() => {
    if (!loaded.current) return;
    const names = new Set(uploads.map((u) => u.name));
    const kept = chosen.filter((name) => names.has(name));
    if (kept.length !== chosen.length) onChange(kept);
  }, [uploads]); // eslint-disable-line react-hooks/exhaustive-deps

  function toggle(name, checked) {
    const next = chosen.filter((n) => n !== name);
    if (checked) next.push(name);
    onChange(next);
  }

  async function uploadFiles(list) {
    const files = Array.from(list || []);
    if (files.length === 0 || disabled || uploading) return;
    setError(null);
    setUploading(true);
    const rows = files.map((f) => ({ name: uploadName(f), sent: 0, total: f.size, done: false, error: null }));
    setProgress(rows);
    const update = (i, patch) => setProgress((prev) => prev.map((row, j) => (j === i ? { ...row, ...patch } : row)));
    const added = [];
    for (let i = 0; i < files.length; i += 1) {
      const name = rows[i].name;
      try {
        const r = await uploadInChunks(files[i], (b64, append) => api.upload(name, b64, append), {
          onProgress: (sent, total) => update(i, { sent, total }),
        });
        added.push(r && r.name ? String(r.name) : name);
        update(i, { done: true });
      } catch (err) {
        update(i, { error: err.message });
      }
    }
    setUploading(false);
    await refresh();
    if (added.length) onChange(Array.from(new Set([...chosen, ...added])));
  }

  async function remove(name) {
    setError(null);
    try {
      await api.uploadDelete(name);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  async function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    if (disabled || uploading) return;
    try {
      uploadFiles(await droppedFiles(event.dataTransfer));
    } catch (err) {
      setError(err.message);
    }
  }

  const allSelected = uploads.length > 0 && uploads.every((u) => chosen.includes(u.name));
  const busy = disabled || uploading;
  const finished = progress.filter((p) => p.done).length;

  return (
    <div
      className={`uploads${dragging ? " dragging" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        if (!busy) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >
      <div className="toolbar">
        <h3>{title}</h3>
        <div className="chips">
          <button type="button" className="small" disabled={busy} onClick={() => filesRef.current && filesRef.current.click()}>
            Files…
          </button>
          <button type="button" className="small" disabled={busy} onClick={() => folderRef.current && folderRef.current.click()}>
            A folder…
          </button>
          <button type="button" className="small" disabled={loading} onClick={refresh}>
            Refresh
          </button>
          {uploads.length > 0 ? (
            <button
              type="button"
              className="small"
              disabled={disabled}
              onClick={() => onChange(allSelected ? [] : uploads.map((u) => u.name))}
            >
              {allSelected ? "Select none" : "Select all"}
            </button>
          ) : null}
        </div>
      </div>
      <input
        ref={filesRef}
        type="file"
        multiple
        hidden
        onChange={(e) => {
          uploadFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <input
        ref={folderRef}
        type="file"
        multiple
        hidden
        webkitdirectory=""
        directory=""
        onChange={(e) => {
          uploadFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <p className="muted">
        {hint || "Drop files or folders here, or pick them. Any kind: text, zip, tar, gz, html, docx, images - the server opens archives and documents."}
        {dir ? ` Kept on the server in ${dir}.` : ""}
      </p>
      {progress.length ? (
        <ul className="upload-progress">
          {progress.map((p, i) => (
            <li key={`${p.name}-${i}`} className={p.error ? "error-text" : p.done ? "muted" : ""}>
              <span className="mono">{p.name}</span> · {fmtBytes(p.total)} ·{" "}
              {p.error ? p.error : p.done ? "uploaded" : progressText(p.sent, p.total)}
              {!p.done && !p.error ? (
                <span className="rating-bar">
                  <i style={{ width: progressText(p.sent, p.total) }} />
                </span>
              ) : null}
            </li>
          ))}
          {!uploading ? (
            <li className="muted">
              {finished} of {progress.length} uploaded{" "}
              <button type="button" className="link" onClick={() => setProgress([])}>
                ×
              </button>
            </li>
          ) : null}
        </ul>
      ) : null}
      <Alert message={error} onDismiss={() => setError(null)} />
      {uploads.length === 0 ? (
        <p className="muted">{loading ? "Loading…" : "No uploaded files yet."}</p>
      ) : (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>use</th>
                <th>file</th>
                <th>kind</th>
                <th>size</th>
                <th>modified</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {uploads.map((u) => (
                <tr key={u.name}>
                  <td>
                    <input
                      type="checkbox"
                      checked={chosen.includes(u.name)}
                      disabled={disabled}
                      onChange={(e) => toggle(u.name, e.target.checked)}
                      aria-label={`use ${u.name}`}
                    />
                  </td>
                  <td className="text">{u.name}</td>
                  <td>{u.kind || "–"}</td>
                  <td>{fmtBytes(u.bytes)}</td>
                  <td>{fmtTime(u.modified)}</td>
                  <td>
                    <button type="button" className="small" disabled={busy} onClick={() => remove(u.name)}>
                      delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
