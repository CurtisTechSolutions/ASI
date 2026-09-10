import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { asArray, fmtBytes, fmtInt, fmtTime } from "../util.js";
import Alert from "./Alert.jsx";

const ACCEPT = ".txt,.md,.csv,.tsv,.json,.jsonl,.log,.text,.py,.zip,text/*,application/zip";

/** ZIP archives go up as bytes and are unpacked on the server; everything else is read as text. */
function isArchive(file) {
  const type = String(file.type || "").toLowerCase();
  return /\.zip$/i.test(file.name) || type === "application/zip" || type === "application/x-zip-compressed";
}

function describeArchive(archive) {
  const skipped = Array.isArray(archive.skipped) ? archive.skipped : [];
  const detail = skipped
    .slice(0, 3)
    .map((s) => `${s.path} (${s.reason})`)
    .join(", ");
  const more = skipped.length > 3 ? `, … ${skipped.length - 3} more` : "";
  const files = `${archive.extracted} text file${archive.extracted === 1 ? "" : "s"}`;
  return `Unpacked ${archive.name}: ${files}${skipped.length ? `, ${skipped.length} skipped (${detail}${more})` : ""}.`;
}

/**
 * Text files kept in the server's upload directory (GET /api/uploads), with a
 * checkbox per file so a panel can train on a selection of them.
 *
 * Files are uploaded from the file input or by dropping them onto the box:
 * the browser reads each text file and POSTs {name, content} to /api/uploads;
 * a ZIP archive is sent as multipart bytes and unpacked on the server, every
 * text file inside becoming its own upload (named <zip>__<dir>__<file>).
 *
 * Props: selected (array of upload names), onChange(names), disabled, title, hint.
 */
export default function UploadPicker({ selected, onChange, disabled = false, title = "Uploaded files", hint }) {
  const [uploads, setUploads] = useState([]);
  const [uploadDir, setUploadDir] = useState(null);
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(null);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);
  const chosen = asArray(selected);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await api.uploads();
      setUploads(asArray(data && data.uploads));
      setUploadDir(data && data.upload_dir ? String(data.upload_dir) : null);
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

  // Forget selections whose file has disappeared (deleted here or elsewhere).
  useEffect(() => {
    const names = new Set(uploads.map((u) => u.name));
    const kept = chosen.filter((name) => names.has(name));
    if (kept.length !== chosen.length) onChange(kept);
  }, [uploads]); // eslint-disable-line react-hooks/exhaustive-deps

  function toggle(name, checked) {
    const next = chosen.filter((n) => n !== name);
    if (checked) next.push(name);
    onChange(next);
  }

  async function uploadFiles(fileList) {
    const files = Array.from(fileList || []);
    if (files.length === 0 || disabled) return;
    setError(null);
    setNotice(null);
    const added = [];
    const failures = [];
    const notes = [];
    for (let i = 0; i < files.length; i += 1) {
      const file = files[i];
      setUploading(`${i + 1} / ${files.length}: ${file.name}`);
      try {
        const data = isArchive(file) ? await api.uploadFile(file) : await api.upload(file.name, await file.text());
        for (const record of asArray(data && data.uploads)) if (record && record.name) added.push(record.name);
        for (const archive of asArray(data && data.archives)) if (archive && archive.name) notes.push(describeArchive(archive));
      } catch (err) {
        failures.push(`${file.name}: ${err.message}`);
      }
    }
    setUploading(null);
    await refresh();
    if (added.length > 0) onChange(Array.from(new Set([...chosen, ...added])));
    if (failures.length > 0) setError(failures.join("\n"));
    const summary = `Uploaded ${added.length} file${added.length === 1 ? "" : "s"}.`;
    setNotice(failures.length > 0 && notes.length === 0 ? null : [summary, ...notes].join(" "));
  }

  async function remove(name) {
    setError(null);
    setNotice(null);
    try {
      await api.deleteUpload(name);
      await refresh();
    } catch (err) {
      setError(err.message);
    }
  }

  function onDrop(event) {
    event.preventDefault();
    setDragging(false);
    uploadFiles(event.dataTransfer && event.dataTransfer.files);
  }

  const allSelected = uploads.length > 0 && uploads.every((u) => chosen.includes(u.name));
  const busy = disabled || uploading !== null;

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
          <button
            type="button"
            className="small"
            disabled={busy}
            onClick={() => inputRef.current && inputRef.current.click()}
          >
            Upload files…
          </button>
          <button type="button" className="small" disabled={loading} onClick={refresh}>
            Refresh
          </button>
          {uploads.length > 0 ? (
            <button
              type="button"
              className="link"
              disabled={disabled}
              onClick={() => onChange(allSelected ? [] : uploads.map((u) => u.name))}
            >
              {allSelected ? "Select none" : "Select all"}
            </button>
          ) : null}
        </div>
      </div>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          uploadFiles(e.target.files);
          e.target.value = "";
        }}
      />
      <p className="muted">
        {hint || "Drop text files or ZIP archives here or press Upload. Each non-blank line is one training text."}
        {" ZIP archives are unpacked on the server: every text file inside becomes an upload."}
        {uploadDir ? ` Stored on the server in ${uploadDir}.` : ""}
      </p>
      {uploading ? <p className="muted">Uploading {uploading}…</p> : null}
      {notice ? <p className="muted">{notice}</p> : null}
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
                <th>lines</th>
                <th>size</th>
                <th>uploaded</th>
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
                  <td>{fmtInt(u.lines)}</td>
                  <td>{fmtBytes(u.bytes)}</td>
                  <td>{fmtTime(u.modified)}</td>
                  <td>
                    <button type="button" className="link" disabled={busy} onClick={() => remove(u.name)}>
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
