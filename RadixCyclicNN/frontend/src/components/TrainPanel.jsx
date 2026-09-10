import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import LineChart from "./LineChart.jsx";
import UploadPicker from "./UploadPicker.jsx";
import { CheckField, NumberField, TextArea } from "./Fields.jsx";

const MAX_ROWS = 300;

/**
 * Start / stop a training job and show its epoch records live. Before a job
 * runs (or after it finishes) the model's stored history from /api/history is
 * shown instead, so previous training is visible after a reload.
 */
export default function TrainPanel({ status }) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState([]);
  const [wholeFile, setWholeFile] = useState(false);
  const [epochs, setEpochs] = useState("5");
  const [lr, setLr] = useState("0.05");
  const [actLr, setActLr] = useState("0.005");
  const [batchSize, setBatchSize] = useState("256");
  const [autoCompress, setAutoCompress] = useState(true);
  const [formError, setFormError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("train");

  const otherJobRunning = jobIsRunning(status) && !running;

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.history();
      setServerHistory(asArray(data && data.history));
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err.message);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Refresh the stored history once a job reaches a terminal state.
  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  async function handleStart(event) {
    event.preventDefault();
    const texts = splitLines(text);
    if (texts.length === 0 && files.length === 0) {
      setFormError("Enter at least one training text (one per line) or select uploaded files.");
      return;
    }
    setFormError(null);
    await start(() =>
      api.train({
        ...(texts.length > 0 ? { texts } : {}),
        ...(files.length > 0 ? { files, whole_file: wholeFile } : {}),
        epochs: parseInteger(epochs, 5),
        lr: parseNumber(lr, 0.05),
        act_lr: parseNumber(actLr, 0.005),
        batch_size: parseInteger(batchSize, 256),
        auto_compress: autoCompress,
      }),
    );
  }

  const jobHistory = asArray(job && job.history);
  const showingJob = running || jobHistory.length > 0;
  const history = showingJob ? jobHistory : serverHistory;
  const rows = history.slice(-MAX_ROWS);
  const lossSeries = [{ name: "loss", color: "#7c3aed", values: history.map((r, i) => ({ x: i + 1, y: r.loss })) }];

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Train</h2>
        <TextArea
          label="Training texts"
          hint="one per line"
          value={text}
          onChange={setText}
          rows={10}
          disabled={running}
          placeholder={"the quick brown fox jumps over the lazy dog\nhello world"}
        />
        <UploadPicker
          selected={files}
          onChange={setFiles}
          disabled={running}
          title="Training files"
          hint="Upload text files and tick the ones to train on (in addition to the texts above)."
        />
        <CheckField
          label="Treat each selected file as one text (instead of one text per line)"
          checked={wholeFile}
          onChange={setWholeFile}
          disabled={running || files.length === 0}
        />
        <div className="row">
          <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} step={1} disabled={running} />
          <NumberField
            label="Batch size"
            value={batchSize}
            onChange={setBatchSize}
            min={1}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <NumberField label="Learning rate" value={lr} onChange={setLr} min={0} disabled={running} />
          <NumberField label="Activation lr" value={actLr} onChange={setActLr} min={0} disabled={running} />
        </div>
        <CheckField
          label="Auto-compress after each epoch"
          checked={autoCompress}
          onChange={setAutoCompress}
          disabled={running}
        />
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : "Start training"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />
      </form>

      <div className="card">
        <div className="toolbar">
          <h2>Epochs</h2>
          <button type="button" className="small" onClick={loadHistory}>
            Reload history
          </button>
        </div>
        <JobStatus job={job} emptyText="No training job in this session. Enter texts and press Start." />
        <Alert message={historyError} onDismiss={() => setHistoryError(null)} />
        {!showingJob && history.length > 0 ? (
          <p className="muted">Showing the model's stored training history ({history.length} epochs).</p>
        ) : null}
        {history.length > 0 ? (
          <>
            <LineChart series={lossSeries} xLabel="epoch (record #)" yLabel="loss" height={200} />
            {history.length > MAX_ROWS ? (
              <p className="muted">
                Showing the last {MAX_ROWS} of {history.length} epochs.
              </p>
            ) : null}
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>epoch</th>
                    <th>loss</th>
                    <th>perplexity</th>
                    <th>nodes</th>
                    <th>edges</th>
                    <th>compression</th>
                    <th>merges</th>
                    <th>seconds</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td>{fmtInt(r.epoch)}</td>
                      <td>{fmtNum(r.loss, 4)}</td>
                      <td>{fmtNum(r.perplexity, 3)}</td>
                      <td>{fmtInt(r.nodes)}</td>
                      <td>{fmtInt(r.edges)}</td>
                      <td>{fmtNum(r.compression_ratio, 2)}</td>
                      <td>{fmtInt(r.merges)}</td>
                      <td>{fmtNum(r.seconds, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : running ? (
          <p className="muted">Waiting for the first epoch…</p>
        ) : null}
      </div>
    </>
  );
}
