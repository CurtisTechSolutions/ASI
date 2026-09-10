import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import UploadPicker from "./UploadPicker.jsx";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import LineChart from "./LineChart.jsx";
import { NumberField, TextArea } from "./Fields.jsx";

const MAX_ROWS = 200;

/** GAN-style self-upgrade loop: start / stop, live chart of gap and fake score, latest sample. */
export default function EvolvePanel({ status }) {
  const [corpus, setCorpus] = useState("");
  const [corpusFiles, setCorpusFiles] = useState([]);
  const [samples, setSamples] = useState("8");
  const [realPerGeneration, setRealPerGeneration] = useState("8");
  const [maxLength, setMaxLength] = useState("40");
  const [temperature, setTemperature] = useState("1.0");
  const [generations, setGenerations] = useState("");
  const [checkpointEvery, setCheckpointEvery] = useState("0");
  const [formError, setFormError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("evolve");

  const otherJobRunning = jobIsRunning(status) && !running;

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.evolveHistory();
      setServerHistory(asArray(data && data.history));
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err.message);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // Refresh the server-side history once a job reaches a terminal state.
  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  async function handleStart(event) {
    event.preventDefault();
    const texts = splitLines(corpus);
    if (texts.length === 0 && corpusFiles.length === 0) {
      setFormError("Enter a corpus of real texts (one per line) or select uploaded files.");
      return;
    }
    setFormError(null);
    await start(() =>
      api.evolveStart({
        ...(texts.length > 0 ? { corpus: texts } : {}),
        ...(corpusFiles.length > 0 ? { corpus_files: corpusFiles } : {}),
        generations: generations.trim() === "" ? null : parseInteger(generations, 0),
        samples: parseInteger(samples, 8),
        real_per_generation: parseInteger(realPerGeneration, 8),
        max_length: parseInteger(maxLength, 40),
        temperature: parseNumber(temperature, 1),
        checkpoint_every: parseInteger(checkpointEvery, 0),
      }),
    );
  }

  const jobHistory = asArray(job && job.history);
  const history = running || jobHistory.length > 0 ? jobHistory : serverHistory;
  const rows = history.slice(-MAX_ROWS);
  const latest = history.length ? history[history.length - 1] : null;
  // Plot by record index: the stored history can span several runs whose generation counters restart.
  const xOf = (r, i) => i + 1;
  const series = [
    { name: "gap (real − fake)", color: "#2563eb", values: history.map((r, i) => ({ x: xOf(r, i), y: r.gap })) },
    {
      name: "fake_score_mean",
      color: "#dc2626",
      values: history.map((r, i) => ({ x: xOf(r, i), y: r.fake_score_mean })),
    },
    {
      name: "real_score_mean",
      color: "#16a34a",
      dashed: true,
      values: history.map((r, i) => ({ x: xOf(r, i), y: r.real_score_mean })),
    },
  ];

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Evolve</h2>
        <p className="muted">
          Generator = the model, discriminator = a second network. Each generation samples fakes, teaches the
          discriminator real-vs-fake via 2NRL, and feeds the worst fakes back to the generator as 2NRL garbage.
        </p>
        <TextArea
          label="Real corpus"
          hint="one text per line"
          value={corpus}
          onChange={setCorpus}
          rows={8}
          disabled={running}
          placeholder={"the quick brown fox jumps over the lazy dog\nhello world"}
        />
        <UploadPicker
          selected={corpusFiles}
          onChange={setCorpusFiles}
          disabled={running}
          title="Corpus files"
          hint="Uploaded files used as the real corpus (in addition to the texts above)."
        />
        <div className="row">
          <NumberField
            label="Generations"
            hint="blank = forever"
            value={generations}
            onChange={setGenerations}
            min={0}
            step={1}
            disabled={running}
            placeholder="∞"
          />
          <NumberField
            label="Checkpoint every"
            hint="generations, 0 = off"
            value={checkpointEvery}
            onChange={setCheckpointEvery}
            min={0}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <NumberField
            label="Fakes per generation"
            value={samples}
            onChange={setSamples}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Real per generation"
            value={realPerGeneration}
            onChange={setRealPerGeneration}
            min={1}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <NumberField
            label="Max length"
            value={maxLength}
            onChange={setMaxLength}
            min={3}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Temperature"
            value={temperature}
            onChange={setTemperature}
            min={0.01}
            disabled={running}
          />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : "Start evolving"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop(api.evolveStop)}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />
      </form>

      <div className="card">
        <div className="toolbar">
          <h2>Generations</h2>
          <button type="button" className="small" onClick={loadHistory}>
            Reload history
          </button>
        </div>
        <JobStatus job={job} emptyText="No evolve job in this session." />
        <Alert message={historyError} onDismiss={() => setHistoryError(null)} />
        <LineChart
          series={series}
          xLabel="generation (record #)"
          yLabel="per-char log-prob"
          emptyText={running ? "Waiting for the first generation…" : "No generations yet."}
        />
        {latest ? (
          <>
            <h3>Latest sample (generation {fmtInt(latest.generation)})</h3>
            <pre className="sample">{String(latest.sample ?? "")}</pre>
          </>
        ) : null}
        {rows.length > 0 ? (
          <>
            {history.length > MAX_ROWS ? (
              <p className="muted">
                Showing the last {MAX_ROWS} of {history.length} generations.
              </p>
            ) : null}
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>gen</th>
                    <th>fake</th>
                    <th>real</th>
                    <th>gap</th>
                    <th>gen loss</th>
                    <th>nodes</th>
                    <th>edges</th>
                    <th>compression</th>
                    <th>seconds</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i}>
                      <td>{fmtInt(r.generation)}</td>
                      <td>{fmtNum(r.fake_score_mean, 3)}</td>
                      <td>{fmtNum(r.real_score_mean, 3)}</td>
                      <td>{fmtNum(r.gap, 3)}</td>
                      <td>{fmtNum(r.gen_loss, 4)}</td>
                      <td>{fmtInt(r.nodes)}</td>
                      <td>{fmtInt(r.edges)}</td>
                      <td>{fmtNum(r.compression_ratio, 2)}</td>
                      <td>{fmtNum(r.seconds, 2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
    </>
  );
}
