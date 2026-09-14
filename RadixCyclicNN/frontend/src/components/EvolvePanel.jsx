import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import UploadPicker from "./UploadPicker.jsx";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import LineChart from "./LineChart.jsx";
import { NumberField, SelectField, TextArea } from "./Fields.jsx";

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
  const [blatantMode, setBlatantMode] = useState("none");
  const [blatantMargin, setBlatantMargin] = useState("1.0");
  const [blatantBoost, setBlatantBoost] = useState("4");
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
        blatant_mode: blatantMode,
        blatant_margin: parseNumber(blatantMargin, 1),
        blatant_boost: parseNumber(blatantBoost, 4),
      }),
    );
  }

  const countKind = Boolean(status && status.kind === "count");

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
          discriminator real-vs-fake via 2NRL, and feeds the failed fakes back to the generator: as 2NRL garbage,
          weighted by how badly they failed before the model is inverted, or as local path inversions (see below).
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
        <fieldset className="schedule">
          <legend>Failures: train on them, blatantly fail on purpose, then invert</legend>
          <div className="row">
            <SelectField
              label="Failure handling"
              value={blatantMode}
              onChange={setBlatantMode}
              disabled={running}
              options={[
                ["none", "off: the worst half is uniform 2NRL garbage"],
                ["fail_invert", countKind ? "penalise every failure, the worse the more (count model)" : "train on every failure, the worse the more, then invert the model"],
                ["activation", countKind ? "penalise failed paths locally (count model)" : "local: invert the activation function (a) along failed paths"],
                ["state", countKind ? "penalise failed paths locally (count model)" : "local: invert the trained node value (z) along failed paths"],
              ]}
            />
            <NumberField
              label="Margin"
              hint="per-char log-prob below the real texts = blatant"
              value={blatantMargin}
              onChange={setBlatantMargin}
              min={0}
              disabled={running || blatantMode === "none"}
            />
            <NumberField
              label="Max boost"
              hint="largest learning-rate multiplier"
              value={blatantBoost}
              onChange={setBlatantBoost}
              min={1}
              disabled={running || blatantMode !== "fail_invert"}
            />
          </div>
          <p className="muted">
            A failure is a fake the discriminator scores below the real texts; how far below (g, per-char log-prob)
            is how bad it is. <b>Train then invert</b>: the model is trained on every failure with its learning
            rates (weights, states and the activation functions) multiplied by 1 + g / margin (capped at the max
            boost) - the worse the response, the more the activation function updates - so it blatantly fails on
            purpose; then the whole model is inverted, turning what it now does confidently into what it confidently
            avoids, and fine-tuned on the real texts. <b>Local</b>: no negative pass; every other node on a failed
            path has its activation amplitude (or trained value) moved toward its negation by g / (2 · margin): a
            slight attenuation for a slightly worse fake, a neutralised path at the margin, a full inversion at twice
            it. Blatant fakes (beyond the margin) leave the 2NRL garbage set, so a generation whose bad fakes were
            all blatant skips the negative pass and the global inversion.
          </p>
        </fieldset>
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
                    <th>failures</th>
                    <th>blatant</th>
                    <th>boost</th>
                    <th>2NRL</th>
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
                      <td>{fmtInt(r.failures)}</td>
                      <td>{fmtInt(r.blatant)}</td>
                      <td title={r.flipped ? `${fmtInt(r.flipped)} nodes / edges changed` : undefined}>{fmtNum(r.boost_mean, 2)}</td>
                      <td>{r.twonrl === undefined ? "–" : r.twonrl ? "yes" : "skipped"}</td>
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
