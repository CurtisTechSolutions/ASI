import { useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines, yesNo } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { NumberField, TextArea } from "./Fields.jsx";

function PhaseTable({ title, rows }) {
  return (
    <>
      <h3>{title}</h3>
      {rows.length === 0 ? (
        <p className="muted">No records yet.</p>
      ) : (
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
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

/**
 * 2NRL: train on bad data, invert the network, fine-tune on good data.
 * Also exposes a manual Invert button.
 */
export default function TwoNRLPanel({ status }) {
  const [bad, setBad] = useState("");
  const [good, setGood] = useState("");
  const [negEpochs, setNegEpochs] = useState("3");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.05");
  const [posLr, setPosLr] = useState("0.01");
  const [formError, setFormError] = useState(null);
  const [inverting, setInverting] = useState(false);
  const [invertError, setInvertError] = useState(null);
  const [invertNotice, setInvertNotice] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("2nrl");

  const anyJobRunning = jobIsRunning(status) || running;
  const otherJobRunning = jobIsRunning(status) && !running;

  async function handleStart(event) {
    event.preventDefault();
    const badTexts = splitLines(bad);
    const goodTexts = splitLines(good);
    if (badTexts.length === 0 || goodTexts.length === 0) {
      setFormError("Provide at least one bad text and one good text (one per line).");
      return;
    }
    setFormError(null);
    await start(() =>
      api.twoNrl({
        bad: badTexts,
        good: goodTexts,
        neg_epochs: parseInteger(negEpochs, 3),
        pos_epochs: parseInteger(posEpochs, 3),
        neg_lr: parseNumber(negLr, 0.05),
        pos_lr: parseNumber(posLr, 0.01),
      }),
    );
  }

  async function handleInvert() {
    setInverting(true);
    setInvertError(null);
    setInvertNotice(null);
    try {
      const stats = await api.invert();
      setInvertNotice(`Network inverted. inverted = ${yesNo(stats && stats.inverted)}`);
    } catch (err) {
      setInvertError(err.message);
    } finally {
      setInverting(false);
    }
  }

  const history = asArray(job && job.history);
  const negative = history.filter((r) => r && r.phase === "negative");
  const positive = history.filter((r) => r && r.phase === "positive");
  const lastLoss = (rows) => (rows.length ? rows[rows.length - 1].loss : undefined);

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>2NRL</h2>
        <p className="muted">
          Phase 1 trains on the bad data, phase 2 inverts every weight and activation, phase 3 fine-tunes on the good
          data with a smaller learning rate.
        </p>
        <TextArea
          label="Bad / garbage texts"
          hint="one per line"
          value={bad}
          onChange={setBad}
          rows={6}
          disabled={running}
          placeholder={"asdf qwer zxcv\nlorem ipsum garbage"}
        />
        <TextArea
          label="Good / correct texts"
          hint="one per line"
          value={good}
          onChange={setGood}
          rows={6}
          disabled={running}
          placeholder={"the quick brown fox jumps over the lazy dog"}
        />
        <div className="row">
          <NumberField
            label="Negative epochs"
            value={negEpochs}
            onChange={setNegEpochs}
            min={1}
            step={1}
            disabled={running}
          />
          <NumberField
            label="Positive epochs"
            value={posEpochs}
            onChange={setPosEpochs}
            min={1}
            step={1}
            disabled={running}
          />
        </div>
        <div className="row">
          <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={running} />
          <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={running} />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : "Run 2NRL"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <Alert message={formError} onDismiss={() => setFormError(null)} />
        <Alert message={error} onDismiss={clearError} />

        <h3>Manual inversion</h3>
        <p className="muted">
          Flips the sign of every edge weight and activation amplitude. Inverting twice restores the network.
        </p>
        <div className="actions">
          <button type="button" onClick={handleInvert} disabled={inverting || anyJobRunning}>
            {inverting ? "Inverting…" : "Invert network"}
          </button>
        </div>
        <Alert kind="ok" message={invertNotice} onDismiss={() => setInvertNotice(null)} />
        <Alert message={invertError} onDismiss={() => setInvertError(null)} />
      </form>

      <div className="card">
        <h2>Phases</h2>
        <JobStatus job={job} emptyText="No 2NRL job yet." />
        {history.length > 0 ? (
          <dl className="kv">
            <dt>last negative loss</dt>
            <dd>{fmtNum(lastLoss(negative), 4)}</dd>
            <dt>last positive loss</dt>
            <dd>{fmtNum(lastLoss(positive), 4)}</dd>
          </dl>
        ) : running ? (
          <p className="muted">Waiting for the first epoch…</p>
        ) : null}
        <PhaseTable title="Negative phase (bad data)" rows={negative} />
        <PhaseTable title="Positive phase (good data, after inversion)" rows={positive} />
      </div>
    </>
  );
}
