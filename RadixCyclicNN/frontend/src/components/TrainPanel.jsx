import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { asArray, countingKind, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import LineChart from "./LineChart.jsx";
import UploadPicker from "./UploadPicker.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const MAX_ROWS = 300;
const PREVIEW_DEBOUNCE_MS = 350;

/** Rates as multiples of their base rate so both curves share one axis (absolute when the base is 0). */
function multiplierSeries(points, key, base, name, color) {
  const scale = base > 0 ? base : 1;
  return { name, color, values: points.map((p) => ({ x: p.epoch, y: p[key] / scale })) };
}

/**
 * Start / stop a training job and show its epoch records live. Before a job
 * runs (or after it finishes) the model's stored history from /api/history is
 * shown instead, so previous training is visible after a reload.
 *
 * The learning rates can follow a schedule: an expression of the epoch
 * (a "graph function") that the server evaluates once per epoch; the panel
 * previews the resulting curve through POST /api/schedule/preview while typing.
 */
export default function TrainPanel({ status }) {
  const [text, setText] = useStoredState("train.text", "");
  const [files, setFiles] = useState([]);
  const [wholeFile, setWholeFile] = useStoredState("train.wholeFile", false);
  const [epochs, setEpochs] = useStoredState("train.epochs", "5");
  const [lr, setLr] = useStoredState("train.lr", "0.05");
  const [actLr, setActLr] = useStoredState("train.actLr", "0.005");
  const [lrSchedule, setLrSchedule] = useStoredState("train.lrSchedule", "");
  const [actLrSchedule, setActLrSchedule] = useStoredState("train.actLrSchedule", "");
  const [presetName, setPresetName] = useStoredState("train.presetName", "");
  const [reverseSchedule, setReverseSchedule] = useStoredState("train.reverseSchedule", false);
  const [presets, setPresets] = useState([]);
  const [scheduleHelp, setScheduleHelp] = useState(null);
  const [preview, setPreview] = useState(null);
  const [previewError, setPreviewError] = useState(null);
  const [batchSize, setBatchSize] = useStoredState("train.batchSize", "256");
  const [autoCompress, setAutoCompress] = useStoredState("train.autoCompress", true);
  const [formError, setFormError] = useState(null);
  const [serverHistory, setServerHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("train");

  const otherJobRunning = jobIsRunning(status) && !running;
  const countKind = Boolean(status && status.kind === "count");
  // no learning rates, batches or schedules: the count and resonant models learn by counting
  const rateless = countingKind(status);
  const goEngine = Boolean(status && status.engine === "go");
  // Go server: what one training text is (each text is handled by its own goroutine)
  const [split, setSplit] = useStoredState("train.split", "lines");
  const [pageLines, setPageLines] = useStoredState("train.pageLines", "50");
  const scheduled = !rateless && (lrSchedule.trim() !== "" || actLrSchedule.trim() !== "");

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

  // Presets and the list of names an expression may use (cosmetic: the server validates).
  useEffect(() => {
    let alive = true;
    api
      .schedule()
      .then((info) => {
        if (!alive || !info || typeof info !== "object") return;
        setPresets(asArray(info.presets).filter((p) => p && typeof p.name === "string"));
        setScheduleHelp(info);
      })
      .catch(() => {
        // Presets are a convenience; typing an expression still works.
      });
    return () => {
      alive = false;
    };
  }, []);

  // Preview the curve while typing (debounced); blank expressions mean constant rates.
  useEffect(() => {
    if (!scheduled) {
      setPreview(null);
      setPreviewError(null);
      return undefined;
    }
    let alive = true;
    const timer = setTimeout(async () => {
      try {
        const data = await api.schedulePreview({
          lr_schedule: lrSchedule.trim(),
          act_lr_schedule: actLrSchedule.trim(),
          epochs: parseInteger(epochs, 5),
          lr: parseNumber(lr, 0.05),
          act_lr: parseNumber(actLr, 0.005),
          reverse_schedule: reverseSchedule,
        });
        if (!alive) return;
        setPreview(data && typeof data === "object" ? data : null);
        setPreviewError(null);
      } catch (err) {
        if (!alive) return;
        setPreview(null);
        setPreviewError(err.message);
      }
    }, PREVIEW_DEBOUNCE_MS);
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [scheduled, lrSchedule, actLrSchedule, epochs, lr, actLr, reverseSchedule]);

  // Refresh the stored history once a job reaches a terminal state.
  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  function choosePreset(name) {
    setPresetName(name);
    const preset = presets.find((p) => p.name === name);
    if (preset) {
      setLrSchedule(String(preset.lr ?? ""));
      setActLrSchedule(String(preset.act_lr ?? ""));
    } else if (name === "") {
      setLrSchedule("");
      setActLrSchedule("");
    }
  }

  async function handleStart(event) {
    event.preventDefault();
    const texts = splitLines(text);
    if (texts.length === 0 && files.length === 0) {
      setFormError("Enter at least one training text (one per line) or select uploaded files.");
      return;
    }
    setFormError(null);
    const goSplit = goEngine && split !== "lines";
    await start(() =>
      api.train({
        // the Go server cuts the pasted text and the files into paragraphs / pages itself
        ...(goSplit ? (text.trim() ? { text } : {}) : texts.length > 0 ? { texts } : {}),
        ...(goEngine ? { split, ...(split === "pages" ? { page_lines: parseInteger(pageLines, 50) } : {}) } : {}),
        ...(files.length > 0 ? { files, whole_file: wholeFile } : {}),
        epochs: parseInteger(epochs, 5),
        lr: parseNumber(lr, 0.05),
        act_lr: parseNumber(actLr, 0.005),
        ...(scheduled && lrSchedule.trim() ? { lr_schedule: lrSchedule.trim() } : {}),
        ...(scheduled && actLrSchedule.trim() ? { act_lr_schedule: actLrSchedule.trim() } : {}),
        ...(scheduled && reverseSchedule ? { reverse_schedule: true } : {}),
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

  const previewPoints = asArray(preview && preview.points).filter((p) => p && typeof p === "object");
  const previewSeries = [
    multiplierSeries(previewPoints, "lr", parseNumber(lr, 0.05), "learning rate", "#2563eb"),
    multiplierSeries(previewPoints, "act_lr", parseNumber(actLr, 0.005), "activation lr", "#ea580c"),
  ];
  const first = previewPoints[0];
  const last = previewPoints[previewPoints.length - 1];
  const presetOptions = [
    ["", "custom (blank = constant rates)"],
    ...presets.map((p) => [p.name, p.description ? `${p.name} — ${p.description}` : p.name]),
  ];
  const helpNames = scheduleHelp
    ? [...asArray(scheduleHelp.variables), ...asArray(scheduleHelp.constants), ...asArray(scheduleHelp.functions)].join(", ")
    : "epoch, i, epochs, t, lr0, act_lr0, lr, sin, cos, exp, log, sqrt, min, max, clamp, pi, e";

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Train</h2>
        <TextArea
          label="Training texts"
          hint={goEngine && split !== "lines" ? `cut into ${split} by the server` : "one per line"}
          value={text}
          onChange={setText}
          rows={10}
          disabled={running}
          placeholder={"the quick brown fox jumps over the lazy dog\nhello world"}
        />
        {goEngine ? (
          <div className="row">
            <SelectField
              label="Texts are"
              hint="each text is one unit of work for the goroutines"
              value={split}
              onChange={setSplit}
              disabled={running}
              options={[
                ["lines", "lines (one text per non-blank line)"],
                ["paragraphs", "paragraphs (blank-line separated blocks)"],
                ["pages", "pages (form feeds, or every N lines)"],
              ]}
            />
            {split === "pages" ? (
              <NumberField label="Lines per page" hint="when a file has no form feeds" value={pageLines} onChange={setPageLines} min={1} step={1} disabled={running} />
            ) : null}
          </div>
        ) : null}
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
            disabled={running || rateless}
            hint={rateless ? `not used by the ${status.model_label || status.kind} model` : undefined}
          />
        </div>
        {countKind ? (
          <p className="muted">
            Count / reward model: every epoch counts one more traversal of each text's path (edge weight = the
            edge's share of its node's traversals, all time and recently, plus rewards). There are no learning
            rates or batches to set.
          </p>
        ) : null}
        {rateless && !countKind ? (
          <p className="muted">
            Resonant model: every epoch walks each text carrying its <b>phase</b> and counts each traversal into
            its edge's circular mean, so an edge learns the phase at which it fires and how coherently. The cycle
            decisions the text made train the metacognitive layer beside it. There are no learning rates or
            batches to set.
          </p>
        ) : null}
        <div className="row" hidden={rateless}>
          <NumberField label="Learning rate" hint="lr0" value={lr} onChange={setLr} min={0} disabled={running} />
          <NumberField
            label="Activation lr"
            hint="act_lr0"
            value={actLr}
            onChange={setActLr}
            min={0}
            disabled={running}
          />
        </div>
        <fieldset className="schedule" hidden={rateless}>
          <legend>Learning-rate schedule (graph function of the epoch)</legend>
          <SelectField
            label="Preset"
            value={presetName}
            onChange={choosePreset}
            options={presetOptions}
            disabled={running}
          />
          <div className="row">
            <TextField
              label="Learning rate per epoch"
              hint="expression; blank = constant"
              value={lrSchedule}
              onChange={(v) => {
                setLrSchedule(v);
                setPresetName("");
              }}
              placeholder="linear(lr0, 4 * lr0)"
              disabled={running}
            />
            <TextField
              label="Activation lr per epoch"
              hint="may use lr, the epoch's learning rate"
              value={actLrSchedule}
              onChange={(v) => {
                setActLrSchedule(v);
                setPresetName("");
              }}
              placeholder="lr / 10"
              disabled={running}
            />
          </div>
          <CheckField
            label="Reverse the schedule (play it backwards: the last epoch's rates come first, so a ramp up becomes a ramp down)"
            checked={reverseSchedule}
            onChange={setReverseSchedule}
            disabled={running}
          />
          <p className="muted">
            Variables and functions: {helpNames}; helpers <code>linear(a, b)</code>, <code>geometric(a, b)</code>,{" "}
            <code>cosine(a, b)</code>, <code>step(a, factor, every)</code>, <code>warmup(a, b, n)</code>;{" "}
            <code>epoch</code> counts from 1, <code>t</code> runs from 0 (first epoch) to 1 (last).
          </p>
          <Alert message={previewError} onDismiss={() => setPreviewError(null)} />
          {scheduled && previewPoints.length > 0 ? (
            <div className="schedule-preview">
              <LineChart series={previewSeries} xLabel="epoch" yLabel="× base rate" height={180} />
              <p className="muted">
                lr {fmtNum(first.lr, 4)} → {fmtNum(last.lr, 4)} · activation lr {fmtNum(first.act_lr, 4)} →{" "}
                {fmtNum(last.act_lr, 4)} over {fmtInt(previewPoints.length)} epochs
              </p>
            </div>
          ) : null}
        </fieldset>
        {countKind ? (
          <p className="muted">
            The <b>score function</b> - how an edge's share of its node, the sliding window and the rewards weigh
            against each other - is a setting of the network rather than of a training run, and lives on the{" "}
            <a href="#network">Network settings</a> tab, beside the traversal and the encoder.
          </p>
        ) : null}
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
                    <th>lr</th>
                    <th>act lr</th>
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
                      <td>{fmtNum(r.lr, 4)}</td>
                      <td>{fmtNum(r.act_lr, 4)}</td>
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
