import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import {
  asArray,
  fmtBytes,
  fmtInt,
  fmtNum,
  fmtTime,
  jobIsRunning,
  parseInteger,
  parseNumber,
  splitLines,
  yesNo,
} from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import UploadPicker from "./UploadPicker.jsx";
import { NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const MAX_ROWS = 100;
const SLOW_NOTE = "Ollama is working; this can take a minute or two.";

/** Server-side Ollama defaults reported by /api/status as {"ollama": {"url", "model"}}. */
function ollamaDefaults(status) {
  const o = status && status.ollama && typeof status.ollama === "object" ? status.ollama : {};
  return {
    url: typeof o.url === "string" ? o.url : "",
    model: typeof o.model === "string" ? o.model : "",
  };
}

/** URL to send as an override: only when the field differs from the server default. */
function urlOverride(url, defaults) {
  const value = String(url ?? "").trim();
  return value && value !== defaults.url ? value : "";
}

/** {"url", "model"} overrides for the corpus / review requests (omitted = server default). */
function requestOverrides(url, model, defaults) {
  const out = {};
  const value = urlOverride(url, defaults);
  if (value) out.url = value;
  if (model) out.model = model;
  return out;
}

/** Coerce a text list from the API: strings only, blank entries dropped. */
function asTexts(value) {
  return asArray(value)
    .map((t) => (typeof t === "string" ? t : String(t ?? "")))
    .filter((t) => t.trim() !== "");
}

function unique(texts) {
  return Array.from(new Set(texts));
}

/** fmtBytes with a GB step, for model files. */
function fmtSize(value) {
  const gb = 1024 * 1024 * 1024;
  if (typeof value === "number" && Number.isFinite(value) && value >= gb) return `${(value / gb).toFixed(2)} GB`;
  return fmtBytes(value);
}

function verdictOf(review) {
  const v = review && review.verdict;
  return v === "pass" || v === "fail" ? v : "unrated";
}

/** Chips for the corpus lines held as 2NRL garbage / good data, with a clear button each. */
function HeldChips({ heldBad, heldGood, onClearBad, onClearGood }) {
  if (heldBad.length === 0 && heldGood.length === 0) return null;
  return (
    <div className="chips held">
      {heldBad.length > 0 ? (
        <span className="chip bad">
          held as 2NRL garbage: {fmtInt(heldBad.length)} lines
          <button type="button" className="link" onClick={onClearBad} aria-label="Clear held garbage lines" title="Clear">
            ×
          </button>
        </span>
      ) : null}
      {heldGood.length > 0 ? (
        <span className="chip good">
          held as 2NRL good: {fmtInt(heldGood.length)} lines
          <button type="button" className="link" onClick={onClearGood} aria-label="Clear held good lines" title="Clear">
            ×
          </button>
        </span>
      ) : null}
    </div>
  );
}

/** Ollama URL + model selection (GET /api/ollama/models). Unavailable servers are reported inline. */
function ConnectionCard({ defaults, url, setUrl, model, setModel }) {
  const [models, setModels] = useState([]);
  const [info, setInfo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const data = await api.ollamaModels(urlOverride(url, defaults));
      const seen = new Set();
      const list = asArray(data && data.models).filter((m) => {
        if (!m || typeof m.name !== "string" || seen.has(m.name)) return false;
        seen.add(m.name);
        return true;
      });
      setModels(list);
      setInfo(data && typeof data === "object" ? data : {});
      const names = list.map((m) => m.name);
      const preferred = data && typeof data.model === "string" ? data.model : "";
      // Keep a still-listed choice, otherwise preselect the default model when the server offers it.
      setModel((current) => (current && names.includes(current) ? current : names.includes(preferred) ? preferred : ""));
      if (url === null && data && typeof data.url === "string" && data.url) setUrl(data.url);
    } catch (err) {
      setModels([]);
      setInfo(null);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const available = Boolean(info && info.available);
  const defaultModel = info && typeof info.model === "string" && info.model ? info.model : defaults.model;
  const options = [
    ["", defaultModel ? `server default (${defaultModel})` : "server default"],
    ...models.map((m) => [m.name, typeof m.size === "number" ? `${m.name} · ${fmtSize(m.size)}` : m.name]),
  ];
  const selected = models.find((m) => m.name === model) || null;
  const details = selected && selected.details && typeof selected.details === "object" ? selected.details : null;
  const detailText = details
    ? Object.entries(details)
        .filter(([, v]) => v !== null && v !== undefined && typeof v !== "object")
        .map(([k, v]) => `${k} ${String(v)}`)
        .join(" · ")
    : "";

  let summary;
  if (loading) summary = "Checking the Ollama server…";
  else if (!info) summary = error ? "The API could not be reached." : "Not checked yet.";
  else if (available) summary = `${fmtInt(models.length)} model${models.length === 1 ? "" : "s"} at ${String(info.url ?? "")}`;
  else summary = `Ollama unavailable at ${String(info.url ?? "")}`;

  return (
    <form
      className="card wide"
      onSubmit={(event) => {
        event.preventDefault();
        load();
      }}
    >
      <div className="toolbar">
        <h2>Ollama</h2>
        <div className="job-status">
          <span className={`badge ${available ? "done" : info ? "error" : ""}`}>
            {available ? "available" : info ? "unavailable" : "unknown"}
          </span>
          <span className="muted">{summary}</span>
        </div>
      </div>
      <p className="muted">
        A local Ollama server writes training corpora from a prompt and acts as an adversarial reviewer of the model's
        samples. Every Ollama call can take a minute or two.
      </p>
      <div className="row">
        <TextField
          label="Ollama URL"
          hint={defaults.url ? `server default ${defaults.url}` : "blank = server default"}
          value={url ?? ""}
          onChange={setUrl}
          placeholder={defaults.url || "http://127.0.0.1:11434"}
          disabled={loading}
        />
        <SelectField
          label="Model"
          hint={loading ? "loading…" : models.length > 0 ? `${models.length} listed` : "none listed"}
          value={model}
          onChange={setModel}
          options={options}
          disabled={loading}
        />
      </div>
      <div className="actions">
        <button type="submit" disabled={loading}>
          {loading ? "Checking…" : "Refresh models"}
        </button>
        <span className="muted note">Press Refresh after changing the URL to list its models.</span>
      </div>
      {selected ? (
        <p className="muted">
          {selected.name}: {fmtSize(selected.size)}
          {selected.modified_at ? ` · modified ${fmtTime(selected.modified_at)}` : ""}
          {detailText ? ` · ${detailText}` : ""}
        </p>
      ) : null}
      <Alert message={error} onDismiss={() => setError(null)} />
      {info && !available ? (
        <>
          <Alert message={info.error || "Ollama is not available."} />
          <p className="muted">
            Check the URL and that Ollama is running, then press Refresh. The requests below still use the settings
            above.
          </p>
        </>
      ) : null}
    </form>
  );
}

/** Ask Ollama for a corpus (POST /api/ollama/corpus, train=false); the lines go to CorpusResultCard. */
function CorpusCard({ overrides, onResult }) {
  const [prompt, setPrompt] = useState("");
  const [lines, setLines] = useState("20");
  const [style, setStyle] = useState("good");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (prompt.trim() === "") {
      setError("Enter a prompt describing the corpus to write.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.ollamaCorpus({
        prompt,
        lines: Math.max(1, parseInteger(lines, 20)),
        style,
        ...overrides,
        train: false,
      });
      onResult(data && typeof data === "object" ? data : {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Corpus from a prompt</h2>
      <TextArea
        label="Prompt"
        hint="what the lines should be about"
        value={prompt}
        onChange={setPrompt}
        rows={5}
        disabled={loading}
        placeholder="Short English sentences about the weather, one per line."
      />
      <div className="row">
        <NumberField label="Lines" value={lines} onChange={setLines} min={1} step={1} disabled={loading} />
        <SelectField
          label="Style"
          value={style}
          onChange={setStyle}
          disabled={loading}
          options={[
            ["good", "good (well-formed text)"],
            ["garbage", "garbage (2NRL negative data)"],
          ]}
        />
      </div>
      <div className="actions">
        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Generating…" : "Generate"}
        </button>
        {loading ? <span className="muted note">{SLOW_NOTE}</span> : null}
      </div>
      <Alert message={error} onDismiss={() => setError(null)} />
    </form>
  );
}

/** Generated lines plus what to do with them: train, save as upload, hold as 2NRL data. */
function CorpusResultCard({ corpus, status, heldBad, heldGood, onHoldBad, onHoldGood, onClearBad, onClearGood }) {
  const texts = corpus ? asTexts(corpus.texts) : [];
  const [epochs, setEpochs] = useState("5");
  const [lr, setLr] = useState("0.05");
  const [batchSize, setBatchSize] = useState("256");
  const [saveName, setSaveName] = useState("ollama-corpus.txt");
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState(null);
  const [saved, setSaved] = useState(null);
  const [trainSeen, setTrainSeen] = useState(false);
  const train = useJob("train");

  // The hook adopts the server's last train job on mount; only show it once it is ours or running.
  useEffect(() => {
    if (train.running) setTrainSeen(true);
  }, [train.running]);

  // A new corpus invalidates the record of the previous save.
  useEffect(() => {
    setSaved(null);
    setSaveError(null);
  }, [corpus]);

  const otherJobRunning = jobIsRunning(status) && !train.running;

  async function handleTrain() {
    if (texts.length === 0) return;
    setTrainSeen(true);
    await train.start(() =>
      api.train({
        texts,
        epochs: parseInteger(epochs, 5),
        lr: parseNumber(lr, 0.05),
        batch_size: parseInteger(batchSize, 256),
      }),
    );
  }

  async function handleSave(event) {
    event.preventDefault();
    const name = saveName.trim();
    if (!name) {
      setSaveError("Enter a file name for the upload.");
      return;
    }
    setSaving(true);
    setSaveError(null);
    setSaved(null);
    try {
      const data = await api.upload(name, `${texts.join("\n")}\n`);
      const record = asArray(data && data.uploads)[0];
      setSaved(record && typeof record === "object" ? record : { name });
    } catch (err) {
      setSaveError(err.message);
    } finally {
      setSaving(false);
    }
  }

  const history = asArray(train.job && train.job.history);
  const rows = history.slice(-MAX_ROWS);
  const savedText = saved
    ? `Saved ${String(saved.name ?? saveName)}: ${fmtInt(saved.lines)} lines, ${fmtBytes(saved.bytes)}` +
      (saved.replaced !== undefined ? `, replaced existing file: ${yesNo(saved.replaced)}` : "") +
      (saved.modified ? `, ${fmtTime(saved.modified)}` : "") +
      ". It is now selectable in the file pickers."
    : null;

  return (
    <div className="card">
      <h2>Generated lines</h2>
      {!corpus ? (
        <p className="muted">
          Describe a corpus in the prompt and press Generate. The lines can then be trained on, saved as an upload, or
          held as garbage / good data for 2NRL.
        </p>
      ) : texts.length === 0 ? (
        <p className="muted">Ollama returned no usable lines. Try a more specific prompt or another model.</p>
      ) : (
        <>
          <p className="muted">
            {fmtInt(texts.length)} lines · style {String(corpus.style ?? "–")} · model {String(corpus.model ?? "–")}
          </p>
          <div className="lines">
            {texts.map((t, i) => (
              <div className="line" key={i}>
                <span className="n">{i + 1}</span>
                <span className="t">{t}</span>
              </div>
            ))}
          </div>

          <h3>Train on these lines</h3>
          <div className="inline-form">
            <NumberField label="Epochs" value={epochs} onChange={setEpochs} min={1} step={1} disabled={train.running} />
            <NumberField label="Learning rate" value={lr} onChange={setLr} min={0} disabled={train.running} />
            <NumberField
              label="Batch size"
              value={batchSize}
              onChange={setBatchSize}
              min={1}
              step={1}
              disabled={train.running}
            />
            <button
              type="button"
              className="primary"
              disabled={train.running || train.busy || otherJobRunning}
              onClick={handleTrain}
            >
              {train.busy ? "Starting…" : "Train on these"}
            </button>
            <button type="button" className="danger" disabled={!train.running} onClick={() => train.stop()}>
              Stop
            </button>
          </div>
          {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
          <Alert message={train.error} onDismiss={train.clearError} />
        </>
      )}

      {trainSeen ? (
        <>
          <h3>Training job</h3>
          <JobStatus job={train.job} emptyText="No training job yet." />
          {rows.length > 0 ? (
            <>
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
                        <td>{fmtNum(r.compression_ratio, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : train.running ? (
            <p className="muted">Waiting for the first epoch…</p>
          ) : null}
        </>
      ) : null}

      {texts.length > 0 ? (
        <>
          <h3>Save as upload</h3>
          <form className="inline-form" onSubmit={handleSave}>
            <TextField
              label="File name"
              value={saveName}
              onChange={setSaveName}
              placeholder="ollama-corpus.txt"
              disabled={saving}
            />
            <button type="submit" disabled={saving}>
              {saving ? "Saving…" : "Save as upload"}
            </button>
          </form>
          <Alert message={saveError} onDismiss={() => setSaveError(null)} />
          <Alert kind="ok" message={savedText} onDismiss={() => setSaved(null)} />

          <h3>Use for 2NRL</h3>
          <p className="muted">
            Held lines feed the "Apply as 2NRL" step of the review card: garbage lines join the bad set, good lines the
            good set.
          </p>
          <div className="actions">
            <button type="button" onClick={() => onHoldBad(texts)}>
              Use as 2NRL garbage
            </button>
            <button type="button" onClick={() => onHoldGood(texts)}>
              Use as 2NRL good
            </button>
          </div>
        </>
      ) : null}
      <HeldChips heldBad={heldBad} heldGood={heldGood} onClearBad={onClearBad} onClearGood={onClearGood} />
    </div>
  );
}

/** Have Ollama rate samples from the model, or pasted texts (POST /api/ollama/review, apply=none). */
function ReviewCard({ overrides, onResult }) {
  const [count, setCount] = useState("8");
  const [prefix, setPrefix] = useState("");
  const [maxLength, setMaxLength] = useState("60");
  const [temperature, setTemperature] = useState("1.0");
  const [threshold, setThreshold] = useState("6");
  const [texts, setTexts] = useState("");
  const [context, setContext] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const given = splitLines(texts);
  const usingGiven = given.length > 0;

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const body = { threshold: parseNumber(threshold, 6), ...overrides, apply: "none" };
      if (usingGiven) {
        body.texts = given;
      } else {
        body.count = Math.max(1, parseInteger(count, 8));
        body.prefix = prefix;
        body.max_length = Math.max(1, parseInteger(maxLength, 60));
        body.temperature = parseNumber(temperature, 1);
      }
      if (context.trim() !== "") body.context = context.trim();
      const data = await api.ollamaReview(body);
      onResult(data && typeof data === "object" ? data : {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="card" onSubmit={handleSubmit}>
      <h2>Adversarial review</h2>
      <p className="muted">
        Ollama rates each sample from the model 0–10; ratings at or above the threshold pass. Paste texts below to
        review those instead of sampling.
      </p>
      <div className="row">
        <NumberField
          label="Samples"
          hint="from the model"
          value={count}
          onChange={setCount}
          min={1}
          step={1}
          disabled={loading || usingGiven}
        />
        <NumberField
          label="Max length"
          value={maxLength}
          onChange={setMaxLength}
          min={1}
          step={1}
          disabled={loading || usingGiven}
        />
      </div>
      <div className="row">
        <NumberField
          label="Temperature"
          value={temperature}
          onChange={setTemperature}
          min={0.01}
          disabled={loading || usingGiven}
        />
        <NumberField
          label="Pass threshold"
          hint="0–10"
          value={threshold}
          onChange={setThreshold}
          min={0}
          max={10}
          disabled={loading}
        />
      </div>
      <TextField
        label="Prefix"
        hint="optional; samples continue it"
        value={prefix}
        onChange={setPrefix}
        placeholder="the quick br"
        disabled={loading || usingGiven}
      />
      <TextArea
        label="Texts to review"
        hint="optional, one per line; replaces sampling"
        value={texts}
        onChange={setTexts}
        rows={5}
        disabled={loading}
        placeholder={"the quick brown fox jumps over the lazy dog\nasdf qwer zxcv"}
      />
      {usingGiven ? (
        <p className="muted">
          {fmtInt(given.length)} given text{given.length === 1 ? "" : "s"} will be reviewed; the sampling settings are
          ignored.
        </p>
      ) : null}
      <TextField
        label="Context for the reviewer"
        hint="optional"
        value={context}
        onChange={setContext}
        placeholder="The model is trained on short English sentences about the weather."
        disabled={loading}
      />
      <div className="actions">
        <button type="submit" className="primary" disabled={loading}>
          {loading ? "Reviewing…" : "Review"}
        </button>
        {loading ? <span className="muted note">{SLOW_NOTE}</span> : null}
      </div>
      <Alert message={error} onDismiss={() => setError(null)} />
    </form>
  );
}

/** Review table plus the "Apply as 2NRL" step (POST /api/2nrl with the failed / passed texts). */
function ReviewResultCard({ review, status, heldBad, heldGood, onClearBad, onClearGood }) {
  const [goodFiles, setGoodFiles] = useState([]);
  const [negEpochs, setNegEpochs] = useState("3");
  const [posEpochs, setPosEpochs] = useState("3");
  const [negLr, setNegLr] = useState("0.05");
  const [posLr, setPosLr] = useState("0.01");
  const [nrlSeen, setNrlSeen] = useState(false);
  const nrl = useJob("2nrl");

  useEffect(() => {
    if (nrl.running) setNrlSeen(true);
  }, [nrl.running]);

  const reviews = review ? asArray(review.reviews).filter((r) => r && typeof r === "object") : [];
  const passed = reviews.filter((r) => verdictOf(r) === "pass").length;
  const failed = reviews.filter((r) => verdictOf(r) === "fail").length;
  const unrated = reviews.length - passed - failed;
  const bad = unique([...(review ? asTexts(review.bad) : []), ...heldBad]);
  const good = unique([...(review ? asTexts(review.good) : []), ...heldGood]);
  const hasGood = good.length > 0 || goodFiles.length > 0;
  const otherJobRunning = jobIsRunning(status) && !nrl.running;

  let reason = null;
  if (bad.length === 0 && !hasGood) reason = "Nothing to apply yet: run a review, or hold lines from the corpus card.";
  else if (bad.length === 0) reason = "No bad texts: nothing failed the review and no garbage lines are held.";
  else if (!hasGood)
    reason = "No good texts: nothing passed the review; hold good lines from the corpus card or pick extra good files.";

  async function handleRun() {
    if (reason) return;
    setNrlSeen(true);
    await nrl.start(() =>
      api.twoNrl({
        bad,
        ...(good.length > 0 ? { good } : {}),
        ...(goodFiles.length > 0 ? { good_files: goodFiles } : {}),
        neg_epochs: parseInteger(negEpochs, 3),
        pos_epochs: parseInteger(posEpochs, 3),
        neg_lr: parseNumber(negLr, 0.05),
        pos_lr: parseNumber(posLr, 0.01),
      }),
    );
  }

  const passRate =
    review && typeof review.pass_rate === "number" && Number.isFinite(review.pass_rate)
      ? `${Math.round(review.pass_rate * 100)}%`
      : "–";
  const history = asArray(nrl.job && nrl.job.history);
  const rows = history.slice(-MAX_ROWS);

  return (
    <div className="card">
      <h2>Review results</h2>
      {!review ? (
        <p className="muted">Press Review to have Ollama rate samples from the model, or the texts you paste.</p>
      ) : reviews.length === 0 ? (
        <p className="muted">No reviews were returned.</p>
      ) : (
        <>
          <div className="chips">
            <span className="stat">
              mean rating <b>{fmtNum(review.mean_rating, 2)}</b>
            </span>
            <span className="stat">
              pass rate <b>{passRate}</b>
            </span>
            <span className="stat">
              passed <b>{fmtInt(passed)}</b> · failed <b>{fmtInt(failed)}</b> · unrated <b>{fmtInt(unrated)}</b>
            </span>
            <span className="stat">
              source <b>{String(review.source ?? "–")}</b>
            </span>
            <span className="stat">
              model <b>{String(review.model ?? "–")}</b>
            </span>
            <span className="stat">
              threshold <b>{fmtNum(review.threshold, 1)}</b>
            </span>
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>#</th>
                  <th>rating</th>
                  <th>verdict</th>
                  <th>text</th>
                  <th>critique</th>
                </tr>
              </thead>
              <tbody>
                {reviews.map((r, i) => {
                  const verdict = verdictOf(r);
                  const rating = typeof r.rating === "number" && Number.isFinite(r.rating) ? r.rating : null;
                  const width = rating === null ? 0 : Math.max(0, Math.min(100, rating * 10));
                  return (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>
                        <span className="rating">
                          <span className={`rating-bar ${verdict}`} aria-hidden="true">
                            <i style={{ width: `${width}%` }} />
                          </span>
                          {rating === null ? "–" : fmtNum(rating, 1)}
                        </span>
                      </td>
                      <td>
                        <span className={`badge ${verdict}`}>{verdict}</span>
                      </td>
                      <td className="wrap">{String(r.text ?? "")}</td>
                      <td className="wrap">{String(r.critique ?? "")}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}

      <h3>Apply as 2NRL</h3>
      <p className="muted">
        Texts that failed or could not be rated become the bad set; texts that passed, held good lines and extra good
        files become the good set.
      </p>
      <div className="chips">
        <span className="stat">
          bad <b>{fmtInt(bad.length)}</b>
          <span>
            ({fmtInt(failed)} failed · {fmtInt(unrated)} unrated · {fmtInt(heldBad.length)} held)
          </span>
        </span>
        <span className="stat">
          good <b>{fmtInt(good.length)}</b>
          <span>
            ({fmtInt(passed)} passed · {fmtInt(heldGood.length)} held · {fmtInt(goodFiles.length)} files)
          </span>
        </span>
      </div>
      <HeldChips heldBad={heldBad} heldGood={heldGood} onClearBad={onClearBad} onClearGood={onClearGood} />
      <UploadPicker
        selected={goodFiles}
        onChange={setGoodFiles}
        disabled={nrl.running}
        title="Extra good files"
        hint="Uploaded files added to the good set for the positive phase."
      />
      <div className="row">
        <NumberField
          label="Negative epochs"
          value={negEpochs}
          onChange={setNegEpochs}
          min={1}
          step={1}
          disabled={nrl.running}
        />
        <NumberField
          label="Positive epochs"
          value={posEpochs}
          onChange={setPosEpochs}
          min={1}
          step={1}
          disabled={nrl.running}
        />
      </div>
      <div className="row">
        <NumberField label="Negative lr" value={negLr} onChange={setNegLr} min={0} disabled={nrl.running} />
        <NumberField label="Positive lr" value={posLr} onChange={setPosLr} min={0} disabled={nrl.running} />
      </div>
      <div className="actions">
        <button
          type="button"
          className="primary"
          disabled={Boolean(reason) || nrl.running || nrl.busy || otherJobRunning}
          onClick={handleRun}
        >
          {nrl.busy ? "Starting…" : "Run 2NRL"}
        </button>
        <button type="button" className="danger" disabled={!nrl.running} onClick={() => nrl.stop()}>
          Stop
        </button>
      </div>
      {reason ? <p className="muted">{reason}</p> : null}
      {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
      <Alert message={nrl.error} onDismiss={nrl.clearError} />
      {nrlSeen ? (
        <>
          <JobStatus job={nrl.job} emptyText="No 2NRL job yet." />
          {rows.length > 0 ? (
            <>
              {history.length > MAX_ROWS ? (
                <p className="muted">
                  Showing the last {MAX_ROWS} of {history.length} epochs.
                </p>
              ) : null}
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>phase</th>
                      <th>epoch</th>
                      <th>loss</th>
                      <th>perplexity</th>
                      <th>nodes</th>
                      <th>compression</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((r, i) => (
                      <tr key={i}>
                        <td>{r && r.phase ? String(r.phase) : "–"}</td>
                        <td>{fmtInt(r && r.epoch)}</td>
                        <td>{fmtNum(r && r.loss, 4)}</td>
                        <td>{fmtNum(r && r.perplexity, 3)}</td>
                        <td>{fmtInt(r && r.nodes)}</td>
                        <td>{fmtNum(r && r.compression_ratio, 2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : nrl.running ? (
            <p className="muted">Waiting for the first epoch…</p>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

/**
 * Ollama tab: connection (URL + model), a corpus written from a prompt with
 * train / save / hold-for-2NRL actions, and an adversarial review whose failed
 * and passed texts can be applied as a 2NRL run. The URL, model and held lines
 * are shared by the cards; each card keeps its own form state.
 */
export default function OllamaPanel({ status }) {
  const defaults = ollamaDefaults(status);
  const [url, setUrl] = useState(null);
  const [model, setModel] = useState("");
  const [corpus, setCorpus] = useState(null);
  const [review, setReview] = useState(null);
  const [heldBad, setHeldBad] = useState([]);
  const [heldGood, setHeldGood] = useState([]);

  // Seed the URL field with the server default once /api/status reports it (until the user edits it).
  useEffect(() => {
    if (url === null && defaults.url) setUrl(defaults.url);
  }, [defaults.url, url]);

  const overrides = requestOverrides(url, model, defaults);
  const clearBad = () => setHeldBad([]);
  const clearGood = () => setHeldGood([]);

  return (
    <>
      <ConnectionCard defaults={defaults} url={url} setUrl={setUrl} model={model} setModel={setModel} />
      <CorpusCard overrides={overrides} onResult={setCorpus} />
      <CorpusResultCard
        corpus={corpus}
        status={status}
        heldBad={heldBad}
        heldGood={heldGood}
        onHoldBad={setHeldBad}
        onHoldGood={setHeldGood}
        onClearBad={clearBad}
        onClearGood={clearGood}
      />
      <ReviewCard overrides={overrides} onResult={setReview} />
      <ReviewResultCard
        review={review}
        status={status}
        heldBad={heldBad}
        heldGood={heldGood}
        onClearBad={clearBad}
        onClearGood={clearGood}
      />
    </>
  );
}
