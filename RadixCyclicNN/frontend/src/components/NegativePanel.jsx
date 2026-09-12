import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { asArray, fmtInt, fmtNum, fmtTime, parseInteger, parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const DECISION_CLASS = { reject: "fail", suspect: "unrated", pass: "pass" };

/** The blamed fragments of a judged text, marked inside it. */
export function Highlighted({ text, spans }) {
  const ranges = asArray(spans)
    .filter((s) => s && Number.isFinite(s.start) && Number.isFinite(s.end) && s.end > s.start)
    .sort((a, b) => a.start - b.start);
  const parts = [];
  let at = 0;
  ranges.forEach((span, i) => {
    const start = Math.max(at, span.start);
    if (start >= span.end) return;
    if (start > at) parts.push(<span key={`t${i}`}>{text.slice(at, start)}</span>);
    parts.push(
      <mark key={`m${i}`} title={`${span.reason || "blamed"} · blame ${fmtNum(span.blame, 2)} · ${fmtInt(span.fails)} failures`}>
        {text.slice(start, span.end)}
      </mark>,
    );
    at = span.end;
  });
  if (at < text.length) parts.push(<span key="tail">{text.slice(at)}</span>);
  return <p className="blamed mono">{parts.length ? parts : text}</p>;
}

/** One verdict: the sentence, the reasons behind it and the fragments carrying them. */
export function Verdict({ verdict }) {
  if (!verdict) return null;
  const decision = verdict.decision || verdict.verdict;
  const reasons = asArray(verdict.reasons);
  return (
    <div className="verdict">
      <p>
        <span className={`badge ${DECISION_CLASS[decision] || "unrated"}`}>{decision}</span>{" "}
        <span className="muted">
          risk {fmtNum(verdict.risk, 2)} · peak {fmtNum(verdict.peak, 2)} · coverage {fmtNum(verdict.coverage, 2)}
          {typeof verdict.ratio === "number" ? ` · ratio ${fmtNum(verdict.ratio, 2)}` : ""}
          {verdict.rule ? ` · rule: ${verdict.rule}` : ""}
        </span>
      </p>
      <Highlighted text={String(verdict.text ?? "")} spans={verdict.spans} />
      <p className="critique">{verdict.why}</p>
      {reasons.length ? (
        <p className="chips">
          {reasons.map((r) => (
            <span className="chip bad" key={r.reason}>
              {r.reason} <small>{fmtNum(r.blame, 1)}</small>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}

/**
 * The negative network: a copy of the network that keeps only its negative
 * portions. It is trained on failures alone - the tutor (the Ollama reviewer,
 * the code judge, the evolve discriminator, a thumbs down) says what went
 * wrong and why - and it filters the positive model's output: the pair is the
 * GAN at output time.
 */
export default function NegativePanel({ status }) {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [notice, setNotice] = useState(null);
  const [busy, setBusy] = useState("");

  // filter
  const [count, setCount] = useState("3");
  const [prefix, setPrefix] = useState("");
  const [maxLength, setMaxLength] = useState("60");
  const [mode, setMode] = useState("sample");
  const [overSample, setOverSample] = useState("3");
  const [threshold, setThreshold] = useState("");
  const [minCoverage, setMinCoverage] = useState("");
  const [ratio, setRatio] = useState("0");
  const [peak, setPeak] = useState("");
  const [noRatio, setNoRatio] = useState(false);
  const [strict, setStrict] = useState(false);
  const [learn, setLearn] = useState(false);
  const [candidates, setCandidates] = useState("");
  const [filtered, setFiltered] = useState(null);

  // judge / teach
  const [judgeText, setJudgeText] = useState("");
  const [verdicts, setVerdicts] = useState([]);
  const [lesson, setLesson] = useState("");
  const [reason, setReason] = useState("gibberish");
  const [severity, setSeverity] = useState("1");
  const [source, setSource] = useState("frontend");
  const [note, setNote] = useState("");
  const [cleared, setCleared] = useState("");

  const refresh = useCallback(async () => {
    try {
      setInfo(await api.negative());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function run(what, work, message) {
    setBusy(what);
    setError(null);
    setNotice(null);
    try {
      const result = await work();
      if (message) setNotice(typeof message === "function" ? message(result) : message);
      await refresh();
      return result;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy("");
    }
  }

  const stats = (info && info.stats) || {};
  const reasons = asArray(info && info.reasons);
  const journal = asArray(info && info.journal);
  const settings = (info && info.settings) || {};
  const running = Boolean(status && status.job && status.job.state === "running");
  const disabled = Boolean(busy) || running;

  async function handleFilter(event) {
    event.preventDefault();
    const given = splitLines(candidates);
    const body = {
      count: parseInteger(count, 3),
      prefix,
      mode,
      max_length: parseInteger(maxLength, 60),
      over_sample: parseInteger(overSample, 3),
      strict,
      learn,
      no_ratio: noRatio,
      ratio: parseNumber(ratio, 0),
    };
    if (threshold !== "") body.threshold = parseNumber(threshold, undefined);
    if (minCoverage !== "") body.min_coverage = parseNumber(minCoverage, undefined);
    if (peak !== "") body.peak = parseNumber(peak, undefined);
    if (given.length) body.texts = given;
    const data = await run("filter", () => api.negativeFilter(body));
    if (data) setFiltered(data);
  }

  async function handleJudge(event) {
    event.preventDefault();
    const texts = splitLines(judgeText);
    if (!texts.length) {
      setError("Enter a text to judge (one per line).");
      return;
    }
    const data = await run("judge", () => api.negativeJudge({ texts }));
    if (data) setVerdicts(asArray(data.verdicts));
  }

  async function handleBlame(event) {
    event.preventDefault();
    const texts = splitLines(lesson);
    if (!texts.length) {
      setError("Enter the failed text (one per line).");
      return;
    }
    const done = await run(
      "blame",
      () => api.negativeBlame({ texts, reason, severity: parseNumber(severity, 1), source, note }),
      (data) => `blamed ${texts.length} text(s) for ${reason}: ${fmtInt(data.stats.edges)} edges know about it now`,
    );
    if (done) setLesson("");
  }

  async function handleClear(event) {
    event.preventDefault();
    const texts = splitLines(cleared);
    if (!texts.length) {
      setError("Enter the text the tutor passed (one per line).");
      return;
    }
    const done = await run(
      "clear",
      () => api.negativeClear({ texts }),
      (data) => `cleared ${fmtInt(data.matched)} of ${texts.length} text(s); ${fmtInt(data.unmatched)} shared nothing with a failure`,
    );
    if (done) setCleared("");
  }

  const rejected = asArray(filtered && filtered.rejected);
  const kept = asArray(filtered && filtered.texts);

  return (
    <>
      <form className="card wide" onSubmit={handleFilter}>
        <h2>Filter</h2>
        <p className="muted">
          The GAN at output time: the positive model over-samples candidates, the negative network vetoes the ones
          built out of known failure. Two signals reject - blame (risk over the threshold) and the likelihood ratio
          (it reads more like failure than like the training data). Paste candidates to judge those instead.
        </p>
        <div className="row">
          <NumberField label="Texts wanted" value={count} onChange={setCount} min="1" step="1" disabled={disabled} />
          <NumberField label="Over-sample" value={overSample} onChange={setOverSample} min="1" step="1" disabled={disabled}
                       hint="candidates per text" />
          <NumberField label="Max length" value={maxLength} onChange={setMaxLength} min="0" step="1" disabled={disabled} />
          <SelectField label="Mode" value={mode} onChange={setMode} disabled={disabled}
                       options={[["sample", "sample"], ["beam", "beam"], ["dijkstra", "dijkstra"]]} />
        </div>
        <div className="row">
          <TextField label="Prefix" value={prefix} onChange={setPrefix} placeholder="(from scratch)" disabled={disabled} />
          <NumberField label="Threshold" value={threshold} onChange={setThreshold} min="0" disabled={disabled}
                       placeholder={fmtNum(settings.threshold, 2)} hint="blame per transition" />
          <NumberField label="Min coverage" value={minCoverage} onChange={setMinCoverage} min="0" max="1" disabled={disabled}
                       placeholder={fmtNum(settings.min_coverage, 2)} hint="known-failing share" />
          <NumberField label="Ratio" value={ratio} onChange={setRatio} disabled={disabled || noRatio} hint="nats/char" />
          <NumberField label="Peak" value={peak} onChange={setPeak} min="0" disabled={disabled}
                       placeholder="off" hint="blame on one fragment" />
        </div>
        <div className="checks">
          <CheckField label="No ratio rule (blame only)" checked={noRatio} onChange={setNoRatio} disabled={disabled} />
          <CheckField label="Strict (drop suspect too)" checked={strict} onChange={setStrict} disabled={disabled} />
          <CheckField label="Blame what is rejected" checked={learn} onChange={setLearn} disabled={disabled} />
        </div>
        <TextArea label="Candidates" value={candidates} onChange={setCandidates} rows={3} disabled={disabled}
                  hint="optional: judge these instead of generating" placeholder="one candidate per line" />
        <div className="actions">
          <button type="submit" className="primary" disabled={disabled}>
            {busy === "filter" ? "Filtering…" : "Run the pair"}
          </button>
          {filtered ? (
            <span className="note">
              {fmtInt(asArray(filtered.verdicts).length)} candidates · {fmtInt(asArray(filtered.kept).length)} passed ·{" "}
              {fmtInt(rejected.length)} vetoed
            </span>
          ) : null}
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
      </form>

      {filtered ? (
        <div className="card">
          <h2>Output</h2>
          {kept.length ? (
            <ol className="samples">
              {kept.map((text, i) => (
                <li key={`${i}-${text}`}>
                  <pre className="sample">{text}</pre>
                </li>
              ))}
            </ol>
          ) : (
            <p className="muted">Everything was vetoed - that is information, not an error.</p>
          )}
          {rejected.length ? (
            <>
              <h3>Vetoed</h3>
              {rejected.map((verdict, i) => (
                <Verdict verdict={verdict} key={`r${i}`} />
              ))}
            </>
          ) : null}
        </div>
      ) : null}

      <form className="card" onSubmit={handleJudge}>
        <h2>Why</h2>
        <p className="muted">
          Walk a text through the failure structure: how much of it is built out of known failure, which reasons that
          blame carries and which fragments carry it.
        </p>
        <TextArea label="Texts" value={judgeText} onChange={setJudgeText} rows={4} disabled={disabled}
                  placeholder="one text per line" />
        <div className="actions">
          <button type="submit" className="primary" disabled={disabled}>
            {busy === "judge" ? "Judging…" : "Judge"}
          </button>
        </div>
        {verdicts.map((verdict, i) => (
          <Verdict verdict={verdict} key={`v${i}`} />
        ))}
      </form>

      <form className="card" onSubmit={handleBlame}>
        <h2>Blame</h2>
        <p className="muted">
          The negatives come from the tutor. Give it what went wrong, the reason it went wrong and (optionally) the
          tutor's own words - this is the only thing that adds structure to the negative network.
        </p>
        <TextArea label="Failed texts" value={lesson} onChange={setLesson} rows={3} disabled={disabled}
                  placeholder="one failure per line" />
        <div className="row">
          <TextField label="Reason" value={reason} onChange={setReason} disabled={disabled} placeholder="gibberish" />
          <NumberField label="Severity" value={severity} onChange={setSeverity} min="0" disabled={disabled}
                       hint="1 = one ordinary failure" />
          <TextField label="Source" value={source} onChange={setSource} disabled={disabled} hint="who says so" />
        </div>
        <TextField label="Note" value={note} onChange={setNote} disabled={disabled} hint="the tutor's own words" />
        <div className="actions">
          <button type="submit" className="primary" disabled={disabled}>
            {busy === "blame" ? "Blaming…" : "Blame"}
          </button>
        </div>
      </form>

      <form className="card" onSubmit={handleClear}>
        <h2>Clear</h2>
        <p className="muted">
          The tutor passed these: blame and clearing cancel, so a fragment that shows up in good and bad output alike
          stops carrying the verdict. Nothing is created here.
        </p>
        <TextArea label="Passed texts" value={cleared} onChange={setCleared} rows={3} disabled={disabled}
                  placeholder="one text per line" />
        <div className="actions">
          <button type="submit" disabled={disabled}>
            {busy === "clear" ? "Clearing…" : "Clear"}
          </button>
        </div>
      </form>

      <div className="card wide">
        <div className="toolbar">
          <h2>What the tutor blamed</h2>
          <div className="actions">
            <button type="button" className="small" disabled={disabled} onClick={() => run("save", () => api.negativeSave(), (d) => `saved ${d.path}`)}>
              Save
            </button>
            <button type="button" className="small danger" disabled={disabled}
                    onClick={() => run("reset", () => api.negativeReset(), "the negative network forgot everything")}>
              Forget everything
            </button>
          </div>
        </div>
        <dl className="kv">
          <dt>failures</dt>
          <dd>{fmtInt(stats.failures_total)}</dd>
          <dt>blame on edges</dt>
          <dd>{fmtNum(stats.edge_blame_total, 2)}</dd>
          <dt>cleared</dt>
          <dd>{fmtInt(stats.cleared_total)}</dd>
          <dt>nodes / edges</dt>
          <dd>
            {fmtInt(stats.nodes)} / {fmtInt(stats.edges)}
          </dd>
          <dt>judgements</dt>
          <dd>
            {fmtInt(stats.judgements)} ({fmtInt(stats.rejected)} rejected)
          </dd>
          <dt>sources</dt>
          <dd>
            {Object.entries((stats.sources && typeof stats.sources === "object" ? stats.sources : {}))
              .map(([k, v]) => `${k}=${v}`)
              .join(", ") || "–"}
          </dd>
          <dt>file</dt>
          <dd className="mono">{(info && info.path) || "–"}</dd>
        </dl>
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>reason</th>
                <th>blame</th>
                <th>failures</th>
                <th>edges</th>
                <th>share</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {reasons.length === 0 ? (
                <tr>
                  <td colSpan={6} className="muted">
                    Nothing has been blamed yet. The tutor fills this in: run an Ollama review with “blame”, a codegen
                    round, the evolve loop, or blame a text above.
                  </td>
                </tr>
              ) : (
                reasons.map((r) => (
                  <tr key={r.reason}>
                    <td>{r.reason}</td>
                    <td>{fmtNum(r.blame, 2)}</td>
                    <td>{fmtInt(r.fails)}</td>
                    <td>{fmtInt(r.edges)}</td>
                    <td>{fmtNum(r.share, 2)}</td>
                    <td>
                      <button type="button" className="link" disabled={disabled}
                              onClick={() => run("forget", () => api.negativeForget({ reason: r.reason }),
                                                (d) => `forgot ${r.reason} (${fmtNum(d.blame_removed, 2)} blame off ${fmtInt(d.edges)} edges)`)}>
                        forget
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
        {journal.length ? (
          <>
            <h3>Journal</h3>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>when</th>
                    <th>reason</th>
                    <th>severity</th>
                    <th>source</th>
                    <th>text</th>
                    <th>the tutor said</th>
                  </tr>
                </thead>
                <tbody>
                  {journal.map((entry, i) => (
                    <tr key={`${entry.at}-${i}`}>
                      <td>{fmtTime(entry.at)}</td>
                      <td>{entry.reason}</td>
                      <td>{fmtNum(entry.severity, 2)}</td>
                      <td>{entry.source || "–"}</td>
                      <td className="text">{entry.text}</td>
                      <td className="wrap">{entry.note || "–"}</td>
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
