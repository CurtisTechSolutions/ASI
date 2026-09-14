import { useState } from "react";
import { asArray, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";

const MODES = [
  ["beam", "beam search (the cheapest path)"],
  ["sample", "sampled"],
];

/**
 * The recall tutor, shared by the Speech and Images tabs.
 *
 * Nothing here needs an LLM: the network was taught an encoded utterance or
 * picture, so the right answer is on file. It is given the opening of that text
 * and asked to write the rest; what comes back is run through the codec and
 * compared with the original, marked out of 10, and - with "blame it" - handed
 * to the negative network with the single worst thing wrong with it named
 * (unreadable, truncated, overrun, garbled, silence / blank, clipping / noise,
 * mishearing, distortion / drift).
 *
 * `run(options)` does the request; `disabled` says the tab has nothing to ask
 * about yet (no recording, no image).
 */
export default function RecallCard({ modality, run, disabled }) {
  const [length, setLength] = useState(modality === "speech" ? "240" : "0");
  const [lead, setLead] = useState("");
  const [attempts, setAttempts] = useState("1");
  const [mode, setMode] = useState("beam");
  const [threshold, setThreshold] = useState("6");
  const [listenBack, setListenBack] = useState(false);
  const [blame, setBlame] = useState(true);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  async function ask() {
    setBusy(true);
    setError(null);
    try {
      const options = {
        length: parseInteger(length, 0),
        attempts: parseInteger(attempts, 1),
        mode,
        threshold: parseNumber(threshold, 6),
        blame,
      };
      if (lead.trim() !== "") options.lead = parseInteger(lead, 0);
      if (modality === "speech" && listenBack) options.listen_back = true;
      setResult(await run(options));
    } catch (err) {
      setError(err.message);
      setResult(null);
    } finally {
      setBusy(false);
    }
  }

  const lessons = asArray(result && result.lessons);
  const report = (result && result.report) || null;
  const taught = result && result.negative ? result.negative.taught : null;
  const thing = modality === "speech" ? "utterance" : "picture";

  return (
    <div className="card">
      <h2>What does it remember?</h2>
      <p className="muted">
        The {thing} was encoded into text and trained on, so the right answer is on file and no teacher is needed. The
        network is given the opening of that text — {modality === "speech" ? "the utterance's own token and the waveform header" : "the image header and a few characters of the payload"} — and
        writes the rest; what comes back is decoded and compared with the original. The mark out of 10 is the agreement
        over the payload, and a failure is named.
      </p>
      <div className="row">
        <NumberField
          label="Ask for"
          hint="payload characters (0: the whole thing); the marking compares against exactly that much"
          value={length}
          onChange={setLength}
          min={0}
          step={10}
          disabled={busy}
        />
        <NumberField
          label="Lead"
          hint={modality === "speech" ? "payload characters given away (default 0: the token identifies it)" : "payload characters given away (default 16: a header alone names no picture)"}
          value={lead}
          onChange={setLead}
          min={0}
          step={1}
          disabled={busy}
          placeholder="default"
        />
        <NumberField label="Attempts" hint="it stops at the first pass" value={attempts} onChange={setAttempts} min={1} step={1} disabled={busy} />
      </div>
      <div className="row">
        <SelectField label="First attempt" value={mode} onChange={setMode} options={MODES} disabled={busy} />
        <NumberField label="Pass mark" hint="out of 10" value={threshold} onChange={setThreshold} min={0} max={10} disabled={busy} />
      </div>
      <div className="row">
        {modality === "speech" ? (
          <CheckField label="Listen back (transcribe what it said and compare the words)" checked={listenBack} onChange={setListenBack} disabled={busy} />
        ) : null}
        <CheckField label="Blame it: teach the negative network why each failure failed" checked={blame} onChange={setBlame} disabled={busy} />
      </div>
      <div className="actions">
        <button type="button" className="primary" disabled={busy || disabled} onClick={ask}>
          {busy ? "Asking…" : `Ask it for this ${thing} back`}
        </button>
      </div>
      {disabled ? <p className="muted">Nothing to ask about yet — {modality === "speech" ? "record or choose a recording" : "choose an image"} first.</p> : null}
      <Alert message={error} onDismiss={() => setError(null)} />
      {report ? (
        <p className="muted">
          <b>
            {report.passed}/{report.lessons} remembered
          </b>
          , mean {fmtNum(report.mean_score, 1)}/10, {Math.round((report.mean_agreement || 0) * 100)}% agreement
          {Object.keys(report.reasons || {}).length
            ? ` · ${Object.entries(report.reasons)
                .map(([reason, count]) => `${reason} ×${count}`)
                .join(", ")}`
            : ""}
        </p>
      ) : null}
      {lessons.length ? (
        <table className="data">
          <thead>
            <tr>
              <th>what</th>
              <th>mark</th>
              <th>agreement</th>
              <th>verdict</th>
              <th>why</th>
            </tr>
          </thead>
          <tbody>
            {lessons.map((lesson, i) => {
              const grade = lesson.grade || {};
              const facts = grade.facts || {};
              return (
                <tr key={i}>
                  <td>{(lesson.exercise && (lesson.exercise.label || lesson.exercise.id)) || `#${i + 1}`}</td>
                  <td>{fmtNum(grade.score, 1)}</td>
                  <td>{Math.round((facts.agreement || 0) * 100)}%</td>
                  <td>{grade.passed ? "pass" : <b>{grade.error}</b>}</td>
                  <td className="muted">{grade.comment || "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      ) : null}
      {taught ? (
        <p className="muted">
          Blamed {taught.blamed} failure(s) over {taught.edges} edge(s); cleared {taught.cleared} fragment(s) from what
          it did remember. The Negative tab shows what it now knows.
        </p>
      ) : null}
    </div>
  );
}
