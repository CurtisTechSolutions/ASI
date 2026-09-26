import { useCallback, useEffect, useState } from "react";
import { unitWord } from "../settings.js";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { DEFAULT_BLUR, bandWeights, chargedRows, clampBlur, describeBand, fmtShare, gramUnits, unitStyle } from "../attention.js";
import { jobIsRunning } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, TextField } from "./Fields.jsx";

/** A gram drawn the way the band sees it: every unit as sharp as its position's weight. */
function BlurredGram({ gram, unit, weights }) {
  const units = gramUnits(gram, unit);
  const plain = JSON.stringify(String(gram));
  return (
    <span
      className="band-gram"
      role="img"
      aria-label={`the gram ${plain}`}
      title={`${plain}: the centre sharp, the ends blurred - how the band reads this gram`}
    >
      {units.map((text, i) => (
        <span key={i} className="band-unit" style={unitStyle(weights[i] ?? 1)}>
          {unit && unit !== "char" ? text : text === " " ? "␣" : text}
        </span>
      ))}
    </span>
  );
}

/** The band over one gram: a bar per position, and a sample gram under it, drawn through the band. */
function BandPicture({ weights, gram, unit }) {
  return (
    <div className="band-picture" aria-label={`the band over one gram: ${weights.map(fmtShare).join(", ")}`}>
      <div className="band-bars">
        {weights.map((w, i) => (
          <div key={i} className="band-bar-slot">
            <div className="band-bar" style={{ height: `${Math.max(3, w * 100)}%` }} />
            <small>{fmtShare(w)}</small>
          </div>
        ))}
      </div>
      {gram ? <BlurredGram gram={gram} unit={unit} weights={weights} /> : null}
    </div>
  );
}

/** One side of a correction, gram by gram: the writer rule's mark beside the band's share. */
function PreviewSide({ title, side, unit, weights }) {
  const rows = chargedRows(side);
  return (
    <>
      <h4>
        {title}: <span className="text-display inline">{String((side && side.text) || "")}</span>
      </h4>
      {rows.length === 0 ? (
        <p className="muted">Nothing in it changed.</p>
      ) : (
        <table className="band-table">
          <thead>
            <tr>
              <th>gram</th>
              <th title="the rule with the band off: the step that wrote the changed unit takes all of it">band off</th>
              <th title="each changed unit's one charge, shared by how centrally each gram sees it">band on</th>
              <th title="the step that sees the change most sharply: it takes the verdict">focus</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.index} className={row.focus ? "focus" : undefined}>
                <td>
                  {row.gram === null ? (
                    <span className="muted">the step into END (the text stopped too early)</span>
                  ) : (
                    <BlurredGram gram={row.gram} unit={unit} weights={weights} />
                  )}
                </td>
                <td>{row.writer ? "1" : "–"}</td>
                <td>
                  <span className="band-share">
                    <span style={{ width: `${Math.round(row.charge * 100)}%` }} />
                  </span>{" "}
                  {row.charge > 0 ? fmtShare(row.charge) : "–"}
                </td>
                <td>{row.focus ? "★" : ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </>
  );
}

/**
 * The attention band (../../SPEC-AttentionBand.md): where inside a gram a correction's blame and credit
 * land. Read from GET /api/model/attention, written with POST /api/model/attention - it belongs to the
 * model and is saved with it - and shown on a correction the user types (POST /api/model/attention/preview,
 * which changes nothing), at the blur on the slider, so a blur can be seen before it is applied.
 */
export default function AttentionBandCard({ status, onStatus }) {
  const [info, setInfo] = useState(null);
  const [on, setOn] = useState(false);
  const [blur, setBlur] = useState(String(DEFAULT_BLUR));
  const [wrong, setWrong] = useStoredState("attention.wrong", "the cat sat on the mat");
  const [right, setRight] = useStoredState("attention.right", "the bat sat on the mat");
  const [preview, setPreview] = useState(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const jobRunning = jobIsRunning(status);
  const modelKey = status ? `${status.kind}|${status.encoding}` : "";

  const load = useCallback(async () => {
    setError(null);
    try {
      const data = await api.attention();
      const band = data && data.attention;
      if (!band || typeof band !== "object") return;
      setInfo(band);
      setOn(Boolean(band.on));
      setBlur(String(band.on ? band.blur : band.default_blur ?? DEFAULT_BLUR));
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    setPreview(null);
    load();
  }, [modelKey, load]);

  const blurValue = clampBlur(blur);
  const n = info ? info.ngram : 3;
  const unit = info ? info.unit : "char";
  const weights = bandWeights(n, blurValue);
  const applies = Boolean(info && info.applies);
  const sample = (() => {
    const grams = preview && preview.wrong && Array.isArray(preview.wrong.grams) ? preview.wrong.grams : [];
    return grams.length ? grams[Math.floor(grams.length / 2)] : null;
  })();

  async function apply() {
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      const data = await api.setAttention(on ? { on: true, blur: blurValue } : { on: false });
      setNotice(`The band is now ${describeBand(data && data.attention)}. It is saved with the model when the model is saved.`);
      await load();
      if (onStatus) {
        const next = await api.status();
        if (next && typeof next === "object") onStatus(next);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function runPreview() {
    setBusy(true);
    setError(null);
    try {
      setPreview(await api.attentionPreview({ wrong, right, blur: blurValue }));
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card wide">
      <h2>Attention band</h2>
      <p className="muted">
        A reader&apos;s eye fixes on one point of a line: that point is sharp, and the letters either side of it
        blur with the distance. A gram is this model&apos;s fixation, and the band is how sharply each of its
        positions is seen - <b>1 at the centre</b>, falling to <b>1 − blur</b> at the first and last unit. With the
        band on, every unit a <b>correction</b> changes hands out one charge, shared among the grams that see it by
        how centrally each sees it: the gram with the change at its centre takes the most, and the judged-path
        verdict. Off, each changed unit is charged in full to the step that wrote it - the one whose gram has only
        just reached it, at its edge. A thumbs up, a thumbs down or a training pass marks every unit alike, and no
        band changes those.
      </p>
      {!info ? (
        <p className="muted">Asking the server about the band…</p>
      ) : !applies ? (
        <p className="muted">
          The {String((status && status.model_label) || (status && status.kind) || "active")} model is never
          corrected, so a band would have nothing to spread: it belongs to the count model and the negative
          network. The picture below is only what a band would look like over its grams.
        </p>
      ) : (
        <p>
          Now: <b>{describeBand(info)}</b>
        </p>
      )}
      <div className="row">
        <CheckField
          label="Band on"
          hint="off = the step that wrote a changed unit takes all of it"
          checked={on}
          onChange={setOn}
          disabled={busy || jobRunning || !applies}
        />
        <label className="field">
          <span>
            Blur <em>(how blurred the ends of a gram are: {fmtShare(blurValue)})</em>
          </span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={blurValue}
            onChange={(e) => setBlur(e.target.value)}
            disabled={busy || jobRunning}
          />
        </label>
      </div>
      <BandPicture weights={weights} gram={sample} unit={unit} />
      <p className="muted">
        {n === 1
          ? "A gram of one unit is all centre: every unit sits in one gram, which takes its whole charge - the band changes nothing."
          : n === 2
            ? "A gram of two units has no centre - both are ends - so the band is flat whatever the blur: the two grams that see a unit share it evenly."
            : info && info.stride === n
              ? "This encoding cuts the text into groups that share nothing: each unit sits in one gram, which takes its whole charge, so the band changes nothing here."
              : `Over a gram of ${n} ${unitWord(unit)}: the bars are the band at the blur above.`}
      </p>
      <div className="actions">
        <button type="button" className="primary" disabled={busy || jobRunning || !applies} onClick={apply}>
          {busy ? "Applying…" : on ? `Apply: band on, blur ${fmtShare(blurValue)}` : "Apply: band off"}
        </button>
        <button type="button" className="small" disabled={busy} onClick={load}>
          Reload from the model
        </button>
      </div>
      {jobRunning ? <p className="muted">A job is running; the band can be changed once it finishes.</p> : null}
      <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />

      <h3>Where does a correction land?</h3>
      <div className="row">
        <TextField label="The network wrote" value={wrong} onChange={setWrong} placeholder="the cat sat on the mat" />
        <TextField label="The teacher wrote" value={right} onChange={setRight} placeholder="the bat sat on the mat" />
      </div>
      <div className="actions">
        <button type="button" disabled={busy} onClick={runPreview}>
          {busy ? "Working…" : `Show it at blur ${fmtShare(blurValue)}`}
        </button>
      </div>
      {preview ? (
        <>
          <p className="muted">
            Nothing was changed: this is where the two rules would put the blame and the credit. A step through a
            merged node is charged what its grams add up to, at most one.
          </p>
          <PreviewSide title="Blamed" side={preview.wrong} unit={unit} weights={bandWeights(n, preview.blur)} />
          <PreviewSide title="Taught" side={preview.right} unit={unit} weights={bandWeights(n, preview.blur)} />
        </>
      ) : (
        <p className="muted">Type what the network wrote and what it should have written, and press the button.</p>
      )}
      <Alert message={error} onDismiss={() => setError(null)} />
    </div>
  );
}
