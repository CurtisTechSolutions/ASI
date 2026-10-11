import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { GUARD_UNITS, pct } from "../guard.js";
import { asArray, fmtInt, fmtNum, parseInteger, parseNumber, splitLines } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";
import Verdict from "./Verdict.jsx";

/**
 * The guard: a negative network. It records failures alone - a text blamed with a reason and a severity charges
 * every transition of it (the last few tokens, then the next) under that reason - and good text clears what it
 * crosses. It judges any text by the evidence against its transitions: risk (evidence per transition), coverage
 * (the share of its transitions that are known failures) and its verdict - reject (coverage at least the
 * minimum and risk at least the threshold), suspect (some known failure, under the limit) or pass. Every
 * other tab can put it in front of what a model writes ("Guard"); here it is judged, taught and tuned by hand.
 */
export default function GuardPanel({ onMoved }) {
  const [info, setInfo] = useState(null);
  const [judgeText, setJudgeText] = useStoredState("guard.judgeText", "");
  const [verdicts, setVerdicts] = useState([]);
  const [blameText, setBlameText] = useStoredState("guard.blameText", "");
  const [reason, setReason] = useStoredState("guard.reason", "gibberish");
  const [severity, setSeverity] = useStoredState("guard.severity", "1");
  const [clearText, setClearText] = useStoredState("guard.clearText", "");
  const [amount, setAmount] = useStoredState("guard.amount", "1");
  const [threshold, setThreshold] = useStoredState("guard.threshold", "1");
  const [minCoverage, setMinCoverage] = useStoredState("guard.minCoverage", "0.5");
  const [unit, setUnit] = useStoredState("guard.unit", "word");
  const [order, setOrder] = useStoredState("guard.order", "3");
  const [path, setPath] = useStoredState("guard.path", "guard.json.gz");
  const [armed, setArmed] = useState(false); // "Forget everything" waits for a second click
  const [notice, setNotice] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setInfo(await api.guard());
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  /** Run `fn`; what it answers with is the notice, and a fresh `guard` (or stats) in it replaces the shown one. */
  async function act(fn, moved = true) {
    setBusy(true);
    setNotice(null);
    try {
      const r = await fn();
      setError(null);
      if (moved && onMoved) onMoved();
      return r;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setBusy(false);
    }
  }

  const judges = splitLines(judgeText);
  const blames = splitLines(blameText);
  const clears = splitLines(clearText);

  const judge = (event) => {
    event.preventDefault();
    act(async () => {
      const r = await api.guardJudge(judges);
      setVerdicts(asArray(r && r.verdicts));
      await refresh();
    }, false);
  };

  const blame = (event) => {
    event.preventDefault();
    act(async () => {
      const r = await api.guardBlame(blames, reason.trim(), Math.max(0, parseNumber(severity, 1)));
      setInfo(r.guard);
      setNotice(`blamed ${fmtInt(r.blamed)} transitions of ${blames.length} text${blames.length === 1 ? "" : "s"} as "${reason.trim() || "unspecified"}"`);
    });
  };

  const clear = (event) => {
    event.preventDefault();
    act(async () => {
      const r = await api.guardClear(clears, Math.max(0, parseNumber(amount, 1)));
      setInfo(r.guard);
      setNotice(`cleared ${fmtInt(r.cleared)} known transitions of ${clears.length} text${clears.length === 1 ? "" : "s"}`);
    });
  };

  const settings = (event) => {
    event.preventDefault();
    act(async () => {
      const r = await api.guardSettings({
        threshold: Math.max(0, parseNumber(threshold, 1)),
        min_coverage: Math.min(1, Math.max(0, parseNumber(minCoverage, 0.5))),
      });
      setInfo(r);
      setNotice(`a text is now rejected at risk ${fmtNum(r.threshold, 2)} and coverage ${pct(r.min_coverage)}`);
    });
  };

  const fresh = () =>
    act(async () => {
      const r = await api.guardNew({
        unit,
        order: Math.max(1, parseInteger(order, 3)),
        threshold: Math.max(0, parseNumber(threshold, 1)),
        min_coverage: Math.min(1, Math.max(0, parseNumber(minCoverage, 0.5))),
      });
      setInfo(r);
      setVerdicts([]);
      setNotice(`a new guard of ${r.unit}s, order ${r.order}: it knows no failure yet`);
    });

  const forget = (name) =>
    act(async () => {
      const r = await api.guardForget(name);
      setInfo(r.guard);
      setNotice(`forgot "${name}": ${fmtInt(r.forgotten)} transitions`);
    });

  const reset = () =>
    act(async () => {
      setInfo(await api.guardReset());
      setVerdicts([]);
      setNotice("forgot every failure");
    });

  const reasons = asArray(info && info.reasons);
  const journal = asArray(info && info.journal);
  const totalBlame = reasons.reduce((sum, r) => sum + (Number(r.blame) || 0), 0) || 1;
  return (
    <div className="panel">
      <div className="card wide">
        <h2>The guard</h2>
        <p className="note">
          A negative network: it learns nothing but failures. A text blamed with a reason charges every transition in
          it (the last {info ? info.order - 1 : "few"} tokens, then the next) under that reason; good text clears
          what it crosses. Any text is judged by the evidence left against its transitions - its risk (evidence per
          transition) and coverage (the share of its transitions that are known failures) - and rejected when both
          reach the limits, suspect when some of it is a known failure, passed when none is. Tick "Guard" on the
          Generate, Talk, Converse, Chat, Feedback or Network tab to put it in front of what a model writes: a
          rejected sample is vetoed and another drawn in its place.
        </p>
        {info ? (
          <p className="mono">
            {info.unit}s, order <b>{info.order}</b> · rejects at risk <b>{fmtNum(info.threshold, 2)}</b> and coverage{" "}
            <b>{pct(info.min_coverage)}</b> · {fmtInt(info.transitions)} transitions known · evidence{" "}
            {fmtNum(info.evidence, 2)} · {fmtInt(info.blamed)} texts blamed, {fmtInt(info.cleared)} cleared ·{" "}
            {fmtInt(info.judgements)} judged, {fmtInt(info.rejected)} rejected
          </p>
        ) : null}
        <div className="actions">
          <TextField label="File" value={path} onChange={setPath} hint="on the server" />
          <button
            type="button"
            disabled={busy || !path.trim()}
            onClick={() => act(async () => setNotice(`saved the guard to ${(await api.guardSave(path.trim())).saved ?? path}`), false)}
          >
            Save
          </button>
          <button
            type="button"
            disabled={busy || !path.trim()}
            onClick={() =>
              act(async () => {
                setInfo(await api.guardLoad(path.trim()));
                setVerdicts([]);
                setNotice(`loaded the guard from ${path.trim()}`);
              })
            }
          >
            Load
          </button>
          <button type="button" disabled={busy} onClick={refresh}>
            Refresh
          </button>
        </div>
        <Alert kind="ok" message={notice} onDismiss={() => setNotice(null)} />
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <form className="card" onSubmit={judge}>
        <h2>Judge</h2>
        <p className="muted">One text a line: each judged on its own, the blamed tokens marked inside it.</p>
        <TextArea label={`Texts (${judges.length})`} value={judgeText} onChange={setJudgeText} rows={6} />
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || judges.length === 0}>
            Judge
          </button>
          <button type="button" disabled={verdicts.length === 0} onClick={() => setVerdicts([])}>
            Clear verdicts
          </button>
        </div>
      </form>

      <div className="card">
        <h2>Verdicts</h2>
        {verdicts.length === 0 ? (
          <p className="muted">Judge a text or two.</p>
        ) : (
          <div className="verdicts">
            {verdicts.map((v, i) => (
              <Verdict key={i} verdict={v} unit={info ? info.unit : undefined} />
            ))}
          </div>
        )}
      </div>

      <form className="card" onSubmit={blame}>
        <h2>Blame</h2>
        <p className="muted">Failures, one a line: every transition in them is charged the severity under the reason.</p>
        <TextArea label={`Texts (${blames.length})`} value={blameText} onChange={setBlameText} rows={5} />
        <div className="row">
          <TextField label="Reason" value={reason} onChange={setReason} hint="why they failed" />
          <NumberField label="Severity" value={severity} onChange={setSeverity} min="0" step="0.1" />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || blames.length === 0}>
            Blame
          </button>
        </div>
      </form>

      <form className="card" onSubmit={clear}>
        <h2>Clear</h2>
        <p className="muted">
          Good text, one a line: every transition in it the guard knows is cleared by the amount, so a phrase a
          failure shares with good text stops counting against it.
        </p>
        <TextArea label={`Texts (${clears.length})`} value={clearText} onChange={setClearText} rows={5} />
        <NumberField label="Amount" value={amount} onChange={setAmount} min="0" step="0.1" />
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || clears.length === 0}>
            Clear
          </button>
        </div>
      </form>

      <form className="card wide" onSubmit={settings}>
        <h2>Settings</h2>
        <p className="muted">
          A text is rejected when its coverage reaches the minimum and its risk the threshold. A new guard starts with
          no failures, reading words or characters, a transition being the last order - 1 tokens and the next.
        </p>
        <div className="row">
          <NumberField label="Threshold" value={threshold} onChange={setThreshold} min="0" step="0.1" hint="risk: evidence per transition" />
          <NumberField label="Min coverage" value={minCoverage} onChange={setMinCoverage} min="0" max="1" step="0.05" hint="0 to 1" />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={busy}>
            Apply
          </button>
        </div>
        <div className="row">
          <SelectField label="Unit" value={unit} onChange={setUnit} options={GUARD_UNITS} />
          <NumberField label="Order" value={order} onChange={setOrder} min="1" step="1" hint="tokens in a transition" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy} onClick={fresh}>
            New guard
          </button>
        </div>
      </form>

      <div className="card wide">
        <div className="toolbar">
          <h2>Reasons</h2>
          <button
            type="button"
            className={armed ? "danger" : ""}
            disabled={busy}
            onBlur={() => setArmed(false)}
            onClick={() => {
              if (!armed) {
                setArmed(true);
                return;
              }
              setArmed(false);
              reset();
            }}
          >
            {armed ? "Forget every failure? click again" : "Forget everything"}
          </button>
        </div>
        {reasons.length === 0 ? (
          <p className="muted">No failures yet: blame a text, or tick "teach the guard" on another tab.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>reason</th>
                  <th>blame</th>
                  <th>share</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {reasons.map((r) => (
                  <tr key={r.reason}>
                    <td className="text">{r.reason}</td>
                    <td>{fmtNum(r.blame, 2)}</td>
                    <td>{pct((Number(r.blame) || 0) / totalBlame)}</td>
                    <td className="row-actions">
                      <button type="button" className="small" disabled={busy} onClick={() => forget(r.reason)}>
                        forget
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="card wide">
        <h2>Journal</h2>
        <p className="muted">The texts blamed last, newest first.</p>
        {journal.length === 0 ? (
          <p className="muted">Nothing blamed yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>text</th>
                  <th>reason</th>
                  <th>severity</th>
                  <th>transitions</th>
                </tr>
              </thead>
              <tbody>
                {journal.map((j, i) => (
                  <tr key={i}>
                    <td className="text">{j.text}</td>
                    <td>{j.reason}</td>
                    <td>{fmtNum(j.severity, 2)}</td>
                    <td>{fmtInt(j.transitions)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
