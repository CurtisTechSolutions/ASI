import { useState } from "react";
import { highlightPieces, markAlpha, pct, verdictClass } from "../guard.js";
import { asArray, fmtNum } from "../util.js";

/**
 * A judged text with its blamed spans marked inside it: each token a span covers tinted by the evidence against
 * it (darker the nearer the text's worst), the reason in its tooltip. Spans that cannot be laid on the text (a
 * guard of sounds, say) are listed beneath it as their tokens instead.
 */
export function Highlighted({ text, spans, unit, peak }) {
  const list = asArray(spans);
  const laid = list.length ? highlightPieces(text, list, unit) : null;
  const top = Number.isFinite(Number(peak)) && Number(peak) > 0 ? Number(peak) : Math.max(0, ...list.map((s) => Number(s.evidence) || 0));
  return (
    <>
      <p className="blamed mono">
        {laid
          ? laid.pieces.map((p, i) =>
              p.mark ? (
                <mark
                  key={i}
                  style={{ background: `rgba(220, 38, 38, ${markAlpha(p.mark.evidence, top)})` }}
                  title={`${p.mark.reasons.join(", ")} · evidence ${fmtNum(p.mark.evidence, 2)}`}
                >
                  {p.text}
                </mark>
              ) : (
                <span key={i}>{p.text}</span>
              ),
            )
          : text || <span className="muted">(nothing)</span>}
      </p>
      {list.length && !laid ? (
        <p className="chips">
          {list.map((s, i) => (
            <span className="chip bad" key={i} title={`${s.reason || "blamed"} · evidence ${fmtNum(s.evidence, 2)}`}>
              {asArray(s.tokens).join(" ")} → {s.next}
            </span>
          ))}
        </p>
      ) : null}
    </>
  );
}

/**
 * One verdict of the guard: the badge (reject / suspect / pass), its risk (evidence per transition), coverage
 * (the share of its transitions that are known failures) and peak, the sentence why, the reasons with their
 * shares, and the text with its blamed spans marked. `text` stands in when the verdict (a brief one, with a
 * generated sample) does not carry its own.
 */
export default function Verdict({ verdict, text, unit }) {
  if (!verdict) return null;
  const reasons = asArray(verdict.reasons);
  const said = String(verdict.text ?? text ?? "");
  return (
    <div className="verdict">
      <p>
        <span className={`badge ${verdictClass(verdict.verdict)}`}>{verdict.verdict || "?"}</span>{" "}
        <span className="muted">
          risk {fmtNum(verdict.risk, 2)} · coverage {pct(verdict.coverage)}
          {verdict.peak !== undefined ? ` · peak ${fmtNum(verdict.peak, 2)}` : ""}
          {verdict.transitions !== undefined ? ` · ${verdict.blamed ?? 0} of ${verdict.transitions} transitions blamed` : ""}
        </span>
      </p>
      <Highlighted text={said} spans={verdict.spans} unit={unit} peak={verdict.peak} />
      {verdict.why ? <p className="critique">{verdict.why}</p> : null}
      {reasons.length ? (
        <p className="chips">
          {reasons.map((r) => (
            <span className="chip bad" key={r.reason} title={`blame ${fmtNum(r.blame, 2)}`}>
              {r.reason} <small>{pct(r.share)}</small>
            </span>
          ))}
        </p>
      ) : null}
    </div>
  );
}

/**
 * The guard's verdict on one reply, small: its badge (the sentence why in the tooltip), how many were vetoed to
 * get to it, and a "why" that opens the whole verdict. Nothing when the reply was not guarded.
 */
export function GuardBadge({ guard, text, vetoed, unit }) {
  const [open, setOpen] = useState(false);
  if (!guard || typeof guard !== "object") return null;
  const n = typeof vetoed === "boolean" ? null : Number(vetoed) || 0;
  return (
    <>
      <span className={`badge tight ${verdictClass(guard.verdict)}`} title={guard.why || "the guard's verdict"}>
        {guard.verdict || "?"}
      </span>
      {vetoed === true ? <span className="muted">vetoed by the guard</span> : null}
      {n ? <span className="muted">{n} vetoed</span> : null}
      <button type="button" className="link guard-why" onClick={() => setOpen(!open)}>
        {open ? "hide" : "why"}
      </button>
      {open ? (
        <div className="guard-open">
          <Verdict verdict={guard} text={text} unit={unit} />
        </div>
      ) : null}
    </>
  );
}
