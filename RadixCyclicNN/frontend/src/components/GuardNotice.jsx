import { useState } from "react";
import { asArray, fmtInt } from "../util.js";
import { Verdict } from "./NegativePanel.jsx";

/**
 * What the guard stopped on the way out.
 *
 * Every answer the model writes goes through the pair: the positive model
 * writes and the negative network - built from nothing but the tutor's
 * failures - vetoes what it recognises. `guard` is the report that comes back
 * with the answer; `null` means nothing guarded it (no negative network, or
 * one that has never been taught a failure), and then this shows nothing at
 * all. Open it to read why each vetoed candidate was dropped, with the blamed
 * fragments marked inside it - unless the answer was asked for without the
 * provenance, when the report is the counts alone.
 */
export default function GuardNotice({ guard, what = "candidates" }) {
  const [open, setOpen] = useState(false);
  if (!guard || !guard.on) return null;
  // without provenance the report is the counts alone: nothing to open
  const silent = guard.provenance === false;
  const rejected = silent ? [] : asArray(guard.rejected);
  const vetoed = silent ? Number(guard.vetoed) || 0 : rejected.length;
  const judged = silent ? Number(guard.judged) || 0 : asArray(guard.verdicts).length;
  return (
    <div className={`guard ${vetoed ? "vetoed" : "clear"}`}>
      <p className="muted">
        <b>Guard</b> · the negative network judged {fmtInt(judged)} {what} and vetoed {fmtInt(vetoed)}
        {Number.isFinite(guard.asked) ? ` (${fmtInt(guard.asked)} asked for)` : ""}
        {silent && vetoed ? " · without saying why (provenance off)" : ""}
        {rejected.length ? (
          <>
            {" · "}
            <button type="button" className="link" onClick={() => setOpen(!open)}>
              {open ? "hide" : "why"}
            </button>
          </>
        ) : null}
      </p>
      {open && rejected.length ? (
        <div className="verdicts">
          {rejected.map((verdict, i) => (
            <Verdict verdict={verdict} key={i} />
          ))}
        </div>
      ) : null}
    </div>
  );
}
