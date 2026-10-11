import { useState } from "react";
import { guardNoticeText } from "../guard.js";
import { asArray, fmtNum } from "../util.js";

/**
 * What the guard stopped on the way out. A guarded draw comes back with the guard's summary (`{judged, vetoed,
 * all_vetoed, vetoes}`): it judged every sample the model wrote (over-sampled to make up for the vetoes) and
 * vetoed the ones it rejected; when it rejected all of them the least risky was kept. Open "why" to read each
 * vetoed one with its risk and the guard's sentence. Nothing at all without a summary.
 */
export default function GuardNotice({ guard, what = "samples" }) {
  const [open, setOpen] = useState(false);
  if (!guard || typeof guard !== "object") return null;
  const vetoes = asArray(guard.vetoes);
  const vetoed = Number(guard.vetoed) || 0;
  return (
    <div className={`guard ${vetoed ? "vetoed" : "clear"}`}>
      <p className="muted">
        <b>Guard</b> · {guardNoticeText(guard, what)}
        {vetoes.length ? (
          <>
            {" · "}
            <button type="button" className="link guard-why" onClick={() => setOpen(!open)}>
              {open ? "hide" : "why"}
            </button>
          </>
        ) : null}
      </p>
      {open && vetoes.length ? (
        <ul className="vetoes">
          {vetoes.map((v, i) => (
            <li key={i}>
              <span className="mono">{v.completion || "(nothing)"}</span>{" "}
              <span className="muted">
                risk {fmtNum(v.risk, 2)}
                {v.why ? ` · ${v.why}` : ""}
              </span>
            </li>
          ))}
          {vetoed > vetoes.length ? <li className="muted">and {vetoed - vetoes.length} more</li> : null}
        </ul>
      ) : null}
    </div>
  );
}
