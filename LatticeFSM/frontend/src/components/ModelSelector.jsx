import { LMS } from "../lm.js";
import { fmtInt, fmtNum } from "../util.js";

/** "8 nodes of letters · confidence 0.90 · 12,345 tokens" - the chosen model's brief, from GET /api/status. */
function brief(lm, status) {
  const s = status && status[lm];
  if (!s) return status ? "not made yet - the Train tab makes one" : "";
  if (lm === "chain") {
    return `${s.nodes} nodes of ${s.unit}s · confidence ${fmtNum(s.confidence, 2)} · ${fmtInt(s.tokens)} tokens · ${fmtInt(
      s.vocab,
    )} distinct`;
  }
  return `${s.order}-grams of ${s.unit}s · ${fmtInt(s.tokens)} tokens · ${fmtInt(s.ngrams)} n-grams`;
}

/**
 * The language-model selector at the top of the page: the chain or the N-gram model. The choice is the
 * browser's (remembered in localStorage) - both models live on the server side by side, and every
 * language-model tab (Train, Predict, Generate, Talk, Converse, Chat, Score, Feedback) talks to the one chosen.
 */
export default function ModelSelector({ lm, onChange, status }) {
  const current = LMS.find((m) => m.id === lm) || LMS[0];
  const line = brief(current.id, status);
  return (
    <div className="model-selector" aria-live="polite">
      <label>
        <span>Language model</span>
        <select value={current.id} onChange={(e) => onChange(e.target.value)} title="The model every language-model tab talks to">
          {LMS.map((m) => (
            <option key={m.id} value={m.id}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      {line ? <span className="mono muted">{line}</span> : null}
      <span className="muted description">{current.description}</span>
    </div>
  );
}
