import { useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { useLmInfo } from "../hooks/useLmInfo.js";
import { depthStyle, lmLabel, mean, probStyle } from "../lm.js";
import { showToken } from "../matrix.js";
import { asArray, fmtNum } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, TextArea } from "./Fields.jsx";

/**
 * Score a text (POST /api/{m}/score): the probability the chosen model gave each token of it as it read it,
 * shown token by token - the likelier, the stronger the tint, so a surprise stands out pale - with the
 * perplexity and the bits per token. For the chain each token can be tinted by how deep its walk went instead.
 */
export default function ScorePanel({ lm, tick }) {
  const [info] = useLmInfo(lm, tick);
  const [text, setText] = useStoredState("score.text", "the brain is a network of cells");
  const [byDepth, setByDepth] = useStoredState("score.byDepth", false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => setResult(null), [lm]);

  async function score(event) {
    event.preventDefault();
    setBusy(true);
    try {
      setResult(await api.lmScore(lm, text));
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const tokens = asArray(result && result.tokens);
  const maxDepth = info && info.nodes ? info.nodes - 1 : 8;
  const depths = tokens.map((t) => t.depth);
  const hasDepth = depths.some((d) => d !== undefined && d !== null);
  const tintDepth = byDepth && hasDepth;
  const unit = result ? result.unit : info ? info.unit : "word";
  const lowest = tokens.length ? tokens.reduce((a, b) => (Number(b.p) < Number(a.p) ? b : a)) : null;
  return (
    <div className="panel">
      <form className="card" onSubmit={score}>
        <h2>Score</h2>
        <p className="muted">How likely the {lmLabel(lm)} model finds a text, token by token, as it reads it.</p>
        <TextArea label="Text" value={text} onChange={setText} rows={6} />
        {lm === "chain" ? <CheckField label="Tint by depth" checked={byDepth} onChange={setByDepth} hint="how deep the walk went, instead of the probability" /> : null}
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || !text.trim()}>
            {busy ? "Scoring…" : "Score"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Token by token</h2>
        {!result ? (
          <p className="muted">Press Score.</p>
        ) : (
          <>
            <dl className="kv">
              <dt>perplexity</dt>
              <dd>{fmtNum(result.perplexity, 3)}</dd>
              <dt>bits per token</dt>
              <dd>{fmtNum(result.bits, 3)}</dd>
              <dt>log-probability</dt>
              <dd>{fmtNum(result.logp, 3)}</dd>
              <dt>tokens</dt>
              <dd>{tokens.length}</dd>
              {hasDepth ? (
                <>
                  <dt>mean depth</dt>
                  <dd>
                    {fmtNum(mean(depths), 2)} of {maxDepth}
                  </dd>
                </>
              ) : null}
              {lowest ? (
                <>
                  <dt>least expected</dt>
                  <dd className="mono">
                    {showToken(lowest.token, unit)} ({fmtNum(Number(lowest.p) * 100, 3)}%)
                  </dd>
                </>
              ) : null}
            </dl>
            <p className="text-display score-tokens">
              {tokens.map((t, i) => (
                <span
                  key={i}
                  className="tok boxed"
                  style={tintDepth ? depthStyle(t.depth, maxDepth) : probStyle(t.p)}
                  title={`p ${fmtNum(Number(t.p) * 100, 3)}%${t.depth !== undefined && t.depth !== null ? ` · ${t.depth} nodes deep` : ""}`}
                >
                  {showToken(t.token, unit)}
                  {hasDepth ? <sub>{t.depth}</sub> : null}
                </span>
              ))}
            </p>
            <p className="muted">
              {tintDepth
                ? "The darker, the deeper the chain walked to predict the token."
                : "The darker, the likelier the model found the token (on a log scale: a token it gave 1 in 10,000 or less is untinted)."}
            </p>
          </>
        )}
      </div>
    </div>
  );
}
