import { useState } from "react";
import { api } from "../api.js";
import { fmtInt, fmtNum } from "../util.js";
import Alert from "./Alert.jsx";
import { TextArea } from "./Fields.jsx";

/** Log-probability of a text under the model. */
export default function ScorePanel() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  async function handleSubmit(event) {
    event.preventDefault();
    if (text.trim() === "") {
      setError("Enter a text to score.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const data = await api.score({ text });
      setResult(data && typeof data === "object" ? data : {});
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const perChar = result && typeof result.per_char === "number" ? result.per_char : null;

  return (
    <>
      <form className="card" onSubmit={handleSubmit}>
        <h2>Score</h2>
        <TextArea label="Text" value={text} onChange={setText} rows={6} placeholder="the quick brown fox" />
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Scoring…" : "Score"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Result</h2>
        {!result ? (
          <p className="muted">
            Scores are sums of log transition probabilities along the text's path (START → … → END). Higher (closer to
            0) is more likely; unknown trigrams or missing edges count as log(1e-6).
          </p>
        ) : (
          <dl className="kv">
            <dt>log-probability</dt>
            <dd>{fmtNum(result.log_prob, 4)}</dd>
            <dt>per character</dt>
            <dd>{fmtNum(result.per_char, 4)}</dd>
            <dt>mean char probability</dt>
            <dd>{perChar === null ? "–" : fmtNum(Math.exp(perChar), 4)}</dd>
            <dt>characters</dt>
            <dd>{fmtInt(result.chars)}</dd>
            <dt>transitions</dt>
            <dd>{fmtInt(result.transitions)}</dd>
            <dt>unknown transitions</dt>
            <dd>{fmtInt(result.unknown_transitions)}</dd>
          </dl>
        )}
      </div>
    </>
  );
}
