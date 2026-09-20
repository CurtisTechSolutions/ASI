import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { asArray, fmtInt, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { NumberField } from "./Fields.jsx";

/**
 * The word model's alphabet (../../SPEC-WordNGrams.md). A word n-gram model is
 * the count / reward model over an alphabet whose symbols are words, and this
 * is that alphabet: every word training has read, in the order it first read
 * them, with how many of the graph's three-word windows the word appears in -
 * what the graph actually knows about it, and the one count compression cannot
 * change.
 *
 * The vocabulary grows as training reads new words and is never frozen, pruned
 * or learned: a word is in it because it was seen. A word the model has never
 * read is `<unk>` (id 0) at prediction and scoring time, and two different
 * unread words are the same symbol.
 */
export default function WordsPanel({ status }) {
  const [limit, setLimit] = useStoredState("words.limit", "50");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const load = useCallback(
    async (count) => {
      setLoading(true);
      setError(null);
      try {
        const data = await api.words(count);
        setResult(data && typeof data === "object" ? data : {});
      } catch (err) {
        setError(err.message);
        setResult(null);
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  // reload when the model changes under us: a training run grows the vocabulary
  const vocabulary = status ? status.vocabulary : null;
  useEffect(() => {
    load(parseInteger(limit, 50));
  }, [load, vocabulary]); // eslint-disable-line react-hooks/exhaustive-deps

  const rows = asArray(result && result.words);
  const total = result && typeof result.vocabulary === "number" ? result.vocabulary : null;
  const most = rows.reduce((best, row) => Math.max(best, Number(row && row.trigrams) || 0), 0);

  return (
    <>
      <form
        className="card"
        onSubmit={(event) => {
          event.preventDefault();
          load(parseInteger(limit, 50));
        }}
      >
        <h2>Vocabulary</h2>
        <p className="muted">
          A word model is this model over an alphabet whose symbols are words: one code point per word, the same
          three-symbol window, the same graph. The vocabulary grows as training reads new words and is never frozen,
          pruned or learned — a word is in it because it was seen. Whitespace separates words; punctuation stays
          attached to the word it touches and case is kept, so <code>mat.</code> and <code>mat</code> are two words.
        </p>
        <div className="row">
          <NumberField label="Show" hint="words, most read first (0 = all)" value={limit} onChange={setLimit} min={0} step={10} />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Words</h2>
        {total === null ? (
          <p className="muted">Select the word model and train it on a corpus.</p>
        ) : (
          <p className="muted">
            <b>{fmtInt(total)}</b> word{total === 1 ? "" : "s"} read, {fmtInt(rows.length)} shown. “Windows” is how many
            of the graph’s three-word windows hold the word.
          </p>
        )}
        {rows.length === 0 ? (
          <p className="muted">Nothing has been read yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th className="text">word</th>
                  <th className="num">id</th>
                  <th className="num">windows</th>
                  <th className="num">of the most read</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.id}>
                    <td className="text">
                      <code>{String(row.word)}</code>
                    </td>
                    <td className="num">{fmtInt(row.id)}</td>
                    <td className="num">{fmtInt(row.trigrams)}</td>
                    <td className="num">
                      <span className="rating-bar" title={`${fmtInt(row.trigrams)} of ${fmtInt(most)}`}>
                        <i style={{ width: most > 0 ? `${Math.round((Number(row.trigrams) / most) * 100)}%` : 0 }} />
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
