import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { appendToken, showToken } from "../matrix.js";
import { fmtNum, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import LineChart from "./LineChart.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";

const UNITS = [
  ["word", "words"],
  ["char", "characters"],
  ["sound", "sounds (phonemes, spelled back as English)"],
  ["base64", "base64 letters (any file: text, images, audio, video)"],
];

/** One round of training as the history records it, for the table and the chart. */
function RoundRow({ r }) {
  return (
    <tr>
      <td>{r.round}</td>
      <td>{r.tokens !== undefined ? r.tokens : "-"}</td>
      <td>{r.perplexity === null || r.perplexity === undefined ? "-" : fmtNum(r.perplexity, 2)}</td>
      <td>{r.accuracy === null || r.accuracy === undefined ? "-" : fmtNum(r.accuracy * 100, 1) + "%"}</td>
      <td className="muted">
        {["corpus_lines", "ollama_lines", "continuations", "judged", "approved", "source"]
          .filter((k) => r[k] !== undefined)
          .map((k) => `${k.replace("_", " ")} ${r[k]}`)
          .join(" · ")}
      </td>
    </tr>
  );
}

/**
 * Predict the next thing: the N-gram model the server holds. Type, and the likeliest next tokens are listed as you
 * go - click one to take it; continue the text; reward or punish what it wrote; train it on a corpus or on pasted
 * text, or let it train itself with a local language model through Ollama, round after round.
 */
export default function PredictPanel({ onMoved }) {
  const [info, setInfo] = useState(null);
  const [text, setText] = useStoredState("predict.text", "the brain is");
  const [k, setK] = useStoredState("predict.k", "10");
  const [maxTokens, setMaxTokens] = useStoredState("predict.max", "30");
  const [temperature, setTemperature] = useStoredState("predict.temperature", "0");
  const [order, setOrder] = useStoredState("predict.order", "3");
  const [unit, setUnit] = useStoredState("predict.unit", "word");
  const [data, setData] = useStoredState("predict.data", "data/corpus");
  const [paste, setPaste] = useStoredState("predict.paste", "");
  const [useOllama, setUseOllama] = useStoredState("predict.ollama", true);
  const [model, setModel] = useStoredState("predict.model", "");
  const [topics, setTopics] = useStoredState("predict.topics", "");
  const [rounds, setRounds] = useStoredState("predict.rounds", "3");
  const [path, setPath] = useStoredState("predict.path", "ngram.json.gz");
  const [next, setNext] = useState(null);
  const [completion, setCompletion] = useState(null);
  const [ollama, setOllama] = useState(null);
  const [log, setLog] = useState(null);
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const typing = useRef(0);

  useEffect(() => {
    api.ngram().then(setInfo).catch((err) => setError(err.message));
  }, []);

  // the next tokens follow the text as it is typed (a moment after the last key)
  useEffect(() => {
    if (!info) return undefined;
    window.clearTimeout(typing.current);
    typing.current = window.setTimeout(() => {
      api
        .ngramPredict(text, parseNumber(k, 10))
        .then((r) => {
          setNext(r);
          setError(null);
        })
        .catch((err) => setError(err.message));
    }, 150);
    return () => window.clearTimeout(typing.current);
  }, [text, k, info]);

  async function act(fn) {
    setBusy(true);
    try {
      await fn();
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  const refresh = async (r) => {
    setInfo(r && r.model ? r.model : await api.ngram());
    if (onMoved) onMoved();
  };

  const generate = () =>
    act(async () => {
      setCompletion(
        await api.ngramGenerate({
          text,
          max_tokens: parseNumber(maxTokens, 30),
          temperature: parseNumber(temperature, 0),
          seed: Math.floor(Math.random() * 1e9),
        }),
      );
    });

  const credit = (amount) =>
    act(async () => {
      const written = completion.unit === "base64" ? completion.completion : `${text} ${completion.completion}`;
      const r = await api.ngramCredit(written, amount);
      setNote(`${amount > 0 ? "rewarded" : "punished"} ${r.credited} edges of what it wrote`);
      await refresh(r);
    });

  const maxPerplexity = info && info.history ? info.history.filter((r) => r.perplexity != null) : [];
  const unitName = info ? (info.unit === "base64" ? "base64 letter" : info.unit) : "token";
  return (
    <div className="panel">
      <div className="card wide">
        <h2>Predict the next thing</h2>
        <p className="note">
          An N-gram model over the lattice's adaptive edges: the context is the last {info ? info.order - 1 : "N-1"}{" "}
          {unitName}s, the next one is predicted by interpolated Kneser-Ney, and every edge you reward or punish is
          widened or narrowed - the wider, the likelier, the more stimulated the model is.
        </p>
        {info ? (
          <p className="mono">
            {info.order}-grams of {unitName}s · {info.tokens} tokens trained · {info.vocab} distinct · {info.ngrams}{" "}
            n-grams · {info.credited_edges} credited edges
          </p>
        ) : null}
        <TextArea label="Text" value={text} onChange={setText} rows={3} hint="the next token follows it as you type" />
        {next && next.next ? (
          <div className="next-tokens">
            {next.next.map(({ token, p }) => (
              <button
                key={token}
                type="button"
                className="next-token"
                disabled={next.unit === "base64" || token === "</s>"}
                title={token === "</s>" ? "the end of the sequence" : `take ${showToken(token, next.unit)}`}
                onClick={() => setText(appendToken(text, token, next.unit))}
              >
                <span className="next-bar" style={{ width: `${Math.max(2, Math.round(p * 100))}%` }} />
                <span className="next-label mono">{showToken(token, next.unit)}</span>
                <span className="next-p mono">{fmtNum(p * 100, 1)}%</span>
              </button>
            ))}
          </div>
        ) : null}
        <div className="row">
          <NumberField label="Show" value={k} onChange={setK} min="1" max="50" step="1" hint="next tokens" />
          <NumberField label="Write up to" value={maxTokens} onChange={setMaxTokens} min="1" step="1" hint="tokens" />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min="0" step="0.1" hint="0: the likeliest every time" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy} onClick={generate}>
            Continue it
          </button>
          {completion ? (
            <>
              <button type="button" disabled={busy} onClick={() => credit(1)}>
                Good - reward it
              </button>
              <button type="button" disabled={busy} onClick={() => credit(-1)}>
                Bad - punish it
              </button>
            </>
          ) : null}
        </div>
        {completion ? (
          <p className="text-display">
            {completion.unit === "base64" ? (
              completion.completion
            ) : (
              <>
                <span className="muted">{text} </span>
                <strong>{completion.completion || "(it would end here)"}</strong>
              </>
            )}
          </p>
        ) : null}
        {note ? <p className="note">{note}</p> : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <div className="card wide">
        <h2>Train it</h2>
        <div className="row">
          <SelectField label="Unit" value={unit} onChange={setUnit} options={UNITS} />
          <NumberField label="N" value={order} onChange={setOrder} min="1" max="12" step="1" hint="the context is N-1 units" />
          <TextField label="Data" value={data} onChange={setData} hint="files or directories on the server, comma-separated" />
        </div>
        <div className="actions">
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              act(async () => {
                setInfo(await api.ngramNew({ order: parseNumber(order, 3), unit }));
                setNext(null);
                setCompletion(null);
              })
            }
          >
            New model
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              act(async () => {
                const r = await api.ngramTrain({ data });
                setNote(
                  `trained ${r.trained} tokens; held out: perplexity ${fmtNum(r.held_out.perplexity, 2)}, next-token accuracy ${fmtNum(
                    (r.held_out.accuracy || 0) * 100,
                    1,
                  )}%`,
                );
                await refresh(r);
              })
            }
          >
            Train on the data
          </button>
        </div>
        <TextArea label="Or paste text" value={paste} onChange={setPaste} rows={3} hint="every line one sequence" />
        <div className="actions">
          <button
            type="button"
            disabled={busy || !paste.trim()}
            onClick={() =>
              act(async () => {
                const r = await api.ngramText(paste);
                setNote(`trained ${r.trained} tokens of pasted text`);
                await refresh(r);
              })
            }
          >
            Train on this text
          </button>
          <TextField label="File" value={path} onChange={setPath} />
          <button type="button" disabled={busy} onClick={() => act(async () => setNote(`saved ${(await api.ngramSave(path)).saved}`))}>
            Save
          </button>
          <button type="button" disabled={busy} onClick={() => act(async () => setInfo(await api.ngramLoad(path)))}>
            Load
          </button>
        </div>
      </div>

      <div className="card wide">
        <h2>Train itself with Ollama</h2>
        <p className="note">
          Round 0 trains a fresh model on the data, every tenth line held out. Each round after it a local language model
          (Ollama) writes sentences on the corpus's own topics, continues the model's own contexts, and judges texts the
          model wrote - what it approves is rewarded, what it rejects punished - and the model is measured on the held-out
          lines again; it stops when that stops improving.
        </p>
        <div className="row">
          <CheckField label="Use Ollama" checked={useOllama} onChange={setUseOllama} hint="off: the data only" />
          <TextField label="Model" value={model} onChange={setModel} hint="blank: $LATTICEFSM_OLLAMA_MODEL or llama3.2" />
          <TextField label="Topics" value={topics} onChange={setTopics} hint="comma-separated; blank: the corpus's own words" />
          <NumberField label="Rounds" value={rounds} onChange={setRounds} min="0" max="50" step="1" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy} onClick={() => act(async () => setOllama(await api.ngramOllama(model)))}>
            Is Ollama there?
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() =>
              act(async () => {
                const r = await api.ngramAutotrain({
                  data,
                  ollama: useOllama,
                  model: model || undefined,
                  topics,
                  rounds: parseNumber(rounds, 3),
                });
                setLog(r.log);
                await refresh(r);
              })
            }
          >
            {busy ? "Training…" : "Auto-train"}
          </button>
        </div>
        {ollama ? (
          <p className={`note${ollama.available ? "" : " error-text"}`}>
            {ollama.available
              ? `Ollama at ${ollama.url}: ${ollama.models.join(", ") || "no models pulled"}; it would use ${ollama.model}`
              : ollama.error}
          </p>
        ) : null}
        {log ? <pre className="mono run-log">{log.join("\n")}</pre> : null}
        {info && info.history && info.history.length ? (
          <>
            {maxPerplexity.length > 1 ? (
              <LineChart
                series={[
                  {
                    name: "held-out perplexity",
                    color: "var(--accent)",
                    values: maxPerplexity.map((r, i) => ({ x: i, y: r.perplexity })),
                  },
                ]}
                xLabel="round"
                yLabel="perplexity"
                height={200}
              />
            ) : null}
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>round</th>
                    <th>tokens</th>
                    <th>held-out perplexity</th>
                    <th>next-token accuracy</th>
                    <th>what it learned from</th>
                  </tr>
                </thead>
                <tbody>
                  {info.history.map((r, i) => (
                    <RoundRow key={i} r={r} />
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
