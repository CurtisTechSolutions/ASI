import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { historyPoints, lmLabel, pct } from "../lm.js";
import { asArray, fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, TextField } from "./Fields.jsx";
import JobProgress from "./JobProgress.jsx";
import LineChart from "./LineChart.jsx";

/** "corpus 12 · ollama lines 40 · approved 3": what an auto-training round learned from. */
function roundSources(r) {
  return ["corpus_lines", "ollama_lines", "continuations", "judged", "approved", "source"]
    .filter((k) => r[k] !== undefined)
    .map((k) => `${k.replace("_", " ")} ${r[k]}`)
    .join(" · ");
}

/**
 * A local language model through Ollama, as a source of text and a teacher: is it there (and which models
 * has it pulled); have it write a corpus from a prompt, to train the chosen model on or keep as an upload;
 * and let the N-gram model train itself with it, round after round (auto-train, moved here from Predict).
 */
export default function OllamaPanel({ lm, onMoved }) {
  const [model, setModel] = useStoredState("predict.model", "");
  const [url, setUrl] = useStoredState("ollama.url", "");
  const [prompt, setPrompt] = useStoredState("ollama.prompt", "Write short simple sentences about the brain, one a line.");
  const [lines, setLines] = useStoredState("ollama.lines", "20");
  const [uploadAs, setUploadAs] = useStoredState("ollama.uploadAs", "ollama-corpus.txt");
  const [useOllama, setUseOllama] = useStoredState("predict.ollama", true);
  const [topics, setTopics] = useStoredState("predict.topics", "");
  const [rounds, setRounds] = useStoredState("predict.rounds", "3");
  const [data, setData] = useStoredState("predict.data", "data/corpus");
  const [predictFirst, setPredictFirst] = useStoredState("predict.first", true);
  const [selfCredit, setSelfCredit] = useStoredState("predict.selfCredit", "0");
  const [check, setCheck] = useState(null);
  const [corpus, setCorpus] = useState(null);
  const [auto, setAuto] = useState(null); // {log, history}
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [job, setJob] = useState(null); // a running job's progress (writing, training, auto-training)

  async function act(name, fn) {
    setBusy(name);
    try {
      await fn();
      setError(null);
    } catch (err) {
      setError(err.data && err.data.state === "stopped" ? "stopped; what was learned until then is kept" : err.message);
    } finally {
      setBusy(null);
      setJob(null);
    }
  }

  const ollama = () => ({
    ...(model.trim() ? { ollama_model: model.trim() } : {}),
    ...(url.trim() ? { url: url.trim() } : {}),
  });

  const corpusLines = asArray(corpus && corpus.lines);
  const history = asArray(auto && auto.history);
  const perplexity = historyPoints(history, "perplexity");
  return (
    <div className="panel">
      <div className="card wide">
        <h2>Ollama</h2>
        <p className="note">A local language model (ollama.com) the server talks to: a writer of text and a teacher.</p>
        <div className="row">
          <TextField label="Model" value={model} onChange={setModel} hint="blank: $LATTICEFSM_OLLAMA_MODEL or llama3.2" />
          <TextField label="URL" value={url} onChange={setUrl} hint="blank: the server's default (localhost:11434)" />
        </div>
        <div className="actions">
          <button type="button" disabled={busy !== null} onClick={() => act("check", async () => setCheck(await api.ngramOllama(model, url)))}>
            {busy === "check" ? "Asking…" : "Is Ollama there?"}
          </button>
          {check ? (
            <span className={`note${check.available ? "" : " error-text"}`}>
              {check.available
                ? `Ollama at ${check.url}: ${asArray(check.models).join(", ") || "no models pulled"}; it would use ${check.model}`
                : check.error || `no Ollama at ${check.url}`}
            </span>
          ) : null}
        </div>
        {note ? <p className="note">{note}</p> : null}
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>

      <div className="card">
        <h2>A corpus from a prompt</h2>
        <TextField label="Prompt" value={prompt} onChange={setPrompt} />
        <div className="row">
          <NumberField label="Lines" value={lines} onChange={setLines} min="1" max="500" step="1" />
        </div>
        <div className="actions">
          <button
            type="button"
            className="primary"
            disabled={busy !== null || !prompt.trim()}
            onClick={() => act("corpus", async () => setCorpus(await api.ollamaCorpus({ prompt, lines: Math.max(1, parseInteger(lines, 20)), ...ollama() }, setJob)))}
          >
            {busy === "corpus" ? "Writing…" : "Write"}
          </button>
        </div>
        {busy === "corpus" ? <JobProgress job={job} /> : null}
        {corpus ? (
          <>
            <p className="muted">
              {corpusLines.length} line{corpusLines.length === 1 ? "" : "s"} by {corpus.model || "Ollama"}
            </p>
            <ol className="lines">
              {corpusLines.map((line, i) => (
                <li key={i} className="line">
                  {line}
                </li>
              ))}
            </ol>
            <div className="actions">
              <button
                type="button"
                disabled={busy !== null || !corpusLines.length}
                onClick={() =>
                  act("train", async () => {
                    const r = await api.lmText(lm, corpusLines.join("\n"), {}, setJob);
                    setNote(`${lmLabel(lm)}: trained ${r.trained} tokens of Ollama's lines`);
                    if (onMoved) onMoved();
                  })
                }
              >
                Train on these ({lmLabel(lm)})
              </button>
              <TextField label="Upload as" value={uploadAs} onChange={setUploadAs} />
              <button
                type="button"
                disabled={busy !== null || !corpusLines.length || !uploadAs.trim()}
                onClick={() =>
                  act("upload", async () => {
                    const r = await api.uploadText(uploadAs.trim(), `${corpusLines.join("\n")}\n`);
                    setNote(`saved as the upload ${r.name} (${r.bytes} bytes)`);
                    if (onMoved) onMoved();
                  })
                }
              >
                Save as upload
              </button>
            </div>
            {busy === "train" ? <JobProgress job={job} /> : null}
          </>
        ) : null}
      </div>

      <div className="card">
        <h2>The N-gram model trains itself</h2>
        {lm === "chain" ? (
          <p className="alert info">Auto-training is for the N-gram model only: it trains the N-gram model whichever is selected.</p>
        ) : null}
        <p className="note">
          Round 0 trains a fresh N-gram model on the data, every tenth line held out. Each round after it Ollama writes
          sentences on the corpus's own topics, continues the model's own contexts, and judges texts the model wrote -
          what it approves is rewarded, what it rejects punished - and the model is measured on the held-out lines again;
          it stops when that stops improving.
        </p>
        <div className="row">
          <CheckField label="Use Ollama" checked={useOllama} onChange={setUseOllama} hint="off: the data only" />
          <NumberField label="Rounds" value={rounds} onChange={setRounds} min="0" max="50" step="1" />
          <TextField label="Data" value={data} onChange={setData} hint="on the server, comma-separated" />
          <TextField label="Topics" value={topics} onChange={setTopics} hint="comma-separated; blank: the corpus's own words" />
        </div>
        <div className="row">
          <CheckField label="Predict while learning" checked={predictFirst} onChange={setPredictFirst} />
          <NumberField label="Self-credit" value={selfCredit} onChange={setSelfCredit} min="0" step="0.05" disabled={!predictFirst} />
        </div>
        <div className="actions">
          <button
            type="button"
            className="primary"
            disabled={busy !== null}
            onClick={() =>
              act("auto", async () => {
                const r = await api.ngramAutotrain({
                  data,
                  ollama: useOllama,
                  model: model || undefined,
                  topics,
                  rounds: parseNumber(rounds, 3),
                  predict: predictFirst,
                  self_credit: predictFirst ? parseNumber(selfCredit, 0) : 0,
                }, setJob);
                setAuto({ log: asArray(r && r.log), history: asArray(r && r.model && r.model.history) });
                if (onMoved) onMoved();
              })
            }
          >
            {busy === "auto" ? "Training…" : "Auto-train"}
          </button>
        </div>
        {busy === "auto" ? <JobProgress job={job} /> : null}
        {auto && auto.log.length ? <pre className="mono run-log">{auto.log.join("\n")}</pre> : null}
        {perplexity.length > 1 ? (
          <LineChart
            series={[{ name: "held-out perplexity", color: "var(--accent)", values: perplexity }]}
            xLabel="round"
            yLabel="perplexity"
            height={180}
          />
        ) : null}
        {history.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>round</th>
                  <th>tokens</th>
                  <th>perplexity</th>
                  <th>accuracy</th>
                  <th>learned from</th>
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={i}>
                    <td>{r.round ?? i}</td>
                    <td>{r.tokens ?? "–"}</td>
                    <td>{fmtNum(r.perplexity, 2)}</td>
                    <td>{pct(r.accuracy)}</td>
                    <td className="text muted">{roundSources(r)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>
    </div>
  );
}
