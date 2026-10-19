import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { infoLine, useLmInfo } from "../hooks/useLmInfo.js";
import { UNITS, bestSweepRow, goalShare, historyPoints, lmLabel, pct, roundsNote, sweepSeries } from "../lm.js";
import { fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextArea, TextField } from "./Fields.jsx";
import LineChart from "./LineChart.jsx";
import JobProgress from "./JobProgress.jsx";
import UploadPicker from "./UploadPicker.jsx";

const THRESHOLDS = "0, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 1";

/** "; predicted 1200 tokens before learning each: first guess right 17.2%, perplexity 170.17, 2.31 nodes deep" */
function learnedNote(learned) {
  if (!learned || !learned.predicted) return "";
  const depth = learned.depth !== undefined && learned.depth !== null ? `, ${fmtNum(learned.depth, 2)} nodes deep` : "";
  return `; predicted ${learned.predicted} tokens before learning each: first guess right ${pct(
    learned.accuracy || 0,
  )}, perplexity ${fmtNum(learned.perplexity, 2)}${depth}`;
}

/** "; held out: 900 tokens, perplexity 12.3, accuracy 41.0%, top-5 70.0%, 2.10 nodes deep" */
function heldOutNote(h) {
  if (!h || !h.tokens) return "";
  return `; held out: ${h.tokens} tokens, perplexity ${fmtNum(h.perplexity, 2)}, next-token accuracy ${pct(h.accuracy || 0)}${
    h.top5 !== undefined && h.top5 !== null ? `, top-5 ${pct(h.top5)}` : ""
  }${h.depth !== undefined && h.depth !== null ? `, ${fmtNum(h.depth, 2)} nodes deep` : ""}`;
}

/** "0, 0.5, 1" -> [0, 0.5, 1]: the typed thresholds, comma- or space-separated, anything outside [0, 1] dropped. */
function parseThresholds(text) {
  return String(text ?? "")
    .split(/[,\s]+/)
    .filter((s) => s !== "")
    .map(Number)
    .filter((n) => Number.isFinite(n) && n >= 0 && n <= 1);
}

/**
 * Train the chosen language model: make a new one (the chain: its nodes, unit and confidence; the N-gram
 * model: N and unit), train it on files on the server and/or uploaded files, or on pasted text - guessing
 * every token before it learns it, if asked, and crediting itself for the guesses - and follow the held-out
 * perplexity round by round. For the chain, a confidence sweep measures accuracy and depth at a range of
 * thresholds on the held-out split, and one of them can be made the chain's own.
 */
export default function LmTrainPanel({ lm, tick, onMoved }) {
  const [info, setInfo, infoError] = useLmInfo(lm, tick);
  const [nodes, setNodes] = useStoredState("lmtrain.nodes", "8");
  const [confidence, setConfidence] = useStoredState("lmtrain.confidence", "0.9");
  const [order, setOrder] = useStoredState("lmtrain.order", "3");
  const [chainUnit, setChainUnit] = useStoredState("lmtrain.chainUnit", "char");
  const [ngramUnit, setNgramUnit] = useStoredState("lmtrain.ngramUnit", "word");
  const [data, setData] = useStoredState("lmtrain.data", "data/corpus");
  const [useData, setUseData] = useStoredState("lmtrain.useData", true);
  const [heldOut, setHeldOut] = useStoredState("lmtrain.heldOut", "10");
  const [paste, setPaste] = useStoredState("lmtrain.paste", "");
  const [predictFirst, setPredictFirst] = useStoredState("lmtrain.first", true);
  const [selfCredit, setSelfCredit] = useStoredState("lmtrain.selfCredit", "0");
  const [untilData, setUntilData] = useStoredState("lmtrain.until", false);
  const [untilText, setUntilText] = useStoredState("lmtrain.untilText", true);
  const [maxRounds, setMaxRounds] = useStoredState("lmtrain.maxRounds", "20");
  const [goal, setGoal] = useStoredState("lmtrain.goal", "95");
  const [thresholds, setThresholds] = useStoredState("lmtrain.thresholds", THRESHOLDS);
  const [uploads, setUploads] = useState([]);
  const [sweep, setSweep] = useState(null);
  const [rounds, setRounds] = useState(null); // {rounds: [{round, right, of}], reached} of the last run until N of N
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(null);
  const [job, setJob] = useState(null); // the running job's progress: training runs in the background

  const chain = lm === "chain";
  const unit = chain ? chainUnit : ngramUnit;
  const setUnit = chain ? setChainUnit : setNgramUnit;

  async function act(name, fn) {
    setBusy(name);
    try {
      await fn();
      setError(null);
    } catch (err) {
      // a stopped job kept what it learned: say so, and show the model as it now is
      if (err.data && err.data.state === "stopped") {
        setError(null);
        setNote(`stopped after ${err.data.progress ? err.data.progress.tokens : 0} tokens; what was learned until then is kept`);
        if (onMoved) onMoved();
      } else {
        setError(err.message);
      }
    } finally {
      setBusy(null);
      setJob(null);
    }
  }

  const learning = () => ({ predict: predictFirst, self_credit: predictFirst ? parseNumber(selfCredit, 0) : 0 });

  // keep drilling until the goal - a share of every token that can be right (100: all of them) - or the rounds run out
  const untilBody = (on) =>
    on ? { until: true, max_rounds: Math.max(1, parseInteger(maxRounds, 20)), threshold: goalShare(goal) } : {};

  const untilResult = (r) => {
    const list = r && Array.isArray(r.rounds) ? r.rounds : null;
    setRounds(list && list.length ? { rounds: list, reached: Boolean(r.reached), need: r.need, best: r.best } : null);
    return list && list.length ? `; ${roundsNote(list, r.reached, r.need, r.best)}` : "";
  };

  // what to train on: the server paths (when ticked) and the selected uploads; neither means the server's default
  const sources = () => ({
    ...(useData && data.trim() ? { data: data.trim() } : {}),
    ...(uploads.length ? { uploads } : {}),
  });

  const moved = (r) => {
    if (r && r.model) setInfo(r.model);
    if (onMoved) onMoved();
  };

  const makeNew = () =>
    act("new", async () => {
      const body = chain
        ? { nodes: parseInteger(nodes, 8), unit, confidence: parseNumber(confidence, 0.9) }
        : { order: parseInteger(order, 3), unit };
      setInfo(await api.lmNew(lm, body));
      setSweep(null);
      setNote(`a new ${lmLabel(lm)} model`);
      if (onMoved) onMoved();
    });

  const train = () =>
    act("train", async () => {
      const r = await api.lmTrain(lm, { ...sources(), held_out: parseInteger(heldOut, 10), ...learning(), ...untilBody(untilData) }, setJob);
      setNote(`trained ${r.trained} tokens${learnedNote(r.learned)}${heldOutNote(r.held_out)}${untilResult(r)}`);
      moved(r);
    });

  const trainText = () =>
    act("text", async () => {
      const r = await api.lmText(lm, paste, { ...learning(), ...untilBody(untilText) }, setJob);
      setNote(`trained ${r.trained} tokens of pasted text${learnedNote(r.learned)}${untilResult(r)}`);
      moved(r);
    });

  const runSweep = () =>
    act("sweep", async () => {
      const list = parseThresholds(thresholds);
      const r = await api.chainSweep({
        ...sources(),
        held_out: parseInteger(heldOut, 10),
        ...(list.length ? { thresholds: list } : {}),
      }, setJob);
      setSweep(Array.isArray(r && r.rows) ? r.rows : []);
    });

  const pickConfidence = (value) =>
    act("confidence", async () => {
      setInfo(await api.chainConfidence(value));
      setConfidence(String(value));
      setNote(`the chain's confidence is now ${fmtNum(value, 2)}`);
      if (onMoved) onMoved();
    });

  const history = info && Array.isArray(info.history) ? info.history : [];
  const perplexity = historyPoints(history, "perplexity");
  const series = sweepSeries(sweep);
  const best = bestSweepRow(sweep);

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Train the {lmLabel(lm)} model</h2>
        {info ? <p className="mono">{infoLine(lm, info)}</p> : null}
        <Alert message={infoError} />
        <h3>A new model</h3>
        <p className="note">
          {chain
            ? "A chain of N nodes: N-1 input nodes hold the last N-1 inputs and the output node predicts the next. A prediction walks back from the output node through the input nodes, nearest first, and stops as soon as the confidence it has gathered reaches the threshold - 0: the output node alone, 1: as deep as it can go."
            : "An N-gram model: the context is the last N-1 units, the next is predicted by interpolated Kneser-Ney."}
        </p>
        <div className="row">
          <SelectField label="Unit" value={unit} onChange={setUnit} options={UNITS} />
          {chain ? (
            <>
              <NumberField label="Nodes" value={nodes} onChange={setNodes} min="2" max="16" step="1" hint="2-16, the output node one of them" />
              <NumberField label="Confidence" value={confidence} onChange={setConfidence} min="0" max="1" step="0.05" hint="the walk stops on reaching it" />
            </>
          ) : (
            <NumberField label="N" value={order} onChange={setOrder} min="1" max="12" step="1" hint="the context is N-1 units" />
          )}
        </div>
        <div className="actions">
          <button type="button" className="danger" disabled={busy !== null} onClick={makeNew}>
            {busy === "new" ? "Making…" : "New model"}
          </button>
          <span className="note muted">replaces the {lmLabel(lm)} model the server holds; save it first on the Checkpoints tab</span>
        </div>
      </div>

      <div className="card">
        <h2>Train on files</h2>
        <div className="row">
          <CheckField label="Server files" checked={useData} onChange={setUseData} />
          <TextField label="Data" value={data} onChange={setData} disabled={!useData} hint="files or directories on the server, comma-separated" />
        </div>
        <UploadPicker selected={uploads} onChange={setUploads} disabled={busy !== null} title="Uploaded files to train on" />
        <div className="row">
          <NumberField label="Held out" value={heldOut} onChange={setHeldOut} min="0" step="1" hint="every Nth line, to measure on (0: none)" />
        </div>
        <div className="row">
          <CheckField
            label="Predict while learning"
            checked={predictFirst}
            onChange={setPredictFirst}
            hint="guess every next token before it is learned, and measure the guesses"
          />
          <NumberField
            label="Self-credit"
            value={selfCredit}
            onChange={setSelfCredit}
            min="0"
            step="0.05"
            disabled={!predictFirst}
            hint="reward a right guess, punish a wrong one, by this much (0: off)"
          />
        </div>
        <div className="row">
          <CheckField label="Until N of N" checked={untilData} onChange={setUntilData} hint="re-train until every token is predicted right - on a corpus this can take long" />
          <NumberField label="Max rounds" value={maxRounds} onChange={setMaxRounds} min="1" step="1" hint="shared with the text card" />
          <NumberField label="Goal %" value={goal} onChange={setGoal} min="1" max="100" step="1" hint="of every token that can be right; 100: N of N" />
        </div>
        <div className="actions">
          <button type="button" className="primary" disabled={busy !== null} onClick={train}>
            {busy === "train" ? "Training…" : "Train"}
          </button>
          <span className="note muted">
            {useData && data.trim() ? data.trim() : ""}
            {useData && data.trim() && uploads.length ? " and " : ""}
            {uploads.length ? `${uploads.length} upload${uploads.length === 1 ? "" : "s"}` : ""}
            {!(useData && data.trim()) && !uploads.length ? "nothing chosen: the server's data/corpus" : ""}
          </span>
        </div>
        {busy === "train" ? <JobProgress job={job} /> : null}
      </div>

      <div className="card">
        <h2>Train on text</h2>
        <TextArea label="Paste text" value={paste} onChange={setPaste} rows={8} hint="every line one sequence" />
        <CheckField
          label="Until N of N"
          checked={untilText}
          onChange={setUntilText}
          hint="re-train on it, every guess credited ±1, until every token is predicted right"
        />
        <div className="actions">
          <button type="button" disabled={busy !== null || !paste.trim()} onClick={trainText}>
            {busy === "text" ? "Training…" : "Train on this text"}
          </button>
        </div>
        {busy === "text" ? <JobProgress job={job} /> : null}
        <p className="muted">Predict while learning, self-credit, max rounds and the goal (in the card beside) apply here too.</p>
      </div>

      <div className="card wide">
        <h2>Results</h2>
        {note ? <p className="note">{note}</p> : <p className="muted">Train, and what it learned is reported here.</p>}
        <Alert message={error} onDismiss={() => setError(null)} />
        {rounds ? (
          <>
            <h3>
              Until the goal: <span className={rounds.reached ? "ok-text" : "error-text"}>{roundsNote(rounds.rounds, rounds.reached, rounds.need, rounds.best)}</span>
            </h3>
            <div className="table-wrap">
              <table className="data">
                <thead>
                  <tr>
                    <th>round</th>
                    <th>right</th>
                    <th>of</th>
                    <th>share</th>
                  </tr>
                </thead>
                <tbody>
                  {rounds.rounds.map((r, i) => (
                    <tr key={i}>
                      <td>{r.round ?? i + 1}</td>
                      <td>{r.right}</td>
                      <td>{r.of}</td>
                      <td>{r.of ? pct(r.right / r.of) : "–"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : null}
        {perplexity.length > 1 ? (
          <LineChart
            series={[{ name: "held-out perplexity", color: "var(--accent)", values: perplexity }]}
            xLabel="round"
            yLabel="perplexity"
            height={200}
          />
        ) : null}
        {history.length ? (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>round</th>
                  <th>source</th>
                  <th>tokens</th>
                  <th>held-out perplexity</th>
                  <th>accuracy</th>
                  {chain ? <th>depth</th> : <th>top-5</th>}
                  <th>guessed right while learning</th>
                  {chain ? <th>depth while learning</th> : null}
                </tr>
              </thead>
              <tbody>
                {history.map((r, i) => (
                  <tr key={i}>
                    <td>{r.round ?? i}</td>
                    <td className="text">{r.source || "–"}</td>
                    <td>{r.tokens ?? "–"}</td>
                    <td>{r.perplexity === null || r.perplexity === undefined ? "–" : fmtNum(r.perplexity, 2)}</td>
                    <td>{pct(r.accuracy)}</td>
                    {chain ? <td>{fmtNum(r.depth, 2)}</td> : <td>{pct(r.top5)}</td>}
                    <td>{r.learned ? `${pct(r.learned_accuracy)} of ${r.learned}` : "–"}</td>
                    {chain ? <td>{fmtNum(r.learned_depth, 2)}</td> : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </div>

      {chain ? (
        <div className="card wide">
          <h2>Confidence sweep</h2>
          <p className="note">
            The same chain measured on the held-out lines of the data above at each threshold: the higher it is, the
            deeper the walks go before they stop - more context, slower, and not always more accurate. The chain is
            not changed until you pick one.
          </p>
          <div className="row">
            <TextField label="Thresholds" value={thresholds} onChange={setThresholds} hint="0 to 1, comma-separated" />
          </div>
          <div className="actions">
            <button type="button" disabled={busy !== null} onClick={runSweep}>
              {busy === "sweep" ? "Sweeping…" : "Sweep"}
            </button>
            {info ? <span className="note muted">now {fmtNum(info.confidence, 2)}</span> : null}
          </div>
          {busy === "sweep" ? <JobProgress job={job} /> : null}
          {sweep && sweep.length ? (
            <>
              <div className="row">
                <LineChart
                  series={[{ name: "accuracy", color: "var(--accent)", values: series.accuracy }]}
                  xLabel="confidence"
                  yLabel="accuracy"
                  height={200}
                />
                <LineChart
                  series={[{ name: "mean depth", color: "var(--accent-2)", values: series.depth }]}
                  xLabel="confidence"
                  yLabel="nodes deep"
                  height={200}
                />
              </div>
              <div className="table-wrap">
                <table className="data">
                  <thead>
                    <tr>
                      <th>confidence</th>
                      <th>accuracy</th>
                      <th>perplexity</th>
                      <th>mean depth</th>
                      <th>tokens</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {sweep.map((r) => (
                      <tr key={r.confidence}>
                        <td>
                          {fmtNum(r.confidence, 2)}
                          {best === r ? <span className="badge done tight">best</span> : null}
                        </td>
                        <td>{pct(r.accuracy)}</td>
                        <td>{fmtNum(r.perplexity, 2)}</td>
                        <td>{fmtNum(r.depth, 2)}</td>
                        <td>{r.tokens ?? "–"}</td>
                        <td>
                          <button type="button" className="small" disabled={busy !== null} onClick={() => pickConfidence(r.confidence)}>
                            use this confidence
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : sweep ? (
            <p className="muted">The sweep measured nothing (no held-out lines?).</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
