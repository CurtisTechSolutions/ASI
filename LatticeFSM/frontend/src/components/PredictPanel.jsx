import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { infoLine, useLmInfo } from "../hooks/useLmInfo.js";
import { appendToken, showToken } from "../matrix.js";
import { chainBoxes, joinContinuation, lmLabel, unitWord } from "../lm.js";
import { fmtNum, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import HearButton from "./HearButton.jsx";
import { NumberField, TextArea } from "./Fields.jsx";

/** An input as a box shows it: a space as ␣, an empty node as a dot. */
function boxText(input, unit) {
  if (input === null || input === undefined) return "·";
  return showToken(String(input), unit) || "·";
}

/**
 * The chain as the prediction walked it: the input nodes 1 … N-1 holding the last inputs, then the output
 * node. The walk goes from the output node back, nearest first, and stops once its gathered confidence
 * reaches the threshold; the nodes it traversed show their own confidence and the confidence gathered up to
 * them, the ones it never reached are greyed.
 */
export function ChainWalk({ prediction, unit }) {
  const boxes = chainBoxes(prediction);
  const walk = Array.isArray(prediction.walk) ? prediction.walk : [];
  // the output node shows what the whole walk predicts: the likeliest next token
  const top = Array.isArray(prediction.next) && prediction.next.length ? prediction.next[0].token : null;
  return (
    <>
      <div className="chain-boxes" role="list" aria-label="the chain's nodes">
        {boxes.map((b) => (
          <div
            key={b.node}
            role="listitem"
            className={`chain-box${b.output ? " output" : ""}${b.step ? " walked" : " off"}`}
            title={
              b.step
                ? `depth ${b.depth}: guesses ${b.step.guess ?? "-"}, confidence ${fmtNum(b.step.confidence, 3)}, gathered ${fmtNum(
                    b.step.gathered,
                    3,
                  )}; seen ${b.step.seen ?? 0}, tested ${b.step.tested ?? 0}, right ${b.step.right ?? 0}`
                : b.output
                  ? "the output node"
                  : "not traversed: the walk stopped nearer the output node"
            }
          >
            <span className="chain-label">{b.output ? "output" : `node ${b.node}`}</span>
            <span className="chain-input mono">{b.output ? (top ? boxText(top, unit) : "?") : boxText(b.input, unit)}</span>
            {b.step ? (
              <span className="chain-conf mono">
                c {fmtNum(b.step.confidence, 2)}
                <br />Σ {fmtNum(b.step.gathered, 2)}
              </span>
            ) : (
              <span className="chain-conf muted">–</span>
            )}
          </div>
        ))}
      </div>
      <p className="note">
        walked {prediction.depth ?? walk.length - 1} input node{(prediction.depth ?? walk.length - 1) === 1 ? "" : "s"} deep,
        gathering confidence {fmtNum(prediction.confidence, 3)}
        {walk.length ? ` (stopped at ${walk.length - 1 === 0 ? "the output node" : `node ${boxes.length - (walk.length - 1)}`})` : ""}
      </p>
      {walk.length ? (
        <div className="table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th>depth</th>
                <th>node</th>
                <th>input</th>
                <th>guess</th>
                <th>confidence</th>
                <th>gathered</th>
                <th>seen</th>
                <th>right / tested</th>
                <th>rewarded / punished</th>
              </tr>
            </thead>
            <tbody>
              {walk.map((s, i) => {
                const depth = s.depth ?? i;
                return (
                  <tr key={i}>
                    <td>{depth}</td>
                    <td>{depth === 0 ? "output" : `node ${boxes.length - depth}`}</td>
                    <td className="mono">{depth === 0 ? "–" : boxText(s.input, unit)}</td>
                    <td className="mono">{s.guess === undefined || s.guess === null ? "–" : boxText(s.guess, unit)}</td>
                    <td>{fmtNum(s.confidence, 3)}</td>
                    <td>{fmtNum(s.gathered, 3)}</td>
                    <td>{s.seen ?? "–"}</td>
                    <td>
                      {s.right ?? 0} / {s.tested ?? 0}
                    </td>
                    <td>
                      {fmtNum(s.rewarded ?? 0, 1)} / {fmtNum(s.punished ?? 0, 1)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </>
  );
}

/**
 * Predict the next thing with the chosen language model. Type, and the likeliest next tokens are listed as you
 * go - click one to take it; continue the text, and reward or punish what it wrote. For the chain, the input
 * nodes are drawn holding what you typed, with the walk the prediction took through them: which nodes it
 * traversed, each one's confidence, and the confidence gathered.
 */
export default function PredictPanel({ lm, tick, onMoved }) {
  const [info, setInfo, infoError] = useLmInfo(lm, tick);
  const [text, setText] = useStoredState("predict.text", "the brain is");
  const [k, setK] = useStoredState("predict.k", "10");
  const [maxTokens, setMaxTokens] = useStoredState("predict.max", "30");
  const [temperature, setTemperature] = useStoredState("predict.temperature", "0");
  const [next, setNext] = useState(null);
  const [completion, setCompletion] = useState(null);
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const typing = useRef(0);

  // a switched model has its own next tokens and its own continuation
  useEffect(() => {
    setNext(null);
    setCompletion(null);
    setNote(null);
  }, [lm]);

  // the next tokens follow the text as it is typed (a moment after the last key)
  useEffect(() => {
    if (!info) return undefined;
    window.clearTimeout(typing.current);
    typing.current = window.setTimeout(() => {
      api
        .lmPredict(lm, text, parseNumber(k, 10))
        .then((r) => {
          setNext(r);
          setError(null);
        })
        .catch((err) => setError(err.message));
    }, 150);
    return () => window.clearTimeout(typing.current);
  }, [text, k, info, lm]);

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

  const generate = () =>
    act(async () => {
      const r = await api.lmGenerate(lm, {
        text,
        max_tokens: parseNumber(maxTokens, 30),
        temperature: parseNumber(temperature, 0),
        seed: Math.floor(Math.random() * 1e9),
      });
      setCompletion({ ...r, prompt: text });
    });

  const credit = (amount) =>
    act(async () => {
      const written = joinContinuation(completion.prompt, completion.completion, completion.unit);
      const r = await api.lmCredit(lm, written, amount);
      setNote(`${amount > 0 ? "rewarded" : "punished"} ${r.credited} ${lm === "chain" ? "nodes" : "edges"} of what it wrote`);
      if (r && r.model) setInfo(r.model);
      if (onMoved) onMoved();
    });

  const unit = info ? info.unit : next ? next.unit : "word";
  const chain = lm === "chain";
  return (
    <div className="panel">
      <div className="card wide">
        <h2>Predict the next thing</h2>
        <p className="note">
          {chain
            ? `The chain: ${info ? info.nodes - 1 : "N-1"} input nodes hold the last ${unitWord(unit)}s and the output node predicts the next one. The prediction walks back from the output node, nearest input first, only until it has gathered the confidence it needs (${
                info ? fmtNum(info.confidence, 2) : "the threshold"
              }) - the more confident the nodes near the output, the shorter the walk.`
            : `An N-gram model over the lattice's adaptive edges: the context is the last ${info ? info.order - 1 : "N-1"} ${unitWord(
                unit,
              )}s, the next one is predicted by interpolated Kneser-Ney, and every edge you reward or punish is widened or narrowed.`}
        </p>
        {info ? <p className="mono">{infoLine(lm, info)}</p> : null}
        <Alert message={infoError ? `${lmLabel(lm)}: ${infoError}` : null} />
        <TextArea label="Text" value={text} onChange={setText} rows={3} hint="the next token follows it as you type" />
        {chain && next && Array.isArray(next.walk) ? <ChainWalk prediction={next} unit={next.unit || unit} /> : null}
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
              <HearButton text={joinContinuation(completion.prompt, completion.completion, completion.unit)} unit={completion.unit} label="what it wrote" />
            </>
          ) : null}
        </div>
        {completion ? (
          <p className="text-display">
            {completion.unit === "base64" ? (
              completion.completion
            ) : (
              <>
                <span className="muted">{completion.prompt}</span>
                {completion.unit === "char" ? "" : " "}
                <strong>{completion.completion || "(it would end here)"}</strong>
              </>
            )}
          </p>
        ) : null}
        {note ? <p className="note">{note}</p> : null}
        <Alert message={error} onDismiss={() => setError(null)} />
        <p className="muted">Training is on the Train tab, Ollama's auto-training on the Ollama tab, saving and loading on the Checkpoints tab.</p>
      </div>
    </div>
  );
}
