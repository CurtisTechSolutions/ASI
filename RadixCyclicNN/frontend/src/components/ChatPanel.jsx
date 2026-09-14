import { Fragment, useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useJob } from "../hooks/useJob.js";
import { asArray, fmtInt, fmtNum, jobIsRunning, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import JobStatus from "./JobStatus.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

const isObject = (value) => Boolean(value) && typeof value === "object";

/**
 * The model in conversation with an LLM that marks it.
 *
 * Every other teacher here talks at the network; this one talks *to* it. The
 * partner says a short line, the model replies by continuing it (the same
 * search the Converse tab uses), they take turns, and then the LLM marks every
 * reply out of 10 against the line it answered — and the conversation as a
 * whole. What failed blames the negative network, what passed clears it, and
 * 2NRL trains the model on both, with the partner's own lines joining the
 * positive phase: they are what a good reply here would have looked like. A
 * reply the model could only repeat is punished whatever the judge made of it,
 * and with "Avoid repeated words" on that includes a reply repeating its own
 * words - though a reply that catches itself repeating first backs up to where
 * the walk went round and explores other ways on ("Explore").
 *
 * The transcript reads newest first: a new exchange is appended to the top and
 * pushes the older ones down, so the latest reply is where the eye already is
 * and a running conversation never has to be scrolled to.
 */
export default function ChatPanel({ status }) {
  const [topic, setTopic] = useState("");
  const [opening, setOpening] = useState("");
  const [persona, setPersona] = useState("");
  const [conversations, setConversations] = useState("1");
  const [turns, setTurns] = useState("4");
  const [context, setContext] = useState("12");
  const [maxLength, setMaxLength] = useState("60");
  const [mode, setMode] = useState("beam");
  const [threshold, setThreshold] = useState("6");
  const [provider, setProvider] = useState("ollama");
  const [partnerModel, setPartnerModel] = useState("");
  const [url, setUrl] = useState("");
  const [guard, setGuard] = useState(true);
  const [avoidWordRepeats, setAvoidWordRepeats] = useState(true);
  const [explore, setExplore] = useState("3");
  const [blame, setBlame] = useState(true);
  const [learn, setLearn] = useState(true);
  const [teachPartner, setTeachPartner] = useState(true);
  const [history, setHistory] = useState([]);
  const [historyError, setHistoryError] = useState(null);
  const { job, running, busy, error, start, stop, clearError } = useJob("chat");

  const loadHistory = useCallback(async () => {
    try {
      const data = await api.chatHistory();
      setHistory(asArray(data && data.history).filter(isObject));
      setHistoryError(null);
    } catch (err) {
      setHistoryError(err.message);
    }
  }, []);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  // while a job runs its own records are the live view; afterwards the server's history is reloaded
  const jobState = job ? job.state : null;
  useEffect(() => {
    if (jobState && jobState !== "running") loadHistory();
  }, [jobState, loadHistory]);

  const live = asArray(job && job.history).filter(isObject);
  const records = running || live.length ? live : history;
  const spoken = records.filter((r) => r.kind === "exchange");
  const held = records.filter((r) => r.kind === "conversation");
  const card = [...records].reverse().find((r) => r.kind === "report") || null;
  // newest first: a new exchange is appended to the top of the transcript and pushes the older ones down
  const newestFirst = [...spoken].reverse();
  const saidTwice = held.reduce((total, r) => total + (Number(r.repeats) || 0), 0);

  const otherJobRunning = jobIsRunning(status) && !running;

  async function handleStart(event) {
    event.preventDefault();
    await start(() =>
      api.chatStart({
        conversations: Math.max(0, parseInteger(conversations, 1)),
        turns: Math.max(1, parseInteger(turns, 4)),
        context: Math.max(0, parseInteger(context, 12)),
        max_length: Math.max(1, parseInteger(maxLength, 60)),
        mode,
        threshold: parseNumber(threshold, 6),
        provider,
        guard,
        avoid_word_repeats: avoidWordRepeats,
        explore: Math.max(0, parseInteger(explore, 3)),
        blame,
        learn,
        teach_partner: teachPartner,
        ...(topic.trim() ? { topic: topic.trim() } : {}),
        ...(opening.trim() ? { opening: opening.trim() } : {}),
        ...(persona.trim() ? { persona: persona.trim() } : {}),
        ...(partnerModel.trim() ? { partner_model: partnerModel.trim() } : {}),
        ...(url.trim() ? { url: url.trim() } : {}),
      }),
    );
  }

  return (
    <>
      <form className="card" onSubmit={handleStart}>
        <h2>Chat</h2>
        <p className="muted">
          The other side of the line is a real language model. It says a short line, the network replies by
          continuing it — the same search the Converse tab uses, so a reply is a real walk of the graph — and then
          the LLM marks every reply out of 10 against the line it answered. What failed blames the negative
          network, what passed clears it, and 2NRL trains the model on both. The partner's own lines join the
          positive phase: they are what a good reply here would have looked like.
        </p>
        <div className="row">
          <TextField label="Topic" hint="what to talk about" value={topic} onChange={setTopic}
                     placeholder="(the partner chooses)" disabled={running} />
          <TextField label="Partner is" hint="who it is being" value={persona} onChange={setPersona}
                     placeholder="a curious child" disabled={running} />
        </div>
        <TextField label="Opening line" hint="optional: spoken as given, instead of the partner opening"
                   value={opening} onChange={setOpening} placeholder="tell me about the cat" disabled={running} />
        <div className="row">
          <NumberField label="Conversations" hint="0 = until you stop it" value={conversations}
                       onChange={setConversations} min={0} step={1} disabled={running} />
          <NumberField label="Replies each" value={turns} onChange={setTurns} min={1} step={1} disabled={running} />
          <NumberField label="Context" hint="characters a reply picks up" value={context} onChange={setContext}
                       min={0} step={1} disabled={running} />
          <NumberField label="Max length" value={maxLength} onChange={setMaxLength} min={1} step={1} disabled={running} />
        </div>
        <div className="row">
          <SelectField label="Replies" value={mode} onChange={setMode} disabled={running}
                       options={[["beam", "beam (the most likely unheard one)"], ["sample", "sample (stochastic)"]]} />
          <NumberField label="Explore" hint="times a reply may back up out of a repeat" value={explore}
                       onChange={setExplore} min={0} step={1} disabled={running || !avoidWordRepeats} />
          <NumberField label="Pass mark" hint="out of 10" value={threshold} onChange={setThreshold} min={0} max={10}
                       disabled={running} />
          <SelectField label="Partner" value={provider} onChange={setProvider} disabled={running}
                       options={[["ollama", "Ollama (local)"], ["chatgpt", "ChatGPT (server key)"]]} />
          <TextField label="Model" value={partnerModel} onChange={setPartnerModel} placeholder="(the server's default)"
                     disabled={running} />
        </div>
        <TextField label="URL" value={url} onChange={setUrl} placeholder="(the server's default)" disabled={running} />
        <div className="checks">
          <CheckField label="Veto bad replies before they are spoken" checked={guard} onChange={setGuard}
                      disabled={running} hint="the negative network guards the conversation" />
          <CheckField label="Blame what failed" checked={blame} onChange={setBlame} disabled={running} />
          <CheckField label="Train on the marks" checked={learn} onChange={setLearn} disabled={running} hint="2NRL" />
          <CheckField label="Learn the partner's lines too" checked={teachPartner} onChange={setTeachPartner}
                      disabled={running || !learn} />
          <CheckField label="Avoid repeated words" checked={avoidWordRepeats} onChange={setAvoidWordRepeats}
                      disabled={running} hint="a reply may not say the same word or phrase twice in a row" />
        </div>
        <div className="actions">
          <button type="submit" className="primary" disabled={running || busy || otherJobRunning}>
            {busy ? "Starting…" : running ? "Talking…" : "Start"}
          </button>
          <button type="button" className="danger" disabled={!running} onClick={() => stop()}>
            Stop
          </button>
        </div>
        {otherJobRunning ? <p className="muted">Another job is running; wait for it to finish.</p> : null}
        <JobStatus job={job} emptyText="They have never spoken. Press Start and they will." />
        <Alert message={error} onDismiss={clearError} />
      </form>

      <div className="card">
        <h2>Transcript</h2>
        {spoken.length > 1 ? (
          <p className="muted">Newest first: the latest exchange is at the top and the transcript grows downwards.</p>
        ) : null}
        {saidTwice ? (
          <p className="muted">
            {fmtInt(saidTwice)} repl{saidTwice === 1 ? "y" : "ies"} the model could only repeat: punished with the
            failures, whatever the judge made of {saidTwice === 1 ? "it" : "them"}.
          </p>
        ) : null}
        {spoken.length === 0 ? (
          <p className="muted">
            {running ? "Waiting for the first line…" : "No conversation yet. Press Start."}
          </p>
        ) : (
          <ol className="dialogue" aria-label="conversation" reversed>
            {newestFirst.map((r, i) => {
              const mark = markOf(held, r);
              return (
                <Fragment key={`${r.conversation}-${r.exchange}-${i}`}>
                  <li className="turn a">
                    <div className="speaker">
                      <b>Partner</b>
                      {r.exchange === 1 ? <span className="badge">conversation {fmtInt(r.conversation)}</span> : null}
                    </div>
                    <p className="bubble">{r.said}</p>
                  </li>
                  <li className={`turn b${mark ? ` rated ${mark.verdict === "pass" ? "up" : "down"}` : ""}`}>
                    <div className="speaker">
                      <b>Model</b>
                      {r.fresh ? <span className="badge">new topic</span> : null}
                      {r.repeat ? <span className="badge down">repeat</span> : null}
                      {r.stutter ? <span className="badge down">repeats itself</span> : null}
                      {r.rethink && r.rethink.found ? <span className="badge up">thought again</span> : null}
                      {r.vetoed ? <span className="badge down">{fmtInt(r.vetoed)} vetoed</span> : null}
                      {mark ? (
                        <span className={`badge ${mark.verdict === "pass" ? "pass" : "fail"}`}>
                          {fmtNum(mark.rating, 1)}/10
                        </span>
                      ) : null}
                    </div>
                    <p className="bubble">
                      {r.context ? (
                        <>
                          <span className="context" title="picked up from the line before">{r.context}</span>
                          {String(r.reply ?? "").slice(String(r.context).length)}
                        </>
                      ) : (
                        r.reply
                      )}
                    </p>
                    <div className="meta">
                      cost {fmtNum(r.cost, 3)} · p {fmtNum(r.probability, 4)}
                      {r.context ? <> · picked up “{r.context}”</> : <> · a fresh line</>}
                      {r.rethink ? <> · {rethinkSays(r)}</> : null}
                      {mark && mark.critique ? <> · {mark.critique}</> : null}
                    </div>
                  </li>
                </Fragment>
              );
            })}
          </ol>
        )}
      </div>

      <div className="card">
        <h2>Marks</h2>
        <p className="muted">
          Every reply is marked against the line it answered, and each conversation gets a verdict of its own.
        </p>
        {held.length === 0 ? (
          <p className="muted">Nothing has been marked yet.</p>
        ) : (
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>#</th>
                  <th>replies</th>
                  <th>passed</th>
                  <th>failed</th>
                  <th>mean mark</th>
                  <th>overall</th>
                  <th>vetoed</th>
                  <th>blamed</th>
                  <th>learned</th>
                  <th>ended</th>
                </tr>
              </thead>
              <tbody>
                {held.map((r) => (
                  <tr key={r.conversation}>
                    <td>{fmtInt(r.conversation)}</td>
                    <td>{fmtInt(r.exchanges)}</td>
                    <td>{fmtInt(r.passed)}</td>
                    <td>{fmtInt(r.failed)}</td>
                    <td>{fmtNum(r.mean_rating, 1)}</td>
                    <td title={r.overall_critique || ""}>{fmtNum(r.overall_rating, 1)}</td>
                    <td>{fmtInt(r.vetoed)}</td>
                    <td>{fmtInt(r.blamed)}</td>
                    <td className="muted">{r.action || "—"}</td>
                    <td className="muted">{r.stalled || "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {card ? (
          <p className="muted">
            {fmtInt(card.conversations)} conversation(s), {fmtInt(card.exchanges)} repl(ies):{" "}
            {fmtInt(card.passed)} passed, {fmtInt(card.failed)} failed · mean mark {fmtNum(card.mean_rating, 2)}/10
            {Number.isFinite(card.trend) ? ` · trend ${card.trend >= 0 ? "+" : ""}${fmtNum(card.trend, 2)}` : ""}
            {card.blamed ? ` · blamed ${fmtInt(card.blamed)}` : ""}
            {card.stalled ? ` · ${fmtInt(card.stalled)} ended early` : ""}
          </p>
        ) : null}
        <Alert message={historyError} onDismiss={() => setHistoryError(null)} />
      </div>
    </>
  );
}

/** What an exchange's rethink record says in one line: what it caught itself doing, and how that turned out. */
function rethinkSays(exchange) {
  const r = exchange.rethink;
  const caught = `caught itself saying “${r.noticed}” twice`;
  if (!r.steps) return `${caught}: the words it picked up, not its own`;
  if (r.found) return `${caught}: kept “${r.cut}”, found another way on in ${fmtInt(r.explored)} path(s)`;
  const ending = exchange.repeat ? "said it anyway" : "took a lesser answer";
  return `${caught}: kept “${r.cut}”, weighed ${fmtInt(r.explored)} path(s), ${ending}`;
}

/** The judge's mark for one exchange, once its conversation has been marked. */
function markOf(held, exchange) {
  const record = held.find((r) => r.conversation === exchange.conversation);
  if (!record) return null;
  const review = asArray(record.reviews)[exchange.exchange - 1];
  return isObject(review) && review.rating !== null && review.rating !== undefined ? review : null;
}
