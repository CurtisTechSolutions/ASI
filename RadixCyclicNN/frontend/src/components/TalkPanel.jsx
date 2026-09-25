import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { fmtInt, fmtNum, parseInteger, parseNumber, unitName } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";
import GuardNotice from "./GuardNotice.jsx";

/**
 * Talk to the model in today's format: messages in, an assistant message out,
 * the thinking first and then the text, streamed as they are produced.
 *
 * Every line typed here is one turn of a conversation the model answers the
 * way `converse` answers: the tail of the line is located in the graph and
 * continued, so a reply is a real walk of the network. What arrives first is
 * the **thinking** - the search's own trace, line by line as the search takes
 * each step: what it looked for and whether the graph knew it, how many paths
 * it weighed, what the negative network vetoed and why, where it caught itself
 * repeating and how it backed out, and what it finally said at what cost. Then
 * the **text**, one chunk per node of the walk - a merged node arrives as a
 * word, an unmerged one as a letter, so the radix compression is visible.
 *
 * The panel speaks `POST /v1/messages` with `stream: true` (the Messages
 * dialect; `POST /v1/chat/completions` serves the same conversation in
 * OpenAI's), which is what any client of those formats speaks too: point one
 * at this server and it talks to the model. Newest first, as the Converse and
 * Chat tabs read: a new turn is appended to the top and pushes the older ones
 * down, so the reply being written is where the eye already is.
 */

/** What a turn's rethink record says in one line (the Converse tab's phrasing). */
function rethinkSays(turn) {
  const r = turn.rethink;
  const caught =
    r.kind === "stutter" ? `caught itself saying “${r.noticed}” twice` : `caught itself repeating “${r.noticed}”`;
  if (!r.steps) return `${caught}: the words it picked up, not its own`;
  const learned = r.taught >= 0 ? " (and learned to hand over there)" : "";
  if (r.found) return `${caught}: kept “${r.cut}”, found another way on in ${fmtInt(r.explored)} path(s)${learned}`;
  const ending = turn.repeat ? "said it anyway" : "took a lesser answer";
  return `${caught}: kept “${r.cut}”, weighed ${fmtInt(r.explored)} path(s), ${ending}${learned}`;
}

/** How a reply ended, in words. */
function stopSays(message, units) {
  switch (message.stop) {
    case "end_turn":
      return "reached the end of a text";
    case "max_tokens":
      return `cut at the cap (${units})`;
    case "stop_sequence":
      return `stopped at “${message.stopSequence}”`;
    case "tool_use":
      return "wrote a tool call";
    case "refusal":
      return "the guard vetoed everything it could say";
    default:
      return message.stopped ? "stopped" : message.error ? "failed" : "";
  }
}

let nextId = 1;

export default function TalkPanel({ status }) {
  const [draft, setDraft] = useStoredState("talk.draft", "");
  const [maxTokens, setMaxTokens] = useStoredState("talk.maxTokens", "60");
  const [context, setContext] = useStoredState("talk.context", "12");
  const [k, setK] = useStoredState("talk.k", "5");
  const [explore, setExplore] = useStoredState("talk.explore", "3");
  const [mode, setMode] = useStoredState("talk.mode", "beam");
  const [temperature, setTemperature] = useStoredState("talk.temperature", "1.0");
  const [showThinking, setShowThinking] = useStoredState("talk.showThinking", true);
  const [guard, setGuard] = useStoredState("talk.guard", true);
  const [avoidWordRepeats, setAvoidWordRepeats] = useStoredState("talk.avoidWordRepeats", true);
  const [learn, setLearn] = useStoredState("talk.learn", true);
  const [messages, setMessages] = useState([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const abort = useRef(null);
  const units = unitName(status);
  const kind = status ? status.kind : null;
  const origin = typeof window !== "undefined" && window.location ? window.location.origin : "http://127.0.0.1:8000";

  // leaving the page stops a reply still being written
  useEffect(
    () => () => {
      if (abort.current) abort.current.abort();
    },
    [],
  );

  /** Change the reply being written (always the last message). */
  function patchLast(update) {
    setMessages((prev) => {
      if (!prev.length) return prev;
      const next = prev.slice();
      const last = next[next.length - 1];
      next[next.length - 1] = { ...last, ...(typeof update === "function" ? update(last) : update) };
      return next;
    });
  }

  async function send(event) {
    if (event) event.preventDefault();
    const said = draft.trim();
    if (!said || busy) return;
    setError(null);
    // the conversation so far, both sides heard; a reply with nothing to say is not a message the dialect takes
    const history = messages
      .filter((m) => !m.error && (m.role === "user" || m.text))
      .map((m) => ({ role: m.role, content: m.text }));
    const body = {
      model: "",
      messages: [...history, { role: "user", content: said }],
      max_tokens: parseInteger(maxTokens, 60),
      temperature: parseNumber(temperature, 1),
      thinking: { type: "enabled" },
      mode,
      context: parseInteger(context, 12),
      k: parseInteger(k, 5),
      explore: parseInteger(explore, 3),
      avoid_word_repeats: avoidWordRepeats,
      learn,
      guard,
    };
    setDraft("");
    setMessages((prev) => [
      ...prev,
      { id: nextId++, role: "user", text: said },
      { id: nextId++, role: "assistant", text: "", thinking: "", stop: null, stopSequence: null, turn: null, guard: null, streaming: true },
    ]);
    setBusy(true);
    const controller = new AbortController();
    abort.current = controller;
    try {
      await api.talkStream(
        body,
        (name, data) => {
          if (name === "content_block_delta") {
            const delta = data.delta || {};
            if (delta.type === "thinking_delta") patchLast((m) => ({ thinking: m.thinking + delta.thinking }));
            else if (delta.type === "text_delta") patchLast((m) => ({ text: m.text + delta.text }));
          } else if (name === "message_delta") {
            const record = data.radixnet || {};
            patchLast({
              stop: data.delta ? data.delta.stop_reason : null,
              stopSequence: data.delta ? data.delta.stop_sequence : null,
              turn: record.turn || null,
              guard: record.guard || null,
              outputTokens: data.usage ? data.usage.output_tokens : null,
            });
          } else if (name === "error") {
            patchLast({ error: data && data.error && data.error.message ? data.error.message : "the stream failed" });
          }
        },
        controller.signal,
      );
      patchLast({ streaming: false });
    } catch (err) {
      if (err && err.name === "AbortError") {
        patchLast({ streaming: false, stopped: true });
      } else {
        setError(err.message);
        patchLast({ streaming: false, error: err.message });
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }

  const newestFirst = [...messages].reverse();
  const curl = `curl -N ${origin}/v1/chat/completions -H 'Content-Type: application/json' \\
  -d '{"model": "radixnet", "messages": [{"role": "user", "content": "tell me about the cat"}], "stream": true}'`;

  return (
    <>
      <form className="card" onSubmit={send}>
        <h2>Talk</h2>
        <p className="muted">
          The model in today&apos;s format: a conversation of messages goes in, an assistant message comes back -
          its <b>thinking</b> first, then the <b>text</b>, both streamed as they are produced. A reply is what{" "}
          <b>Converse</b> would say next after your line (the tail of it is picked up and continued), so it is a
          real walk of the graph; the thinking is the search&apos;s own trace - what it looked for, how many
          paths it weighed, what the negative network vetoed and why, where it caught itself repeating - and the
          text arrives one node of the walk at a time. A system prompt is accepted by the API and not read: the
          network continues text and cannot follow an instruction.
        </p>
        <label className="field">
          <span>
            Say something <em>(Enter sends, Shift+Enter starts a new line)</em>
          </span>
          <textarea
            value={draft}
            rows={2}
            placeholder="tell me about the cat"
            spellCheck={false}
            disabled={busy}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
          />
        </label>
        <div className="row">
          <NumberField
            label="Max tokens"
            hint={`${units} the reply may add - a token is one ${unitName(status, false)}`}
            value={maxTokens}
            onChange={setMaxTokens}
            min={1}
            step={1}
          />
          <NumberField label="Context" hint={`${units} of your line it picks up`} value={context} onChange={setContext} min={0} step={1} />
          <NumberField label="K" hint="candidates weighed" value={k} onChange={setK} min={1} step={1} />
          <NumberField
            label="Explore"
            hint="times it may back up out of a repeat (0 = not at all)"
            value={explore}
            onChange={setExplore}
            min={0}
            step={1}
            disabled={!avoidWordRepeats}
          />
        </div>
        <div className="row">
          <SelectField
            label="Mode"
            value={mode}
            onChange={setMode}
            options={[
              ["beam", "beam (the most likely reply the conversation has not heard)"],
              ["sample", "sample (a stochastic walk)"],
            ]}
          />
          <NumberField label="Temperature" value={temperature} onChange={setTemperature} min={0} disabled={mode !== "sample"} />
        </div>
        <CheckField label="Show the thinking" hint="the search's trace, as it happens" checked={showThinking} onChange={setShowThinking} />
        <CheckField
          label="Filter with the negative network"
          hint="a candidate it vetoes is never said, and the thinking says why"
          checked={guard}
          onChange={setGuard}
        />
        <CheckField
          label="Avoid repeated words"
          hint="skip a reply that says the same word or phrase twice in a row"
          checked={avoidWordRepeats}
          onChange={setAvoidWordRepeats}
          disabled={busy}
        />
        <CheckField
          label="Learn where it goes round"
          hint="what a rethink finds out is taught to the graph (this changes the model)"
          checked={learn}
          onChange={setLearn}
          disabled={busy || !explore || explore === "0"}
        />
        <div className="actions">
          <button type="submit" className="primary" disabled={busy || !draft.trim()}>
            {busy ? "Replying…" : "Send"}
          </button>
          <button type="button" disabled={!busy} onClick={() => abort.current && abort.current.abort()}>
            Stop
          </button>
          <button
            type="button"
            className="small"
            disabled={busy || !messages.length}
            onClick={() => {
              setMessages([]);
              setError(null);
            }}
          >
            Clear
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
      </form>

      <div className="card">
        <h2>Conversation</h2>
        {messages.length > 2 ? (
          <p className="muted">Newest first: the latest reply is at the top and the conversation grows downwards.</p>
        ) : null}
        {messages.length === 0 ? (
          <p className="muted">Say something above; the {kind || "active"} model answers here, thinking first.</p>
        ) : (
          <ol className="dialogue" aria-label="conversation" reversed>
            {newestFirst.map((m) => {
              if (m.role === "user") {
                return (
                  <li key={m.id} className="turn a">
                    <div className="speaker">
                      <b>you</b>
                    </div>
                    <p className="bubble">{m.text}</p>
                  </li>
                );
              }
              const turn = m.turn;
              const lines = m.thinking ? m.thinking.split("\n") : [];
              const flags = [
                m.streaming ? "writing" : null,
                m.stopped ? "stopped" : null,
                m.error ? "failed" : null,
                turn && turn.fresh ? "new topic" : null,
                turn && turn.repeat ? "repeat" : null,
                turn && turn.rethink && turn.rethink.found ? "thought again" : null,
                turn && turn.vetoed ? `${fmtInt(turn.vetoed)} vetoed` : null,
              ].filter(Boolean);
              return (
                <li key={m.id} className={`turn b${m.streaming ? " streaming" : ""}`}>
                  <div className="speaker">
                    <b>model</b>
                    {flags.map((f) => (
                      <span key={f} className={`badge${f === "repeat" || f === "failed" ? " down" : ""}${f === "thought again" ? " up" : ""}`}>
                        {f}
                      </span>
                    ))}
                  </div>
                  {lines.length ? (
                    <details className="thinking" open={showThinking || m.streaming}>
                      <summary>
                        thinking · {fmtInt(lines.length)} line{lines.length === 1 ? "" : "s"}
                      </summary>
                      <ol className="thoughts">
                        {lines.map((line, i) => (
                          <li key={i}>{line}</li>
                        ))}
                      </ol>
                    </details>
                  ) : null}
                  <p className="bubble">
                    {turn && turn.context && m.text.startsWith(turn.context) ? (
                      <>
                        <span className="context" title="picked up from your line">
                          {turn.context}
                        </span>
                        {m.text.slice(turn.context.length)}
                      </>
                    ) : (
                      m.text || (m.streaming ? "" : m.error ? m.error : "(nothing to say)")
                    )}
                  </p>
                  <div className="meta">
                    {stopSays(m, units)}
                    {turn ? (
                      <>
                        {" · "}cost {fmtNum(turn.cost, 3)} · p {fmtNum(turn.probability, 4)}
                        {turn.context ? <> · picked up “{turn.context}”</> : null}
                        {turn.skipped ? <> · skipped {fmtInt(turn.skipped)}</> : null}
                        {turn.rethink ? <> · {rethinkSays(turn)}</> : null}
                      </>
                    ) : null}
                    {Number.isFinite(m.outputTokens) ? <> · {fmtInt(m.outputTokens)} {units} written, thinking included</> : null}
                  </div>
                  <GuardNotice guard={m.guard} what="candidates" />
                </li>
              );
            })}
          </ol>
        )}
      </div>

      <div className="card">
        <h2>From any client</h2>
        <p className="muted">
          The same conversation is served in the two shapes every client speaks: <code>POST /v1/chat/completions</code>{" "}
          (OpenAI&apos;s Chat Completions - the thinking is <code>reasoning_content</code>, <code>stream: true</code>{" "}
          sends <code>chat.completion.chunk</code> events) and <code>POST /v1/messages</code> (Anthropic&apos;s
          Messages - <code>thinking</code>, <code>text</code> and <code>tool_use</code> blocks, streamed as
          content-block events). <code>GET /v1/models</code> lists the models in memory as{" "}
          <code>radixnet-&lt;kind&gt;</code>; <code>&quot;radixnet&quot;</code> or an empty model name is whichever
          is active. Usage counts in the model&apos;s units: a token is one {unitName(status, false)}.
        </p>
        <pre className="code wrap">{curl}</pre>
      </div>
    </>
  );
}
