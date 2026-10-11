/**
 * One turn of talking to a language model, shared by the Talk and Voice tabs: what you say is (if asked)
 * learned first, then the model's reply is its own continuation of it - its prediction of what comes next.
 */
import { api } from "./api.js";
import { joinContinuation } from "./lm.js";

/**
 * Say `message` to model `lm`: `{text, reply, tokens, depths, unit, learned}` - the reply the continuation,
 * with its tokens (and, for the chain, how deep each walk went). With `learn`, the message is trained on
 * first (`POST /api/{m}/text`), so the model hears what you say. With `guard`, the guard judges the reply (and
 * vetoes it for another if it rejects it): the turn carries `guard` (its verdict in brief) and `vetoed` (how many).
 */
export async function askModel(lm, message, { learn = false, maxTokens = 40, temperature = 0, guard = false } = {}) {
  let learned = null;
  if (learn) {
    const r = await api.lmText(lm, message);
    learned = r ? r.trained : null;
  }
  const r = await api.lmGenerate(lm, {
    text: message,
    max_tokens: maxTokens,
    temperature,
    seed: Math.floor(Math.random() * 1e9),
    ...(guard ? { guard: true } : {}),
  });
  const sample = r && Array.isArray(r.samples) && r.samples.length ? r.samples[0] : null;
  const unit = r ? r.unit : undefined;
  const completion = r ? String(r.completion ?? (sample ? sample.completion : "") ?? "") : "";
  return {
    text: message,
    unit,
    // base64 continuations come back as the decoded prompt and continuation together: the reply is the rest
    reply: unit === "base64" && completion.startsWith(message) ? completion.slice(message.length) : completion,
    completion,
    tokens: sample && Array.isArray(sample.tokens) ? sample.tokens : [],
    depths: sample && Array.isArray(sample.depths) ? sample.depths : null,
    learned,
    guard: sample && sample.guard ? sample.guard : null,
    vetoed: r && r.guard ? Number(r.guard.vetoed) || 0 : 0,
    guardSummary: r && r.guard ? r.guard : null,
  };
}

/** What a 👍 / 👎 on a reply credits: the message and the reply that continued it, as one text. */
export function turnText(turn) {
  return joinContinuation(turn.text, turn.completion, turn.unit);
}
