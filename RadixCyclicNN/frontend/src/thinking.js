/**
 * The THINK sentinel's records, read for display: what the Think tab, the
 * Converse and Chat transcripts and the Ollama tab's thinking cards show
 * (../../DESIGN.md sections 5.1.2 and 36). Pure functions, checked by
 * ../test/thinking.test.mjs.
 */

import { parseInteger, parseNumber } from "./util.js";

/** Node ids of the four sentinels. Every real node's id is at least FIRST. */
export const START = 0;
export const END = 1;
export const BACK = 2;
export const THINK = 3;
export const FIRST = 4;

/** Whether a node id is one of the sentinels (graph views and paths show them apart from the real nodes). */
export function isSentinel(id) {
  return Number.isInteger(id) && id >= 0 && id < FIRST;
}

/** The defaults of one thought (radixnet.thinking): its length, how deep it may question itself, and how often. */
export const THINK_LENGTH = 60;
export const THINK_DEPTH = 2;
export const THINK_QUESTIONS = 1;

/** What a thinking model may be asked for (Ollama's `think`): on, a level, off, or the model's own choice. */
export const THINK_LEVELS = [
  ["true", "on"],
  ["low", "low"],
  ["medium", "medium"],
  ["high", "high"],
  ["false", "off"],
  ["default", "the model's choice"],
];

/** The `think` field of POST /api/ollama/think for a THINK_LEVELS value: a bool, a level, or "default". */
export function thinkLevel(value) {
  const text = String(value ?? "").trim().toLowerCase();
  if (text === "" || text === "true") return true;
  if (text === "false") return false;
  return text;
}

/** How the `think` a server answered with reads: on, off, a level, or the model's choice (null). */
export function thinkLevelSays(think) {
  if (think === true) return "on";
  if (think === false) return "off";
  if (think === null || think === undefined || think === "default") return "the model's choice";
  return String(think);
}

/** The POST /api/think body for the Think tab's fields (numbers are held as strings and parsed here). */
export function thinkRequest(form) {
  const f = form || {};
  const mode = f.mode === "sample" ? "sample" : "beam";
  const body = {
    mode,
    k: Math.max(1, parseInteger(f.k, 5)),
    max_length: Math.max(0, parseInteger(f.maxLength, THINK_LENGTH)),
    depth: Math.max(0, parseInteger(f.depth, THINK_DEPTH)),
    questions: Math.max(0, parseInteger(f.questions, THINK_QUESTIONS)),
    learn: f.learn !== false,
  };
  // the text is sent as typed: under a character encoding "the cat " and "the cat" end at different nodes
  const about = String(f.about ?? "");
  if (about.trim()) body.about = about;
  if (mode === "sample") {
    body.temperature = Math.max(0, parseNumber(f.temperature, 1));
    const seed = parseInteger(f.seed, null);
    if (seed !== null) body.seed = seed;
  }
  return body;
}

/**
 * The POST /api/ollama/think body: the prompt, how many questions to think about, the thinking level and,
 * to teach the thinking to the network, the run. `lr` / `batchSize` are sent only when given (the sine model
 * reads them; the counting kinds have none). `overrides` are the Ollama URL and model the tab picked.
 */
export function ollamaThinkRequest(form, overrides = {}) {
  const f = form || {};
  const body = {
    prompt: String(f.prompt ?? ""),
    lines: Math.max(1, parseInteger(f.lines, 5)),
    think: thinkLevel(f.think),
    temperature: Math.max(0, parseNumber(f.temperature, 0.7)),
    ...overrides,
    train: Boolean(f.train),
  };
  const saveAs = String(f.saveAs ?? "").trim();
  if (saveAs) body.save_as = saveAs;
  if (body.train) {
    body.epochs = Math.max(0, parseInteger(f.epochs, 5));
    body.questions = f.questions !== false;
    body.with_answers = Boolean(f.withAnswers);
    if (f.lr !== undefined && f.lr !== null && f.lr !== "") body.lr = Math.max(0, parseNumber(f.lr, 0.05));
    if (f.batchSize !== undefined && f.batchSize !== null && f.batchSize !== "") {
      body.batch_size = Math.max(1, parseInteger(f.batchSize, 256));
    }
  }
  return body;
}

const THEN_SAYS = { back: "then backed up", think: "then went back to the thought", end: "then went on" };

/** The questions a thought asked itself (its `questions`, or none). */
export function questionsOf(thought) {
  return thought && Array.isArray(thought.questions) ? thought.questions.filter((q) => q && typeof q === "object") : [];
}

/**
 * One line saying what a thought did - the frontend's copy of radixnet.thinking.summarize:
 * thought “…”; questioned itself once; then backed up.
 */
export function summarizeThought(thought) {
  if (!thought || typeof thought !== "object") return "";
  const text = typeof thought.text === "string" ? thought.text : "";
  let said;
  if (text) said = `thought “${text}”`;
  else if (thought.stopped === "nothing" && thought.depth) said = "had nothing new to think";
  else said = "had nothing to think with yet";
  const parts = [said];
  const asked = questionsOf(thought).length || (Number.isInteger(thought.questioned) ? thought.questioned : 0);
  if (asked) parts.push(`questioned itself ${asked === 1 ? "once" : `${asked} times`}`);
  if (thought.then) parts.push(THEN_SAYS[thought.then] || `then ${thought.then}`);
  return parts.join("; ");
}

/** What set a thought off, in words. */
export function triggerSays(trigger) {
  switch (trigger) {
    case "asked":
      return "asked for";
    case "questioned":
      return "a question the thought above it asked itself";
    case "stutter":
      return "a voice caught itself saying something twice";
    case "repeat":
      return "a voice caught itself repeating the conversation";
    default:
      return trigger ? String(trigger) : "–";
  }
}

/** How a thought stopped, in words. */
export function stoppedSays(stopped) {
  switch (stopped) {
    case "end":
      return "at the end of a thought";
    case "length":
      return "out of length";
    case "nothing":
      return "it had nothing (new) to think";
    default:
      return stopped ? String(stopped) : "–";
  }
}

/** What a thought triggered when it stopped, in words. */
export function thenSays(then) {
  switch (then) {
    case "back":
      return "handed over to BACK: the walk goes round here";
    case "think":
      return "went back to the thought that asked";
    case "end":
      return "ended";
    default:
      return then ? String(then) : "–";
  }
}

/** What a thought taught the model, in words ("" when it taught nothing). */
export function learnedSays(thought) {
  const t = thought || {};
  const learned = [];
  if (Number.isInteger(t.taught) && t.taught >= 0) learned.push(`to stop and think at node ${t.taught}`);
  if (Number.isInteger(t.handed_over) && t.handed_over >= 0) learned.push(`to hand over at node ${t.handed_over}`);
  return learned.join(" and ");
}

/** A thought and the questions it asked itself, in reading order: [{thought, depth}] (depth 0 is the thought). */
export function thoughtTree(thought) {
  const out = [];
  const visit = (t, depth) => {
    if (!t || typeof t !== "object") return;
    out.push({ thought: t, depth });
    for (const q of questionsOf(t)) visit(q, depth + 1);
  };
  visit(thought, 0);
  return out;
}

const TERMINATORS = ".!?";
const SPACES = " \t\n\r";

/**
 * The sentences of a text that end in a question mark, each with the index it starts at - the frontend's
 * copy of radixnet.thinking.questions_in. A sentence runs from the first character after the previous
 * sentence's terminators (. ! ?) to the end of its own; a run of terminators holding a "?" makes it a
 * question. It is how the thinking an LLM wrote shows the questions it asked itself.
 */
export function questionsIn(text) {
  const value = String(text ?? "");
  const out = [];
  const n = value.length;
  let i = 0;
  while (i < n) {
    while (i < n && SPACES.includes(value[i])) i += 1;
    if (i >= n) break;
    const start = i;
    while (i < n && !TERMINATORS.includes(value[i])) i += 1;
    let j = i;
    while (j < n && TERMINATORS.includes(value[j])) j += 1;
    if (i > start && value.slice(i, j).includes("?")) out.push({ start, text: value.slice(start, j).trim() });
    i = j;
  }
  return out;
}

/** A text cut into runs with its questions marked: [{text, question}], which joined give the text back. */
export function questionRuns(text) {
  const value = String(text ?? "");
  const runs = [];
  let at = 0;
  for (const q of questionsIn(value)) {
    if (q.start > at) runs.push({ text: value.slice(at, q.start), question: false });
    runs.push({ text: value.slice(q.start, q.start + q.text.length), question: true });
    at = q.start + q.text.length;
  }
  if (at < value.length) runs.push({ text: value.slice(at), question: false });
  return runs;
}

/** Every thought a conversation had, in order: the turns' rethinks' thoughts (radixnet.thinking.thoughts_of). */
export function thoughtsOf(turns) {
  return (Array.isArray(turns) ? turns : [])
    .map((t) => (t && t.rethink && typeof t.rethink === "object" ? t.rethink.thought : null))
    .filter((t) => t && typeof t === "object");
}

/** The nodes a conversation taught to stop and think at, once each and in order (the CLI's `thought_at`). */
export function thoughtNodes(turns) {
  const nodes = new Set();
  for (const t of thoughtsOf(turns)) if (Number.isInteger(t.taught) && t.taught >= 0) nodes.add(t.taught);
  return [...nodes].sort((a, b) => a - b);
}
