/**
 * Tests for the THINK sentinel's display helpers (`src/thinking.js`) - plain `node --test`.
 * The question finder is checked against the same cases as radixnet.thinking.questions_in and
 * the Go QuestionsIn, and the summary against radixnet.thinking.summarize. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  FIRST,
  THINK,
  THINK_LEVELS,
  isSentinel,
  learnedSays,
  ollamaThinkRequest,
  questionRuns,
  questionsIn,
  stoppedSays,
  summarizeThought,
  thenSays,
  thinkLevel,
  thinkLevelSays,
  thinkRequest,
  thoughtNodes,
  thoughtTree,
  thoughtsOf,
  triggerSays,
} from "../src/thinking.js";

test("the sentinels are the first four node ids", () => {
  assert.equal(THINK, 3);
  assert.equal(FIRST, 4);
  for (const id of [0, 1, 2, 3]) assert.equal(isSentinel(id), true, String(id));
  for (const id of [4, 17, -1, 2.5, "2", null, undefined]) assert.equal(isSentinel(id), false, String(id));
});

test("questions are the sentences ending in a question mark, as Python and Go find them", () => {
  const cases = [
    ["Hmm. Is that right? Yes. Why not?", [[5, "Is that right?"], [25, "Why not?"]]],
    ["no questions here.", []],
    ["  really?!  ", [[2, "really?!"]]],
    ["a? b. c?", [[0, "a?"], [6, "c?"]]],
    ["", []],
    ["???", []], // a run of terminators with no sentence before it
    ["hmm, what did the cat do? it sat.", [[0, "hmm, what did the cat do?"]]],
  ];
  for (const [text, want] of cases) {
    assert.deepEqual(
      questionsIn(text).map((q) => [q.start, q.text]),
      want,
      JSON.stringify(text),
    );
  }
  assert.deepEqual(questionsIn(undefined), []);
});

test("a text cut into runs marks its questions and gives the text back", () => {
  const text = "Hmm. Is that right? Yes, it is. Why?";
  const runs = questionRuns(text);
  assert.deepEqual(runs, [
    { text: "Hmm. ", question: false },
    { text: "Is that right?", question: true },
    { text: " Yes, it is. ", question: false },
    { text: "Why?", question: true },
  ]);
  assert.equal(runs.map((r) => r.text).join(""), text);
  assert.deepEqual(questionRuns("no questions."), [{ text: "no questions.", question: false }]);
  assert.deepEqual(questionRuns(""), []);
});

test("a thought is summarised the way radixnet.thinking.summarize says it", () => {
  assert.equal(
    summarizeThought({ text: "", stopped: "nothing", depth: 0, then: "end", questions: [] }),
    "had nothing to think with yet; then went on",
  );
  assert.equal(
    summarizeThought({ text: "", stopped: "nothing", depth: 1, then: "think" }),
    "had nothing new to think; then went back to the thought",
  );
  const question = { text: "why is that so?", depth: 1, stopped: "end", then: "think", questions: [] };
  assert.equal(
    summarizeThought({ text: "the sky is blue", stopped: "end", then: "back", questions: [question] }),
    "thought “the sky is blue”; questioned itself once; then backed up",
  );
  assert.equal(
    summarizeThought({ text: "hmm", then: "end", questions: [question, question] }),
    "thought “hmm”; questioned itself 2 times; then went on",
  );
  // a record that only counts its questions still says so
  assert.equal(summarizeThought({ text: "hmm", then: "end", questioned: 1 }), "thought “hmm”; questioned itself once; then went on");
  assert.equal(summarizeThought({ text: "hmm", then: "wander" }), "thought “hmm”; then wander");
  assert.equal(summarizeThought(null), "");
});

test("the words for a trigger, a stop, a hand-over and a lesson", () => {
  assert.equal(triggerSays("asked"), "asked for");
  assert.match(triggerSays("stutter"), /twice/);
  assert.match(triggerSays("repeat"), /repeating/);
  assert.match(triggerSays("questioned"), /asked itself/);
  assert.equal(triggerSays("custom"), "custom");
  assert.equal(stoppedSays("length"), "out of length");
  assert.match(thenSays("back"), /BACK/);
  assert.equal(thenSays("end"), "ended");
  assert.equal(learnedSays({ taught: 5, handed_over: -1 }), "to stop and think at node 5");
  assert.equal(learnedSays({ taught: 5, handed_over: 5 }), "to stop and think at node 5 and to hand over at node 5");
  assert.equal(learnedSays({ taught: -1, handed_over: -1 }), "");
  assert.equal(learnedSays(undefined), "");
});

test("a thought and its questions read in order, one level deeper each", () => {
  const deepest = { text: "c", depth: 2, questions: [] };
  const middle = { text: "b", depth: 1, questions: [deepest] };
  const top = { text: "a", depth: 0, questions: [middle, null] };
  assert.deepEqual(
    thoughtTree(top).map(({ thought, depth }) => [thought.text, depth]),
    [["a", 0], ["b", 1], ["c", 2]],
  );
  assert.deepEqual(thoughtTree(null), []);
});

test("a conversation's thoughts, and the nodes it learned to stop and think at", () => {
  const turns = [
    { rethink: { thought: { text: "x", taught: 7 } } },
    { rethink: null },
    { rethink: { thought: null } },
    { rethink: { thought: { text: "y", taught: 5 } } },
    { rethink: { thought: { text: "z", taught: 7 } } },
    { rethink: { thought: { text: "w", taught: -1 } } },
    null,
  ];
  assert.deepEqual(thoughtsOf(turns).map((t) => t.text), ["x", "y", "z", "w"]);
  assert.deepEqual(thoughtNodes(turns), [5, 7]);
  assert.deepEqual(thoughtNodes(undefined), []);
});

test("a thinking level is what the API's think field takes", () => {
  assert.deepEqual(
    THINK_LEVELS.map(([value]) => value),
    ["true", "low", "medium", "high", "false", "default"],
  );
  assert.equal(thinkLevel("true"), true);
  assert.equal(thinkLevel(""), true);
  assert.equal(thinkLevel("false"), false);
  assert.equal(thinkLevel(" High "), "high");
  assert.equal(thinkLevel("default"), "default"); // every server reads it as the model's choice
  assert.equal(thinkLevelSays(true), "on");
  assert.equal(thinkLevelSays(false), "off");
  assert.equal(thinkLevelSays(null), "the model's choice");
  assert.equal(thinkLevelSays("medium"), "medium");
});

test("the Think tab asks for exactly what its fields say", () => {
  assert.deepEqual(thinkRequest({}), { mode: "beam", k: 5, max_length: 60, depth: 2, questions: 1, learn: true });
  assert.deepEqual(
    thinkRequest({
      about: "the cat ",
      mode: "sample",
      k: "3",
      maxLength: "40",
      temperature: "0.5",
      seed: "7",
      depth: "0",
      questions: "2",
      learn: false,
    }),
    { mode: "sample", k: 3, max_length: 40, depth: 0, questions: 2, learn: false, about: "the cat ", temperature: 0.5, seed: 7 },
  );
  // a blank seed draws at random, a blank about thinks about nothing in particular, and beam has no temperature
  const drawn = thinkRequest({ mode: "sample", seed: "", about: "   " });
  assert.equal("seed" in drawn, false);
  assert.equal("about" in drawn, false);
  assert.equal("temperature" in thinkRequest({ mode: "beam", temperature: "2" }), false);
  // out of range is pulled back in, as the fields' own limits say
  assert.equal(thinkRequest({ k: "0" }).k, 1);
  assert.equal(thinkRequest({ depth: "-1" }).depth, 0);
});

test("the Ollama thinking card asks for exactly what its fields say", () => {
  const overrides = { url: "http://ollama:11434", model: "qwen3" };
  assert.deepEqual(ollamaThinkRequest({ prompt: "the sea" }, overrides), {
    prompt: "the sea",
    lines: 5,
    think: true,
    temperature: 0.7,
    url: "http://ollama:11434",
    model: "qwen3",
    train: false,
  });
  // teaching it: the run's epochs, whether to learn where it questions itself, and the answers
  const taught = ollamaThinkRequest(
    { prompt: "x", lines: "3", think: "high", temperature: "0.2", train: true, epochs: "2", questions: false, withAnswers: true, saveAs: " sea.txt " },
    {},
  );
  assert.deepEqual(taught, {
    prompt: "x",
    lines: 3,
    think: "high",
    temperature: 0.2,
    train: true,
    save_as: "sea.txt",
    epochs: 2,
    questions: false,
    with_answers: true,
  });
  // a sine model reads a learning rate and a batch size; the counting kinds are sent none
  const sine = ollamaThinkRequest({ prompt: "x", train: true, lr: "0.1", batchSize: "32" });
  assert.equal(sine.lr, 0.1);
  assert.equal(sine.batch_size, 32);
  const counting = ollamaThinkRequest({ prompt: "x", train: true });
  assert.equal("lr" in counting, false);
  assert.equal("batch_size" in counting, false);
  // off and the model's choice go out as the servers read them; an override cannot switch training on
  assert.equal(ollamaThinkRequest({ prompt: "x", think: "false" }).think, false);
  assert.equal(ollamaThinkRequest({ prompt: "x", think: "default" }).think, "default");
  assert.equal(ollamaThinkRequest({ prompt: "x" }, { train: true }).train, false);
  assert.equal(ollamaThinkRequest({ prompt: "x", lines: "0" }).lines, 1);
});
