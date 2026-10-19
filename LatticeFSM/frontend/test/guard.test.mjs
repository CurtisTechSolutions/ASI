/** Tests for the guard's helpers (`src/guard.js`): the server's tokens, spans laid on a text, the notice's line. */

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  guardNoticeText,
  guardPieces,
  guardTokens,
  highlightPieces,
  markAlpha,
  pct,
  spansAlign,
  tokenMarks,
  verdictClass,
} from "../src/guard.js";

test("guardTokens cuts words as the server does: apostrophes inside words, other characters alone", () => {
  assert.deepEqual(guardTokens("Don't stop, me now!"), ["don't", "stop", ",", "me", "now", "!"]);
  assert.deepEqual(guardTokens("rock'n'roll 'quoted' it's"), ["rock'n'roll", "'", "quoted", "'", "it's"]);
  assert.deepEqual(guardTokens("a_b 3.14"), ["a_b", "3", ".", "14"]);
  assert.deepEqual(guardTokens("ÉCOLE naïve"), ["école", "naïve"]);
  assert.deepEqual(guardTokens("  "), []);
  assert.deepEqual(guardTokens("Keep Case", "word", false), ["Keep", "Case"]);
  // an apostrophe before a digit or at the end ends the word
  assert.deepEqual(guardTokens("dogs' 9'5"), ["dogs", "'", "9", "'", "5"]);
});

test("guardTokens reads characters from the trimmed text, a space inside as _", () => {
  assert.deepEqual(guardTokens("  Hi you ", "char"), ["h", "i", "_", "y", "o", "u"]);
});

test("guardTokens cannot lay sounds or base64 on the text", () => {
  assert.equal(guardTokens("hello", "sound"), null);
  assert.equal(guardPieces("hello", "base64"), null);
});

test("guardPieces join back into the text, with the tokens numbered", () => {
  const text = "  so,  what?\tno ";
  const pieces = guardPieces(text);
  assert.equal(pieces.map((p) => p.text).join(""), text);
  assert.deepEqual(
    pieces.filter((p) => p.index >= 0).map((p) => [p.index, p.text]),
    [
      [0, "so"],
      [1, ","],
      [2, "what"],
      [3, "?"],
      [4, "no"],
    ],
  );
  // runs of space are one gap
  assert.deepEqual(pieces.filter((p) => p.index < 0).map((p) => p.text), ["  ", "  ", "\t", " "]);
  const chars = guardPieces(" ab ", "char");
  assert.equal(chars.map((p) => p.text).join(""), " ab ");
  assert.deepEqual(chars.map((p) => p.index), [-1, 0, 1, -1]);
});

test("spansAlign checks each span's tokens against the text's window", () => {
  const tokens = guardTokens("The cat sat");
  assert.equal(spansAlign(tokens, [{ start: 1, end: 3, tokens: ["Cat", "sat"] }]), true);
  assert.equal(spansAlign(tokens, [{ start: 1, end: 3, tokens: ["dog", "sat"] }]), false);
  assert.equal(spansAlign(tokens, [{ start: 2, end: 4 }]), false); // past the end
  assert.equal(spansAlign(tokens, [{ start: 0, end: 1 }]), true); // no tokens: taken on trust
  assert.equal(spansAlign(tokens, []), true);
  assert.equal(spansAlign(null, []), false);
});

test("tokenMarks keeps the most evidence over a token and each reason once", () => {
  const marks = tokenMarks(4, [
    { start: 0, end: 2, evidence: 1, reason: "rude" },
    { start: 1, end: 3, evidence: 3, reason: "off topic" },
    { start: 1, end: 2, evidence: 0.5, reason: "rude" },
    { start: 3, end: 3, evidence: 9 }, // empty: ignored
    null,
  ]);
  assert.equal(marks[0].evidence, 1);
  assert.deepEqual(marks[0].reasons, ["rude"]);
  assert.equal(marks[1].evidence, 3);
  assert.deepEqual(marks[1].reasons, ["rude", "off topic"]);
  assert.deepEqual(marks[1].spans, [0, 1, 2]);
  assert.equal(marks[2].evidence, 3);
  assert.equal(marks[3], null);
  // a window past the text is cut at its end
  assert.equal(tokenMarks(2, [{ start: 1, end: 5, evidence: 1 }]).length, 2);
});

test("markAlpha tints from faint to strong at the peak", () => {
  assert.equal(markAlpha(0, 2), 0);
  assert.equal(markAlpha(2, 2), 0.6);
  assert.equal(markAlpha(1, 2), 0.38);
  assert.equal(markAlpha(5, 0), 0.6);
  assert.equal(markAlpha("x", 1), 0);
});

test("highlightPieces lays the spans on the text and marks the space inside a span", () => {
  const text = "Hello, you idiot!";
  const spans = [{ start: 2, end: 4, tokens: ["you", "idiot"], evidence: 2, reason: "insult", next: "!" }];
  const laid = highlightPieces(text, spans);
  assert.equal(laid.unit, "word");
  assert.equal(laid.pieces.map((p) => p.text).join(""), text);
  const marked = laid.pieces.filter((p) => p.mark).map((p) => p.text);
  assert.deepEqual(marked, ["you", " ", "idiot"]);
  assert.deepEqual(laid.pieces.find((p) => p.text === "you").mark.reasons, ["insult"]);
  // the space before "you" is outside the span
  assert.equal(laid.pieces[2].text, " ");
  assert.equal(laid.pieces[2].mark, null);
});

test("highlightPieces finds characters when words do not fit, and gives up when nothing does", () => {
  const laid = highlightPieces("ab c", [{ start: 1, end: 3, tokens: ["b", "_"], evidence: 1 }]);
  assert.equal(laid.unit, "char");
  assert.deepEqual(laid.pieces.filter((p) => p.mark).map((p) => p.text), ["b", " "]);
  assert.equal(highlightPieces("ab c", [{ start: 0, end: 1, tokens: ["zz"] }]), null);
  assert.equal(highlightPieces("hello", [{ start: 0, end: 1 }], "sound"), null);
  // no spans: the whole text, unmarked
  assert.deepEqual(highlightPieces("hi there", []).pieces.map((p) => p.mark), [null, null, null]);
});

test("guardNoticeText says what was judged and vetoed", () => {
  assert.equal(guardNoticeText(null), "");
  assert.equal(guardNoticeText({ judged: 9, vetoed: 2, all_vetoed: false }), "the guard judged 9 samples and vetoed 2");
  assert.equal(guardNoticeText({ judged: 1, vetoed: 0 }), "the guard judged 1 sample and vetoed 0");
  assert.equal(
    guardNoticeText({ judged: 3, vetoed: 3, all_vetoed: true }, "replies"),
    "the guard judged 3 replies and vetoed 3 · all vetoed: kept the least risky",
  );
});

test("verdictClass and pct", () => {
  assert.equal(verdictClass("reject"), "fail");
  assert.equal(verdictClass("suspect"), "unrated");
  assert.equal(verdictClass("pass"), "pass");
  assert.equal(verdictClass(undefined), "unrated");
  assert.equal(pct(0.25), "25%");
  assert.equal(pct(null), "–");
});
