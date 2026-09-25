/**
 * Tests for asking a model backwards (`src/backwards.js`) - plain `node --test`:
 * the query and the answer are turned around exactly as the servers'
 * `Encoding.reverse` turns a training text around
 * (../../SPEC-SearchAndTraining.md section 9), or a model trained backwards is
 * asked something it never read. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { test } from "node:test";

import { readingOrder, reverseUnits } from "../src/backwards.js";

/** The code points Python's `split_words` (and Go's `strings.Fields`, Rust's `split_whitespace`) split words on. */
const SEPARATORS = [
  0x9, 0xa, 0xb, 0xc, 0xd, 0x20, 0x85, 0xa0, 0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007,
  0x2008, 0x2009, 0x200a, 0x2028, 0x2029, 0x202f, 0x205f, 0x3000,
];

test("characters are turned around code point by code point", () => {
  assert.equal(reverseUnits("héllo wörld", "chars"), "dlröw olléh");
  assert.equal(reverseUnits("the cat\n", "chars"), "\ntac eht");
  assert.equal(reverseUnits("", "chars"), "");
  assert.equal(reverseUnits(null), "");
  // a character outside the BMP is one code point, not two halves
  assert.equal(reverseUnits("a😀b", "chars"), "b😀a");
  // a combining accent is a code point of its own, so twice is the text again
  const text = "café au lait";
  assert.equal(reverseUnits(reverseUnits(text)), text);
});

test("words are turned around whole, with single spaces", () => {
  assert.equal(reverseUnits("  the  cat\tsat\n", "words"), "sat cat the");
  assert.equal(reverseUnits("one", "words"), "one");
  assert.equal(reverseUnits(" \t ", "words"), "");
  assert.equal(reverseUnits("the cat, sat.", "words"), "sat. cat, the");
  // twice is the text as a word model keeps it
  assert.equal(reverseUnits(reverseUnits("a  b\nc", "words"), "words"), "a b c");
});

test("words split on exactly the separators the servers split on", () => {
  for (const cp of SEPARATORS) {
    const gap = String.fromCodePoint(cp);
    assert.equal(reverseUnits(`a${gap}b`, "words"), "b a", `U+${cp.toString(16)} separates words`);
  }
  // what str.split() adds, and the zero-width and byte-order marks, stay inside a word
  for (const cp of [0x1c, 0x1d, 0x1e, 0x1f, 0x200b, 0xfeff, 0x180e]) {
    const inside = `a${String.fromCodePoint(cp)}b`;
    assert.equal(reverseUnits(`${inside} c`, "words"), `c ${inside}`, `U+${cp.toString(16)} is part of a word`);
  }
});

test("an answer reads the right way round: what came before, then the query", () => {
  // the model was sent "lazy dog" turned around, and went on with " the over" turned around
  assert.deepEqual(readingOrder("god yzal", " eht revo", "chars"), { before: "over the ", gap: "", query: "lazy dog" });
  assert.deepEqual(readingOrder("dog lazy", "the over", "words"), { before: "over the", gap: " ", query: "lazy dog" });
  // nothing came before: no gap to keep
  assert.deepEqual(readingOrder("dog lazy", "", "words"), { before: "", gap: "", query: "lazy dog" });
  // read together, the pieces are the model's whole text turned around
  const { before, gap, query } = readingOrder("god yzal", " eht revo kciuq", "chars");
  assert.equal(before + gap + query, reverseUnits("god yzal eht revo kciuq", "chars"));
});
