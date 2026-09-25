/**
 * Asking a model that was trained backwards (../../SPEC-SearchAndTraining.md
 * section 9).
 *
 * A run with "Read every text backwards" (`reverse` on /api/train) teaches the
 * model every text from its end to its start, so what it learns to continue is
 * what came *before*. Asked the right way round it answers nonsense; asked
 * backwards - the query turned around before it is sent, the answer turned
 * around when it comes back - it says what precedes the text it was given.
 *
 * Both turns are made in the model's own units, exactly as the servers'
 * `Encoding.reverse` makes them: code points for a character model, whole
 * words (written with single spaces) for a word model.
 *
 * Pure - no React, no storage - so `node --test` holds it to those rules.
 */

/**
 * Unicode White_Space: the separators every server splits words on (Python's
 * `split_words`, Go's `strings.Fields`, Rust's `split_whitespace`) - the same
 * 25 code points, and not the four `str.split()` adds.
 */
const WORD_GAP = /\p{White_Space}+/u;

/** `text` read backwards in `units` ("chars" or "words"): its last character, or word, first. */
export function reverseUnits(text, units = "chars") {
  const value = String(text ?? "");
  if (units === "words") return value.split(WORD_GAP).filter(Boolean).reverse().join(" ");
  // Array.from walks code points, so a character outside the BMP stays whole
  return Array.from(value).reverse().join("");
}

/**
 * A backwards answer in reading order. The model was sent the query turned
 * around (`prefix`) and went on with `continuation`; read the right way round
 * the continuation comes first - it is what came before - and the query last.
 * `gap` is the space a word model keeps between two words.
 */
export function readingOrder(prefix, continuation, units = "chars") {
  const before = reverseUnits(continuation, units);
  const query = reverseUnits(prefix, units);
  return { before, gap: units === "words" && before && query ? " " : "", query };
}
