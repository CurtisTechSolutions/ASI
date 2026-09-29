/**
 * What a model of sounds said, in words.
 *
 * A model whose units are phones or syllables answers in sounds
 * (`DH.AH0 # K.AE1.T`), and every server spells each answer back into English
 * beside them (`spelled_prediction` / `spelled_turn` in
 * ../../RadixCyclicNN/radixnet/encoding.py, and the Go and Rust ports): `spelled` is the
 * whole text in words, and `spelled_continuation` (a prediction) or
 * `spelled_reply` (a conversation's turn) the part of those words the model
 * wrote itself. A model of letters, words or acoustic units is its own
 * spelling, and its records carry no such field.
 *
 * Pure - no React, no storage - so `node --test` holds it to those rules.
 */

/**
 * A record's words, split where the model's own part begins: `{ whole, head,
 * tail }` - the English the record spells, what came before the part the
 * model wrote, and that part - or null for a record that carries no spelling.
 * `part` names the field that holds the tail (`spelled_continuation` or
 * `spelled_reply`); a record without it is all tail, and so is one whose tail
 * does not end its whole.
 */
export function spelledParts(record, part = "spelled_continuation") {
  if (!record || typeof record !== "object" || typeof record.spelled !== "string") return null;
  const whole = record.spelled;
  const given = typeof record[part] === "string" ? record[part] : whole;
  const tail = whole.endsWith(given) ? given : whole;
  return { whole, head: whole.slice(0, whole.length - tail.length), tail };
}
