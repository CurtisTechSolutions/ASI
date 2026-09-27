import assert from "node:assert/strict";
import { test } from "node:test";

import { DEFAULTS, Endpointer, bytesToBase64, decodePcm, decodeWav, historyOf, levelOf, pairTranscript } from "../src/voice.js";

test("the endpointer cuts an utterance out of the levels alone", () => {
  const ep = new Endpointer({ threshold: 0.02, minSpeechMs: 250, silenceMs: 700, startFrames: 2 });
  assert.equal(ep.feed(0.001, 0), null); // quiet
  assert.equal(ep.feed(0.05, 50), null); // one loud frame is not a start yet
  assert.equal(ep.feed(0.05, 100), "start"); // two in a row are
  assert.equal(ep.startedAt, 50); // the utterance began at the first loud frame
  for (let t = 150; t < 800; t += 50) assert.equal(ep.feed(0.05, t), null);
  for (let t = 800; t < 1450; t += 50) assert.equal(ep.feed(0.001, t), null); // going quiet
  assert.equal(ep.feed(0.001, 1500), "end"); // 700 ms of quiet ends it
  assert.equal(ep.active, false);
  // a click: loud for less than the minimum is cancelled, not an utterance
  assert.equal(ep.feed(0.5, 2000), null);
  assert.equal(ep.feed(0.5, 2050), "start");
  assert.equal(ep.feed(0.5, 2100), null);
  assert.equal(ep.feed(0.0, 2900), "cancel");
  // and a level that goes on for ever is cut at maxMs
  const long = new Endpointer({ maxMs: 1000, silenceMs: 700, minSpeechMs: 100, startFrames: 1 });
  assert.equal(long.feed(0.1, 0), "start");
  assert.equal(long.feed(0.1, 500), null);
  assert.equal(long.feed(0.1, 1000), "end");
  assert.equal(DEFAULTS.silenceMs, 700);
});

test("the level is the RMS of a frame", () => {
  assert.equal(levelOf(new Float32Array([0, 0, 0])), 0);
  assert.equal(levelOf(new Float32Array([0.5, -0.5, 0.5, -0.5])), 0.5);
  assert.equal(levelOf(new Float32Array(0)), 0);
});

test("an utterance is paired with the dictation's finals by time, each used once", () => {
  const finals = [
    { text: "hello", at: 1000, used: false },
    { text: "the cat sat", at: 5200, used: false }, // committed a little after the utterance ended at 5000
    { text: "later", at: 9000, used: false },
  ];
  assert.equal(pairTranscript(finals, 3000, 5000), "the cat sat");
  assert.deepEqual(finals.map((f) => f.used), [false, true, false]);
  assert.equal(pairTranscript(finals, 3000, 5000), ""); // gone
  assert.equal(pairTranscript(finals, 500, 1200), "hello");
  assert.equal(pairTranscript([{ text: "  ", at: 10, used: false }], 0, 20), "");
});

test("PCM and WAV chunks decode to samples", () => {
  const pcm = new Int16Array([0, 16384, -16384, 32767, -32768]);
  const bytes = new Uint8Array(pcm.buffer);
  const samples = decodePcm(bytesToBase64(bytes));
  assert.deepEqual([...samples].map((v) => Math.round(v * 1000) / 1000), [0, 0.5, -0.5, 1, -1]);
  const wav = new Uint8Array(44 + bytes.length);
  const view = new DataView(wav.buffer);
  const ascii = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) view.setUint8(offset + i, text.charCodeAt(i));
  };
  ascii(0, "RIFF");
  view.setUint32(4, 36 + bytes.length, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, 8000, true);
  view.setUint32(28, 16000, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, bytes.length, true);
  wav.set(bytes, 44);
  const decoded = decodeWav(bytesToBase64(wav));
  assert.equal(decoded.rate, 8000);
  assert.deepEqual([...decoded.samples].map((v) => Math.round(v * 1000) / 1000), [0, 0.5, -0.5, 1, -1]);
  // bytesToBase64 takes any length, in pieces
  const big = new Uint8Array(70000).fill(65);
  assert.equal(atob(bytesToBase64(big)).length, 70000);
});

test("the history is the conversation oldest first, the model's units as it said them", () => {
  const turns = [
    { you: { text: "a dog", heard: true }, model: { text: "a dog sat", spelled: "a dog sat" } },
    { you: { text: "", heard: false }, model: null }, // an utterance no words were heard in
    { you: { text: "the cat", heard: true }, model: { text: "DH AH0 # K AE1 T", spelled: "the cat" } },
  ];
  assert.deepEqual(historyOf(turns), [
    ["You", "the cat"],
    ["Model", "DH AH0 # K AE1 T"],
    ["You", "a dog"],
    ["Model", "a dog sat"],
  ]);
  assert.deepEqual(historyOf(turns, 1), [["You", "a dog"], ["Model", "a dog sat"]]);
});
