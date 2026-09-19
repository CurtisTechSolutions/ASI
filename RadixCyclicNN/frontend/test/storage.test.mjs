/**
 * Tests for the settings store (`src/storage.js`) - plain `node --test`, no
 * dependencies: the module is pure and takes its storage as an argument, so a
 * Map standing in for `localStorage` is all it needs. Run with `npm test`.
 */

import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import {
  MAX_VALUE_CHARS,
  STORAGE_PREFIX,
  browserStorage,
  clearSettings,
  readSetting,
  removeSetting,
  resetBrowserStorage,
  sameShape,
  settingNames,
  storageKey,
  writeSetting,
} from "../src/storage.js";

/** A `localStorage` stand-in; `failOn` makes one method throw the way a full or blocked store does. */
function fakeStorage(entries = {}, failOn = null) {
  const map = new Map(Object.entries(entries));
  const guard = (name) => {
    if (failOn === name) throw new Error(`${name} refused`);
  };
  return {
    map,
    get length() {
      guard("length");
      return map.size;
    },
    key(i) {
      guard("key");
      return [...map.keys()][i] ?? null;
    },
    getItem(k) {
      guard("getItem");
      return map.has(k) ? map.get(k) : null;
    },
    setItem(k, v) {
      guard("setItem");
      map.set(k, String(v));
    },
    removeItem(k) {
      guard("removeItem");
      map.delete(k);
    },
  };
}

afterEach(() => {
  delete globalThis.localStorage;
  resetBrowserStorage();
});

test("a stored setting comes back", () => {
  const storage = fakeStorage();
  assert.equal(writeSetting(storage, "train.epochs", "5"), true);
  assert.equal(storage.map.get(storageKey("train.epochs")), '"5"');
  assert.equal(readSetting(storage, "train.epochs", "10"), "5");
  // every type a panel field holds survives the round trip
  for (const value of ["", "text", 0, 42, 1.5, true, false, null, ["a", "b"], { a: 1 }]) {
    writeSetting(storage, "probe", value);
    assert.deepEqual(readSetting(storage, "probe", value), value);
  }
});

test("the default wins when nothing is stored, or the shape changed", () => {
  const storage = fakeStorage();
  assert.equal(readSetting(storage, "missing", "5"), "5");
  writeSetting(storage, "train.epochs", 5); // a number where the panel now keeps a string
  assert.equal(readSetting(storage, "train.epochs", "10"), "10");
  writeSetting(storage, "train.files", { not: "a list" });
  assert.deepEqual(readSetting(storage, "train.files", []), []);
  storage.map.set(storageKey("broken"), "{not json");
  assert.equal(readSetting(storage, "broken", "default"), "default");
  assert.equal(sameShape("a", "b"), true);
  assert.equal(sameShape(1, "b"), false);
  assert.equal(sameShape({}, []), false);
  assert.equal(sameShape("anything", null), true); // no shape to insist on
});

test("a stored object is merged into the defaults", () => {
  const storage = fakeStorage();
  writeSetting(storage, "codegen.settings", { rounds: "3", gone: "old setting", judge: "yes" });
  const defaults = { rounds: "1", judge: true, strictness: "strict" };
  assert.deepEqual(readSetting(storage, "codegen.settings", defaults), {
    rounds: "3", // kept
    judge: true, // the stored value changed type: the default stands
    strictness: "strict", // added since the value was stored
  });
  // "gone" is not in the defaults any more and does not come back
  assert.equal("gone" in readSetting(storage, "codegen.settings", defaults), false);
});

test("values too large to keep are dropped, and take the older value with them", () => {
  const storage = fakeStorage();
  assert.equal(writeSetting(storage, "train.text", "small"), true);
  const huge = "x".repeat(MAX_VALUE_CHARS + 1);
  assert.equal(writeSetting(storage, "train.text", huge), false);
  assert.equal(readSetting(storage, "train.text", "fallback"), "fallback");
  assert.equal(storage.map.has(storageKey("train.text")), false);
});

test("unserialisable values are not settings", () => {
  const storage = fakeStorage();
  assert.equal(writeSetting(storage, "images.file", undefined), false);
  const cycle = {};
  cycle.self = cycle;
  assert.equal(writeSetting(storage, "cycle", cycle), false);
  assert.equal(settingNames(storage).length, 0);
});

test("a storage that refuses is the same as no storage", () => {
  for (const failing of ["getItem", "setItem", "removeItem", "length"]) {
    const storage = fakeStorage({}, failing);
    assert.equal(readSetting(storage, "train.epochs", "5"), "5");
    assert.doesNotThrow(() => writeSetting(storage, "train.epochs", "7"));
    assert.doesNotThrow(() => clearSettings(storage));
  }
  assert.equal(readSetting(null, "train.epochs", "5"), "5");
  assert.equal(writeSetting(null, "train.epochs", "5"), false);
  assert.deepEqual(settingNames(null), []);
  assert.equal(clearSettings(null), 0);
});

test("only this app's keys are listed and cleared", () => {
  const storage = fakeStorage({ "some.other.app": "1", [storageKey("a")]: '"1"', [storageKey("b")]: '"2"' });
  assert.deepEqual(settingNames(storage).sort(), ["a", "b"]);
  assert.equal(clearSettings(storage), 2);
  assert.deepEqual(settingNames(storage), []);
  assert.equal(storage.map.get("some.other.app"), "1"); // not ours, not touched
  assert.equal(removeSetting(storage, "gone"), true);
});

test("the browser storage is probed once, and never throws", () => {
  globalThis.localStorage = fakeStorage();
  const storage = browserStorage();
  assert.ok(storage);
  assert.equal(browserStorage(), storage); // memoised
  assert.equal(storage.map.has(`${STORAGE_PREFIX}probe`), false); // the probe cleans up after itself

  resetBrowserStorage();
  globalThis.localStorage = fakeStorage({}, "setItem"); // a private window: present but unusable
  assert.equal(browserStorage(), null);

  resetBrowserStorage();
  delete globalThis.localStorage;
  assert.equal(browserStorage(), null);
});
