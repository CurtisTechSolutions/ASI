/**
 * Panel settings remembered in the browser.
 *
 * Every settings field of every panel (epochs, learning rates, modes, the
 * topic of a tutor run, the text in a box) is kept in `localStorage` under the
 * `radixnet.v1.` prefix, so a reload - or coming back tomorrow - finds the
 * forms as they were left. Results, job state, server data and uploaded-file
 * selections are *not* settings and are never stored.
 *
 * Everything here is pure and takes the storage as an argument, so it can be
 * tested against a plain object and used where the browser has no storage at
 * all: `localStorage` throws or is missing in private windows, with site data
 * blocked, and outside a browser, and a write can fail at any time when the
 * quota is full. Every path answers with the caller's fallback instead, so a
 * panel keeps working with plain React state.
 */

/** Namespace of every stored setting; the version lets a future format ignore old values wholesale. */
export const STORAGE_PREFIX = "radixnet.v1.";

/** Values larger than this are not stored (a pasted corpus can be megabytes; the quota is ~5 MB). */
export const MAX_VALUE_CHARS = 256 * 1024;

/** The `localStorage` key a setting is stored under. */
export function storageKey(name) {
  return STORAGE_PREFIX + name;
}

/**
 * Is a stored value still usable in place of `fallback`?
 *
 * The defaults say what a setting looks like; a value of another shape comes
 * from an older version of the app (or a hand-edited store) and is ignored.
 */
export function sameShape(value, fallback) {
  if (fallback === null || fallback === undefined) return true; // no shape to insist on
  if (Array.isArray(fallback)) return Array.isArray(value);
  if (typeof fallback === "object") return Boolean(value) && typeof value === "object" && !Array.isArray(value);
  return typeof value === typeof fallback;
}

/** A stored object is merged into the defaults: new settings appear, unknown ones are dropped. */
function merge(value, fallback) {
  if (!fallback || typeof fallback !== "object" || Array.isArray(fallback)) return value;
  const out = { ...fallback };
  for (const key of Object.keys(fallback)) {
    if (Object.prototype.hasOwnProperty.call(value, key) && sameShape(value[key], fallback[key])) {
      out[key] = value[key];
    }
  }
  return out;
}

/** The stored value of `name`, or `fallback` when there is none, it is unreadable or it changed shape. */
export function readSetting(storage, name, fallback) {
  if (!storage) return fallback;
  let raw;
  try {
    raw = storage.getItem(storageKey(name));
  } catch {
    return fallback;
  }
  if (raw === null || raw === undefined) return fallback;
  let value;
  try {
    value = JSON.parse(raw);
  } catch {
    return fallback;
  }
  if (!sameShape(value, fallback)) return fallback;
  return merge(value, fallback);
}

/** Store `value` under `name`; answers whether it was stored (a full quota or a huge value is not an error). */
export function writeSetting(storage, name, value) {
  if (!storage) return false;
  let raw;
  try {
    raw = JSON.stringify(value);
  } catch {
    return false; // a File, a Set, a cycle: not a setting
  }
  if (raw === undefined) return false;
  if (raw.length > MAX_VALUE_CHARS) {
    removeSetting(storage, name); // too big to keep: forget the older, smaller value too
    return false;
  }
  try {
    storage.setItem(storageKey(name), raw);
    return true;
  } catch {
    return false;
  }
}

/** Has `name` ever been stored? (A panel seeding a field from the server must not overwrite a chosen value.) */
export function hasSetting(storage, name) {
  if (!storage) return false;
  try {
    return storage.getItem(storageKey(name)) !== null;
  } catch {
    return false;
  }
}

export function removeSetting(storage, name) {
  try {
    storage.removeItem(storageKey(name));
    return true;
  } catch {
    return false;
  }
}

/** The names (without the prefix) of every stored setting. */
export function settingNames(storage) {
  if (!storage) return [];
  const names = [];
  try {
    for (let i = 0; i < storage.length; i += 1) {
      const key = storage.key(i);
      if (typeof key === "string" && key.startsWith(STORAGE_PREFIX)) names.push(key.slice(STORAGE_PREFIX.length));
    }
  } catch {
    return names;
  }
  return names;
}

/** Forget every stored setting (other keys in the same origin are left alone); returns how many went. */
export function clearSettings(storage) {
  const names = settingNames(storage);
  let removed = 0;
  for (const name of names) {
    if (removeSetting(storage, name)) removed += 1;
  }
  return removed;
}

let cached;

/**
 * The browser's `localStorage`, or `null` where it cannot be used.
 *
 * Probed once with a real write: a private window or blocked site data may
 * expose the object and throw only when it is used.
 */
export function browserStorage() {
  if (cached !== undefined) return cached;
  cached = null;
  try {
    const storage = globalThis.localStorage;
    if (storage) {
      const probe = `${STORAGE_PREFIX}probe`;
      storage.setItem(probe, "1");
      storage.removeItem(probe);
      cached = storage;
    }
  } catch {
    cached = null;
  }
  return cached;
}

/** Forget the probed storage (tests). */
export function resetBrowserStorage() {
  cached = undefined;
}
