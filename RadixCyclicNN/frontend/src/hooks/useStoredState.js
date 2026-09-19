import { useEffect, useRef, useState } from "react";
import { browserStorage, readSetting, writeSetting } from "../storage.js";

/** Keystrokes are cheap, writes are not: a value is stored once it stops changing. */
const WRITE_DELAY_MS = 250;

/**
 * `useState` that remembers its value in the browser (see `storage.js`).
 *
 * Drop-in for a panel's settings field: the initial value comes from
 * `localStorage` when a value of the same shape was stored under `name`, else
 * from `initialValue`, and every later change is written back (debounced).
 * Nothing is written until the value actually changes, so the defaults of a
 * panel nobody touched leave no trace. Where the browser has no usable
 * storage this is exactly `useState`.
 *
 * `name` must be stable and unique per field ("tutor.topic", "train.epochs");
 * two mounted components sharing a name would share the stored value but not
 * their state, so give each instance its own prefix.
 */
export function useStoredState(name, initialValue) {
  const storage = browserStorage();
  const [value, setValue] = useState(() => readSetting(storage, name, initialValue));
  const stored = useRef(true); // the value just read back is already stored

  useEffect(() => {
    if (stored.current) {
      stored.current = false;
      return undefined;
    }
    const timer = setTimeout(() => writeSetting(storage, name, value), WRITE_DELAY_MS);
    return () => clearTimeout(timer);
  }, [storage, name, value]);

  return [value, setValue];
}
