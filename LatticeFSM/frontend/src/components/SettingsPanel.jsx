import { useEffect, useState } from "react";
import { LMS } from "../lm.js";
import { browserStorage, clearSettings, settingNames } from "../storage.js";
import { fmtInt } from "../util.js";
import { SelectField } from "./Fields.jsx";

/** How long after a change the count is read again: a setting is written once it stops changing. */
const RECOUNT_MS = 600;

const TEXT_SIZES = [
  ["13", "small (13 px)"],
  ["14", "normal (14 px)"],
  ["16", "large (16 px)"],
  ["18", "larger (18 px)"],
];

/**
 * Settings of this browser, none of them sent to the server: the language model the tabs talk to (the
 * header's selector), the text size, and how many settings are remembered here - with a two-click "forget
 * every saved setting" that reloads the page with the defaults.
 */
export default function SettingsPanel({ lm, setLm, textSize, setTextSize }) {
  const storage = browserStorage();
  const [armed, setArmed] = useState(false);
  const [count, setCount] = useState(() => (storage ? settingNames(storage).length : 0));

  // a setting is stored a moment after it changes (useStoredState), so count again once it has been
  useEffect(() => {
    if (!storage) return undefined;
    const timer = setTimeout(() => setCount(settingNames(storage).length), RECOUNT_MS);
    return () => clearTimeout(timer);
  }, [storage, lm, textSize]);

  return (
    <div className="panel">
      <div className="card">
        <h2>Settings</h2>
        <SelectField
          label="Language model"
          value={lm}
          onChange={setLm}
          options={LMS.map((m) => [m.id, `${m.label} - ${m.description}`])}
          hint="the header's selector: what Train, Predict, Generate, Talk and the rest talk to"
        />
        <SelectField label="Text size" value={textSize} onChange={setTextSize} options={TEXT_SIZES} />
      </div>

      <div className="card">
        <h2>This browser</h2>
        {!storage ? (
          <p className="muted">
            This browser keeps no site data (a private window, or site data blocked), so every setting lasts until the
            page is closed.
          </p>
        ) : (
          <>
            <p className="muted">
              {fmtInt(count)} setting{count === 1 ? " is" : "s are"} remembered here: every panel&apos;s fields (the
              text in the boxes, the numbers, the paths, the checkboxes) and the two above. Results, conversations and
              chosen uploads are not. Nothing here leaves this browser.
            </p>
            <div className="actions">
              <button
                type="button"
                className="danger small"
                onBlur={() => setArmed(false)}
                onClick={() => {
                  if (!armed) {
                    setArmed(true);
                    return;
                  }
                  clearSettings(storage);
                  window.location.reload();
                }}
              >
                {armed ? "Forget them all? Click again" : "Forget every saved setting"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
