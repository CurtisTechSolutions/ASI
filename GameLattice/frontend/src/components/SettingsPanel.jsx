import { useEffect, useState } from "react";
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
 * Settings of this browser, none of them sent to the server: the text size, where the API is, and how many panel
 * settings are remembered here - with a two-click "forget every saved setting" that reloads with the defaults.
 */
export default function SettingsPanel({ textSize, setTextSize, health }) {
  const storage = browserStorage();
  const [armed, setArmed] = useState(false);
  const [count, setCount] = useState(() => (storage ? settingNames(storage).length : 0));

  useEffect(() => {
    if (!storage) return undefined;
    const timer = setTimeout(() => setCount(settingNames(storage).length), RECOUNT_MS);
    return () => clearTimeout(timer);
  }, [storage, textSize]);

  return (
    <div className="panel">
      <div className="card">
        <h2>Settings</h2>
        <SelectField label="Text size" value={textSize} onChange={setTextSize} options={TEXT_SIZES} />
        <h3>The server</h3>
        <p className="muted">
          The API is under <code>/api/</code> on the page&apos;s own origin: <code>gamelattice serve</code> serves this
          page and the API together, and in development Vite proxies <code>/api</code> to it (127.0.0.1:8000, or{" "}
          <code>VITE_PROXY_TARGET</code>). Its host and port are set where it is started (<code>make serve HOST=… PORT=…</code>).
          {health ? ` Connected: gamelattice ${health.version}, ${health.language} on ${health.shape.join(" × ")}.` : ""}
        </p>
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
              {fmtInt(count)} setting{count === 1 ? " is" : "s are"} remembered here under <code>gamelattice.v1.</code>:
              every panel&apos;s fields (the language, the states, the window, the arms, the episodes, the text, Nim&apos;s
              settings) and the text size. Results and traces are not. Nothing here leaves this browser.
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
                {armed ? "Forget them all? Click again" : "Reset settings"}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
