import { useEffect, useState } from "react";
import { browserStorage, clearSettings, settingNames } from "../storage.js";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { fmtInt } from "../util.js";
import BackwardsField from "./BackwardsField.jsx";
import SearchFields from "./SearchFields.jsx";
import TraversalFields from "./TraversalFields.jsx";
import TrainingPlanFields from "./TrainingPlanFields.jsx";

/** How long after a change the count is read again: the settings are written once they stop changing. */
const RECOUNT_MS = 600;

/** "Forget everything this browser remembers": two clicks, then a reload with the defaults. */
function BrowserCard({ changed }) {
  const storage = browserStorage();
  const [armed, setArmed] = useState(false);
  const [count, setCount] = useState(() => (storage ? settingNames(storage).length : 0));

  // a setting is stored a moment after it changes (useStoredState), so count again once it has been
  useEffect(() => {
    if (!storage) return undefined;
    const timer = setTimeout(() => setCount(settingNames(storage).length), RECOUNT_MS);
    return () => clearTimeout(timer);
  }, [storage, changed]);

  return (
    <div className="card">
      <h2>This browser</h2>
      {!storage ? (
        <p className="muted">
          This browser keeps no site data (a private window, or site data blocked), so every setting lasts until
          the page is closed.
        </p>
      ) : (
        <>
          <p className="muted">
            {fmtInt(count)} setting(s) are remembered here - these, and every panel&apos;s own fields (epochs,
            rates, prefixes, prompts, the text in the boxes). Results, ratings and uploaded-file choices are not.
            Nothing here leaves this browser or is saved with a model.
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
  );
}

/**
 * Settings: the site-wide settings of this browser - what every search and
 * every training run starts from.
 *
 * The **traversal** (what a search looks for), the **sampling filters** and
 * the **diversity** (how a sampled walk and a beam choose), whether a query is
 * asked **backwards** (for a model trained backwards), and **how a run walks
 * its texts** (the order, the curriculum, the replay and the early stop). The
 * Predict, Generate and Train tabs show the same controls, so a change made
 * there is a change made here. What belongs to the model instead -
 * its encoding and its score function - is on the Model settings tab.
 */
export default function SettingsPanel({ status }) {
  const { network, search, training, backwards } = useSiteSettings();
  return (
    <>
      <div className="card wide">
        <h2>Settings</h2>
        <p className="muted">
          The settings of this browser: what every search and every training run starts from, whichever model is
          loaded. They are remembered here and shared by the tabs that use them - Predict, Generate and Train show
          the same controls. What belongs to a model and is saved with it - the encoding it reads text in, its
          score function - is on the <a href="#model">Model settings</a> tab.
        </p>
      </div>

      <div className="card">
        <h2>Traversal</h2>
        <TraversalFields />
        {network.shared ? (
          <div className="actions">
            <button type="button" className="small" disabled={network.traversal === "reward"} onClick={network.reset}>
              Back to the default (reward)
            </button>
          </div>
        ) : null}
      </div>

      <div className="card">
        <h2>Sampling and diversity</h2>
        <SearchFields />
        {search.shared ? (
          <div className="actions">
            <button type="button" className="small" disabled={!search.active} onClick={search.reset}>
              Back to the defaults (all off)
            </button>
          </div>
        ) : null}
      </div>

      <div className="card">
        <h2>Backwards</h2>
        <BackwardsField status={status} />
      </div>

      <div className="card wide">
        <h2>How a run walks its texts</h2>
        <TrainingPlanFields />
        {training.shared ? (
          <div className="actions">
            <button type="button" className="small" disabled={!training.active} onClick={training.reset}>
              Back to the defaults (all off)
            </button>
          </div>
        ) : null}
      </div>

      <BrowserCard
        changed={[network.traversal, network.penaltyScale, network.meritScale, search.values, training.values, backwards.on]}
      />
    </>
  );
}
