import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { browserStorage, clearSettings } from "./storage.js";
import { useStoredState } from "./hooks/useStoredState.js";
import StatusBar from "./components/StatusBar.jsx";
import BodyPanel from "./components/BodyPanel.jsx";
import GamePanel from "./components/GamePanel.jsx";
import LatticePanel from "./components/LatticePanel.jsx";
import EncoderPanel from "./components/EncoderPanel.jsx";
import NimPanel from "./components/NimPanel.jsx";
import ResultsPanel from "./components/ResultsPanel.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";

const TABS = [
  { id: "body", label: "Body", Component: BodyPanel },
  { id: "game", label: "Game", Component: GamePanel },
  { id: "lattice", label: "Lattice", Component: LatticePanel },
  { id: "encoder", label: "Encoder", Component: EncoderPanel },
  { id: "nim", label: "Nim", Component: NimPanel },
  { id: "results", label: "Results", Component: ResultsPanel },
  { id: "settings", label: "Settings", Component: SettingsPanel },
];

/** "Reset saved settings": forget every remembered panel setting and reload with the defaults. Two clicks. */
function SettingsReset() {
  const storage = browserStorage();
  const [armed, setArmed] = useState(false);
  if (!storage) return null;
  return (
    <>
      {" · "}
      <button
        type="button"
        className="link"
        title="Panel settings (the language, the states, the episodes, the text) are remembered in this browser."
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
        {armed ? "reset them? click again" : "settings saved in this browser"}
      </button>
    </>
  );
}

function tabFromHash() {
  const id = window.location.hash.replace(/^#/, "");
  return TABS.some((t) => t.id === id) ? id : TABS[0].id;
}

/**
 * Single-page layout: header, status bar, tab strip, panels. All panels stay mounted (hidden when inactive) so
 * form state survives tab switches; the active tab is mirrored in the URL hash. `tick` counts the times a panel
 * moved a body, and every panel that shows one refreshes on it. `read` is the last string classified or run (its
 * trace), shared by the Body tab that makes it and the Game tab that shows one step of it in detail.
 */
export default function App() {
  const [status, setStatus] = useState(null);
  const [health, setHealth] = useState(null);
  const [textSize, setTextSize] = useStoredState("settings.textSize", "14");
  const [tab, setTab] = useState(tabFromHash);
  const [tick, setTick] = useState(0);
  const [read, setRead] = useState(null);
  const [step, setStep] = useState(0);
  const onMoved = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    const onHashChange = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((h) => {
        if (alive && h && h.version !== undefined) setHealth(h);
      })
      .catch(() => {
        // the status bar reports connectivity; the version is cosmetic
      });
    return () => {
      alive = false;
    };
  }, [tick]);

  useEffect(() => {
    document.body.style.fontSize = /^\d+$/.test(textSize) ? `${textSize}px` : "";
  }, [textSize]);

  const selectTab = useCallback((id) => {
    setTab(id);
    if (window.location.hash !== `#${id}`) window.history.replaceState(null, "", `#${id}`);
  }, []);
  const goToGame = useCallback(() => selectTab("game"), [selectTab]);
  const goToBody = useCallback(() => selectTab("body"), [selectTab]);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-row">
          <h1>GameLattice</h1>
        </div>
        <p className="tagline">
          A RadixCyclicNN whose step is a game, naming the states of a LatticeFSM lattice: at every character the walk
          window&apos;s nodes buy seats in an auction, play a congestion game over the lattice&apos;s states, and the
          aggregate names where we are; the lattice says where this takes us.
        </p>
      </header>
      <StatusBar onStatus={setStatus} tick={tick} />
      <nav className="tabs" aria-label="Panels">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={t.id === tab ? "active" : ""} onClick={() => selectTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      {TABS.map(({ id, Component }) => (
        <div key={id} hidden={id !== tab}>
          <Component
            status={status}
            health={health}
            tick={tick}
            onMoved={onMoved}
            read={read}
            setRead={setRead}
            step={step}
            setStep={setStep}
            goToGame={goToGame}
            goToBody={goToBody}
            textSize={textSize}
            setTextSize={setTextSize}
          />
        </div>
      ))}
      <footer className="app-footer">
        gamelattice {health ? health.version : ""} · the API is under /api/, the routes in ../README.md
        <SettingsReset />
      </footer>
    </div>
  );
}
