import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { browserStorage, clearSettings } from "./storage.js";
import StatusBar from "./components/StatusBar.jsx";
import MatrixPanel from "./components/MatrixPanel.jsx";
import RunPanel from "./components/RunPanel.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import TimePanel from "./components/TimePanel.jsx";
import MachinePanel from "./components/MachinePanel.jsx";
import CompressPanel from "./components/CompressPanel.jsx";
import WalkPanel from "./components/WalkPanel.jsx";

const TABS = [
  { id: "matrix", label: "Matrix", Component: MatrixPanel },
  { id: "run", label: "Run", Component: RunPanel },
  { id: "walk", label: "Walk", Component: WalkPanel },
  { id: "train", label: "Train", Component: TrainPanel },
  { id: "time", label: "Time", Component: TimePanel },
  { id: "compress", label: "Compress", Component: CompressPanel },
  { id: "machine", label: "Machine", Component: MachinePanel },
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
        title="Panel settings (the string, the stimulation, episodes, the file) are remembered in this browser."
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
 * Single-page layout: header, status bar, tab strip, panels. All panels stay
 * mounted (hidden when inactive) so form state survives tab switches; the
 * active tab is mirrored in the URL hash. `tick` counts the times a panel
 * moved the machine, and every panel that shows the machine refreshes on it.
 */
export default function App() {
  const [stats, setStats] = useState(null);
  const [tab, setTab] = useState(tabFromHash);
  const [version, setVersion] = useState(null);
  const [tick, setTick] = useState(0);
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
        if (alive && h && h.version !== undefined) setVersion(String(h.version));
      })
      .catch(() => {
        // the status bar reports connectivity; the version is cosmetic
      });
    return () => {
      alive = false;
    };
  }, []);

  const selectTab = useCallback((id) => {
    setTab(id);
    if (window.location.hash !== `#${id}`) window.history.replaceState(null, "", `#${id}`);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-row">
          <h1>LatticeFSM</h1>
        </div>
        <p className="tagline">
          A finite state machine over a dense 3D matrix of adaptive edges: every cell a record of its own life, every
          walk weighed by the edge's own function, and the wide channel the easier one the more stimulated the machine is.
        </p>
      </header>
      <StatusBar onStats={setStats} tick={tick} />
      <nav className="tabs" aria-label="Panels">
        {TABS.map((t) => (
          <button key={t.id} type="button" className={t.id === tab ? "active" : ""} onClick={() => selectTab(t.id)}>
            {t.label}
          </button>
        ))}
      </nav>
      {TABS.map(({ id, Component }) => (
        <div key={id} hidden={id !== tab}>
          <Component stats={stats} tick={tick} onMoved={onMoved} />
        </div>
      ))}
      <footer className="app-footer">
        latticefsm {version || ""} · the API is under /api/, the routes in rust/README.md
        <SettingsReset />
      </footer>
    </div>
  );
}
