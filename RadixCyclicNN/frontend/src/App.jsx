import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import StatusBar from "./components/StatusBar.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";
import ScorePanel from "./components/ScorePanel.jsx";
import TwoNRLPanel from "./components/TwoNRLPanel.jsx";
import EvolvePanel from "./components/EvolvePanel.jsx";
import OllamaPanel from "./components/OllamaPanel.jsx";
import CheckpointPanel from "./components/CheckpointPanel.jsx";
import GraphView from "./components/GraphView.jsx";

const TABS = [
  { id: "train", label: "Train", Component: TrainPanel },
  { id: "predict", label: "Predict", Component: PredictPanel },
  { id: "generate", label: "Generate", Component: GeneratePanel },
  { id: "score", label: "Score", Component: ScorePanel },
  { id: "2nrl", label: "2NRL", Component: TwoNRLPanel },
  { id: "evolve", label: "Evolve", Component: EvolvePanel },
  { id: "ollama", label: "Ollama", Component: OllamaPanel },
  { id: "checkpoints", label: "Checkpoints", Component: CheckpointPanel },
  { id: "graph", label: "Graph", Component: GraphView, single: true },
];

function tabFromHash() {
  const id = window.location.hash.replace(/^#/, "");
  return TABS.some((t) => t.id === id) ? id : TABS[0].id;
}

/**
 * Single-page layout: header, status bar, tab strip, panels. All panels stay
 * mounted (hidden when inactive) so form state and job polling survive tab
 * switches; the active tab is mirrored in the URL hash.
 */
export default function App() {
  const [status, setStatus] = useState(null);
  const [tab, setTab] = useState(tabFromHash);
  const [version, setVersion] = useState(null);

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
        // The status bar reports connectivity; the version is cosmetic.
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
        <h1>RadixCyclicNN</h1>
        <p className="tagline">
          self-compressing cyclic graph · sine activation · Dijkstra prediction · 2NRL · GAN-style evolution
        </p>
      </header>

      <StatusBar onStatus={setStatus} />

      <nav className="tabs" role="tablist" aria-label="Panels">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={tab === t.id}
            aria-controls={`panel-${t.id}`}
            className={tab === t.id ? "active" : ""}
            onClick={() => selectTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {TABS.map(({ id, Component, single }) => (
          <section
            key={id}
            id={`panel-${id}`}
            role="tabpanel"
            hidden={tab !== id}
            className={`panel${single ? " single" : ""}`}
          >
            <Component status={status} />
          </section>
        ))}
      </main>

      <footer className="app-footer">
        RadixCyclicNN{version ? ` v${version}` : ""} · API {status ? "connected" : "unreachable"} · built with Vite +
        React, no other dependencies.
      </footer>
    </div>
  );
}
