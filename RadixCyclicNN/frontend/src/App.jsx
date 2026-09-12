import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import StatusBar from "./components/StatusBar.jsx";
import ModelSelector from "./components/ModelSelector.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";
import ConversePanel from "./components/ConversePanel.jsx";
import ScorePanel from "./components/ScorePanel.jsx";
import TwoNRLPanel from "./components/TwoNRLPanel.jsx";
import EvolvePanel from "./components/EvolvePanel.jsx";
import OllamaPanel from "./components/OllamaPanel.jsx";
import TutorPanel from "./components/TutorPanel.jsx";
import CodeGenPanel from "./components/CodeGenPanel.jsx";
import ImagesPanel from "./components/ImagesPanel.jsx";
import CheckpointPanel from "./components/CheckpointPanel.jsx";
import GraphView from "./components/GraphView.jsx";

// pythonOnly tabs need the Python server (its sine network, the corpus / review calls to Ollama, the
// sandbox, the image encoder); the Go server (`radixnet-count serve`) runs the count / reward model only
// and hides them. The Tutor tab is not one of them: both servers run the lessons.
const TABS = [
  { id: "train", label: "Train", Component: TrainPanel },
  { id: "predict", label: "Predict", Component: PredictPanel },
  { id: "generate", label: "Generate", Component: GeneratePanel },
  { id: "converse", label: "Converse", Component: ConversePanel },
  { id: "score", label: "Score", Component: ScorePanel },
  { id: "2nrl", label: "2NRL", Component: TwoNRLPanel },
  { id: "evolve", label: "Evolve", Component: EvolvePanel, pythonOnly: true },
  { id: "ollama", label: "Ollama", Component: OllamaPanel, pythonOnly: true },
  { id: "tutor", label: "Tutor", Component: TutorPanel },
  { id: "code", label: "Code", Component: CodeGenPanel, pythonOnly: true },
  { id: "images", label: "Images", Component: ImagesPanel, pythonOnly: true },
  { id: "checkpoints", label: "Checkpoints", Component: CheckpointPanel },
  { id: "graph", label: "Graph", Component: GraphView, single: true },
];

/** The engine behind the API: "go" when the Go server answers, else "python". */
export function engineOf(status, health) {
  const fromStatus = status && typeof status.engine === "string" ? status.engine : null;
  const fromHealth = health && typeof health.engine === "string" ? health.engine : null;
  return fromStatus || fromHealth || "python";
}

function tabsFor(engine) {
  return engine === "go" ? TABS.filter((t) => !t.pythonOnly) : TABS;
}

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
  const [health, setHealth] = useState(null);
  const engine = engineOf(status, health);
  const tabs = tabsFor(engine);
  const activeTab = tabs.some((t) => t.id === tab) ? tab : tabs[0].id;

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
        if (!alive || !h) return;
        setHealth(h);
        if (h.version !== undefined) setVersion(String(h.version));
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
        <div className="app-header-row">
          <h1>RadixCyclicNN</h1>
          <ModelSelector status={status} onStatus={setStatus} />
          {engine === "go" ? (
            <span
              className="badge engine"
              title="This API is served by the Go implementation of the count / reward model (radixnet-count serve): a goroutine per text, counters bumped without locks (racy by default, --exact for reproducible counts)"
            >
              Go engine · {status && status.workers ? `${status.workers} goroutines` : "one goroutine per text"}
              {status && status.counting === "racy" ? " · racy counting" : ""}
            </span>
          ) : null}
        </div>
        <p className="tagline">
          self-compressing cyclic graph · sine activation or count / reward edges · Dijkstra and top-K / bottom-K
          prediction · 2NRL · GAN-style evolution
        </p>
      </header>

      <StatusBar onStatus={setStatus} />

      <nav className="tabs" role="tablist" aria-label="Panels">
        {tabs.map((t) => (
          <button
            key={t.id}
            type="button"
            role="tab"
            aria-selected={activeTab === t.id}
            aria-controls={`panel-${t.id}`}
            className={activeTab === t.id ? "active" : ""}
            onClick={() => selectTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main>
        {tabs.map(({ id, Component, single }) => (
          <section
            key={id}
            id={`panel-${id}`}
            role="tabpanel"
            hidden={activeTab !== id}
            className={`panel${single ? " single" : ""}`}
          >
            <Component status={status} />
          </section>
        ))}
      </main>

      <footer className="app-footer">
        RadixCyclicNN{version ? ` v${version}` : ""} · API {status ? "connected" : "unreachable"}
        {engine === "go" ? " (Go server)" : ""} · built with Vite + React, no other dependencies.
      </footer>
    </div>
  );
}
