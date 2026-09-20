import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { browserStorage, clearSettings } from "./storage.js";
import StatusBar from "./components/StatusBar.jsx";
import ModelSelector from "./components/ModelSelector.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";
import ConversePanel from "./components/ConversePanel.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import ScorePanel from "./components/ScorePanel.jsx";
import TwoNRLPanel from "./components/TwoNRLPanel.jsx";
import NegativePanel from "./components/NegativePanel.jsx";
import EvolvePanel from "./components/EvolvePanel.jsx";
import OllamaPanel from "./components/OllamaPanel.jsx";
import TutorPanel from "./components/TutorPanel.jsx";
import CodeGenPanel from "./components/CodeGenPanel.jsx";
import AgentPanel from "./components/AgentPanel.jsx";
import ImagesPanel from "./components/ImagesPanel.jsx";
import SpeechPanel from "./components/SpeechPanel.jsx";
import CheckpointPanel from "./components/CheckpointPanel.jsx";
import NetworkSettingsPanel from "./components/NetworkSettingsPanel.jsx";
import GraphView from "./components/GraphView.jsx";
import { NetworkSettingsProvider } from "./hooks/useNetworkSettings.jsx";

// Both servers now run every tab: the lessons, the evolve loop, the Ollama corpus and review, code
// generation, tool use and the image and speech encoders. The Go side's images use a thumbnail rather than
// the diffusion VAE, and its speech needs the words to come with the audio, which is what this page dictates
// anyway; a tab that still needed the Python server would carry `pythonOnly: true` and be hidden when the Go
// server (`radixnet-count serve`) answers.
const TABS = [
  { id: "train", label: "Train", Component: TrainPanel },
  { id: "predict", label: "Predict", Component: PredictPanel },
  { id: "generate", label: "Generate", Component: GeneratePanel },
  { id: "converse", label: "Converse", Component: ConversePanel },
  { id: "chat", label: "Chat", Component: ChatPanel },
  { id: "score", label: "Score", Component: ScorePanel },
  { id: "2nrl", label: "2NRL", Component: TwoNRLPanel },
  { id: "negative", label: "Negative", Component: NegativePanel },
  { id: "evolve", label: "Evolve", Component: EvolvePanel },
  { id: "ollama", label: "Ollama", Component: OllamaPanel },
  { id: "tutor", label: "Tutor", Component: TutorPanel },
  { id: "code", label: "Code", Component: CodeGenPanel },
  { id: "agent", label: "Agent", Component: AgentPanel },
  { id: "images", label: "Images", Component: ImagesPanel },
  { id: "speech", label: "Speech", Component: SpeechPanel },
  { id: "checkpoints", label: "Checkpoints", Component: CheckpointPanel },
  { id: "network", label: "Network settings", Component: NetworkSettingsPanel },
  { id: "graph", label: "Graph", Component: GraphView, single: true },
];

/**
 * "Reset saved settings": forget every remembered panel setting and reload
 * with the defaults. Two clicks, so a stray one costs nothing; hidden where
 * the browser has no usable storage (a private window, blocked site data).
 */
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
        title="Panel settings (epochs, rates, prefixes, prompts, the text in the boxes) are remembered in this browser. Results, ratings and uploaded-file choices are not."
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
    <NetworkSettingsProvider>
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
            prediction · 2NRL · GAN-style evolution · a negative network that filters the output
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
          {engine === "go" ? " (Go server)" : ""} · built with Vite + React, no other dependencies
          <SettingsReset />
        </footer>
      </div>
    </NetworkSettingsProvider>
  );
}
