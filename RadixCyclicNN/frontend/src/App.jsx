import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { browserStorage, clearSettings } from "./storage.js";
import { wordKind } from "./util.js";
import StatusBar from "./components/StatusBar.jsx";
import ModelSelector from "./components/ModelSelector.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";
import TalkPanel from "./components/TalkPanel.jsx";
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
import ModelSettingsPanel from "./components/ModelSettingsPanel.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import GraphView from "./components/GraphView.jsx";
import WordsPanel from "./components/WordsPanel.jsx";
import { SiteSettingsProvider } from "./hooks/useSiteSettings.jsx";

// The Python and Go servers run every tab: the lessons, the evolve loop, the Ollama corpus and review, code
// generation, tool use and the image and speech encoders. The Go side's images use a thumbnail rather than
// the diffusion VAE, and its speech needs the words to come with the audio, which is what this page dictates
// anyway; a tab that still needed the Python server would carry `pythonOnly: true` and be hidden when the Go
// server (`radixnet-count serve`) answers.
//
// The Rust server (`radixnet serve`) lists the routes it answers in `/api/status` (`routes`), and a tab whose
// `route` is not among them is hidden when it answers - so a tab appears the moment the port serves it, and
// never before. The model's own tabs (`model: true`) are always there.
const TABS = [
  { id: "train", label: "Train", Component: TrainPanel, model: true },
  { id: "predict", label: "Predict", Component: PredictPanel, model: true },
  { id: "generate", label: "Generate", Component: GeneratePanel, model: true },
  { id: "talk", label: "Talk", Component: TalkPanel, route: "/v1/messages" },
  { id: "converse", label: "Converse", Component: ConversePanel, route: "/api/converse" },
  { id: "chat", label: "Chat", Component: ChatPanel, route: "/api/chat/start" },
  { id: "score", label: "Score", Component: ScorePanel, model: true },
  { id: "words", label: "Words", Component: WordsPanel, wordOnly: true, model: true },
  { id: "2nrl", label: "2NRL", Component: TwoNRLPanel, model: true },
  { id: "negative", label: "Negative", Component: NegativePanel, route: "/api/negative" },
  { id: "evolve", label: "Evolve", Component: EvolvePanel, route: "/api/evolve/start" },
  { id: "ollama", label: "Ollama", Component: OllamaPanel, route: "/api/ollama/models" },
  { id: "tutor", label: "Tutor", Component: TutorPanel, route: "/api/tutor/start" },
  { id: "code", label: "Code", Component: CodeGenPanel, route: "/api/codegen/start" },
  { id: "agent", label: "Agent", Component: AgentPanel, route: "/api/agent/start" },
  { id: "images", label: "Images", Component: ImagesPanel, route: "/api/images" },
  { id: "speech", label: "Speech", Component: SpeechPanel, route: "/api/speech" },
  { id: "checkpoints", label: "Checkpoints", Component: CheckpointPanel, route: "/api/checkpoints" },
  { id: "model", label: "Model settings", Component: ModelSettingsPanel, model: true },
  { id: "settings", label: "Settings", Component: SettingsPanel, model: true },
  { id: "graph", label: "Graph", Component: GraphView, single: true, model: true },
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

/** Whether the Rust server's `/api/status` lists `route` (as "GET /api/x" or "POST /api/x"). */
export function rustServes(status, route) {
  const routes = status && Array.isArray(status.routes) ? status.routes : [];
  return Boolean(route) && routes.some((r) => typeof r === "string" && r.split(" ")[1] === route);
}

function tabsFor(engine, status) {
  // `wordOnly` belongs to the word model, whose symbols are words: there is no vocabulary to show anywhere else
  const words = wordKind(status);
  return TABS.filter(
    (t) =>
      !(engine === "go" && t.pythonOnly) &&
      // the Rust server lists the routes it answers; a tab it cannot serve would 404
      !(engine === "rust" && !t.model && !rustServes(status, t.route)) &&
      !(t.wordOnly && !words),
  );
}

/** Tabs that were renamed: an old link still lands where it meant to. */
const TAB_ALIASES = { network: "model" };

function tabFromHash() {
  const raw = window.location.hash.replace(/^#/, "");
  const id = TAB_ALIASES[raw] || raw;
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
  const tabs = tabsFor(engine, status);
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
    <SiteSettingsProvider>
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
            {engine === "rust" ? (
              <span
                className="badge engine"
                title="This API is served by the Rust implementation of the count / reward model (radixnet serve): a thread pool over the texts, atomic counting, and no dependencies. The tabs shown are the routes it answers."
              >
                Rust engine · {status && status.workers ? `${status.workers} threads` : "one thread per core"}
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
              <Component status={status} onStatus={setStatus} />
            </section>
          ))}
        </main>

        <footer className="app-footer">
          RadixCyclicNN{version ? ` v${version}` : ""} · API {status ? "connected" : "unreachable"}
          {engine === "go" ? " (Go server)" : engine === "rust" ? " (Rust server)" : ""} · built with Vite + React,
          no other dependencies
          <SettingsReset />
        </footer>
      </div>
    </SiteSettingsProvider>
  );
}
