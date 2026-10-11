import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { browserStorage, clearSettings } from "./storage.js";
import { useStoredState } from "./hooks/useStoredState.js";
import { lmId } from "./lm.js";
import StatusBar from "./components/StatusBar.jsx";
import ModelSelector from "./components/ModelSelector.jsx";
import LmTrainPanel from "./components/LmTrainPanel.jsx";
import GeneratePanel from "./components/GeneratePanel.jsx";
import NetworkPanel from "./components/NetworkPanel.jsx";
import ConversePanel from "./components/ConversePanel.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import ScorePanel from "./components/ScorePanel.jsx";
import NodesPanel from "./components/NodesPanel.jsx";
import FeedbackPanel from "./components/FeedbackPanel.jsx";
import OllamaPanel from "./components/OllamaPanel.jsx";
import VoicePanel from "./components/VoicePanel.jsx";
import CheckpointPanel from "./components/CheckpointPanel.jsx";
import SettingsPanel from "./components/SettingsPanel.jsx";
import MatrixPanel from "./components/MatrixPanel.jsx";
import RunPanel from "./components/RunPanel.jsx";
import TrainPanel from "./components/TrainPanel.jsx";
import TimePanel from "./components/TimePanel.jsx";
import MachinePanel from "./components/MachinePanel.jsx";
import CompressPanel from "./components/CompressPanel.jsx";
import WalkPanel from "./components/WalkPanel.jsx";
import TalkPanel from "./components/TalkPanel.jsx";
import PredictPanel from "./components/PredictPanel.jsx";
import CubePanel from "./components/CubePanel.jsx";

// the language-model tabs first (they talk to the model the header's selector picks), then the lattice's own
const TABS = [
  { id: "train", label: "Train", Component: LmTrainPanel },
  { id: "predict", label: "Predict", Component: PredictPanel },
  { id: "generate", label: "Generate", Component: GeneratePanel },
  { id: "talk", label: "Talk", Component: TalkPanel },
  { id: "network", label: "Network", Component: NetworkPanel },
  { id: "converse", label: "Converse", Component: ConversePanel },
  { id: "chat", label: "Chat", Component: ChatPanel },
  { id: "score", label: "Score", Component: ScorePanel },
  { id: "nodes", label: "Nodes", Component: NodesPanel },
  { id: "feedback", label: "Feedback", Component: FeedbackPanel },
  { id: "ollama", label: "Ollama", Component: OllamaPanel },
  { id: "voice", label: "Voice", Component: VoicePanel },
  { id: "checkpoints", label: "Checkpoints", Component: CheckpointPanel },
  { id: "settings", label: "Settings", Component: SettingsPanel },
  { id: "matrix", label: "Matrix", Component: MatrixPanel },
  { id: "cube", label: "3D", Component: CubePanel },
  { id: "run", label: "Run", Component: RunPanel },
  { id: "walk", label: "Walk", Component: WalkPanel },
  { id: "teach", label: "Teach", Component: TrainPanel },
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
 * moved the machine (or a language model), and every panel that shows one
 * refreshes on it. `lm` is the language model the header's selector picks
 * ("chain" or "ngram", remembered in this browser), handed to every panel.
 */
export default function App() {
  const [stats, setStats] = useState(null);
  const [status, setStatus] = useState(null);
  const [storedLm, setStoredLm] = useStoredState("lm", "chain");
  const [textSize, setTextSize] = useStoredState("settings.textSize", "14");
  const lm = lmId(storedLm);
  const setLm = useCallback((value) => setStoredLm(lmId(value)), [setStoredLm]);
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

  useEffect(() => {
    document.body.style.fontSize = /^\d+$/.test(textSize) ? `${textSize}px` : "";
  }, [textSize]);

  const selectTab = useCallback((id) => {
    setTab(id);
    if (window.location.hash !== `#${id}`) window.history.replaceState(null, "", `#${id}`);
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-header-row">
          <h1>LatticeFSM</h1>
          <ModelSelector lm={lm} onChange={setLm} status={status} />
        </div>
        <p className="tagline">
          A finite state machine over a dense 3D matrix of adaptive edges: every cell a record of its own life, every
          walk weighed by the edge's own function, and the wide channel the easier one the more stimulated the machine is.
        </p>
      </header>
      <StatusBar onStats={setStats} onStatus={setStatus} tick={tick} />
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
            stats={stats}
            status={status}
            tick={tick}
            onMoved={onMoved}
            lm={lm}
            setLm={setLm}
            textSize={textSize}
            setTextSize={setTextSize}
          />
        </div>
      ))}
      <footer className="app-footer">
        latticefsm {version || ""} · the API is under /api/, the routes in rust/README.md
        <SettingsReset />
      </footer>
    </div>
  );
}
