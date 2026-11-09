import { useCallback, useEffect, useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { nimTiles, recordLine } from "../nim.js";
import { fmtInt, fmtNum, parseInteger } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField } from "./Fields.jsx";
import JobProgress from "./JobProgress.jsx";

const FIRST = [
  ["alternate", "alternate (we move first in odd games)"],
  ["first", "we always move first"],
  ["second", "we always move second"],
];

/**
 * The Nim tab: the body that plays (the move tape read by the encoder, naming the states of an act lattice whose
 * row is the forward model; an action decoder proposing moves the referee replays). Make one, babble (random
 * moves: only the forward model learns), and play against a random opponent; the totals of the last batch as
 * tiles, and its games one by one.
 */
export default function NimPanel({ onMoved, tick }) {
  const [states, setStates] = useStoredState("nim.states", "13");
  const [width, setWidth] = useStoredState("nim.window", "8");
  const [seed, setSeed] = useStoredState("nim.seed", "1");
  const [grade, setGrade] = useStoredState("nim.grade", true);
  const [babbleGames, setBabbleGames] = useStoredState("nim.babble", "200");
  const [games, setGames] = useStoredState("nim.games", "100");
  const [first, setFirst] = useStoredState("nim.first", "alternate");
  const [info, setInfo] = useState(null);
  const [report, setReport] = useState(null);
  const [job, setJob] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const refresh = useCallback(async () => {
    try {
      setInfo(await api.nim());
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh, tick]);

  async function act(what) {
    setBusy(true);
    try {
      if (what === "new") {
        setInfo(await api.nimNew({ states: parseInteger(states, 13), window: parseInteger(width, 8), seed: parseInteger(seed, 1), grade }));
        setReport(null);
      } else {
        const body =
          what === "babble"
            ? { games: parseInteger(babbleGames, 200) }
            : { games: parseInteger(games, 100), ...(first === "alternate" ? {} : { first: first === "first" }) };
        const r = await (what === "babble" ? api.nimBabble : api.nimPlay)(body, setJob);
        setReport({ ...r, what });
        setInfo(r.nim);
      }
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setJob(null);
      setBusy(false);
      if (onMoved) onMoved();
    }
  }

  const records = report && Array.isArray(report.records) ? report.records : [];
  return (
    <div className="panel">
      <div className="card">
        <h2>A Nim body</h2>
        <p className="note">
          Heaps of 3, 5 and 7, normal play. The encoder reads the tape of moves as words and names a state of the act
          lattice; the lattice&apos;s row is the forward model; the decoder proposes moves the referee replays, and a
          refusal lands on the edge that proposed. E2 measured it at chance: under 0.04 bits about the position.
        </p>
        <div className="row">
          <NumberField label="States" value={states} onChange={setStates} min="2" step="1" />
          <NumberField label="Window" value={width} onChange={setWidth} min="1" step="1" />
          <NumberField label="Seed" value={seed} onChange={setSeed} min="0" step="1" />
        </div>
        <CheckField label="Grade won games" checked={grade} onChange={setGrade} hint="credit every move of a won game once more, at its end" />
        <div className="actions">
          <button type="button" disabled={busy} onClick={() => act("new")}>
            New
          </button>
        </div>
        {info ? (
          <p className="muted">
            {info.settings.states} states, window {info.settings.window}, seed {info.settings.seed}: {fmtInt(info.games)} games so far · I{" "}
            {fmtNum(info.information_bits, 3)} bits over {info.states_used} states · agreement {fmtNum(info.agreement, 2)} · diversity{" "}
            {fmtNum(info.diversity, 2)} bits
          </p>
        ) : null}
      </div>
      <div className="card">
        <h2>Babble, then play</h2>
        <div className="inline-form">
          <NumberField label="Babble games" value={babbleGames} onChange={setBabbleGames} min="1" step="50" />
          <div className="actions">
            <button type="button" disabled={busy} onClick={() => act("babble")} title="random moves: the forward model learns the rules' dynamics">
              Babble
            </button>
          </div>
        </div>
        <div className="inline-form">
          <NumberField label="Games" value={games} onChange={setGames} min="1" step="50" />
          <SelectField label="Who moves first" value={first} onChange={setFirst} options={FIRST} />
          <div className="actions">
            <button type="button" className="primary" disabled={busy} onClick={() => act("play")}>
              Play
            </button>
          </div>
        </div>
        <JobProgress job={job} />
        <Alert message={error} onDismiss={() => setError(null)} />
      </div>
      {report ? (
        <div className="card wide">
          <h3>
            {report.what === "babble" ? "Babbled" : "Played"} {fmtInt(report.totals.games)} games{report.stopped ? " (stopped)" : ""}
          </h3>
          <div className="tiles">
            {nimTiles(report).map((t) => (
              <span key={t.key} className="tile" title={t.title}>
                <small>{t.label}</small>
                <b>{t.value}</b>
              </span>
            ))}
          </div>
          <div className="table-wrap">
            <table className="data">
              <thead>
                <tr>
                  <th>game</th>
                  <th>first</th>
                  <th>won</th>
                  <th>plies</th>
                  <th>our moves</th>
                  <th>first try</th>
                  <th>refusals</th>
                  <th>abstained</th>
                  <th>optimal</th>
                  <th>lessons</th>
                </tr>
              </thead>
              <tbody>
                {records
                  .slice(-200)
                  .reverse()
                  .map((r) => (
                    <tr key={r.game} title={recordLine(r)}>
                      <td className="mono">{r.game}</td>
                      <td>{r.first ? "us" : "them"}</td>
                      <td className={r.won ? "ok" : "bad"}>{r.won ? "won" : "lost"}</td>
                      <td className="mono">{r.plies}</td>
                      <td className="mono">{r.our_moves}</td>
                      <td className="mono">{r.first_try}</td>
                      <td className="mono">{r.refusals}</td>
                      <td className="mono">{r.abstained}</td>
                      <td className="mono">{r.optimal}</td>
                      <td className="mono">{r.lessons}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          {records.length > 200 ? <p className="muted">The last 200 games, newest first.</p> : null}
        </div>
      ) : null}
    </div>
  );
}
