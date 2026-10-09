import { useState } from "react";
import { api } from "../api.js";
import { useStoredState } from "../hooks/useStoredState.js";
import { focusIndex } from "../matrix.js";
import { fmtNum, parseInteger, parseNumber } from "../util.js";
import Alert from "./Alert.jsx";
import { CheckField, NumberField, SelectField, TextField } from "./Fields.jsx";

/**
 * Where a run starts and how it moves: the focus on the central vertical
 * vector (set, none, or learned from the input), skips past a node when a
 * two-edge path is more efficient, and the nodes rearranging themselves.
 */
export default function WalkPanel({ stats, onMoved }) {
  const [focus, setFocus] = useStoredState("walk.focus", "0");
  const [noFocus, setNoFocus] = useStoredState("walk.noFocus", true);
  const [learn, setLearn] = useStoredState("walk.learn", false);
  const [probe, setProbe] = useStoredState("walk.probe", "abba");
  const [skip, setSkip] = useStoredState("walk.skip", false);
  const [margin, setMargin] = useStoredState("walk.margin", "0");
  const [every, setEvery] = useStoredState("walk.rearrangeEvery", "0");
  const [axis, setAxis] = useStoredState("walk.axis", "states");
  const [swapI, setSwapI] = useStoredState("walk.swapI", "0");
  const [swapJ, setSwapJ] = useStoredState("walk.swapJ", "12");
  const [note, setNote] = useState(null);
  const [error, setError] = useState(null);

  const n = stats && stats.shape ? stats.shape[0] : 13;
  const symbol = stats && stats.alphabet && stats.center ? stats.alphabet[stats.center[1]] : "g";
  const col = stats && stats.center ? stats.center[2] : 6;
  const node = noFocus ? null : focusIndex(n, parseNumber(focus, 0));

  async function act(fn, say) {
    try {
      const r = await fn();
      setNote(say(r));
      setError(null);
      if (onMoved) onMoved();
    } catch (err) {
      setError(err.message);
    }
  }

  return (
    <div className="panel">
      <div className="card wide">
        <h2>Focus</h2>
        <p className="note">
          Focus is a number from 0 to 1 that picks a node of the central vertical vector - the column ({"0.."}
          {n - 1}, {symbol}, {col}) through the central node - and every run starts from that node's state. Low or no
          focus is the top node; high focus the bottom one; 0.5 the central node. The range is cut into {n} equal bands,
          one per node. Learned, the focus is read off each input by a small function of it, trained by the same
          rewards and punishments as the edges. Now:{" "}
          <b>
            {stats && stats.learn_focus
              ? "learned from the input"
              : stats && stats.focus !== null && stats.focus !== undefined
                ? `${fmtNum(stats.focus, 2)}, node (${stats.focus_node[0]}, ${symbol}, ${col})`
                : `none, the start state ${stats ? stats.start : 0}`}
          </b>
          .
        </p>
        <div className="row">
          <label className="field">
            <span>
              Focus <em>({noFocus ? "none" : `${fmtNum(parseNumber(focus, 0), 2)} → node (${node}, ${symbol}, ${col})`})</em>
            </span>
            <input type="range" min="0" max="1" step="0.01" value={focus} disabled={noFocus || learn} onChange={(e) => setFocus(e.target.value)} />
          </label>
        </div>
        <CheckField label="No focus: start from the start state" checked={noFocus} onChange={setNoFocus} disabled={learn} />
        <CheckField label="Learn the focus from the input" checked={learn} onChange={setLearn} />
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.focus({ focus: noFocus ? null : parseNumber(focus, 0), learn }),
                (r) =>
                  r.stats.learn_focus
                    ? "the focus is learned from the input"
                    : r.stats.focus === null
                      ? `no focus: runs start from state ${r.stats.start}`
                      : `focus ${fmtNum(r.stats.focus, 2)}: runs start from state ${r.stats.origin}`,
              )
            }
          >
            Apply
          </button>
        </div>
        <div className="row">
          <TextField label="What the learner reads off" value={probe} onChange={setProbe} />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.focus({ text: probe }),
                (r) => `the learner reads "${r.learned.text}" as focus ${fmtNum(r.learned.focus, 3)}: state ${r.learned.state}`,
              )
            }
          >
            Ask the learner
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Skip a node</h2>
        <p className="note">
          With skips on, before each move the machine looks one symbol further: if the best two-edge path through any
          node beats the path its own next step begins by more than the margin (in summed log-probability), it reads
          both symbols in one move, straight past the node between. Both edges are traversed and credited; the node
          skipped is not visited. {stats ? `${stats.skips} skips so far.` : ""}
        </p>
        <CheckField label="Skip when a two-edge path is more efficient" checked={skip} onChange={setSkip} />
        <div className="row">
          <NumberField label="Margin" value={margin} onChange={setMargin} min="0" step="0.1" hint="log-probability" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.skip({ skip, margin: parseNumber(margin, 0) }),
                (r) => `skips ${r.skip ? "on" : "off"}, margin ${fmtNum(r.skip_margin, 2)}`,
              )
            }
          >
            Apply
          </button>
        </div>
      </div>
      <div className="card wide">
        <h2>Rearrange and swap</h2>
        <p className="note">
          Swapping two states, or two symbols, relabels them everywhere - their edges, their records, the start state -
          so the machine walks as before from its start state; what changes is where they sit, and so which the focus
          picks and which shell of the compressed code they fall in. Rearranging lets the nodes do it themselves: each
          pass, neighbours swap when the one farther from the centre is busier, so the busy nodes move inward. States
          top to bottom: <span className="mono">[{stats && stats.state_order ? stats.state_order.join(", ") : ""}]</span>;
          symbols: <span className="mono">{stats && stats.symbol_order ? stats.symbol_order.join("") : ""}</span>.
        </p>
        <div className="actions">
          <button type="button" onClick={() => act(() => api.rearrange({}), (r) => `${r.swaps.length} swaps (one pass)`)}>
            Rearrange one pass
          </button>
          <button type="button" onClick={() => act(() => api.rearrange({ full: true }), (r) => `${r.swaps.length} swaps (until settled)`)}>
            Rearrange until settled
          </button>
        </div>
        <div className="row">
          <NumberField label="Rearrange every N transitions" value={every} onChange={setEvery} min="0" step="100" hint="0: never" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.skip({ rearrange_every: parseInteger(every, 0) }),
                (r) => (r.rearrange_every ? `rearranging every ${r.rearrange_every} transitions` : "automatic rearranging off"),
              )
            }
          >
            Apply
          </button>
        </div>
        <div className="row">
          <SelectField
            label="Swap"
            value={axis}
            onChange={setAxis}
            options={[
              ["states", "two states"],
              ["symbols", "two symbols"],
            ]}
          />
          <NumberField label="i" value={swapI} onChange={setSwapI} min="0" step="1" />
          <NumberField label="j" value={swapJ} onChange={setSwapJ} min="0" step="1" />
        </div>
        <div className="actions">
          <button
            type="button"
            onClick={() =>
              act(
                () => api.rearrange({ axis, i: parseInteger(swapI, 0), j: parseInteger(swapJ, 0) }),
                () => `swapped ${axis} ${swapI} and ${swapJ}`,
              )
            }
          >
            Swap
          </button>
        </div>
        <Alert message={error} onDismiss={() => setError(null)} />
        {note ? <p className="note">{note}</p> : null}
      </div>
    </div>
  );
}
