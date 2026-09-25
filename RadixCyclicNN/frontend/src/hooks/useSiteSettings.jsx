import { createContext, useContext, useMemo } from "react";
import { useStoredState } from "./useStoredState.js";
import { NetworkSettingsProvider, useNetworkSettings } from "./useNetworkSettings.jsx";
import {
  SEARCH_DEFAULTS,
  TRAINING_DEFAULTS,
  searchActive,
  searchBody,
  searchProblems,
  trainingActive,
  trainingBody,
  trainingProblems,
} from "../settings.js";

/**
 * The site-wide settings of this browser, in one place: the traversal every
 * search runs (`useNetworkSettings`, as it always was), the sampling filters
 * and the beam's diversity every search starts from, how every training run
 * walks its texts (../../SPEC-SearchAndTraining.md), and whether Predict and
 * Generate ask the model backwards (section 9 there).
 *
 * They are edited on the Settings tab, and the Predict, Generate and Train
 * tabs show the same controls: one value, several doors into it - every panel
 * stays mounted while hidden, so per-panel state would drift apart within a
 * session. Everything is remembered in this browser (`useStoredState`, under
 * `site.search.*`, `site.train.*` and `site.query.backwards`; the traversal
 * keeps its `network.*` names), and nothing here is saved with a model.
 */

/** A group used outside the provider: the defaults, and nothing to change them with. */
function fallbackGroup(defaults) {
  return Object.freeze({
    values: defaults,
    problems: Object.freeze({}),
    active: false,
    shared: false,
    set: () => {},
    reset: () => {},
    body: () => ({}),
  });
}

const FALLBACK = Object.freeze({
  search: fallbackGroup(SEARCH_DEFAULTS),
  training: fallbackGroup(TRAINING_DEFAULTS),
  backwards: Object.freeze({ on: false, shared: false, set: () => {} }),
});

const SiteSettings = createContext(null);

/** One group of settings as the tabs use it; `bodyOf(values, ...args)` makes its request fields. */
function group(values, setters, defaults, problemsOf, bodyOf, activeOf) {
  return {
    values,
    problems: problemsOf(values),
    active: activeOf(values),
    shared: true,
    /** `set("topP", "0.9")` - one setting at a time, by name. */
    set: (name, next) => {
      const setter = setters[name];
      if (setter) setter(next);
    },
    reset: () => {
      for (const [name, setter] of Object.entries(setters)) setter(defaults[name]);
    },
    body: (...args) => bodyOf(values, ...args),
  };
}

function SiteSettingsValue({ children }) {
  const [topK, setTopK] = useStoredState("site.search.topK", SEARCH_DEFAULTS.topK);
  const [topP, setTopP] = useStoredState("site.search.topP", SEARCH_DEFAULTS.topP);
  const [minP, setMinP] = useStoredState("site.search.minP", SEARCH_DEFAULTS.minP);
  const [diversity, setDiversity] = useStoredState("site.search.diversity", SEARCH_DEFAULTS.diversity);
  const [order, setOrder] = useStoredState("site.train.order", TRAINING_DEFAULTS.order);
  const [curriculum, setCurriculum] = useStoredState("site.train.curriculum", TRAINING_DEFAULTS.curriculum);
  const [replay, setReplay] = useStoredState("site.train.replay", TRAINING_DEFAULTS.replay);
  const [replaySize, setReplaySize] = useStoredState("site.train.replaySize", TRAINING_DEFAULTS.replaySize);
  const [patience, setPatience] = useStoredState("site.train.patience", TRAINING_DEFAULTS.patience);
  const [minDelta, setMinDelta] = useStoredState("site.train.minDelta", TRAINING_DEFAULTS.minDelta);
  const [queryBackwards, setQueryBackwards] = useStoredState("site.query.backwards", false);

  const search = useMemo(
    () =>
      group(
        { topK, topP, minP, diversity },
        { topK: setTopK, topP: setTopP, minP: setMinP, diversity: setDiversity },
        SEARCH_DEFAULTS,
        searchProblems,
        searchBody,
        searchActive,
      ),
    [topK, topP, minP, diversity, setTopK, setTopP, setMinP, setDiversity],
  );
  const training = useMemo(
    () =>
      group(
        { order, curriculum, replay, replaySize, patience, minDelta },
        {
          order: setOrder,
          curriculum: setCurriculum,
          replay: setReplay,
          replaySize: setReplaySize,
          patience: setPatience,
          minDelta: setMinDelta,
        },
        TRAINING_DEFAULTS,
        trainingProblems,
        trainingBody,
        trainingActive,
      ),
    [
      order,
      curriculum,
      replay,
      replaySize,
      patience,
      minDelta,
      setOrder,
      setCurriculum,
      setReplay,
      setReplaySize,
      setPatience,
      setMinDelta,
    ],
  );
  const backwards = useMemo(
    () => ({ on: queryBackwards, shared: true, set: (next) => setQueryBackwards(Boolean(next)) }),
    [queryBackwards, setQueryBackwards],
  );
  const value = useMemo(() => ({ search, training, backwards }), [search, training, backwards]);
  return <SiteSettings.Provider value={value}>{children}</SiteSettings.Provider>;
}

/** Every site-wide setting: the traversal (`NetworkSettingsProvider`) and the search and training defaults. */
export function SiteSettingsProvider({ children }) {
  return (
    <NetworkSettingsProvider>
      <SiteSettingsValue>{children}</SiteSettingsValue>
    </NetworkSettingsProvider>
  );
}

/**
 * `{network, search, training, backwards}`: the traversal as
 * `useNetworkSettings()` has it, the search and training groups - each
 * `{values, problems, active, set, reset, body}`, where `search.body(mode)`
 * and `training.body()` are the request fields - and `backwards`
 * (`{on, set}`): whether Predict and Generate ask the model backwards, for a
 * model trained with "Read every text backwards" (`../backwards.js`). That one
 * is never sent: it turns the query around before it goes and the answer
 * around when it comes back. The defaults (unchangeable) outside the provider.
 */
export function useSiteSettings() {
  const network = useNetworkSettings();
  const site = useContext(SiteSettings) || FALLBACK;
  return { network, search: site.search, training: site.training, backwards: site.backwards };
}
