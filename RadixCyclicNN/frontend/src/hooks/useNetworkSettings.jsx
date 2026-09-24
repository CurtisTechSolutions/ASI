import { createContext, useContext, useMemo } from "react";
import { useStoredState } from "./useStoredState.js";

/**
 * The traversal every search runs, in one place: the Settings tab is where it
 * is edited, and it is one of the site-wide settings `useSiteSettings.jsx`
 * gathers (this provider sits inside `SiteSettingsProvider`).
 *
 * `useStoredState` remembers a value per name, but two mounted components
 * using the same name would share the *stored* value and not the state - and
 * every panel here stays mounted while it is hidden, so they would drift apart
 * within a session. These settings therefore live once, in a provider at the
 * top of the app, and every panel reads them through `useNetworkSettings()`:
 * one source of truth, several doors into it.
 */

export const TRAVERSALS = [
  ["reward", "reward (follow what was rewarded)"],
  ["punishment", "punishment (avoid what was punished)"],
  ["least-punished", "least-punished (rank by the worst step's blame)"],
];

export const DEFAULTS = Object.freeze({ traversal: "reward", penaltyScale: "1", meritScale: "1" });

/** Used outside the provider: the defaults, and nothing to change them with. */
const FALLBACK = Object.freeze({
  ...DEFAULTS,
  set: () => {},
  reset: () => {},
  body: Object.freeze({}),
  shared: false,
});

const NetworkSettings = createContext(null);

/**
 * The request fields for a traversal; `{}` for the default, so an older server
 * is unaffected. The least-punished traversal has no scales of its own - it
 * reads the blame off the graph rather than pricing a step with it - so it
 * travels as the name alone (../../SPEC-LeastPunished.md).
 */
export function traversalBody(traversal, penaltyScale, meritScale) {
  if (traversal === "least-punished") return { traversal };
  if (traversal !== "punishment") return {};
  const number = (value, fallback) => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
  };
  return { traversal, penalty_scale: number(penaltyScale, 1), merit_scale: number(meritScale, 1) };
}

export function NetworkSettingsProvider({ children }) {
  const [traversal, setTraversal] = useStoredState("network.traversal", DEFAULTS.traversal);
  const [penaltyScale, setPenaltyScale] = useStoredState("network.penaltyScale", DEFAULTS.penaltyScale);
  const [meritScale, setMeritScale] = useStoredState("network.meritScale", DEFAULTS.meritScale);

  const value = useMemo(() => {
    const setters = { traversal: setTraversal, penaltyScale: setPenaltyScale, meritScale: setMeritScale };
    return {
      traversal,
      penaltyScale,
      meritScale,
      shared: true,
      /** `set("traversal", "punishment")` - one setting at a time, by name. */
      set: (name, next) => {
        const setter = setters[name];
        if (setter) setter(next);
      },
      reset: () => {
        setTraversal(DEFAULTS.traversal);
        setPenaltyScale(DEFAULTS.penaltyScale);
        setMeritScale(DEFAULTS.meritScale);
      },
      /** What a /api/predict or /api/generate body needs for these settings. */
      body: traversalBody(traversal, penaltyScale, meritScale),
    };
  }, [traversal, penaltyScale, meritScale, setTraversal, setPenaltyScale, setMeritScale]);

  return <NetworkSettings.Provider value={value}>{children}</NetworkSettings.Provider>;
}

/** The shared settings; the defaults (unchangeable) when rendered outside the provider. */
export function useNetworkSettings() {
  return useContext(NetworkSettings) || FALLBACK;
}
