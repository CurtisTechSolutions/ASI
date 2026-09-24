import { NumberField } from "./Fields.jsx";
import Alert from "./Alert.jsx";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { problemText, searchProblemsFor, searchSettingsFor } from "../settings.js";

/**
 * The sampling filters and the beam's diversity (../../SPEC-SearchAndTraining.md
 * sections 1-2): what a sampled step may draw from, and how far apart the
 * beam's K are picked.
 *
 * The values are site-wide (`useSiteSettings`), so this is the same control
 * wherever it appears - the Settings tab and the Predict and Generate tabs.
 * `mode` is the mode the search will actually run in: the compact version on
 * an action tab shows only what that mode reads (the filters for `sample`,
 * the diversity for `beam`) and nothing for the exact searches, which read
 * neither. Without a mode every field is shown.
 */
export default function SearchFields({ mode = null, compact = false, disabled = false }) {
  const { search } = useSiteSettings();
  const { values, problems, set } = search;
  const reads = mode === null ? { filters: true, diversity: true } : searchSettingsFor(mode);
  if (!reads.filters && !reads.diversity) return null;
  const hint = (text) => (compact ? "shared with Settings" : text);
  const shown = mode === null ? problems : searchProblemsFor(values, mode);

  return (
    <>
      {reads.filters ? (
        <div className="row">
          <NumberField
            label="Top-K"
            hint={hint("the K cheapest steps only; 0 = off")}
            value={values.topK}
            onChange={(value) => set("topK", value)}
            min={0}
            step={1}
            disabled={disabled}
          />
          <NumberField
            label="Top-p"
            hint={hint("nucleus: the cheapest steps holding p of the mass; 1 = off")}
            value={values.topP}
            onChange={(value) => set("topP", value)}
            min={0}
            max={1}
            step="any"
            disabled={disabled}
          />
          <NumberField
            label="Min-p"
            hint={hint("steps at least p times as likely as the best; 0 = off")}
            value={values.minP}
            onChange={(value) => set("minP", value)}
            min={0}
            max={1}
            step="any"
            disabled={disabled}
          />
        </div>
      ) : null}
      {reads.diversity ? (
        <div className="row">
          <NumberField
            label="Diversity"
            hint={hint("what a path pays for repeating one already picked; 0 = off")}
            value={values.diversity}
            onChange={(value) => set("diversity", value)}
            min={0}
            step="any"
            disabled={disabled}
          />
        </div>
      ) : null}
      <Alert message={problemText(shown)} />
      {!compact ? (
        <>
          <p className="muted">
            A <b>sampled</b> walk draws every step from the options its node offers, weighted{" "}
            <code>exp(−(cost − lowest) / temperature)</code>. The filters narrow those options first, in this
            order: <b>top-K</b> keeps the K cheapest; <b>min-p</b> keeps the ones whose weight is at least{" "}
            <code>min_p</code> of the best one's, which scales with the temperature; <b>top-p</b> (nucleus sampling)
            keeps the smallest set of cheapest options whose weights reach <code>top_p</code> of the total. The
            cheapest option always survives, and a step still draws once however few are left, so the same seed
            walks the same way on every server.
          </p>
          <p className="muted">
            A <b>beam</b> hands back the K cheapest finished paths, and in a compressed graph the cheapest few are
            often one text with its ending varied. <b>Diversity</b> picks the K from a larger pool instead: the best
            path first, then each next one by its cost plus <code>diversity × overlap</code>, where the overlap is
            how much of it repeats a path already picked from the start. The costs reported are untouched, so with
            diversity on the list is no longer sorted after its first entry - that is the trade. Only the top K
            are spread; the bottom K still answer what the model finds strangest.
          </p>
          <p className="muted">
            Dijkstra and k-best are exact searches and read neither. The filters are used by <b>sample</b> mode and
            the diversity by <b>beam</b> mode, on the <b>Predict</b> and <b>Generate</b> tabs, which show the same
            controls.
          </p>
        </>
      ) : null}
    </>
  );
}
