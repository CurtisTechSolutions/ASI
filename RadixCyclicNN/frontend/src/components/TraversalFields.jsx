import { NumberField, SelectField } from "./Fields.jsx";
import { TRAVERSALS, useNetworkSettings } from "../hooks/useNetworkSettings.jsx";

/**
 * The traversal the searches run: *what* they look for, as opposed to the
 * mode, which is how they look for it.
 *
 * **reward** is the model's own distribution - the count / reward model's
 * probability carries `exp(reward_scale * reward)`, so the search follows the
 * rewards. **punishment** takes the rewards out of the score altogether and
 * lets the penalties price every step, so the cheapest path is the one that
 * accumulated the least punishment. The two scales say how heavily a
 * punishment counts and how heavily what the corpus did counts; a merit scale
 * of 0 is the pure form, where nothing but the punishments decides.
 * **least-punished** changes no price at all: it changes the *ranking*, so a
 * walk goes by the blame on its worst step before its cost and blame cannot be
 * bought off with rewards elsewhere (../../SPEC-LeastPunished.md). It belongs
 * to the count model, which is the one that keeps the judged paths it reads.
 *
 * The setting is shared (`useNetworkSettings`), so this control is the same
 * control wherever it appears: the Network settings tab, Predict and Generate
 * all read and write one value. `compact` is the version for the action tabs -
 * the fields and one line, without the explanation.
 */
export default function TraversalFields({ compact = false }) {
  const { traversal, penaltyScale, meritScale, set } = useNetworkSettings();
  const punishing = traversal === "punishment";
  return (
    <>
      <div className="row">
        <SelectField
          label="Traversal"
          hint={compact ? "shared with Network settings" : undefined}
          value={traversal}
          onChange={(value) => set("traversal", value)}
          options={TRAVERSALS}
        />
        {punishing ? (
          <NumberField
            label="Penalty scale"
            hint="how heavily a punishment counts"
            value={penaltyScale}
            onChange={(value) => set("penaltyScale", value)}
            min={0}
          />
        ) : null}
        {punishing ? (
          <NumberField
            label="Merit scale"
            hint="0 = nothing but the punishments decides"
            value={meritScale}
            onChange={(value) => set("meritScale", value)}
            min={0}
          />
        ) : null}
      </div>
      {punishing && compact ? (
        <p className="muted">
          The rewards leave the score: only the punishments price a step, so the cheapest path is the one that
          accumulated the <b>least punishment</b>.
        </p>
      ) : null}
      {traversal === "least-punished" && compact ? (
        <p className="muted">
          The prices are untouched; the <b>ranking</b> changes. A walk goes by the blame on its <b>worst step</b>,
          cost only to break ties, and a node offers only the children it has the least against — so blame cannot
          be bought off with rewards elsewhere. The count model keeps the judged paths this reads.
        </p>
      ) : null}
      {!compact ? (
        <>
          <p className="muted">
            <b>reward</b> is what the model believes. The count / reward model's probability carries{" "}
            <code>exp(reward_scale · reward)</code>, so a path the tutor rewarded is cheap and the search follows
            the rewards. This is what every search did before the option existed, and still does by default.
          </p>
          <p className="muted">
            <b>punishment</b> is what the model was punished for. The rewards leave the score altogether and only
            the penalties price the step, so the cheapest path is the one that accumulated the{" "}
            <b>least punishment</b>. Nothing the network was praised for makes a step cheaper here; only what it
            was corrected for makes one dearer. On the negative network it walks the least <i>blamed</i> way
            through the failures instead of the likeliest one.
          </p>
          <p className="muted">
            <b>least-punished</b> is a third thing again: it leaves every price alone and changes what the walks
            are <i>ranked</i> by. A path goes by the punishment on its <b>worst step</b> first and by its summed
            cost only to break ties, and a node offers only the children it has the least against — so a heavily
            punished step cannot be paid for with rewards further along, which the summed cost of the other two
            allows. It belongs to the count model, the one that keeps a record of what each path did.
          </p>
          <p className="muted">
            The traversal is <i>what</i> a search looks for; the mode (dijkstra / k-best / beam / sample) is{" "}
            <i>how</i> it looks. They are independent - every mode can run any of the three. The setting is used
            by the <b>Predict</b> and <b>Generate</b> tabs, which show the same control.
          </p>
        </>
      ) : null}
    </>
  );
}
