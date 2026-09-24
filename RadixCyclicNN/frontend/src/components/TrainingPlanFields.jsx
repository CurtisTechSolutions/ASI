import { NumberField, SelectField } from "./Fields.jsx";
import Alert from "./Alert.jsx";
import { useSiteSettings } from "../hooks/useSiteSettings.jsx";
import { fmtInt } from "../util.js";
import { ORDERS, pacedCounts, problemText, rehearsalCount, replayLine } from "../settings.js";

/**
 * How a training run walks its texts (../../SPEC-SearchAndTraining.md sections
 * 3-6): the order, the curriculum, the rehearsal of the model's replay buffer
 * and the early stop.
 *
 * Site-wide (`useSiteSettings`): the Settings tab and the Train tab show the
 * same control. On the Train tab (`compact`) it previews what the run will do
 * with `texts` texts over `epochs` epochs, and says what the model's buffer
 * holds (`replay`, from the status).
 */
export default function TrainingPlanFields({ compact = false, disabled = false, texts = 0, epochs = 0, replay = null }) {
  const { training } = useSiteSettings();
  const { values, problems, set } = training;
  const hint = (text) => (compact ? undefined : text);
  const counts = pacedCounts(texts, values.curriculum, epochs);
  const curriculumOn = Number(values.curriculum) < 1 && !problems.curriculum;
  const pool = replay && typeof replay === "object" ? Number(replay.texts) || 0 : 0;
  const rehearsed = rehearsalCount(values.replay, texts, pool);
  const replayOn = Number(values.replay) > 0 && !problems.replay;
  const patienceOn = Number(values.patience) > 0 && !problems.patience;

  return (
    <>
      <div className="row">
        <SelectField
          label="Order"
          hint={hint("how every epoch walks the texts")}
          value={values.order}
          onChange={(value) => set("order", value)}
          options={ORDERS}
          disabled={disabled}
        />
        <NumberField
          label="Curriculum"
          hint={hint("the share of the ordered texts the first epoch walks, growing to all; 1 = off")}
          value={values.curriculum}
          onChange={(value) => set("curriculum", value)}
          min={0}
          max={1}
          step="any"
          disabled={disabled}
        />
      </div>
      <div className="row">
        <NumberField
          label="Replay"
          hint={hint("texts rehearsed per epoch, as a share of the run's; 0 = off")}
          value={values.replay}
          onChange={(value) => set("replay", value)}
          min={0}
          step="any"
          disabled={disabled}
        />
        <NumberField
          label="Replay buffer size"
          hint={hint("texts the model keeps to rehearse; 0 drops it, blank leaves it")}
          value={values.replaySize}
          onChange={(value) => set("replaySize", value)}
          min={0}
          step={1}
          placeholder="leave as it is"
          disabled={disabled}
        />
      </div>
      <div className="row">
        <NumberField
          label="Patience"
          hint={hint("stop after this many full epochs without improving; 0 = off")}
          value={values.patience}
          onChange={(value) => set("patience", value)}
          min={0}
          step={1}
          disabled={disabled}
        />
        <NumberField
          label="Min improvement"
          hint={hint("how far the loss must fall below its best to count")}
          value={values.minDelta}
          onChange={(value) => set("minDelta", value)}
          min={0}
          step="any"
          disabled={disabled || !patienceOn}
        />
      </div>
      <Alert message={problemText(problems)} />
      {compact ? (
        <>
          {texts > 0 && epochs > 0 && curriculumOn ? (
            <p className="muted">
              Of the {fmtInt(texts)} texts, the epochs walk {counts.map(fmtInt).join(", ")}.
              {patienceOn ? " Early stopping watches only the epochs that walk all of them." : ""}
            </p>
          ) : null}
          <p className="muted">
            {replayLine(replay)}
            {replayOn && pool > 0 && texts > 0
              ? ` Every epoch rehearses ${fmtInt(rehearsed)} of them after the new texts.`
              : ""}
            {replayOn && pool === 0 ? " Replay has nothing to rehearse until the model keeps a buffer: give it a size, and this run fills it." : ""}
          </p>
        </>
      ) : (
        <>
          <p className="muted">
            <b>Order</b> is the order every epoch walks the texts in. For the count model it also decides what the
            sliding window remembers - the end of an epoch is what &quot;recent&quot; means. <b>Shuffle</b> draws a
            fresh order every epoch from the model&apos;s seed, without touching its random generator.
          </p>
          <p className="muted">
            A <b>curriculum</b> of <code>c</code> starts the run on the first <code>c</code> of the ordered texts
            and grows the share evenly to all of them by the last epoch - with <b>shortest first</b>, the classic
            baby steps. The graph still sees every text before the first epoch; only what each epoch counts
            changes.
          </p>
          <p className="muted">
            The <b>replay buffer</b> is a uniform sample of every text the model was ever trained on, kept with the
            model and saved in its file. A run with <b>replay</b> <code>r</code> walks <code>r</code> times as many
            of those texts as it has new ones, after the new ones, a fresh slice each epoch - rehearsing what it
            read before, so a new corpus does not wash an old one out. Rehearsed texts are not counted as trained
            texts. A <b>size</b> sets the buffer&apos;s capacity from this run on (0 drops it); blank leaves it as
            it is.
          </p>
          <p className="muted">
            <b>Patience</b> ends a run that has stopped improving: after that many full epochs whose loss did not
            fall at least <b>min improvement</b> below the best so far. An epoch still inside the curriculum walks
            fewer texts and does not count. The epoch that stops the run is marked in the history.
          </p>
          <p className="muted">
            These are for training - the <b>Train</b> tab shows the same controls. Thumbs up and down, 2NRL and the
            negative network&apos;s blame walk their texts as they always have, and never touch the buffer.
          </p>
        </>
      )}
    </>
  );
}
