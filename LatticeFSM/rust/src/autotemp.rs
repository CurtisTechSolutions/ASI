//! Auto-temperature: when a reply is no answer - nothing, no words, or a loop - draw it again a little hotter, rung
//! by rung up a ladder, until one is an answer or the ladder ends.  The same rules as `autotemp.py`.
//!
//! Temperature is a re-roll, not a repair: a hotter draw escapes a loop or an early `</s>` only by chance, and the
//! hotter it is the less it follows what the model learned (on `data/corpus`, 0.3 still reads, 1.0 is word salad).
//! So the ladder starts where it was asked, climbs in small steps, stops early (`MAX_TEMPERATURE`), and draws a few
//! times at each rung before climbing.

/// The hottest rung, by default: above it the replies stop following the model.
pub const MAX_TEMPERATURE: f64 = 0.8;
/// How much hotter each rung is than the one before.
pub const STEP: f64 = 0.2;
/// Draws at each rung above 0 before climbing (at 0 a draw is the same every time, so one).
pub const TRIES: usize = 3;
/// The shortest stretch of text, in characters, that said again counts as a loop.
pub const MIN_LOOP: usize = 12;

/// Why a draw is no answer, or `None` when it is one: `text` the reply (without its prompt), `tokens` the prompt's
/// tokens and the reply's, the reply's from `start`, `context` how many tokens the model predicts from.  A greedy
/// draw (`temperature` 0) loops exactly when a context comes back (`repeats_context`); a sampled one is read for a
/// loop (`why_no_answer`).
pub fn why_draw(text: &str, temperature: f64, tokens: &[String], start: usize, context: usize) -> Option<&'static str> {
    if temperature <= 0.0 {
        return match why_empty(text) {
            Some(why) => Some(why),
            None if repeats_context(tokens, start, context) => Some("a loop"),
            None => None,
        };
    }
    why_no_answer(text)
}

fn why_empty(text: &str) -> Option<&'static str> {
    let t = text.trim();
    if t.is_empty() {
        Some("nothing")
    } else if !t.chars().any(char::is_alphanumeric) {
        Some("no words")
    } else {
        None
    }
}

/// Whether a greedy reply has begun going round: some token of it, from `start`, was drawn after the same `k`
/// tokens as an earlier one.  A model at temperature 0 draws the same after the same context, so from there it
/// writes what it wrote before, for ever - exact, where reading the text for a loop can only guess.  (Contexts
/// still reaching back to `<s>` are each their own.)
pub fn repeats_context(tokens: &[String], start: usize, k: usize) -> bool {
    let mut seen = std::collections::HashSet::new();
    (start.max(k)..tokens.len()).any(|g| !seen.insert(&tokens[g - k..g]))
}

/// Why `text` (a reply, without its prompt) is no answer, or `None` when it is one: `"nothing"` (empty), `"no words"`
/// (no letter or digit, say "."), `"a loop"` (`loop_period`: it says something again).
pub fn why_no_answer(text: &str) -> Option<&'static str> {
    if let Some(why) = why_empty(text) {
        return Some(why);
    }
    if loop_period(text.trim()).is_some() {
        return Some("a loop");
    }
    None
}

/// Where `text` says something again: the shortest `p` such that a stretch of at least `MIN_LOOP` characters comes
/// back `p` characters later, after no more new text than its own length (`r >= p / 2`).  A reply cut off by its
/// token limit has seldom gone round twice in full - "is a directed cycle is recorded in a brain is a directed
/// cycle is recorded in a" goes round its 43 characters not quite twice - so most of a period, said again, is
/// enough; "the cat sat on the mat" says only "the " again.
pub fn loop_period(text: &str) -> Option<usize> {
    let s: Vec<char> = text.chars().collect();
    let n = s.len();
    for p in 1..n {
        let need = MIN_LOOP.max(p.div_ceil(2));
        if need + p > n {
            break;
        }
        let mut run = 0;
        for i in 0..n - p {
            if s[i] == s[i + p] {
                run += 1;
                if run >= need {
                    return Some(p);
                }
            } else {
                run = 0;
            }
        }
    }
    None
}

/// The temperatures to try, coolest first: `start`, then `STEP` hotter each rung up to `top` (`start` alone when it
/// is already as hot).
pub fn ladder(start: f64, top: f64) -> Vec<f64> {
    let start = start.max(0.0);
    let mut out = vec![start];
    let mut k = 1;
    loop {
        // rounded to 1e-9 so that 0.2 * 3 is 0.6, as in Python
        let t = ((start + STEP * k as f64) * 1e9).round() / 1e9;
        if t > top + 1e-9 {
            break;
        }
        out.push(t);
        k += 1;
    }
    out
}

/// Every draw to make, in order: (temperature, the seed's offset from the sample's own seed).  The first is the
/// draw a plain request makes, so when it is an answer nothing changes.
pub fn attempts(start: f64, top: f64, tries: usize) -> Vec<(f64, u64)> {
    let mut out = Vec::new();
    for t in ladder(start, top) {
        let n = if t <= 0.0 { 1 } else { tries.max(1) };
        for _ in 0..n {
            let k = out.len() as u64;
            out.push((t, 7919 * k));
        }
    }
    out
}

/// How bad a reason is, for keeping the least bad when nothing was an answer: a loop still says something.
pub fn badness(why: &str) -> usize {
    match why {
        "a loop" => 1,
        "rejected by the guard" => 2,
        "no words" => 3,
        _ => 4,
    }
}

/// One draw on the way up the ladder: its temperature and seed, its text, and why it was no answer.
#[derive(Clone, Debug)]
pub struct Tried {
    pub temperature: f64,
    pub seed: u64,
    pub text: String,
    pub why: Option<&'static str>,
}

/// A reply drawn up the ladder: the draw kept (`pick`, its temperature and seed), whether it is an answer (`found`;
/// if not, `why`), and every draw made.
pub struct Climbed<T> {
    pub pick: T,
    pub temperature: f64,
    pub seed: u64,
    pub found: bool,
    pub why: Option<&'static str>,
    pub tried: Vec<Tried>,
}

/// Draw a reply rung by rung (`attempts(start, top, tries)`, every seed `seed` plus its offset) until one is an
/// answer: `draw(temperature, seed)` gives a reply's tokens, its text and anything else to keep; `why_draw` reads it,
/// with the prompt's tokens `prompt` and the model's `context`; `check` may find more wrong with an answer (the
/// guard, say) and keep what it found in it.  When none is an answer the least bad is kept (`badness`; the coolest
/// of equals).
#[allow(clippy::too_many_arguments)]
pub fn climb<T>(
    prompt: &[String],
    context: usize,
    start: f64,
    top: f64,
    tries: usize,
    seed: u64,
    mut draw: impl FnMut(f64, u64) -> (Vec<String>, String, T),
    mut check: impl FnMut(&str, &mut T) -> Option<&'static str>,
) -> Climbed<T> {
    let mut kept: Vec<(T, Tried)> = Vec::new();
    for (t, offset) in attempts(start, top, tries) {
        let s = seed.wrapping_add(offset);
        let (toks, text, mut extra) = draw(t, s);
        let mut all = prompt.to_vec();
        all.extend(toks);
        let mut why = why_draw(&text, t, &all, prompt.len(), context);
        if why.is_none() {
            why = check(&text, &mut extra);
        }
        let done = why.is_none();
        kept.push((
            extra,
            Tried {
                temperature: t,
                seed: s,
                text,
                why,
            },
        ));
        if done {
            break;
        }
    }
    let found = kept.last().is_some_and(|(_, x)| x.why.is_none());
    let mut pick = kept.len() - 1;
    if !found {
        pick = 0;
        for (k, (_, x)) in kept.iter().enumerate() {
            if badness(x.why.unwrap_or("")) < badness(kept[pick].1.why.unwrap_or("")) {
                pick = k;
            }
        }
    }
    let tried: Vec<Tried> = kept.iter().map(|(_, x)| x.clone()).collect();
    let (extra, x) = kept.swap_remove(pick);
    Climbed {
        pick: extra,
        temperature: x.temperature,
        seed: x.seed,
        found,
        why: x.why,
        tried,
    }
}

/// How a climb went, in a line: "found at temperature 0.4: 0 nothing, 0.2 no words, 0.4 an answer" or "no answer up
/// to 0.8 (kept the one at 0.2: a loop): ...".
pub fn climb_note(found: bool, temperature: f64, why: Option<&str>, tried: &[Tried]) -> String {
    let steps: Vec<String> = tried
        .iter()
        .map(|x| format!("{} {}", fmt_t(x.temperature), x.why.unwrap_or("an answer")))
        .collect();
    if found {
        format!("found at temperature {}: {}", fmt_t(temperature), steps.join(", "))
    } else {
        let top = tried.iter().map(|x| x.temperature).fold(0.0, f64::max);
        format!(
            "no answer up to {} (kept the one at {}: {}): {}",
            fmt_t(top),
            fmt_t(temperature),
            why.unwrap_or(""),
            steps.join(", ")
        )
    }
}

/// A temperature as Python's `f"{t:g}"` writes one.
fn fmt_t(t: f64) -> String {
    let s = format!("{t:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-0" {
        "0".to_string()
    } else {
        s.to_string()
    }
}
