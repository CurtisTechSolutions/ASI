//! The unit diff between what the network wrote and what the teacher
//! corrected (`radixnet/diff.py`, `go/radixnet/diff.go`).
//!
//! The tutor's marking used to be a verdict on a whole sentence: the failed
//! attempt was garbage, the correction gospel, and every edge of either path
//! moved by the same amount.  Most of a corrected sentence is word for word
//! what the network wrote, so this aligns the two and says *where* they
//! differ, so only the nodes the mistake ran through move.
//!
//! A plain longest-common-subsequence diff over units, after the shared prefix
//! and suffix are trimmed; the same tie-break as the other two, so all three
//! mark the same characters (and, under a word encoding, the same words).

use crate::encoding::Encoding;
use crate::json::Json;

/// Above this many cells (|a| x |b| after trimming) the middle is marked
/// changed as a whole instead of aligned.
pub const MAX_CELLS: usize = 4_000_000;

/// Changes any closer than a trigram are one change under the default
/// encoding; another encoding uses its own n.
pub const MIN_EQUAL_RUN: usize = 3;

/// One step of the alignment: `op` is `equal`, `replace`, `delete` or
/// `insert`; `wrong[a0..a1]` against `right[b0..b1]`, in the encoding's units.
#[derive(Clone, Debug, PartialEq)]
pub struct Edit {
    pub op: &'static str,
    pub a0: usize,
    pub a1: usize,
    pub b0: usize,
    pub b1: usize,
    pub wrong: String,
    pub right: String,
}

impl Edit {
    /// `{"op", "wrong", "right"}`, as a lesson record carries it.
    pub fn to_json(&self) -> Json {
        Json::obj([
            ("op", Json::str(self.op)),
            ("wrong", Json::str(self.wrong.clone())),
            ("right", Json::str(self.right.clone())),
        ])
    }

    /// `{"op", "wrong", "right", "at": [a0, a1], "to": [b0, b1]}`, as the
    /// copy editor's `changes` carry it (Python's `_correction_entry`): the
    /// lesson record's three fields and where the change sits on each side.
    pub fn to_json_with_spans(&self) -> Json {
        Json::obj([
            ("op", Json::str(self.op)),
            ("wrong", Json::str(self.wrong.clone())),
            ("right", Json::str(self.right.clone())),
            ("at", Json::ints([self.a0 as i64, self.a1 as i64])),
            ("to", Json::ints([self.b0 as i64, self.b1 as i64])),
        ])
    }

    /// An edit as a record carries it (the inverse of
    /// [`Edit::to_json_with_spans`]); an unknown `op` reads as `replace`, and
    /// a missing span as empty.
    pub fn from_json(doc: &Json) -> Edit {
        let op = match doc.at("op").as_str() {
            Some("equal") => "equal",
            Some("delete") => "delete",
            Some("insert") => "insert",
            _ => "replace",
        };
        let span = |key: &str| -> (usize, usize) {
            let pair = doc.at(key).as_array();
            let at = |i: usize| pair.get(i).and_then(|v| v.as_i64()).unwrap_or(0).max(0) as usize;
            (at(0), at(1))
        };
        let (a0, a1) = span("at");
        let (b0, b1) = span("to");
        Edit {
            op,
            a0,
            a1,
            b0,
            b1,
            wrong: doc.at("wrong").as_str().unwrap_or("").to_string(),
            right: doc.at("right").as_str().unwrap_or("").to_string(),
        }
    }
}

/// A half-open range of unit positions.
pub type Span = (usize, usize);

fn atoms(enc: &Encoding, text: &str) -> Vec<String> {
    let units = enc.units(text);
    (0..units.len()).map(|i| units.at(i)).collect()
}

struct Sides<'a> {
    a: &'a [String],
    b: &'a [String],
    enc: &'a Encoding,
}

impl Sides<'_> {
    fn edit(&self, op: &'static str, a0: usize, a1: usize, b0: usize, b1: usize) -> Edit {
        let join = |part: &[String]| {
            let refs: Vec<&str> = part.iter().map(|s| s.as_str()).collect();
            self.enc.join(&refs)
        };
        Edit {
            op,
            a0,
            a1,
            b0,
            b1,
            wrong: join(&self.a[a0..a1]),
            right: join(&self.b[b0..b1]),
        }
    }
}

/// Aligns `wrong` against `right` one unit of `encoding` at a time; equal runs
/// included, in order.
pub fn edits(wrong: &str, right: &str, encoding: &Encoding) -> Vec<Edit> {
    let a = atoms(encoding, wrong);
    let b = atoms(encoding, right);
    let sides = Sides {
        a: &a,
        b: &b,
        enc: encoding,
    };
    let (n, m) = (a.len(), b.len());
    let mut head = 0;
    while head < n && head < m && a[head] == b[head] {
        head += 1;
    }
    let mut tail = 0;
    while tail < n - head && tail < m - head && a[n - 1 - tail] == b[m - 1 - tail] {
        tail += 1;
    }
    let (mid_a, mid_b) = (&a[head..n - tail], &b[head..m - tail]);
    let mut out = Vec::new();
    if head > 0 {
        out.push(sides.edit("equal", 0, head, 0, head));
    }
    if !mid_a.is_empty() || !mid_b.is_empty() {
        if mid_a.is_empty() {
            out.push(sides.edit("insert", head, head, head, m - tail));
        } else if mid_b.is_empty() {
            out.push(sides.edit("delete", head, n - tail, head, head));
        } else if mid_a.len() * mid_b.len() > MAX_CELLS {
            // too big to align: one change covering the middle
            out.push(sides.edit("replace", head, n - tail, head, m - tail));
        } else {
            out.extend(align(mid_a, mid_b, head, head, &sides));
        }
    }
    if tail > 0 {
        out.push(sides.edit("equal", n - tail, n, m - tail, m));
    }
    coalesce(merge(out, &sides), &sides, encoding.n)
}

/// The unit ranges the two disagree on: the spans of `wrong`, then of
/// `right`.  An insertion is an empty span on the side that lacks the text,
/// kept where it belongs: the model blames the step that walked past it.
pub fn changed_spans(wrong: &str, right: &str, encoding: &Encoding) -> (Vec<Span>, Vec<Span>) {
    let mut left = Vec::new();
    let mut other = Vec::new();
    for e in edits(wrong, right, encoding) {
        if e.op == "equal" {
            continue;
        }
        left.push((e.a0, e.a1));
        other.push((e.b0, e.b1));
    }
    (left, other)
}

/// The changes for a lesson record (equal runs dropped); `limit` 0 keeps all.
pub fn summary(wrong: &str, right: &str, limit: usize, encoding: &Encoding) -> Vec<Edit> {
    let mut out: Vec<Edit> = edits(wrong, right, encoding)
        .into_iter()
        .filter(|e| e.op != "equal")
        .collect();
    if limit > 0 && out.len() > limit {
        out.truncate(limit);
    }
    out
}

/// Whether the half-open range `[lo, hi)` meets any changed span.  An empty
/// span is an insertion point: the step that walked straight past the
/// position is the one at fault.
pub fn spans_touch(lo: usize, hi: usize, spans: &[Span]) -> bool {
    spans.iter().any(|&(s_lo, s_hi)| {
        let end = if s_hi == s_lo { s_lo + 1 } else { s_hi };
        s_lo < hi && lo < end
    })
}

/// The longest common subsequence of two trimmed middles, walked forward.
fn align(a: &[String], b: &[String], off_a: usize, off_b: usize, sides: &Sides) -> Vec<Edit> {
    let (n, m) = (a.len(), b.len());
    // lcs[i][j] = length of the longest common subsequence of a[i..] and b[j..]
    let width = m + 1;
    let mut lcs = vec![0usize; (n + 1) * width];
    for i in (0..n).rev() {
        for j in (0..m).rev() {
            lcs[i * width + j] = if a[i] == b[j] {
                lcs[(i + 1) * width + j + 1] + 1
            } else {
                lcs[(i + 1) * width + j].max(lcs[i * width + j + 1])
            };
        }
    }
    let at = |i: usize, j: usize| lcs[i * width + j];
    let mut out = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < n || j < m {
        if i < n && j < m && a[i] == b[j] {
            let (si, sj) = (i, j);
            while i < n && j < m && a[i] == b[j] {
                i += 1;
                j += 1;
            }
            out.push(sides.edit("equal", off_a + si, off_a + i, off_b + sj, off_b + j));
            continue;
        }
        let (si, sj) = (i, j);
        // the same tie-break as the other two: a deletion first when both are equally good
        while (i < n || j < m) && !(i < n && j < m && a[i] == b[j]) {
            if i < n && (j >= m || at(i + 1, j) >= at(i, j + 1)) {
                i += 1;
            } else {
                j += 1;
            }
        }
        let op = if i > si && j > sj {
            "replace"
        } else if i > si {
            "delete"
        } else {
            "insert"
        };
        out.push(sides.edit(op, off_a + si, off_a + i, off_b + sj, off_b + j));
    }
    out
}

fn joins(last: &Edit, edit: &Edit) -> bool {
    if last.a1 != edit.a0 || last.b1 != edit.b0 {
        return false;
    }
    if last.op == edit.op {
        return true;
    }
    last.op != "equal" && edit.op != "equal" // delete + insert = replace
}

/// Joins neighbouring edits of the same kind (the trimming can split a run).
fn merge(items: Vec<Edit>, sides: &Sides) -> Vec<Edit> {
    let mut out: Vec<Edit> = Vec::new();
    for edit in items {
        if let Some(last) = out.last() {
            if joins(last, &edit) {
                let op = if last.op == edit.op { last.op } else { "replace" };
                let merged = sides.edit(op, last.a0, edit.a1, last.b0, edit.b1);
                *out.last_mut().expect("a last edit") = merged;
                continue;
            }
        }
        out.push(edit);
    }
    out
}

/// Swallows equal runs shorter than a gram between two changes ("mat" ->
/// "park", not "m" -> "p" and "t" -> "rk").
fn coalesce(items: Vec<Edit>, sides: &Sides, min_run: usize) -> Vec<Edit> {
    let mut out: Vec<Edit> = Vec::new();
    for edit in items {
        let n = out.len();
        if n >= 2
            && edit.op != "equal"
            && out[n - 1].op == "equal"
            && out[n - 2].op != "equal"
            && out[n - 1].a1 - out[n - 1].a0 < min_run
        {
            out.pop();
            let before = out.pop().expect("a change before the gap");
            let op = if edit.a1 > before.a0 && edit.b1 > before.b0 {
                "replace"
            } else {
                before.op
            };
            out.push(sides.edit(op, before.a0, edit.a1, before.b0, edit.b1));
            continue;
        }
        out.push(edit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::parse_encoding;

    fn ops(wrong: &str, right: &str) -> Vec<(&'static str, String, String)> {
        edits(wrong, right, &Encoding::default())
            .into_iter()
            .map(|e| (e.op, e.wrong, e.right))
            .collect()
    }

    #[test]
    fn an_added_letter_is_one_insertion() {
        assert_eq!(
            ops("the cat sit", "the cat sits"),
            vec![
                ("equal", "the cat sit".to_string(), "the cat sit".to_string()),
                ("insert", String::new(), "s".to_string()),
            ]
        );
    }

    #[test]
    fn close_changes_are_one_change() {
        let changes = summary(
            "the cat sat on the mat",
            "the cat sat on the park",
            0,
            &Encoding::default(),
        );
        assert_eq!(changes.len(), 1, "{changes:?}");
        assert_eq!((changes[0].wrong.as_str(), changes[0].right.as_str()), ("mat", "park"));
        let (left, right) = changed_spans(
            "the cat sat on the mat",
            "the cat sat on the park",
            &Encoding::default(),
        );
        assert_eq!(left, vec![(19, 22)]);
        assert_eq!(right, vec![(19, 23)]);
    }

    #[test]
    fn a_word_encoding_diffs_whole_words() {
        let enc = parse_encoding("word:2:1").unwrap();
        let changes = summary("he go to school", "he goes to school", 0, &enc);
        assert_eq!(changes.len(), 1);
        assert_eq!(
            (changes[0].op, changes[0].wrong.as_str(), changes[0].right.as_str()),
            ("replace", "go", "goes")
        );
        assert_eq!((changes[0].a0, changes[0].a1), (1, 2));
    }

    #[test]
    fn a_change_carries_its_spans_when_asked() {
        let changes = summary("Hi howe are you??", "Hi, how are you?", 0, &Encoding::default());
        let docs: Vec<String> = changes.iter().map(|e| e.to_json_with_spans().render(0)).collect();
        assert_eq!(
            docs,
            vec![
                "{\"op\":\"insert\",\"wrong\":\"\",\"right\":\",\",\"at\":[2,2],\"to\":[2,3]}",
                "{\"op\":\"delete\",\"wrong\":\"e\",\"right\":\"\",\"at\":[6,7],\"to\":[7,7]}",
                "{\"op\":\"delete\",\"wrong\":\"?\",\"right\":\"\",\"at\":[15,16],\"to\":[15,15]}",
            ]
        );
        assert_eq!(
            changes[0].to_json().render(0),
            "{\"op\":\"insert\",\"wrong\":\"\",\"right\":\",\"}",
            "the lesson record's shape is untouched"
        );
        for edit in &changes {
            assert_eq!(&Edit::from_json(&edit.to_json_with_spans()), edit);
        }
        let bare = Edit::from_json(&crate::json::parse("{\"op\": \"odd\", \"wrong\": \"x\"}").unwrap());
        assert_eq!((bare.op, bare.a0, bare.a1, bare.wrong.as_str()), ("replace", 0, 0, "x"));
    }

    #[test]
    fn identical_texts_have_nothing_to_blame() {
        assert!(summary("same", "same", 0, &Encoding::default()).is_empty());
        assert!(
            spans_touch(3, 5, &[(4, 4)]),
            "an insertion point touches the step past it"
        );
        assert!(!spans_touch(0, 2, &[(4, 6)]));
    }
}
