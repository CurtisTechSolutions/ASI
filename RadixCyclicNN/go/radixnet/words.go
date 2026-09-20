package radixnet

// The word view of a word encoding: what alphabet the graph has actually read.
//
// Under `--encoding word:n:s` a gram is n words rather than n characters
// (encoding.go), and a label is text like any other.  There is no vocabulary
// object to consult, and that is the point of the design: a gram is text, so
// the alphabet the graph knows is whatever its grams are made of.  It grows as
// training reads new words and there is nothing to freeze, prune or learn.
//
// What is here is the *reporting* side of that - `radixnet-count words`,
// `GET /api/words` and the frontend's Words tab - plus the one thing every
// number of a model needs beside it: what it is counted in.
//
// The Go twin of the Python `Encoding.units_name` / `Encoding.vocabulary`.

import "sort"

// UnitsName is what this encoding counts in: "chars" or "words".
//
// Every length, count and score of a model is in these, so the CLI, the API
// and the frontend all carry it beside the number - a per-word number read as
// per-character is read wrong.
func (e Encoding) UnitsName() string {
	if e.Unit == Words {
		return "words"
	}
	return "chars"
}

// Vocabulary is the words of a word encoding's grams and how many grams hold
// each.
//
// `grams` is the graph's gram index, which is the one count that survives
// compression: a phrase merged into one node is still made of the grams that
// built it.
func (e Encoding) Vocabulary(grams []string) map[string]int {
	counts := make(map[string]int, len(grams))
	for _, gram := range grams {
		u := e.Units(gram)
		for i := 0; i < u.Len(); i++ {
			counts[u.Slice(i, i+1)]++
		}
	}
	return counts
}

// GramIndex is every gram the graph holds, in the order they were first read.
func (g *Graph) GramIndex() []string {
	out := make([]string, 0, len(g.index))
	for gram := range g.index {
		out = append(out, gram)
	}
	sort.Strings(out) // a map iterates by the race; the caller wants one answer
	return out
}

// WordRow is one word of the alphabet: how many grams hold it, and its rank in
// the listing.
//
// There is no id to give a word - the graph has no vocabulary, so nothing
// records the order the words were first read - so ID is the row's rank, which
// is what a reader can actually check.
type WordRow struct {
	Word  string `json:"word"`
	ID    int    `json:"id"`
	Grams int    `json:"grams"`
	// Trigrams is Grams under its old name, kept so a client written against the
	// word model still reads this.
	Trigrams int `json:"trigrams"`
}

// TopWords is the alphabet, most read first, ties alphabetically.
// `limit <= 0` is all of them.  A model that does not count in words has none.
//
// The order is the Python implementation's, word for word: two ports asked for
// the same alphabet hand back the same rows in the same order.
func (m *Model) TopWords(limit int) []WordRow {
	enc := m.Encoding()
	if enc.Unit != Words {
		return nil
	}
	counts := enc.Vocabulary(m.G.GramIndex())
	rows := make([]WordRow, 0, len(counts))
	for word, n := range counts {
		rows = append(rows, WordRow{Word: word, Grams: n, Trigrams: n})
	}
	sortWordRows(rows)
	for i := range rows {
		rows[i].ID = i
	}
	if limit > 0 && len(rows) > limit {
		rows = rows[:limit]
	}
	return rows
}

// sortWordRows orders by grams descending, then alphabetically - a map iterates
// by the race, and a listing has to be one answer.
func sortWordRows(rows []WordRow) {
	sort.Slice(rows, func(i, j int) bool {
		if rows[i].Grams != rows[j].Grams {
			return rows[i].Grams > rows[j].Grams
		}
		return rows[i].Word < rows[j].Word
	})
}
