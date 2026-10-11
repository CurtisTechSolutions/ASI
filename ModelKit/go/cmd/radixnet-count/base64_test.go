package main

import "testing"

// --decode-to numbers several outputs before the extension exactly as the
// Python CLI does, so the two write the same file names: splitExt is
// os.path.splitext, a base name's leading dots included.
func TestSplitExtIsPythons(t *testing.T) {
	cases := []struct{ path, root, ext string }{
		{"out.png", "out", ".png"},
		{"out", "out", ""},
		{".hidden", ".hidden", ""},
		{"dir.d/out", "dir.d/out", ""},
		{"a/b.tar.gz", "a/b.tar", ".gz"},
		{"..png", "..png", ""},
		{"a..png", "a.", ".png"},
		{"out.", "out", "."},
		{"/tmp/x/.profile", "/tmp/x/.profile", ""},
		{"dir/.a.b", "dir/.a", ".b"},
		{"...", "...", ""},
		{"a/.", "a/.", ""},
		{"x.y/", "x.y/", ""},
		{"", "", ""},
	}
	for _, c := range cases {
		if root, ext := splitExt(c.path); root != c.root || ext != c.ext {
			t.Errorf("splitExt(%q) = %q, %q; Python says %q, %q", c.path, root, ext, c.root, c.ext)
		}
	}
}
