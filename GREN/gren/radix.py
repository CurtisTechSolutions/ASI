"""A radix tree with the mechanics of a trie, over signature tokens.

Both halves are load-bearing and neither alone suffices:

  from the TRIE   a terminal flag on ANY node, internal ones included, so a
                  general class can be a strict prefix of a specific one --
                  "a two-player perfect-information board game" is a real answer
                  whose shared rules can be used while identification continues.
  from the RADIX  path compression, so only the tokens that actually
                  DISCRIMINATE cost a node.

Their interaction is one rule, and it is the single bug this structure is prone
to: A TERMINAL NODE IS NEVER MERGED AWAY. Without that clause, compression
silently deletes the general classes that are the reason for using a trie.
"""
class RadixGameTree:
    def __init__(self):
        self.segment = [()]            # node -> compressed run of tokens
        self.children = [{}]           # node -> first token of a child run -> node
        self.parent = [-1]
        self.terminal = [False]
        self.game = [None]
        self.count = [0]

    def _new(self, seg, parent):
        self.segment.append(tuple(seg)); self.children.append({})
        self.parent.append(parent); self.terminal.append(False)
        self.game.append(None); self.count.append(0)
        return len(self.segment) - 1

    def split(self, node, i):
        """node keeps segment[:i]; a new child B takes segment[i:] along with
        node's children, terminal flag and game. node becomes non-terminal."""
        seg = self.segment[node]
        assert 0 < i < len(seg), (i, seg)
        b = self._new(seg[i:], node)
        self.children[b] = self.children[node]
        for c in self.children[b].values(): self.parent[c] = b
        self.terminal[b] = self.terminal[node]; self.game[b] = self.game[node]
        self.count[b] = self.count[node]
        self.segment[node] = seg[:i]
        self.children[node] = {seg[i]: b}
        self.terminal[node] = False; self.game[node] = None
        return node, b

    def insert(self, tokens, game):
        tokens = tuple(tokens); node = 0; pos = 0
        while True:
            self.count[node] += 1
            seg = self.segment[node]
            i = 0
            while i < len(seg) and pos < len(tokens) and seg[i] == tokens[pos]:
                i += 1; pos += 1
            if i < len(seg):
                self.split(node, i)                 # diverged mid-run
                seg = self.segment[node]
            if pos == len(tokens):
                self.terminal[node] = True; self.game[node] = game
                return node
            t = tokens[pos]
            nxt = self.children[node].get(t)
            if nxt is None:
                leaf = self._new(tokens[pos:], node)
                self.children[node][t] = leaf
                self.terminal[leaf] = True; self.game[leaf] = game
                self.count[leaf] = 1
                return leaf
            node = nxt

    def merge_child(self, p):
        """Radix compression, with the trie clause. `not self.terminal[p]` is the
        whole interaction between the two structures."""
        if self.terminal[p]: return False           # <- the clause
        kids = list(self.children[p].values())
        if len(kids) != 1: return False
        c = kids[0]
        self.segment[p] = self.segment[p] + self.segment[c]
        self.children[p] = self.children[c]
        for g in self.children[p].values(): self.parent[g] = p
        self.terminal[p] = self.terminal[c]; self.game[p] = self.game[c]
        self.count[p] = max(self.count[p], self.count[c])
        self.segment[c] = (); self.children[c] = {}
        return True

    def compress(self):
        merged = total = 0
        changed = True
        while changed:
            changed = False
            for n in range(len(self.segment)):
                if self.segment[n] == () and n != 0: continue
                if self.merge_child(n): merged += 1; changed = True
        return merged

    def path(self, node):
        out = []
        while node != 0:
            out.append(self.segment[node]); node = self.parent[node]
        return tuple(t for seg in reversed(out) for t in seg)

    def terminals(self):
        return [n for n in range(len(self.segment)) if self.terminal[n]]

    def lookup(self, tokens):
        tokens = tuple(tokens); node = 0; pos = 0
        while True:
            seg = self.segment[node]
            if tokens[pos:pos+len(seg)] != seg: return None
            pos += len(seg)
            if pos == len(tokens):
                return node if self.terminal[node] else None
            nxt = self.children[node].get(tokens[pos])
            if nxt is None: return None
            node = nxt

    def retrieve(self, known, beam=32):
        """Partial evidence: walk, scoring a node by how many of its tokens the
        evidence confirms. Unknown tokens neither confirm nor deny, so the walk
        BRANCHES rather than guessing -- which is how partial evidence yields a
        cluster instead of a point."""
        import heapq
        out = []
        heap = [(-0.0, 0)]
        seen = 0
        while heap and seen < beam * 8:
            negscore, node = heapq.heappop(heap); seen += 1
            score = -negscore
            if self.terminal[node] and self.game[node] is not None:
                out.append((score, self.game[node]))
            for c in self.children[node].values():
                hits = sum(1 for t in self.segment[c] if t in known)
                heapq.heappush(heap, (-(score + hits), c))
        out.sort(key=lambda t: -t[0])
        return out[:beam]
