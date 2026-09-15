"""The cyclic similarity graph. Distance IS similarity: Jaccard distance over
mechanic sets, which is a proper metric (Levandowsky & Winter 1971), so the
geometry is consistent rather than a figure of speech. `check_metric` asserts it.

Cyclic because similarity is not hierarchical: any three games with overlapping
mechanics close a triangle, and a tree would have to break one of its edges.
"""
import itertools

def distance(a, b):
    u = a | b
    return 1.0 - (len(a & b) / len(u)) if u else 1.0

class SimilarityGraph:
    def __init__(self, tau=0.75):
        self.names, self.mech, self.adj, self.tau = [], [], [], tau

    def add(self, name, mechanics):
        i = len(self.names)
        self.names.append(name); self.mech.append(frozenset(mechanics)); self.adj.append({})
        for j in range(i):
            d = distance(self.mech[i], self.mech[j])
            if d <= self.tau:
                self.adj[i][j] = d; self.adj[j][i] = d
        return i

    def index(self, name): return self.names.index(name)
    def dist(self, a, b): return distance(self.mech[a], self.mech[b])

    def neighbours(self, i, k=8):
        return sorted(self.adj[i].items(), key=lambda kv: kv[1])[:k]

    def check_metric(self):
        """Triangle-inequality violations. Must be zero: without it there are
        triples whose distances cannot be realised at once, and every clustering
        would be arguing with the data."""
        n = len(self.names); bad = 0
        for a, b, c in itertools.permutations(range(n), 3):
            if self.dist(a, c) > self.dist(a, b) + self.dist(b, c) + 1e-12: bad += 1
        return bad

    def matrix(self):
        n = len(self.names)
        return [[self.dist(i, j) for j in range(n)] for i in range(n)]
