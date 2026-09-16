"""Saving and loading a trained cortex.

Without this every run retrains from zero, which caps every experiment at what
fits in one process: no long runs, no accumulating training across sessions, no
comparing a model against its earlier self.

What is saved is the LEARNED state -- weights, vocabularies, region membership,
each region's decision rule, the similarity graph. What is not saved is the game
ADAPTERS: those are code, they are reconstructed by name from `games.ALL` on
load, and a checkpoint that outlived its adapters should fail loudly rather than
resurrect a stale copy of them.

JSON, optionally gzipped by extension. Writes are atomic (`.tmp` then
`os.replace`), so a run killed mid-save leaves the previous checkpoint intact.
"""
import gzip, json, os, time

FORMAT = 1

def _open(path, mode, gz=None):
    """`gz` overrides the extension check. The atomic write goes to `<path>.tmp`,
    which does not end in .gz, so compression must follow the FINAL path or the
    file is written plain and read back as gzip."""
    if gz is None: gz = path.endswith(".gz")
    return gzip.open(path, mode + "t", encoding="utf-8") if gz \
        else open(path, mode, encoding="utf-8")

def save(cortex, path):
    """Atomic: write to .tmp, then replace. Returns bytes written."""
    from cortex.games import ALL
    doc = {
        "format": FORMAT, "saved_at": time.time(),
        "tau_new": cortex.tau_new, "seed": cortex.seed, "nh": cortex.nh,
        "graph": cortex.graph.to_dict(),
        "route_of": cortex.route_of,
        "log": cortex.log,
        "regions": [{
            "id": r.id,
            "games": [g.name for g in r.games],
            "vocab": r.vocab.to_dict(),
            "net": r.net.to_dict() if r.net else None,
            "rule": r.rule, "plays": r.plays, "hits": r.hits,
            "mech": sorted(r.mech),
        } for r in cortex.regions],
    }
    unknown = [n for r in doc["regions"] for n in r["games"] if n not in ALL]
    if unknown:
        raise ValueError(f"cannot save: no adapter for {unknown}")
    tmp = path + ".tmp"
    with _open(tmp, "w", gz=path.endswith(".gz")) as f: json.dump(doc, f)
    os.replace(tmp, path)
    return os.path.getsize(path)

def load(path):
    """Rebuild a Cortex. Game adapters are looked up by name in games.ALL; a
    name with no adapter is an error, not a silent drop -- a region missing one
    of its games would have a vocabulary it can no longer explain."""
    from cortex.cortex import Cortex, Region
    from cortex.graph import SimilarityGraph
    from cortex.vocabulary import Vocabulary
    from cortex.sbnn import SBNN
    from cortex.games import ALL
    with _open(path, "r") as f: doc = json.load(f)
    if doc.get("format") != FORMAT:
        raise ValueError(f"checkpoint format {doc.get('format')}, expected {FORMAT}")
    c = Cortex(tau_new=doc["tau_new"], seed=doc["seed"], nh=doc["nh"])
    c.graph = SimilarityGraph.from_dict(doc["graph"])
    c.route_of = dict(doc["route_of"]); c.log = doc["log"]
    c.regions = []
    for rd in doc["regions"]:
        r = Region(rd["id"], seed=doc["seed"] + rd["id"], nh=doc["nh"])
        missing = [n for n in rd["games"] if n not in ALL]
        if missing: raise ValueError(f"no adapter for {missing}")
        r.games = [ALL[n] for n in rd["games"]]
        r.vocab = Vocabulary.from_dict(rd["vocab"])
        r.net = SBNN.from_dict(rd["net"]) if rd["net"] else None
        r.rule = rd["rule"]; r.plays = rd["plays"]; r.hits = rd["hits"]
        r.mech = frozenset(rd["mech"])
        c.regions.append(r)
    return c

class CheckpointManager:
    """Rotating checkpoints with a `latest` pointer, as RadixCyclicNN does."""
    def __init__(self, directory, keep=10, gzip_it=True):
        self.dir, self.keep, self.gz = directory, keep, gzip_it
        os.makedirs(directory, exist_ok=True)

    def _path(self, tag):
        ext = ".json.gz" if self.gz else ".json"
        return os.path.join(self.dir, f"{int(time.time()*1000)}-{tag}{ext}")

    def save(self, cortex, tag="ckpt"):
        p = self._path(tag); n = save(cortex, p)
        with open(os.path.join(self.dir, "latest"), "w") as f: f.write(os.path.basename(p))
        keep_name = os.path.basename(p)
        for old in self.list()[self.keep:]:
            if old == keep_name: continue          # never rotate away what we just wrote
            try: os.remove(os.path.join(self.dir, old))
            except OSError: pass
        return {"path": p, "bytes": n}

    def list(self):
        """Only files this manager named: `<millis>-<tag>.json[.gz]`, newest
        first BY TIMESTAMP. Sorting the raw names string-wise put any other
        checkpoint in the directory ahead of the timestamped ones and rotation
        then deleted the newest real save."""
        out = []
        for f in os.listdir(self.dir):
            if not f.endswith((".json", ".json.gz")): continue
            head = f.split("-", 1)[0]
            if not head.isdigit(): continue
            out.append((int(head), f))
        return [f for _, f in sorted(out, reverse=True)]

    def latest(self):
        p = os.path.join(self.dir, "latest")
        if not os.path.exists(p): return None
        with open(p) as f: name = f.read().strip()
        full = os.path.join(self.dir, name)
        return full if os.path.exists(full) else None

    def load_latest(self):
        p = self.latest()
        return load(p) if p else None
