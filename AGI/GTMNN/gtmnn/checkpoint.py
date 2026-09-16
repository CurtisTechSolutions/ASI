"""Rotating checkpoints with a `latest` pointer. Same contract as
RadixCyclicNN/radixnet/checkpoint.py."""
import json, os, time


class CheckpointManager:
    def __init__(self, directory, keep=5, prefix="gtmnn"):
        self.dir, self.keep, self.prefix = directory, int(keep), prefix
        os.makedirs(directory, exist_ok=True)

    def _stamp(self):
        return time.strftime("%Y%m%d-%H%M%S")

    def path_for(self, tag=None):
        return os.path.join(self.dir, f"{self.prefix}-{tag or self._stamp()}.json.gz")

    def save(self, net, tag=None, meta=None):
        path = self.path_for(tag)
        size = net.save(path)
        with open(os.path.join(self.dir, "latest"), "w", encoding="utf-8") as f:
            f.write(os.path.basename(path))
        if meta is not None:
            with open(path + ".meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f)
        self.rotate()
        return path, size

    def list(self):
        """Newest first, by MTIME. Sorting by name would put a `gtmnn-ck.*` tag
        ahead of every timestamp and rotation would delete the newest save --
        which is exactly the bug this ordering exists to avoid."""
        out = []
        for n in os.listdir(self.dir):
            if not n.startswith(self.prefix) or n.endswith(".meta.json"): continue
            p = os.path.join(self.dir, n)
            if os.path.isfile(p): out.append((os.path.getmtime(p), p))
        return [p for _, p in sorted(out, reverse=True)]

    def latest(self):
        ptr = os.path.join(self.dir, "latest")
        if os.path.exists(ptr):
            with open(ptr, encoding="utf-8") as f: name = f.read().strip()
            p = os.path.join(self.dir, name)
            if os.path.exists(p): return p
        got = self.list()
        return got[0] if got else None

    def rotate(self):
        got = self.list()
        removed = []
        for p in got[self.keep:]:
            try:
                os.remove(p)
                if os.path.exists(p + ".meta.json"): os.remove(p + ".meta.json")
                removed.append(p)
            except OSError:
                pass
        return removed

    def resume(self, cls, backend="auto"):
        p = self.latest()
        return cls.load(p, backend=backend) if p else None
