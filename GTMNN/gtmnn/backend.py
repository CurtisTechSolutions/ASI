"""The population forward pass is the cost floor, so it is the one thing worth
vectorising. Everything else -- the auction, the solver, Shapley -- is O(M) or
O(M*K) work on 64 seats and stays in pure Python whatever the backend.
"""
import time


class PythonBackend:
    name = "python"

    def __init__(self, device=None):
        self.device = None
        self.forwards = 0
        self.steps = 0

    def forward_population(self, pool, feat, ids=None):
        ids = ids if ids is not None else pool.living()
        self.forwards += len(ids)
        return ids, [pool.forward(i, feat) for i in ids]

    def learn_batch(self, pool, pending, lr, act_lr, clip=5.0):
        """pending: (micro_id, target, phi, feat), where target is the profile
        row the credit was earned on. One version bump for the whole batch
        rather than one per micro."""
        v = pool.version
        for i, target, phi, feat in pending:
            if phi == 0.0: continue
            pool.learn(i, target, phi, lr, act_lr, feat, clip)
        pool.version = v + 1
        self.steps += len(pending)

    def describe(self):
        return {"name": self.name, "device": None, "forwards": self.forwards,
                "steps": self.steps}


def get_backend(name="auto", device=None):
    """`auto` and `python` both give the pure-Python backend.

    There is no TorchBackend in this build. DESIGN.md 15 specifies one -- the
    whole population as a single (N, R, H) batched contraction -- and it is the
    right optimisation, but torch is not a dependency of this repository and
    shipping a vectorised path that has never been executed would be a
    performance claim with no measurement behind it. Asking for it says so
    rather than silently returning the slow one."""
    if name in ("auto", "python", None): return PythonBackend(device)
    if name == "torch":
        raise NotImplementedError(
            "TorchBackend is specified in DESIGN.md 15 but not implemented in this "
            "build: torch is not a dependency here, and an unexercised vectorised "
            "path is a performance claim with nothing behind it. Use backend='python'.")
    raise ValueError(f"unknown backend {name!r}")
