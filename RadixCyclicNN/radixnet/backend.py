"""Training backends.

The graph exports its edges once per epoch as a CSR structure (:class:`CSR`)
together with the per-node parameters (:class:`NodeParams`).  A backend wraps
that mutable state (:meth:`Backend.prepare`), runs vectorised mini-batch
gradient steps (:meth:`Backend.step`) and hands the updated values back
(:meth:`Backend.finalize`).

Two implementations exist:

* :class:`PythonBackend` - always available; a tight pure-Python loop over
  flat lists with ``math.sin``/``cos``/``exp``/``log`` bound to locals and no
  per-edge object allocation.
* ``TorchBackend`` (``radixnet.backend_torch``) - optional, imported lazily,
  runs on ``cuda`` > ``mps`` > ``cpu``.

The learning rule is a *one-hop local rule*.  For a transition ``p -> c``
observed in training the loss is the negative log-softmax of the edge score
``w * f_p(z_p) * f_c(z_c)`` among all children of ``p``; gradients touch only
``w`` on ``p``'s out-edges, the states ``z`` and the activation parameters
``(a, b, h, k)`` of ``p`` and its children.  Nothing propagates deeper, so the
vanishing-gradient problem simply does not arise.
"""

from __future__ import annotations

import importlib
import importlib.util
import math
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CSR",
    "NodeParams",
    "Backend",
    "PythonBackend",
    "get_backend",
    "torch_available",
    "describe_backends",
    "MIN_B",
]

MIN_B = 1e-3
"""Lower bound applied to every node's ``b`` after each update."""


class CSR:
    """Compressed-sparse-row view of the graph's edges (node-id order).

    ``indptr[p] .. indptr[p + 1]`` is the CSR position range of ``p``'s
    out-edges; ``indices[j]`` is the child id, ``edge_ids[j]`` the graph edge
    id and ``weights[j]`` the weight at position ``j``.  ``edge_pos`` maps an
    edge id back to its CSR position (used to turn ``(parent, edge)``
    transitions into ``(parent, position)`` pairs for :meth:`Backend.step`).
    """

    __slots__ = ("indptr", "indices", "edge_ids", "weights", "edge_pos")

    def __init__(
        self,
        indptr: list[int],
        indices: list[int],
        edge_ids: list[int],
        weights: list[float],
        edge_pos: dict[int, int],
    ) -> None:
        if len(indices) != len(edge_ids) or len(indices) != len(weights):
            raise ValueError("indices, edge_ids and weights must have the same length")
        if not indptr or indptr[0] != 0 or indptr[-1] != len(indices):
            raise ValueError("indptr must start at 0 and end at len(indices)")
        self.indptr = indptr
        self.indices = indices
        self.edge_ids = edge_ids
        self.weights = weights
        self.edge_pos = edge_pos

    @property
    def n_nodes(self) -> int:
        """Number of rows (all node ids, dead ones included)."""
        return len(self.indptr) - 1

    @property
    def n_edges(self) -> int:
        """Number of CSR entries (alive edges)."""
        return len(self.indices)

    def row(self, p: int) -> range:
        """CSR position range of node ``p``'s out-edges."""
        return range(self.indptr[p], self.indptr[p + 1])

    def __repr__(self) -> str:
        return f"CSR(n_nodes={self.n_nodes}, n_edges={self.n_edges})"


class NodeParams:
    """Per-node learnable values in node-id order: state ``z`` and ``a, b, h, k``."""

    __slots__ = ("z", "a", "b", "h", "k")

    def __init__(
        self,
        z: list[float],
        a: list[float],
        b: list[float],
        h: list[float],
        k: list[float],
    ) -> None:
        n = len(z)
        if not (len(a) == len(b) == len(h) == len(k) == n):
            raise ValueError("z, a, b, h, k must have the same length")
        self.z = z
        self.a = a
        self.b = b
        self.h = h
        self.k = k

    def __len__(self) -> int:
        return len(self.z)

    def copy(self) -> "NodeParams":
        """Deep-enough copy (fresh lists)."""
        return NodeParams(list(self.z), list(self.a), list(self.b), list(self.h), list(self.k))

    def __repr__(self) -> str:
        return f"NodeParams(n={len(self.z)})"


@runtime_checkable
class Backend(Protocol):
    """Interface implemented by :class:`PythonBackend` and ``TorchBackend``."""

    name: str
    device: str

    def prepare(self, csr: CSR, params: NodeParams) -> Any:
        """Wrap / upload the mutable training state for one epoch."""

    def step(
        self,
        state: Any,
        parents: list[int],
        positions: list[int],
        lr: float,
        act_lr: float,
        clip: float = 5.0,
    ) -> float:
        """One mini-batch gradient step; returns the mean loss *before* the update."""

    def finalize(self, state: Any) -> tuple[list[float], NodeParams]:
        """Download weights (CSR order) and node parameters."""

    def node_activations(self, state: Any) -> list[float]:
        """``f_i(z_i)`` for every node id."""


class _PyState:
    """Mutable state of :class:`PythonBackend` for one epoch.

    ``gw`` (per CSR position) and ``gz/ga/gb/gh/gk`` (per node) are persistent
    gradient accumulators: they stay zero between steps and only the entries
    touched by a batch are ever written or reset, so a step costs
    ``O(batch * fan-out)`` regardless of graph size.
    """

    __slots__ = (
        "indptr", "indices", "weights", "z", "a", "b", "h", "k",
        "gw", "gz", "ga", "gb", "gh", "gk",
    )

    def __init__(self, csr: CSR, params: NodeParams) -> None:
        n = len(params)
        if csr.n_nodes != n:
            raise ValueError(f"CSR has {csr.n_nodes} rows but params has {n} nodes")
        self.indptr = list(csr.indptr)
        self.indices = list(csr.indices)
        self.weights = list(csr.weights)
        self.z = list(params.z)
        self.a = list(params.a)
        self.b = list(params.b)
        self.h = list(params.h)
        self.k = list(params.k)
        self.gw = [0.0] * len(self.indices)
        self.gz = [0.0] * n
        self.ga = [0.0] * n
        self.gb = [0.0] * n
        self.gh = [0.0] * n
        self.gk = [0.0] * n


class PythonBackend:
    """Pure-Python reference backend (always available).

    Besides the :class:`Backend` interface it exposes :meth:`loss` and
    :meth:`gradients`, which evaluate the batch without updating anything;
    the tests use them for finite-difference gradient checks.
    """

    __slots__ = ()

    name = "python"
    device = "cpu"

    def __init__(self, device: str | None = None) -> None:
        if device not in (None, "cpu"):
            raise ValueError(f"PythonBackend only supports device 'cpu', got {device!r}")

    # -- Backend interface ---------------------------------------------------

    def prepare(self, csr: CSR, params: NodeParams) -> _PyState:
        """Copy the CSR weights and node parameters into a mutable state."""
        return _PyState(csr, params)

    def step(
        self,
        state: _PyState,
        parents: list[int],
        positions: list[int],
        lr: float,
        act_lr: float,
        clip: float = 5.0,
    ) -> float:
        """One mini-batch step (see the module docstring for the rule).

        Gradients are summed over the batch, divided by the batch size, clipped
        element-wise to ``[-clip, clip]`` and applied once.  ``b`` is kept
        ``>= MIN_B``.  An empty batch is a no-op returning ``0.0``.
        """
        n = len(parents)
        if n == 0:
            return 0.0
        total, touched, rows = self._accumulate(state, parents, positions)
        scale = 1.0 / n
        indptr = state.indptr
        w = state.weights
        gw = state.gw
        for p in rows:
            for j in range(indptr[p], indptr[p + 1]):
                g = gw[j] * scale
                if g > clip:
                    g = clip
                elif g < -clip:
                    g = -clip
                w[j] -= lr * g
                gw[j] = 0.0
        z = state.z
        a = state.a
        b = state.b
        h = state.h
        k = state.k
        gz = state.gz
        ga = state.ga
        gb = state.gb
        gh = state.gh
        gk = state.gk
        for c in touched:
            g = gz[c] * scale
            if g > clip:
                g = clip
            elif g < -clip:
                g = -clip
            z[c] -= lr * g
            gz[c] = 0.0
            g = ga[c] * scale
            if g > clip:
                g = clip
            elif g < -clip:
                g = -clip
            a[c] -= act_lr * g
            ga[c] = 0.0
            g = gb[c] * scale
            if g > clip:
                g = clip
            elif g < -clip:
                g = -clip
            nb = b[c] - act_lr * g
            b[c] = nb if nb > MIN_B else MIN_B
            gb[c] = 0.0
            g = gh[c] * scale
            if g > clip:
                g = clip
            elif g < -clip:
                g = -clip
            h[c] -= act_lr * g
            gh[c] = 0.0
            g = gk[c] * scale
            if g > clip:
                g = clip
            elif g < -clip:
                g = -clip
            k[c] -= act_lr * g
            gk[c] = 0.0
        return total * scale

    def finalize(self, state: _PyState) -> tuple[list[float], NodeParams]:
        """Return copies of the weights (CSR order) and the node parameters."""
        params = NodeParams(
            list(state.z), list(state.a), list(state.b), list(state.h), list(state.k)
        )
        return list(state.weights), params

    def node_activations(self, state: _PyState) -> list[float]:
        """``f_i(z_i)`` for every node in the state."""
        sin = math.sin
        return [
            ai * sin(bi * (zi - hi)) + ki
            for zi, ai, bi, hi, ki in zip(state.z, state.a, state.b, state.h, state.k)
        ]

    # -- debugging helpers ---------------------------------------------------

    def loss(self, state: _PyState, parents: list[int], positions: list[int]) -> float:
        """Mean batch loss at the current parameters (no update)."""
        n = len(parents)
        if n == 0:
            return 0.0
        total, touched, rows = self._accumulate(state, parents, positions)
        self._reset(state, touched, rows)
        return total / n

    def gradients(self, state: _PyState, parents: list[int], positions: list[int]) -> dict:
        """Mean (unclipped) gradients of the batch loss without applying them.

        Returns ``{"loss": float, "w": {csr_pos: g}, "z": {node: g}, "a": {...},
        "b": {...}, "h": {...}, "k": {...}}``; entries not listed are zero.
        """
        n = len(parents)
        out: dict[str, Any] = {"loss": 0.0, "w": {}, "z": {}, "a": {}, "b": {}, "h": {}, "k": {}}
        if n == 0:
            return out
        total, touched, rows = self._accumulate(state, parents, positions)
        scale = 1.0 / n
        out["loss"] = total * scale
        indptr = state.indptr
        gw = state.gw
        for p in rows:
            for j in range(indptr[p], indptr[p + 1]):
                out["w"][j] = gw[j] * scale
        for c in touched:
            out["z"][c] = state.gz[c] * scale
            out["a"][c] = state.ga[c] * scale
            out["b"][c] = state.gb[c] * scale
            out["h"][c] = state.gh[c] * scale
            out["k"][c] = state.gk[c] * scale
        self._reset(state, touched, rows)
        return out

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _reset(state: _PyState, touched: list[int], rows: list[int]) -> None:
        """Zero the accumulators written by :meth:`_accumulate`."""
        indptr = state.indptr
        gw = state.gw
        for p in rows:
            for j in range(indptr[p], indptr[p + 1]):
                gw[j] = 0.0
        gz, ga, gb, gh, gk = state.gz, state.ga, state.gb, state.gh, state.gk
        for c in touched:
            gz[c] = 0.0
            ga[c] = 0.0
            gb[c] = 0.0
            gh[c] = 0.0
            gk[c] = 0.0

    @staticmethod
    def _accumulate(
        state: _PyState, parents: list[int], positions: list[int]
    ) -> tuple[float, list[int], list[int]]:
        """Sum the batch loss and its gradients into the state's accumulators.

        Returns ``(loss_sum, touched_nodes, touched_parents)``; the caller
        scales, applies and resets.  Activation partials are computed once per
        distinct node and cached for the duration of the batch.
        """
        if len(positions) != len(parents):
            raise ValueError("parents and positions must have the same length")
        sin = math.sin
        cos = math.cos
        exp = math.exp
        log = math.log
        indptr = state.indptr
        indices = state.indices
        w = state.weights
        z, a, b, h, k = state.z, state.a, state.b, state.h, state.k
        gw = state.gw
        gz, ga, gb, gh, gk = state.gz, state.ga, state.gb, state.gh, state.gk
        cache: dict[int, tuple[float, float, float, float, float]] = {}
        cache_get = cache.get
        touched: list[int] = []
        rows: dict[int, None] = {}
        total = 0.0
        for i in range(len(parents)):
            p = parents[i]
            t = positions[i]
            lo = indptr[p]
            hi = indptr[p + 1]
            if t < lo or t >= hi:
                raise ValueError(f"position {t} is not an out-edge of node {p}")
            if hi - lo == 1:
                # single child: softmax is 1, loss and every gradient are exactly 0
                continue
            pp = cache_get(p)
            if pp is None:
                ap = a[p]
                bp = b[p]
                d = z[p] - h[p]
                u = bp * d
                su = sin(u)
                cu = cos(u)
                abc = ap * bp * cu
                pp = (ap * su + k[p], abc, su, ap * d * cu, -abc)
                cache[p] = pp
                touched.append(p)
            fp = pp[0]
            rows[p] = None
            scores: list[float] = []
            facts: list[tuple[float, float, float, float, float]] = []
            m = -math.inf
            for j in range(lo, hi):
                c = indices[j]
                cc = cache_get(c)
                if cc is None:
                    ac = a[c]
                    bc = b[c]
                    d = z[c] - h[c]
                    u = bc * d
                    su = sin(u)
                    cu = cos(u)
                    abc = ac * bc * cu
                    cc = (ac * su + k[c], abc, su, ac * d * cu, -abc)
                    cache[c] = cc
                    touched.append(c)
                s = w[j] * fp * cc[0]
                scores.append(s)
                facts.append(cc)
                if s > m:
                    m = s
            ex: list[float] = []
            ssum = 0.0
            for s in scores:
                v = exp(s - m)
                ex.append(v)
                ssum += v
            total += m + log(ssum) - scores[t - lo]
            inv = 1.0 / ssum
            gfp = 0.0
            for idx in range(hi - lo):
                j = lo + idx
                g = ex[idx] * inv
                if j == t:
                    g -= 1.0
                cc = facts[idx]
                wj = w[j]
                fc = cc[0]
                gw[j] += g * fp * fc
                gfp += g * wj * fc
                gfc = g * wj * fp
                c = indices[j]
                gz[c] += gfc * cc[1]
                ga[c] += gfc * cc[2]
                gb[c] += gfc * cc[3]
                gh[c] += gfc * cc[4]
                gk[c] += gfc
            gz[p] += gfp * pp[1]
            ga[p] += gfp * pp[2]
            gb[p] += gfp * pp[3]
            gh[p] += gfp * pp[4]
            gk[p] += gfp
        return total, touched, list(rows)

    def __repr__(self) -> str:
        return "PythonBackend(device='cpu')"


# -- factory ------------------------------------------------------------------

_ACCELERATORS: tuple[bool, bool] | None = None


def torch_available() -> bool:
    """True if ``torch`` can be imported (checked without importing it)."""
    try:
        return importlib.util.find_spec("torch") is not None
    except (ImportError, ValueError):
        return False


def _accelerators() -> tuple[bool, bool]:
    """``(cuda, mps)`` availability; imports torch on first call only."""
    global _ACCELERATORS
    if _ACCELERATORS is None:
        cuda = mps = False
        if torch_available():
            try:
                torch = importlib.import_module("torch")
                cuda = bool(torch.cuda.is_available())
                mps_backend = getattr(torch.backends, "mps", None)
                mps = bool(mps_backend is not None and mps_backend.is_available())
            except Exception:  # noqa: BLE001 - a broken torch install must not break us
                cuda = mps = False
        _ACCELERATORS = (cuda, mps)
    return _ACCELERATORS


def _torch_backend(device: str | None) -> Backend:
    try:
        module = importlib.import_module(f"{__package__}.backend_torch")
    except ImportError as exc:
        raise ImportError(f"torch backend unavailable: {exc}") from exc
    return module.TorchBackend(device=device)


def get_backend(name: str = "auto", device: str | None = None) -> Backend:
    """Return a backend by name.

    * ``"python"`` - :class:`PythonBackend`.
    * ``"torch"`` - ``TorchBackend`` on ``device`` (auto: cuda > mps > cpu);
      raises ``ImportError`` with a clear message if torch is missing.
    * ``"auto"`` - torch if it is importable *and* a GPU (cuda or mps) is
      available, else python.
    """
    key = (name or "auto").lower()
    if key == "python":
        return PythonBackend(device=device)
    if key == "torch":
        return _torch_backend(device)
    if key == "auto":
        if torch_available() and any(_accelerators()):
            try:
                return _torch_backend(device)
            except ImportError:
                return PythonBackend()
        return PythonBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'auto', 'python' or 'torch'")


def describe_backends() -> dict:
    """Availability summary: ``{"python", "torch", "cuda", "mps", "default"}``."""
    torch_ok = torch_available()
    cuda, mps = _accelerators() if torch_ok else (False, False)
    return {
        "python": True,
        "torch": torch_ok,
        "cuda": cuda,
        "mps": mps,
        "default": "torch" if (torch_ok and (cuda or mps)) else "python",
    }
