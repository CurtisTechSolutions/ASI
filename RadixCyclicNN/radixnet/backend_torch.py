"""Optional ``torch`` backend: the one-hop learning rule of
:class:`radixnet.backend.PythonBackend`, fully vectorised over a mini-batch
and runnable on ``cuda`` / ``mps`` / ``cpu``.

``torch`` is imported lazily by :class:`TorchBackend`, so ``import radixnet``
never depends on it.  The numerics follow the pure-Python backend operation by
operation (same multiplication order, the same ``1 / batch`` reciprocal
scaling, the same clip and ``b >= MIN_B`` guard), so in ``float64`` the two
backends agree to roughly ``1e-12`` after a step; ``DESIGN.md`` section 6
asks for ``1e-6``.

Vectorisation strategy of :meth:`TorchBackend.step`:

1. gather the CSR row range ``[lo, hi)`` of every batch parent; rows with a
   single child are dropped (their loss and every gradient are exactly zero);
2. expand the rows into one flat *segment* tensor of ``(transition, CSR
   position)`` pairs with ``repeat_interleave`` and ``arange``;
3. compute the edge scores, a segment-wise numerically stable softmax
   (``scatter_reduce("amax")`` for the row maxima, ``index_add_`` for the row
   sums) and the loss;
4. evaluate the per-segment gradients of the design's formulas and reduce
   them with ``index_add_`` onto the *unique* touched CSR positions and node
   ids;
5. scale by ``1 / batch``, clip, apply (``lr`` for ``w`` and ``z``, ``act_lr``
   for ``a, b, h, k``) and re-apply the ``b >= MIN_B`` guard to the touched
   nodes only.

No Python loop runs over transitions or edges.  The host round trips per step
are the input validation (which also yields the segment size), the two
``unique`` calls and the returned loss.
"""

from __future__ import annotations

import threading
import warnings
from typing import Any, NamedTuple

from .backend import CSR, MIN_B, NodeParams

__all__ = ["TorchBackend", "TorchState", "select_device", "resolve_dtype"]

# Row indices of ``TorchState.node``.
_Z, _A, _B, _H, _K = 0, 1, 2, 3, 4
_PARAM_NAMES = ("z", "a", "b", "h", "k")

_DTYPE_ALIASES = {
    "float64": "float64", "double": "float64", "f64": "float64", "fp64": "float64",
    "float32": "float32", "float": "float32", "single": "float32", "f32": "float32", "fp32": "float32",
}

_torch_module: Any = None
_import_lock = threading.Lock()


def _import_torch() -> Any:
    """Import ``torch`` once (thread-safe) and cache the module.

    Raises ``ImportError`` with a clear message when torch is missing or
    broken.  The ``"Failed to initialize NumPy"`` warning torch emits on
    import is silenced: this project deliberately has no numpy.
    """
    global _torch_module
    if _torch_module is None:
        with _import_lock:
            if _torch_module is None:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore", message="Failed to initialize NumPy", category=UserWarning
                        )
                        import torch
                except Exception as exc:  # noqa: BLE001 - any import failure means "no torch"
                    raise ImportError(
                        "torch backend unavailable: torch is not installed or failed to import "
                        f"({exc}); install it with `pip install torch` or use backend='python'"
                    ) from exc
                _torch_module = torch
    return _torch_module


def _mps_available(torch: Any) -> bool:
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def select_device(torch: Any, device: str | None = None) -> str:
    """Resolve a device request to a normalised device string.

    ``None`` / ``"auto"`` pick ``cuda`` > ``mps`` > ``cpu``.  Explicit requests
    (``"cpu"``, ``"cuda"``, ``"cuda:<index>"``, ``"mps"``) are honoured as
    given.  Raises ``ValueError`` for an unknown device name and
    ``RuntimeError`` when the requested device is not available in this torch
    build or on this machine.
    """
    want = "auto" if device is None else str(device).strip().lower()
    if want in ("", "auto"):
        if torch.cuda.is_available():
            return "cuda"
        if _mps_available(torch):
            return "mps"
        return "cpu"
    kind, _, index = want.partition(":")
    if kind == "cpu":
        if index:
            raise ValueError(f"unsupported device {device!r}: use plain 'cpu'")
        return "cpu"
    if kind == "cuda":
        if index and not index.isdigit():
            raise ValueError(f"unsupported device {device!r}: expected 'cuda' or 'cuda:<index>'")
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"device {device!r} requested but CUDA is not available "
                f"(torch {torch.__version__}, CUDA build: {torch.version.cuda or 'none'})"
            )
        if not index:
            return "cuda"
        count = torch.cuda.device_count()
        if int(index) >= count:
            raise RuntimeError(
                f"device {device!r} requested but only {count} CUDA device(s) are available"
            )
        return f"cuda:{int(index)}"
    if kind == "mps":
        if index:
            raise ValueError(f"unsupported device {device!r}: use plain 'mps'")
        if not _mps_available(torch):
            raise RuntimeError(
                f"device {device!r} requested but the MPS backend is not available "
                f"(torch {torch.__version__})"
            )
        return "mps"
    raise ValueError(f"unknown device {device!r}; expected 'auto', 'cpu', 'cuda[:N]' or 'mps'")


def resolve_dtype(torch: Any, dtype: Any, device: str) -> Any:
    """Turn ``dtype`` (``None``, a name or a ``torch.dtype``) into a torch dtype.

    ``None`` selects ``float64`` (numerical parity with the Python backend)
    except on ``mps``, which has no float64 support and therefore gets
    ``float32``.  Only ``float32`` and ``float64`` are accepted.
    """
    if dtype is None:
        return torch.float32 if device == "mps" else torch.float64
    if isinstance(dtype, torch.dtype):
        resolved = dtype
    else:
        key = _DTYPE_ALIASES.get(str(dtype).strip().lower().removeprefix("torch."))
        if key is None:
            raise ValueError(f"unsupported dtype {dtype!r}; expected 'float64' or 'float32'")
        resolved = torch.float64 if key == "float64" else torch.float32
    if resolved not in (torch.float32, torch.float64):
        raise ValueError(f"unsupported dtype {dtype!r}; expected torch.float64 or torch.float32")
    if resolved is torch.float64 and device == "mps":
        raise ValueError("the MPS backend does not support float64; use dtype='float32'")
    return resolved


class TorchState:
    """Mutable per-epoch state of :class:`TorchBackend`.

    All tensors live on the backend device.  ``indptr`` / ``indices`` are the
    CSR structure (``int64``), ``weights`` the edge weights in CSR order and
    ``node`` a ``(5, n_nodes)`` tensor whose rows are ``z, a, b, h, k``.
    """

    __slots__ = ("indptr", "indices", "weights", "node")

    def __init__(self, indptr: Any, indices: Any, weights: Any, node: Any) -> None:
        self.indptr = indptr
        self.indices = indices
        self.weights = weights
        self.node = node

    @property
    def n_nodes(self) -> int:
        """Number of CSR rows (all node ids, dead ones included)."""
        return int(self.node.shape[1])

    @property
    def n_edges(self) -> int:
        """Number of CSR entries."""
        return int(self.weights.shape[0])

    def __repr__(self) -> str:
        return f"TorchState(n_nodes={self.n_nodes}, n_edges={self.n_edges}, device={self.node.device})"


class _Forward(NamedTuple):
    """Intermediates of the vectorised forward pass (shared by loss / gradients / step).

    ``m`` transitions survive the single-child filter and expand into ``S``
    segment entries (one per candidate edge of every kept transition).
    """

    parents: Any    # (m,)  kept parent ids
    seg_row: Any    # (S,)  kept-transition index of every segment entry
    seg_pos: Any    # (S,)  CSR position of every segment entry
    seg_child: Any  # (S,)  child id at that position
    seg_w: Any      # (S,)  weight at that position
    tgt_seg: Any    # (m,)  segment index of the observed child
    fp: tuple       # parent partials (f, df/dz, df/da, df/db, df/dh), each (m,)
    fc: tuple       # child partials, each (S,)
    prob: Any       # (S,)  softmax of the row scores
    total: Any      # 0-d   summed loss of the kept transitions


class _Gradients(NamedTuple):
    """Batch-summed (unscaled) gradients on the unique touched positions / nodes."""

    positions: Any  # (Ue,) unique CSR positions
    gw: Any         # (Ue,) dL/dw summed over the batch
    nodes: Any      # (Un,) unique node ids
    gn: Any         # (5, Un) rows dL/dz, dL/da, dL/db, dL/dh, dL/dk


class TorchBackend:
    """Vectorised torch implementation of the :class:`~radixnet.backend.Backend` protocol.

    ``device`` is ``None``/``"auto"`` (``cuda`` > ``mps`` > ``cpu``) or an
    explicit ``"cpu"`` / ``"cuda[:N]"`` / ``"mps"``; ``dtype`` defaults to
    ``float64`` (``float32`` on ``mps``).  Like the Python backend it also
    offers the debugging helpers :meth:`loss` and :meth:`gradients`.
    """

    __slots__ = ("device", "dtype", "_torch", "_device")

    name = "torch"

    def __init__(self, device: str | None = None, dtype: Any = None) -> None:
        torch = _import_torch()
        self._torch = torch
        self.device: str = select_device(torch, device)
        self.dtype = resolve_dtype(torch, dtype, self.device)
        self._device = torch.device(self.device)

    # -- Backend interface ---------------------------------------------------

    def prepare(self, csr: CSR, params: NodeParams) -> TorchState:
        """Upload the CSR structure, the weights and the node parameters."""
        n = len(params)
        if csr.n_nodes != n:
            raise ValueError(f"CSR has {csr.n_nodes} rows but params has {n} nodes")
        torch = self._torch
        dev = self._device
        indptr = torch.tensor(csr.indptr, dtype=torch.int64, device=dev)
        indices = torch.tensor(csr.indices, dtype=torch.int64, device=dev)
        weights = torch.tensor(csr.weights, dtype=self.dtype, device=dev)
        node = torch.tensor(
            [params.z, params.a, params.b, params.h, params.k], dtype=self.dtype, device=dev
        )
        return TorchState(indptr, indices, weights, node)

    def step(
        self,
        state: TorchState,
        parents: list[int],
        positions: list[int],
        lr: float,
        act_lr: float,
        clip: float = 5.0,
    ) -> float:
        """One mini-batch step (see the module docstring for the rule).

        Gradients are summed over the batch, divided by the batch size, clipped
        element-wise to ``[-clip, clip]`` and applied once; ``b`` is kept
        ``>= MIN_B``.  Returns the mean loss *before* the update.  An empty
        batch is a no-op returning ``0.0``.
        """
        n = len(parents)
        if len(positions) != n:
            raise ValueError("parents and positions must have the same length")
        if clip < 0.0:
            raise ValueError(f"clip must be non-negative, got {clip!r}")
        if n == 0:
            return 0.0
        fwd = self._forward(state, parents, positions)
        if fwd is None:
            return 0.0
        grads = self._backward(fwd)
        scale = 1.0 / n
        gw = grads.gw.mul_(scale).clamp_(-clip, clip)
        state.weights.index_add_(0, grads.positions, gw, alpha=-lr)
        gn = grads.gn.mul_(scale).clamp_(-clip, clip)
        gn[_Z].mul_(-lr)
        gn[_A:].mul_(-act_lr)
        state.node.index_add_(1, grads.nodes, gn)
        b = state.node[_B]
        b[grads.nodes] = b[grads.nodes].clamp_min(MIN_B)
        return fwd.total.item() * scale

    def finalize(self, state: TorchState) -> tuple[list[float], NodeParams]:
        """Download the weights (CSR order) and the node parameters as lists."""
        z, a, b, h, k = state.node.tolist()
        return state.weights.tolist(), NodeParams(z, a, b, h, k)

    def node_activations(self, state: TorchState) -> list[float]:
        """``f_i(z_i) = a * sin(b * (z - h)) + k`` for every node id."""
        z, a, b, h, k = state.node.unbind(0)
        return (a * self._torch.sin(b * (z - h)) + k).tolist()

    # -- debugging helpers ---------------------------------------------------

    def loss(self, state: TorchState, parents: list[int], positions: list[int]) -> float:
        """Mean batch loss at the current parameters (no update)."""
        n = len(parents)
        if len(positions) != n:
            raise ValueError("parents and positions must have the same length")
        if n == 0:
            return 0.0
        fwd = self._forward(state, parents, positions)
        return 0.0 if fwd is None else fwd.total.item() / n

    def gradients(self, state: TorchState, parents: list[int], positions: list[int]) -> dict:
        """Mean (unclipped) gradients of the batch loss without applying them.

        Same shape as ``PythonBackend.gradients``: ``{"loss": float, "w":
        {csr_pos: g}, "z": {node: g}, "a": {...}, "b": {...}, "h": {...},
        "k": {...}}``; entries not listed are zero.
        """
        n = len(parents)
        if len(positions) != n:
            raise ValueError("parents and positions must have the same length")
        out: dict[str, Any] = {"loss": 0.0, "w": {}, "z": {}, "a": {}, "b": {}, "h": {}, "k": {}}
        if n == 0:
            return out
        fwd = self._forward(state, parents, positions)
        if fwd is None:
            return out
        grads = self._backward(fwd)
        scale = 1.0 / n
        out["loss"] = fwd.total.item() * scale
        out["w"] = dict(zip(grads.positions.tolist(), grads.gw.mul_(scale).tolist()))
        nodes = grads.nodes.tolist()
        for name, row in zip(_PARAM_NAMES, grads.gn.mul_(scale).tolist()):
            out[name] = dict(zip(nodes, row))
        return out

    # -- internals -----------------------------------------------------------

    def _partials(self, cols: Any) -> tuple:
        """Activation and its partials for the ``(5, K)`` parameter columns ``cols``.

        Returns ``(f, df/dz, df/da, df/db, df/dh)``; ``df/dk`` is 1.  The
        expressions mirror ``PythonBackend`` term by term.
        """
        torch = self._torch
        z, a, b, h, k = cols.unbind(0)
        d = z - h
        u = b * d
        su = torch.sin(u)
        cu = torch.cos(u)
        abc = a * b * cu
        return a * su + k, abc, su, a * d * cu, -abc

    def _forward(self, state: TorchState, parents: list[int], positions: list[int]) -> _Forward | None:
        """Scores, softmax and loss of the batch; ``None`` if no transition has a choice.

        Validates every ``(parent, position)`` pair *before* any gather (an
        out-of-range index would be a fatal device-side assert on CUDA).
        """
        torch = self._torch
        dev = self._device
        dtype = self.dtype
        n_nodes = state.n_nodes
        indptr = state.indptr
        p_all = torch.tensor(parents, dtype=torch.int64, device=dev)
        t_all = torch.tensor(positions, dtype=torch.int64, device=dev)
        p_safe = p_all.clamp(0, n_nodes - 1)
        lo = indptr[p_safe]
        hi = indptr[p_safe + 1]
        valid = (p_all == p_safe) & (t_all >= lo) & (t_all < hi)
        lens = hi - lo
        multi = lens > 1
        # one host round trip: validity, number of kept rows and the segment size
        ok, m, seg_size = torch.stack((valid.all().long(), multi.sum(), (lens * multi).sum())).tolist()
        if not ok:
            bad = valid.tolist().index(False)
            p, t = parents[bad], positions[bad]
            if not 0 <= p < n_nodes:
                raise ValueError(f"parent {p} is not a node id (0..{n_nodes - 1})")
            raise ValueError(f"position {t} is not an out-edge of node {p}")
        if m == 0:
            return None
        if m < len(parents):
            # kept rows first, original order preserved; no host sync (m is known)
            keep = torch.argsort(multi.to(torch.int8), descending=True, stable=True)[:m]
            p_all, t_all, lo, lens = p_all[keep], t_all[keep], lo[keep], lens[keep]
        start = torch.cumsum(lens, 0) - lens
        seg_row = torch.repeat_interleave(torch.arange(m, device=dev), lens, output_size=seg_size)
        seg_pos = torch.arange(seg_size, device=dev) - start[seg_row] + lo[seg_row]
        seg_child = state.indices[seg_pos]
        seg_w = state.weights[seg_pos]
        tgt_seg = start + (t_all - lo)
        fp = self._partials(state.node[:, p_all])
        fc = self._partials(state.node[:, seg_child])
        scores = seg_w * fp[0][seg_row] * fc[0]
        row_max = torch.full((m,), float("-inf"), dtype=dtype, device=dev)
        row_max.scatter_reduce_(0, seg_row, scores, reduce="amax", include_self=False)
        ex = torch.exp(scores - row_max[seg_row])
        row_sum = torch.zeros(m, dtype=dtype, device=dev).index_add_(0, seg_row, ex)
        prob = ex * (1.0 / row_sum)[seg_row]
        total = (row_max + torch.log(row_sum) - scores[tgt_seg]).sum()
        return _Forward(p_all, seg_row, seg_pos, seg_child, seg_w, tgt_seg, fp, fc, prob, total)

    def _backward(self, fwd: _Forward) -> _Gradients:
        """Batch-summed gradients of the design's formulas, reduced onto unique positions / nodes."""
        torch = self._torch
        m = fwd.parents.shape[0]
        fp_seg = fwd.fp[0][fwd.seg_row]
        fc = fwd.fc[0]
        g = fwd.prob.clone()
        g[fwd.tgt_seg] -= 1.0
        dw = g * fp_seg * fc
        gfp = torch.zeros(m, dtype=self.dtype, device=self._device).index_add_(
            0, fwd.seg_row, g * fwd.seg_w * fc
        )
        gfc = g * fwd.seg_w * fp_seg
        child = torch.stack((gfc * fwd.fc[1], gfc * fwd.fc[2], gfc * fwd.fc[3], gfc * fwd.fc[4], gfc))
        parent = torch.stack((gfp * fwd.fp[1], gfp * fwd.fp[2], gfp * fwd.fp[3], gfp * fwd.fp[4], gfp))
        uniq_pos, inv_pos = torch.unique(fwd.seg_pos, return_inverse=True)
        gw = torch.zeros(uniq_pos.shape[0], dtype=self.dtype, device=self._device).index_add_(0, inv_pos, dw)
        uniq_node, inv_node = torch.unique(torch.cat((fwd.seg_child, fwd.parents)), return_inverse=True)
        gn = torch.zeros((5, uniq_node.shape[0]), dtype=self.dtype, device=self._device).index_add_(
            1, inv_node, torch.cat((child, parent), 1)
        )
        return _Gradients(uniq_pos, gw, uniq_node, gn)

    def __repr__(self) -> str:
        return f"TorchBackend(device={self.device!r}, dtype={str(self.dtype).removeprefix('torch.')})"
