"""Learning-rate schedules: a rate as a *graph function* of the epoch.

A schedule is a small arithmetic expression evaluated once per epoch::

    linear(lr0, 4 * lr0)          # ramp from the base rate to four times it
    lr0 * 1.25 ** i               # grow 25 % every epoch
    0.05 + 0.45 * t               # from 0.05 at the first epoch to 0.5 at the last
    warmup(lr0 / 10, lr0, 3)      # warm up over three epochs, then hold
    lr / 10                       # (activation schedule) a tenth of this epoch's learning rate

Variables: ``epoch`` (1-based within the training call), ``i`` (0-based),
``epochs`` (total), ``t`` (0 at the first epoch, 1 at the last), ``lr0`` (the
schedule's base rate: ``lr`` or ``act_lr`` of the config), ``act_lr0``, and -
for the activation schedule - ``lr`` (the epoch's learning rate).
Helpers: ``linear(a, b)``, ``geometric(a, b)``, ``cosine(a, b)``,
``step(a, factor, every)``, ``warmup(a, b, n)``; plus ``sin cos tan exp log
log2 log10 sqrt pow abs floor ceil round min max tanh clamp`` and ``pi``, ``e``.
Conditionals are allowed (``0.1 if epoch < 3 else 0.5``).

Expressions are validated against a whitelist of AST nodes before they are
compiled, so no attribute access, names, strings or calls outside the list
can run; every value must be finite and non-negative.
"""

from __future__ import annotations

import ast
import math
from typing import Any

__all__ = [
    "PRESETS",
    "Schedule",
    "ScheduleError",
    "describe",
    "parse_schedule",
    "preview_points",
]


class ScheduleError(ValueError):
    """An expression that cannot be parsed, uses something not allowed, or yields a bad rate."""


_BINOPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv)
_UNARY = (ast.UAdd, ast.USub, ast.Not)
_COMPARE = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)
_BOOL = (ast.And, ast.Or)

VARIABLES = ("epoch", "i", "epochs", "t", "lr0", "act_lr0", "lr")
CONSTANTS = {"pi": math.pi, "e": math.e}
FUNCTIONS: dict[str, Any] = {
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "exp": math.exp, "log": math.log, "log2": math.log2,
    "log10": math.log10, "sqrt": math.sqrt, "pow": math.pow, "abs": abs, "floor": math.floor, "ceil": math.ceil,
    "round": round, "min": min, "max": max, "tanh": math.tanh,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
}
HELPERS = ("linear", "geometric", "cosine", "step", "warmup")

PRESETS: tuple[dict, ...] = (
    {"name": "constant", "lr": "lr0", "act_lr": "act_lr0", "description": "the base rates every epoch"},
    {"name": "linear ramp x4", "lr": "linear(lr0, 4 * lr0)", "act_lr": "linear(act_lr0, 4 * act_lr0)",
     "description": "grow linearly from the base rate to four times it"},
    {"name": "exponential ramp x4", "lr": "geometric(lr0, 4 * lr0)", "act_lr": "geometric(act_lr0, 4 * act_lr0)",
     "description": "grow by the same factor every epoch, reaching four times the base rate"},
    {"name": "cosine ramp x4", "lr": "cosine(lr0, 4 * lr0)", "act_lr": "cosine(act_lr0, 4 * act_lr0)",
     "description": "a smooth S-shaped rise from the base rate to four times it"},
    {"name": "step x1.5 every 2 epochs", "lr": "step(lr0, 1.5, 2)", "act_lr": "step(act_lr0, 1.5, 2)",
     "description": "multiply by 1.5 every two epochs"},
    {"name": "grow 25% per epoch", "lr": "lr0 * 1.25 ** i", "act_lr": "act_lr0 * 1.25 ** i",
     "description": "compound growth of a quarter per epoch"},
    {"name": "warm-up over 3 epochs", "lr": "warmup(lr0 / 10, lr0, 3)", "act_lr": "warmup(act_lr0 / 10, act_lr0, 3)",
     "description": "start at a tenth of the rate, reach it after three epochs, then hold"},
    {"name": "activation follows lr / 10", "lr": "lr0", "act_lr": "lr / 10",
     "description": "keep the activation rate at a tenth of whatever the learning rate is"},
)


def _allowed_names() -> set[str]:
    return set(VARIABLES) | set(CONSTANTS) | set(FUNCTIONS) | set(HELPERS)


def _validate(tree: ast.AST) -> None:
    callable_names = set(FUNCTIONS) | set(HELPERS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Expression, ast.Load, ast.IfExp)):
            continue
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ScheduleError(f"only numbers are allowed as constants, not {node.value!r}")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINOPS):
                raise ScheduleError(f"operator {type(node.op).__name__} is not allowed")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY):
                raise ScheduleError(f"operator {type(node.op).__name__} is not allowed")
        elif isinstance(node, ast.Compare):
            if not all(isinstance(op, _COMPARE) for op in node.ops):
                raise ScheduleError("only <, <=, >, >=, == and != comparisons are allowed")
        elif isinstance(node, ast.BoolOp):
            if not isinstance(node.op, _BOOL):
                raise ScheduleError("only 'and' / 'or' are allowed")
        elif isinstance(node, ast.Name):
            if node.id not in _allowed_names():
                raise ScheduleError(f"unknown name {node.id!r}; allowed: {', '.join(sorted(_allowed_names()))}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in callable_names:
                raise ScheduleError(f"only these functions may be called: {', '.join(sorted(callable_names))}")
            if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
                raise ScheduleError(f"{node.func.id}() takes positional arguments only")
        elif isinstance(node, (ast.And, ast.Or, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
                               ast.UAdd, ast.USub, ast.Not, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
            continue
        else:
            raise ScheduleError(f"{type(node).__name__} is not allowed in a schedule expression")


class _Floats(ast.NodeTransformer):
    """Integer literals become floats, so ``2 ** 100000000`` overflows instead of allocating a huge integer."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, int) and not isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(float(node.value)), node)
        return node


def _environment(epoch: int, epochs: int, base: float, extra: dict[str, float]) -> dict[str, Any]:
    i = epoch - 1
    t = i / (epochs - 1) if epochs > 1 else 0.0
    env: dict[str, Any] = dict(CONSTANTS)
    env.update(FUNCTIONS)
    env.update(epoch=float(epoch), i=float(i), epochs=float(epochs), t=float(t), lr0=float(base), act_lr0=float(base), lr=float(base))
    env.update({key: float(value) for key, value in extra.items()})
    env.update(
        linear=lambda a, b: a + (b - a) * t,
        geometric=lambda a, b: a * (b / a) ** t,
        cosine=lambda a, b: b + (a - b) * (1.0 + math.cos(math.pi * t)) / 2.0,
        step=lambda a, factor, every: a * factor ** math.floor(i / max(1.0, every)),
        warmup=lambda a, b, n: b if i >= n else a + (b - a) * i / max(1.0, n),
    )
    return env


class Schedule:
    """A validated, compiled schedule expression; call it with the epoch to get the rate."""

    __slots__ = ("expression", "_code")

    def __init__(self, expression: str) -> None:
        text = (expression or "").strip()
        if not text:
            raise ScheduleError("the schedule expression is empty")
        try:
            tree = ast.parse(text, mode="eval")
        except SyntaxError as exc:
            raise ScheduleError(f"invalid schedule expression {text!r}: {exc.msg}") from None
        _validate(tree)
        tree = ast.fix_missing_locations(_Floats().visit(tree))
        self.expression = text
        self._code = compile(tree, "<schedule>", "eval")

    def __call__(self, epoch: int, epochs: int, base: float, **extra: float) -> float:
        """The rate for ``epoch`` (1-based) of ``epochs``; ``base`` is ``lr0``; extra names (``act_lr0``, ``lr``)."""
        env = _environment(epoch, epochs, base, extra)
        try:
            value = eval(self._code, {"__builtins__": {}}, env)  # noqa: S307 - the AST was whitelisted above
        except (ArithmeticError, ValueError, TypeError) as exc:
            raise ScheduleError(f"schedule {self.expression!r} failed at epoch {epoch}: {exc}") from None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ScheduleError(f"schedule {self.expression!r} must give a number, got {type(value).__name__}")
        rate = float(value)
        if not math.isfinite(rate) or rate < 0:
            raise ScheduleError(f"schedule {self.expression!r} gave {rate} at epoch {epoch}; rates must be finite and >= 0")
        return rate

    def preview(self, epochs: int, base: float, **extra: float) -> list[float]:
        return [self(epoch, epochs, base, **extra) for epoch in range(1, epochs + 1)]

    def __repr__(self) -> str:
        return f"Schedule({self.expression!r})"


def parse_schedule(expression: str | None) -> Schedule | None:
    """``None`` / blank -> ``None`` (a constant rate); otherwise a validated :class:`Schedule`."""
    if expression is None or not str(expression).strip():
        return None
    if not isinstance(expression, str):
        raise ScheduleError("a schedule must be a string expression")
    return Schedule(expression)


def preview_points(
    lr_schedule: str | None,
    act_lr_schedule: str | None,
    epochs: int,
    lr: float,
    act_lr: float,
    reverse: bool = False,
) -> list[dict]:
    """``[{"epoch", "lr", "act_lr"}]`` for every epoch; the activation schedule sees the epoch's ``lr``.

    ``reverse`` plays the schedule backwards: the rates of the last epoch come
    first (a ramp up becomes a ramp down, a warm-up a cool-down); the pairing
    of ``lr`` and ``act_lr`` within an epoch is kept.
    """
    if epochs < 0:
        raise ScheduleError("epochs must be >= 0")
    lr_fn = parse_schedule(lr_schedule)
    act_fn = parse_schedule(act_lr_schedule)
    points = []
    for epoch in range(1, epochs + 1):
        rate = lr_fn(epoch, epochs, lr, act_lr0=act_lr, lr=lr) if lr_fn else float(lr)
        act = act_fn(epoch, epochs, act_lr, act_lr0=act_lr, lr=rate) if act_fn else float(act_lr)
        points.append({"epoch": epoch, "lr": rate, "act_lr": act})
    if reverse:
        points = [{"epoch": i + 1, "lr": p["lr"], "act_lr": p["act_lr"]} for i, p in enumerate(reversed(points))]
    return points


def describe() -> dict:
    """What a schedule expression may use (for help texts and the frontend)."""
    return {
        "variables": list(VARIABLES),
        "constants": sorted(CONSTANTS),
        "functions": sorted(FUNCTIONS),
        "helpers": [
            "linear(a, b): a at the first epoch, b at the last",
            "geometric(a, b): a to b by a constant factor per epoch (a > 0)",
            "cosine(a, b): a to b along a half cosine",
            "step(a, factor, every): a times factor every `every` epochs",
            "warmup(a, b, n): a to b over the first n epochs, then b",
        ],
        "presets": [dict(p) for p in PRESETS],
    }
