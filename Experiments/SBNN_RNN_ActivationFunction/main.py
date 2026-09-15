"""A self-building RNN whose hidden units carry the author's sine activation.

The activation is the one from ``Research/SineWaveActivationFunction.md``::

    f(x) = a * sin(b * (x - h)) + k        defaults a = -1, b = 1/3, h = 0, k = 0

which at the defaults is exactly ``-sin(x/3)``.  The formula and its partials
are identical to ``RadixCyclicNN/radixnet/activation.py`` -- the reference
implementation -- so this file is that same contract inside a conventional RNN
trained by BPTT, which is what section 10's table claims it is.

All four parameters are learnable **per unit**: amplitude, frequency, phase,
offset -- the four things you can do to a wave.  They are updated by their own
gradients at a *tenth* of the weight learning rate, which section 8 asks for by
name: the activation is the shape of the space the weights are searching in, and
moving the space as fast as you move the search makes both diverge.  Per unit
also matters here specifically, because this network grows: a unit built on a
stall is born at the defaults, exactly ``-sin(x/3)``, and learns its own wave
from there.
"""
import numpy as np

# The author's defaults.  -1 * sin((1/3) * (x - 0)) + 0.
DEFAULT_A = -1.0
DEFAULT_B = 1.0 / 3.0
DEFAULT_H = 0.0
DEFAULT_K = 0.0

MIN_B = 1e-3
"""Floor on the learned frequency, as in ``RadixCyclicNN/radixnet/backend.py``.

``b`` is a frequency; at zero the unit is the constant ``k`` and ``df/db`` is
``a*(x-h)``, which cannot bring it back in any useful direction.  The floor is a
guard on a parameter, not a change to the formula.
"""


class SineActivation:
    """``f(x) = a * sin(b * (x - h)) + k``, one wave per unit, all four learnable.

    Parameters are ``(n_units, 1)`` columns so they broadcast against the RNN's
    ``(hidden_size, 1)`` pre-activation columns.
    """

    PARAMS = ("a", "b", "h", "k")
    DEFAULTS = (DEFAULT_A, DEFAULT_B, DEFAULT_H, DEFAULT_K)

    def __init__(self, n_units, a=DEFAULT_A, b=DEFAULT_B, h=DEFAULT_H, k=DEFAULT_K):
        self.n_units = n_units
        self.a = np.full((n_units, 1), float(a))
        self.b = np.full((n_units, 1), float(b))
        self.h = np.full((n_units, 1), float(h))
        self.k = np.full((n_units, 1), float(k))

    def forward(self, x):
        """The formula, unchanged: ``a * sin(b * (x - h)) + k``."""
        return self.a * np.sin(self.b * (x - self.h)) + self.k

    def backward(self, x, grad_output):
        """Return ``(dL/dx, {param: dL/dparam})`` for ``dL/df = grad_output``.

        Section 8's partials, with ``u = b * (x - h)``::

            f     = a * sin(u) + k
            df/dx = a * b * cos(u)
            df/da = sin(u)
            df/db = a * (x - h) * cos(u)
            df/dh = -a * b * cos(u)
            df/dk = 1
        """
        d = x - self.h
        u = self.b * d
        sin_u, cos_u = np.sin(u), np.cos(u)
        slope = self.a * self.b * cos_u                  # df/dx
        grads = {
            "a": (grad_output * sin_u).sum(axis=1, keepdims=True),
            "b": (grad_output * self.a * d * cos_u).sum(axis=1, keepdims=True),
            "h": (grad_output * -slope).sum(axis=1, keepdims=True),
            "k": grad_output.sum(axis=1, keepdims=True),
        }
        return grad_output * slope, grads

    def update_params(self, grads, lr):
        """One SGD step on ``a, b, h, k``.  Callers pass ``lr = weight_lr / 10``."""
        for name in self.PARAMS:
            setattr(self, name, getattr(self, name) - lr * grads[name])
        np.maximum(self.b, MIN_B, out=self.b)

    def grow(self, amount):
        """Extend by ``amount`` units, each born at the defaults -- i.e. ``-sin(x/3)``."""
        for name, default in zip(self.PARAMS, self.DEFAULTS):
            grown = np.vstack([getattr(self, name), np.full((amount, 1), float(default))])
            setattr(self, name, grown)
        self.n_units += amount

    def zero_grads(self):
        return {name: np.zeros((self.n_units, 1)) for name in self.PARAMS}

    def report(self):
        """Mean value of each knob -- how far the population has moved off the defaults."""
        return {name: float(np.mean(getattr(self, name))) for name in self.PARAMS}


def gradient_check(seed=0, eps=1e-6, trials=200):
    """Central differences against the partials above.  Printed before every run.

    The numbers this file prints are only worth reading if the derivatives are
    right, which is the rule ``Experiments/ActivationFunctionTest/self_building_sinewave.py``
    follows too.  Returns the worst absolute error over all five partials.
    """
    rng = np.random.default_rng(seed)
    n, worst = 3, 0.0
    for _ in range(trials):
        act = SineActivation(n)
        act.a = rng.uniform(-2.0, 2.0, (n, 1))
        act.b = rng.uniform(0.05, 1.5, (n, 1))
        act.h = rng.uniform(-5.0, 5.0, (n, 1))
        act.k = rng.uniform(-1.0, 1.0, (n, 1))

        x = rng.uniform(-20.0, 20.0, (n, 1))
        seed_grad = rng.uniform(-1.0, 1.0, (n, 1))
        dx, grads = act.backward(x, seed_grad)

        num = (act.forward(x + eps) - act.forward(x - eps)) / (2 * eps) * seed_grad
        worst = max(worst, float(np.max(np.abs(num - dx))))

        for name in SineActivation.PARAMS:
            p = getattr(act, name)
            setattr(act, name, p + eps); up = act.forward(x)
            setattr(act, name, p - eps); dn = act.forward(x)
            setattr(act, name, p)
            num = ((up - dn) / (2 * eps) * seed_grad).sum(axis=1, keepdims=True)
            worst = max(worst, float(np.max(np.abs(num - grads[name]))))
    return worst


class SelfBuildingRNN:
    """
    A simple RNN that can grow its hidden layer size ("self-building") when
    training progress stalls. Every hidden unit owns a sine activation whose
    four parameters are learned alongside the weights.
    """

    def __init__(self, input_size, hidden_size, output_size, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = lr

        scale = 0.1
        self.Wxh = rng.normal(0, scale, (hidden_size, input_size))
        self.Whh = rng.normal(0, scale, (hidden_size, hidden_size))
        self.bh = np.zeros((hidden_size, 1))

        self.Why = rng.normal(0, scale, (output_size, hidden_size))
        self.by = np.zeros((output_size, 1))

        self.activation = SineActivation(hidden_size)

        # growth control
        self.loss_history = []
        self.patience = 15
        self.min_delta = 1e-4
        self.max_hidden_size = 256
        self.growth_amount = 4

    def forward(self, inputs):
        """
        inputs: list/array of shape (seq_len, input_size, 1)
        returns hidden states, outputs, and caches for backprop
        """
        h_prev = np.zeros((self.hidden_size, 1))
        self.h_states = [h_prev]
        self.raw_states = []
        self.inputs = inputs
        outputs = []

        for x in inputs:
            raw = self.Wxh @ x + self.Whh @ h_prev + self.bh
            h = self.activation.forward(raw)
            y = self.Why @ h + self.by

            self.raw_states.append(raw)
            self.h_states.append(h)
            outputs.append(y)
            h_prev = h

        return outputs

    def backward(self, targets):
        """
        Basic backpropagation through time (BPTT) for MSE loss.
        targets: list of arrays shape (output_size, 1)
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)
        dWhy = np.zeros_like(self.Why)
        dby = np.zeros_like(self.by)

        d_act = self.activation.zero_grads()

        dh_next = np.zeros((self.hidden_size, 1))
        total_loss = 0.0

        seq_len = len(self.inputs)

        for t in reversed(range(seq_len)):
            x = self.inputs[t]
            h = self.h_states[t + 1]
            h_prev = self.h_states[t]
            y_pred = self.Why @ h + self.by
            y_true = targets[t]

            dy = y_pred - y_true  # MSE gradient
            total_loss += float(np.sum((y_pred - y_true) ** 2))

            dWhy += dy @ h.T
            dby += dy

            dh = self.Why.T @ dy + dh_next

            # the pre-activation of this step is all the activation needs to
            # differentiate itself -- no cached state to get out of step with
            draw, d_act_t = self.activation.backward(self.raw_states[t], dh)
            for name in SineActivation.PARAMS:
                d_act[name] += d_act_t[name]

            dbh += draw
            dWxh += draw @ x.T
            dWhh += draw @ h_prev.T

            dh_next = self.Whh.T @ draw

        # clip gradients to avoid explosion
        for g in (dWxh, dWhh, dbh, dWhy, dby):
            np.clip(g, -5, 5, out=g)

        # update weights
        self.Wxh -= self.lr * dWxh
        self.Whh -= self.lr * dWhh
        self.bh -= self.lr * dbh
        self.Why -= self.lr * dWhy
        self.by -= self.lr * dby

        # a tenth of the weight rate: section 8 of the paper
        self.activation.update_params(d_act, lr=self.lr * 0.1)

        avg_loss = total_loss / seq_len
        return avg_loss

    def maybe_grow(self):
        """
        Self-building logic: if loss has plateaued over `patience` epochs,
        grow the hidden layer by `growth_amount` neurons.
        """
        if self.hidden_size >= self.max_hidden_size:
            return False

        if len(self.loss_history) < self.patience:
            return False

        recent = self.loss_history[-self.patience:]
        improvement = recent[0] - recent[-1]

        if improvement < self.min_delta:
            self._grow_hidden(self.growth_amount)
            return True

        return False

    def _grow_hidden(self, amount):
        rng = np.random.default_rng()
        new_size = self.hidden_size + amount
        scale = 0.1

        Wxh_new = rng.normal(0, scale, (new_size, self.input_size))
        Wxh_new[: self.hidden_size, :] = self.Wxh

        Whh_new = rng.normal(0, scale, (new_size, new_size))
        Whh_new[: self.hidden_size, : self.hidden_size] = self.Whh

        bh_new = np.zeros((new_size, 1))
        bh_new[: self.hidden_size] = self.bh

        Why_new = rng.normal(0, scale, (self.output_size, new_size))
        Why_new[:, : self.hidden_size] = self.Why

        self.Wxh, self.Whh, self.bh, self.Why = Wxh_new, Whh_new, bh_new, Why_new
        self.activation.grow(amount)      # new units start at exactly -sin(x/3)
        self.hidden_size = new_size

        print(f"[Self-Build] Hidden layer grown to {self.hidden_size} units.")

    def train(self, inputs, targets, epochs=200, verbose_every=20):
        for epoch in range(1, epochs + 1):
            self.forward(inputs)
            loss = self.backward(targets)
            self.loss_history.append(loss)

            grew = self.maybe_grow()

            if epoch % verbose_every == 0 or grew:
                p = self.activation.report()
                print(
                    f"Epoch {epoch:4d} | Loss: {loss:.6f} | "
                    f"Hidden size: {self.hidden_size} | "
                    f"a={p['a']:+.4f} b={p['b']:.4f} h={p['h']:+.4f} k={p['k']:+.4f}"
                )

    def predict(self, inputs):
        outputs = self.forward(inputs)
        return outputs


def make_sequence_dataset(seq_len=10, input_size=1, output_size=1, seed=0):
    """
    Generates a simple sine-wave based sequence prediction task:
    given x_t, predict x_{t+1} = sin(x_t) (toy example).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 4 * np.pi, seq_len + 1)
    series = np.sin(t)

    inputs = [np.array([[series[i]]]) for i in range(seq_len)]
    targets = [np.array([[series[i + 1]]]) for i in range(seq_len)]

    return inputs, targets


def main():
    input_size = 1
    output_size = 1
    initial_hidden_size = 4

    inputs, targets = make_sequence_dataset(seq_len=20, input_size=input_size)

    model = SelfBuildingRNN(
        input_size=input_size,
        hidden_size=initial_hidden_size,
        output_size=output_size,
        lr=0.05,
    )

    print(f"Gradient check (max abs error over f, a, b, h, k): {gradient_check():.2e}")
    print("Training Self-Building RNN with f(x) = a*sin(b*(x-h)) + k ...\n")
    model.train(inputs, targets, epochs=400, verbose_every=25)

    print("\nFinal predictions vs targets:")
    preds = model.predict(inputs)
    for i, (p, y) in enumerate(zip(preds, targets)):
        print(f"t={i:2d} | pred={p.item():+.4f} | target={y.item():+.4f}")

    print(f"\nFinal hidden layer size: {model.hidden_size}")
    p = model.activation.report()
    print("Mean learned activation: "
          f"a={p['a']:+.4f} b={p['b']:.4f} h={p['h']:+.4f} k={p['k']:+.4f}"
          f"   (started at a={DEFAULT_A:+.1f} b={DEFAULT_B:.4f} h={DEFAULT_H:+.1f} k={DEFAULT_K:+.1f})")


if __name__ == "__main__":
    main()
