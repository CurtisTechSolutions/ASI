import numpy as np


class CustomActivation:
    """
    Custom parametric activation: f(x) = x * sigmoid(alpha * x) + beta * tanh(x)
    Learnable parameters alpha, beta allow the activation to adapt during training.
    """

    def __init__(self, alpha=1.0, beta=0.1):
        self.alpha = alpha
        self.beta = beta

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def forward(self, x):
        s = self._sigmoid(self.alpha * x)
        t = np.tanh(x)
        self._last_x = x
        self._last_s = s
        self._last_t = t
        return x * s + self.beta * t

    def backward(self, grad_output):
        x = self._last_x
        s = self._last_s
        t = self._last_t

        # d/dx [x * sigmoid(alpha*x)] = sigmoid(alpha*x) + x*alpha*sigmoid*(1-sigmoid)
        d_swish = s + x * self.alpha * s * (1 - s)
        # d/dx [beta * tanh(x)] = beta * (1 - tanh(x)^2)
        d_tanh = self.beta * (1 - t ** 2)

        dx = grad_output * (d_swish + d_tanh)

        # gradients for learnable params (summed over batch)
        d_alpha = np.sum(grad_output * x * x * s * (1 - s))
        d_beta = np.sum(grad_output * t)

        return dx, d_alpha, d_beta

    def update_params(self, d_alpha, d_beta, lr=0.001):
        self.alpha -= lr * d_alpha
        self.beta -= lr * d_beta


class SelfBuildingRNN:
    """
    A simple RNN that can grow its hidden layer size ("self-building") when
    training progress stalls. Uses a custom activation function for hidden
    state updates.
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

        self.activation = CustomActivation(alpha=1.0, beta=0.1)

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

        d_alpha_total = 0.0
        d_beta_total = 0.0

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

            # re-run activation backward using cached forward values from this step
            self.activation._last_x = self.raw_states[t]
            self.activation._last_s = self.activation._sigmoid(
                self.activation.alpha * self.raw_states[t]
            )
            self.activation._last_t = np.tanh(self.raw_states[t])

            draw, d_alpha, d_beta = self.activation.backward(dh)
            d_alpha_total += d_alpha
            d_beta_total += d_beta

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

        self.activation.update_params(d_alpha_total, d_beta_total, lr=self.lr * 0.1)

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
        self.hidden_size = new_size

        print(f"[Self-Build] Hidden layer grown to {self.hidden_size} units.")

    def train(self, inputs, targets, epochs=200, verbose_every=20):
        for epoch in range(1, epochs + 1):
            self.forward(inputs)
            loss = self.backward(targets)
            self.loss_history.append(loss)

            grew = self.maybe_grow()

            if epoch % verbose_every == 0 or grew:
                print(
                    f"Epoch {epoch:4d} | Loss: {loss:.6f} | "
                    f"Hidden size: {self.hidden_size} | "
                    f"alpha={self.activation.alpha:.4f} beta={self.activation.beta:.4f}"
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

    print("Training Self-Building RNN with custom activation function...\n")
    model.train(inputs, targets, epochs=400, verbose_every=25)

    print("\nFinal predictions vs targets:")
    preds = model.predict(inputs)
    for i, (p, y) in enumerate(zip(preds, targets)):
        print(f"t={i:2d} | pred={p.item():+.4f} | target={y.item():+.4f}")

    print(f"\nFinal hidden layer size: {model.hidden_size}")


if __name__ == "__main__":
    main()
