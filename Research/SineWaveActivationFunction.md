# Sine wave activation function instead of a Sigmoid function

Instead of the traditional sigmoid function frequently used in ANNs, we use a sine wave rotated 45 degrees.

This was drawn from the idea that a brain operates like an analog computer. Hence, we assume sine waves are the standard of information encoding/decoding. This led me to see the similarities between a sigmoid function and a sine wave.


Take the code snippet below for instance
```python
class HyperParameters:
    # Sine parameters
    a: float = -1
    h: float = 0
    b: float = 1/3
    k: float = 0
    decay_window: int = 1_000

    def activation_function(self, x: float) -> float:
        return self.a * math.sin(self.b * (x - self.h)) + self.k
```