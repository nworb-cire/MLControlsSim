import numpy as np


class LataccelTokenizer:
    VOCAB_SIZE = 1024
    LATACCEL_RANGE = [-5, 5]

    def __init__(self):
        self.bins = np.linspace(self.LATACCEL_RANGE[0], self.LATACCEL_RANGE[1], self.VOCAB_SIZE)

    def encode(self, value: float | np.ndarray | list[float]) -> int | np.ndarray:
        value = self.clip(value)
        return np.digitize(value, self.bins, right=True)

    def decode(self, token: int | np.ndarray) -> float | np.ndarray:
        return self.bins[token]

    def clip(self, value: float | np.ndarray | list[float]) -> float | np.ndarray:
        return np.clip(value, self.LATACCEL_RANGE[0], self.LATACCEL_RANGE[1])
