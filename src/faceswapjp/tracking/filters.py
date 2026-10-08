"""One Euro filter (Casiez et al. 2012): adaptive low-pass, strong at rest, light during fast motion."""

from __future__ import annotations

import math

import numpy as np


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2 * math.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    def __init__(self, min_cutoff: float = 1.0, beta: float = 5.0, d_cutoff: float = 1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self._x: np.ndarray | None = None
        self._dx: np.ndarray | None = None

    def reset(self) -> None:
        self._x = self._dx = None

    def __call__(self, x: np.ndarray, dt: float) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if self._x is None:
            self._x, self._dx = x.copy(), np.zeros_like(x)
            return x.copy()
        dx = (x - self._x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self._dx = a_d * dx + (1 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        tau = 1.0 / (2 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self._x = a * x + (1 - a) * self._x
        return self._x.copy()


def smoothing_to_cutoff(strength: float) -> float | None:
    """Map a 0..1 UI strength to a min cutoff frequency (Hz). 0 disables smoothing."""
    if strength <= 0:
        return None
    strength = min(strength, 1.0)
    return 10 ** (1.0 - 2.0 * strength)  # 0.5 -> 1 Hz, 1.0 -> 0.1 Hz


class LandmarkSmoother:
    """Smooths 5-point landmarks in face-size-normalized units so settings are resolution independent."""

    def __init__(self, strength: float = 0.5, beta: float = 5.0):
        cutoff = smoothing_to_cutoff(strength)
        self._filter = OneEuroFilter(cutoff, beta) if cutoff else None
        self._scale: float | None = None

    def __call__(self, kps: np.ndarray, dt: float) -> np.ndarray:
        if self._filter is None:
            return kps
        if self._scale is None:
            self._scale = float(np.linalg.norm(kps[0] - kps[1])) or 1.0  # inter-ocular distance
        return (self._filter(kps / self._scale, dt) * self._scale).astype(np.float32)
