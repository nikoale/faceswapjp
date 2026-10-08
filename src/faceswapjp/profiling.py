"""Tiny per-stage timer used by the video pipeline and `faceswapjp bench`."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager


class Profiler:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.totals[name] += time.perf_counter() - t0
            self.counts[name] += 1

    def add(self, name: str, seconds: float) -> None:
        self.totals[name] += seconds
        self.counts[name] += 1

    def report(self, frames: int) -> dict[str, dict[str, float]]:
        """Per stage: total seconds, ms per call and ms per frame."""
        frames = max(frames, 1)
        return {
            k: {"total_s": round(v, 3), "calls": self.counts[k], "ms_per_call": round(1000 * v / max(self.counts[k], 1), 1),
                "ms_per_frame": round(1000 * v / frames, 1)}
            for k, v in sorted(self.totals.items(), key=lambda kv: -kv[1])
        }


class NullProfiler(Profiler):
    @contextmanager
    def __call__(self, name: str):
        yield

    def add(self, name: str, seconds: float) -> None:
        pass


NULL = NullProfiler()
