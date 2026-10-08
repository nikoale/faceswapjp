"""Plugin interfaces. Implementations live in analysis/, swappers/, etc."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..analysis.face import Face


class FaceAnalyzer(Protocol):
    def detect(self, image_bgr8: np.ndarray) -> list[Face]:
        """Detect faces in a uint8 BGR image, with embeddings."""
        ...


class Swapper(Protocol):
    name: str
    input_size: int
    template: str  # alignment template understood by compositing.align ("arcface_128" ...)

    def swap(self, aligned_bgr: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """aligned_bgr: float32 [0,1] (S,S,3). Returns float32 [0,1] (S,S,3) BGR."""
        ...
