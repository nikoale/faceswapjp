"""Plugin interfaces. Implementations live in analysis/, swappers/, occluders/, enhancers/."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ..analysis.face import Face


class FaceAnalyzer(Protocol):
    def detect(self, image_bgr8: np.ndarray, with_embedding: bool = True) -> list[Face]:
        """Detect faces in a uint8 BGR image (with identity embeddings unless disabled)."""
        ...

    def embed(self, image_bgr8: np.ndarray, face: Face) -> np.ndarray:
        """L2-normalized identity embedding for one detected face."""
        ...


class Swapper(Protocol):
    name: str
    input_size: int
    template: str  # alignment template understood by compositing.align ("arcface_128" ...)

    def swap(self, aligned_bgr: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        """aligned_bgr: float32 [0,1] (S,S,3). Returns float32 [0,1] (S,S,3) BGR."""
        ...


class Occluder(Protocol):
    name: str

    def mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        """crop_bgr: float32 [0,1] (W,W,3) aligned target face. Returns float32 (W,W), 1 = may be replaced."""
        ...


class Enhancer(Protocol):
    name: str
    input_size: int
    template: str

    def enhance(self, aligned_bgr: np.ndarray) -> np.ndarray:
        """float32 [0,1] (S,S,3) in its own template -> same shape."""
        ...
