from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray  # (4,) x1, y1, x2, y2 in frame pixels
    kps: np.ndarray  # (5, 2) eyes, nose, mouth corners
    det_score: float
    embedding: np.ndarray | None = None  # (512,) L2-normalized ArcFace embedding
    landmarks_106: np.ndarray | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def best_similarity(embedding: np.ndarray | None, references: np.ndarray) -> float:
    """Highest cosine similarity between an embedding and one or more reference embeddings."""
    if embedding is None:
        return -1.0
    return float(np.max(np.atleast_2d(references) @ embedding))
