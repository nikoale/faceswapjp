"""Face detection / recognition via InsightFace model packs (e.g. buffalo_l)."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..config import DEFAULT_DET_SIZE
from ..runtime import ProviderSpec
from .face import Face


class InsightFaceAnalyzer:
    def __init__(
        self,
        model_dir: Path,
        providers: list[ProviderSpec],
        det_size: tuple[int, int] = DEFAULT_DET_SIZE,
        det_thresh: float = 0.5,
        with_landmarks: bool = False,
    ):
        from insightface.app import FaceAnalysis

        modules = ["detection", "recognition"] + (["landmark_2d_106"] if with_landmarks else [])
        self._app = FaceAnalysis(name=str(model_dir), allowed_modules=modules, providers=providers)
        self._app.prepare(ctx_id=0, det_size=det_size, det_thresh=det_thresh)

    def detect(self, image_bgr8: np.ndarray) -> list[Face]:
        if image_bgr8.dtype != np.uint8 or image_bgr8.ndim != 3:
            raise ValueError("detect() expects a uint8 BGR image")
        with warnings.catch_warnings():
            # insightface calls a deprecated scikit-image API on every face
            warnings.simplefilter("ignore", FutureWarning)
            detected = self._app.get(image_bgr8)
        faces = []
        for f in detected:
            emb = getattr(f, "normed_embedding", None)
            faces.append(
                Face(
                    bbox=np.asarray(f.bbox, dtype=np.float32),
                    kps=np.asarray(f.kps, dtype=np.float32),
                    det_score=float(f.det_score),
                    embedding=None if emb is None else np.asarray(emb, dtype=np.float32),
                    landmarks_106=getattr(f, "landmark_2d_106", None),
                )
            )
        return faces
