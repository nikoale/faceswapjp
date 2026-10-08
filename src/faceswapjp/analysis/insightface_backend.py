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
        self._recognizer = self._app.models["recognition"]
        self._landmarker = self._app.models.get("landmark_2d_106")

    def detect(self, image_bgr8: np.ndarray, with_embedding: bool = True) -> list[Face]:
        if image_bgr8.dtype != np.uint8 or image_bgr8.ndim != 3:
            raise ValueError("detect() expects a uint8 BGR image")
        bboxes, kpss = self._app.det_model.detect(image_bgr8, max_num=0, metric="default")
        faces = [
            Face(bbox=b[:4].astype(np.float32), kps=k.astype(np.float32), det_score=float(b[4]))
            for b, k in zip(bboxes, kpss if kpss is not None else [])
        ]
        if with_embedding:
            for face in faces:
                face.embedding = self.embed(image_bgr8, face)
        if self._landmarker is not None:
            for face in faces:
                face.landmarks_106 = self._landmark_106(image_bgr8, face)
        return faces

    def embed(self, image_bgr8: np.ndarray, face: Face) -> np.ndarray:
        """L2-normalized ArcFace embedding of one detected face."""
        from insightface.utils import face_align

        with warnings.catch_warnings():
            # insightface calls a deprecated scikit-image API here
            warnings.simplefilter("ignore", FutureWarning)
            aligned = face_align.norm_crop(image_bgr8, landmark=face.kps, image_size=self._recognizer.input_size[0])
        emb = self._recognizer.get_feat(aligned).flatten().astype(np.float32)
        return emb / np.linalg.norm(emb)

    def _landmark_106(self, image_bgr8: np.ndarray, face: Face) -> np.ndarray:
        from insightface.app.common import Face as IFace

        f = IFace(bbox=face.bbox, kps=face.kps, det_score=face.det_score)
        return np.asarray(self._landmarker.get(image_bgr8, f), dtype=np.float32)
