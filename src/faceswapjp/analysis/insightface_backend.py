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
        # Prepare the crop-detector input size now, so its one-time setup doesn't stall the first frames.
        try:
            self._app.det_model.detect(np.zeros((256, 256, 3), np.uint8), input_size=(256, 256), max_num=0)
        except Exception:  # noqa: BLE001 - warm-up only
            pass
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

    @staticmethod
    def roi_window(bbox: np.ndarray, shape: tuple[int, ...], scale: float = 2.4) -> tuple[int, int, int, int]:
        """Square crop around a face box, clamped to the image."""
        h, w = shape[:2]
        x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        side = max(x1 - x0, y1 - y0) * scale
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        return (int(max(0, cx - side / 2)), int(max(0, cy - side / 2)),
                int(min(w, cx + side / 2)), int(min(h, cy + side / 2)))

    def detect_roi(self, image_bgr8: np.ndarray, bbox: np.ndarray, window: tuple[int, int, int, int] | None = None,
                   size: int = 256) -> Face | None:
        """Re-detect one known face inside a crop (much cheaper than a full-frame pass).

        `window` lets the caller keep the crop fixed between frames, which keeps the landmarks
        steadier than re-centering the crop on every frame. Returns the detection overlapping
        `bbox` best, in full-frame coordinates, or None.
        """
        rx0, ry0, rx1, ry1 = window or self.roi_window(bbox, image_bgr8.shape)
        if rx1 - rx0 < 16 or ry1 - ry0 < 16:
            return None
        x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        crop = np.ascontiguousarray(image_bgr8[ry0:ry1, rx0:rx1])
        bboxes, kpss = self._app.det_model.detect(crop, input_size=(size, size), max_num=0, metric="default")
        if bboxes is None or len(bboxes) == 0 or kpss is None:
            return None
        offset = np.array([rx0, ry0], dtype=np.float32)
        best, best_iou = None, 0.0
        for b, k in zip(bboxes, kpss):
            box = b[:4].astype(np.float32) + np.tile(offset, 2)
            ix0, iy0 = max(box[0], x0), max(box[1], y0)
            ix1, iy1 = min(box[2], x1), min(box[3], y1)
            inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
            union = (box[2] - box[0]) * (box[3] - box[1]) + (x1 - x0) * (y1 - y0) - inter
            overlap = inter / union if union > 0 else 0.0
            if overlap > best_iou:
                best, best_iou = Face(bbox=box, kps=(k + offset).astype(np.float32), det_score=float(b[4])), overlap
        return best if best_iou >= 0.2 else None

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
