"""Per-frame processing shared by still images, video and previews."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.face import Face
from ..compositing.align import estimate_alignment, warp_crop
from ..compositing.blend import from_float, paste_back, to_float
from ..compositing.color import match_color
from ..compositing.mask import box_mask
from ..models.interfaces import FaceAnalyzer, Swapper


@dataclass
class FrameOptions:
    faces: str = "all"  # "all" | "largest"
    mask_blur: float = 0.12
    mask_padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color_strength: float = 0.5
    min_det_score: float = 0.5


@dataclass
class FrameResult:
    frame: np.ndarray  # same dtype/channels as the input
    matte: np.ndarray  # float32 HxW, 1 = replaced
    faces: list[Face]  # faces that were replaced


def to_detection_image(frame: np.ndarray) -> np.ndarray:
    """uint8 BGR view of a frame for the detector."""
    bgr = frame[..., :3]
    if bgr.dtype == np.uint8:
        return np.ascontiguousarray(bgr)
    return from_float(to_float(bgr), np.uint8)


def select_faces(faces: list[Face], mode: str, min_score: float) -> list[Face]:
    faces = [f for f in faces if f.det_score >= min_score]
    if mode == "all":
        return faces
    if mode == "largest":
        return sorted(faces, key=lambda f: f.area, reverse=True)[:1]
    raise ValueError(f"unknown face selection {mode!r}")


class FrameProcessor:
    def __init__(self, analyzer: FaceAnalyzer, swapper: Swapper, options: FrameOptions | None = None):
        self.analyzer = analyzer
        self.swapper = swapper
        self.options = options or FrameOptions()
        self._mask = box_mask(swapper.input_size, self.options.mask_blur, self.options.mask_padding)

    def detect(self, frame: np.ndarray) -> list[Face]:
        return self.analyzer.detect(to_detection_image(frame))

    def swap_face(self, frame: np.ndarray, face: Face, source_embedding: np.ndarray, matte: np.ndarray) -> None:
        size = self.swapper.input_size
        matrix = estimate_alignment(face.kps, size, self.swapper.template)
        aligned = to_float(warp_crop(frame[..., :3], matrix, size))
        swapped = self.swapper.swap(aligned, source_embedding)
        swapped = match_color(swapped, aligned, self._mask, self.options.color_strength)
        paste_back(frame, swapped, self._mask, matrix, matte)

    def process(
        self,
        frame: np.ndarray,
        source_embedding: np.ndarray,
        faces: list[Face] | None = None,
    ) -> FrameResult:
        out = frame.copy()
        matte = np.zeros(frame.shape[:2], dtype=np.float32)
        if faces is None:
            faces = select_faces(self.detect(frame), self.options.faces, self.options.min_det_score)
        for face in faces:
            self.swap_face(out, face, source_embedding, matte)
        return FrameResult(out, matte, faces)
