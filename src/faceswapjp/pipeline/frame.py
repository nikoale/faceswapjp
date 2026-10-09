"""Per-frame processing shared by still images, video and previews."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..analysis.face import Face
from ..compositing.align import crop_to_crop, estimate_alignment, warp_crop
from ..compositing.blend import from_float, paste_back, to_float
from ..compositing.color import match_color
from ..compositing.mask import box_mask
from ..models.interfaces import Enhancer, FaceAnalyzer, Occluder, Swapper
from ..profiling import NULL, Profiler


@dataclass
class FrameOptions:
    faces: str = "all"  # "all" | "largest" (stills; video uses pipeline.video selection)
    mask_blur: float = 0.12
    mask_padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    color_strength: float = 0.5
    min_det_score: float = 0.5
    work_size: int | None = None  # aligned crop size for masks/compositing (default 256, 512 with enhancer)
    enhance_blend: float = 0.8
    # Resolution the swap model effectively runs at. Above its native 128 px the aligned crop is
    # split into pixel-interleaved 128 px tiles that are swapped one by one and re-interleaved
    # ("pixel boost"): 256 = 4 swaps, 512 = 16 swaps per face.
    swap_size: int = 256


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
    if bgr.dtype == np.uint16:
        return cv2.convertScaleAbs(bgr, alpha=1 / 257)  # fast path (~2x numpy)
    return from_float(to_float(bgr), np.uint8)


def select_faces(faces: list[Face], mode: str, min_score: float) -> list[Face]:
    faces = [f for f in faces if f.det_score >= min_score]
    if mode == "all":
        return faces
    if mode == "largest":
        return sorted(faces, key=lambda f: f.area, reverse=True)[:1]
    raise ValueError(f"unknown face selection {mode!r}")


def pixel_boost(swap, crop: np.ndarray, base: int) -> np.ndarray:
    """Run `swap` (base px in, base px out) over an n*base crop as n*n interleaved tiles."""
    n = crop.shape[0] // base
    if n <= 1:
        return swap(crop)
    out = np.empty_like(crop)
    for i in range(n):
        for j in range(n):
            out[i::n, j::n] = swap(np.ascontiguousarray(crop[i::n, j::n]))
    if n > 2:
        # Neighbouring pixels come from different swaps; at 4x4 their small disagreements show
        # up as a fine mesh. A sub-pixel blur removes it while keeping most of the gained detail.
        out = cv2.GaussianBlur(out, (0, 0), 0.6)
    return out


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size:
        return img
    interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_CUBIC
    return np.clip(cv2.resize(img, (size, size), interpolation=interp), 0.0, 1.0)


class FrameProcessor:
    def __init__(
        self,
        analyzer: FaceAnalyzer,
        swapper: Swapper,
        options: FrameOptions | None = None,
        occluders: list[Occluder] | None = None,
        enhancer: Enhancer | None = None,
    ):
        self.analyzer = analyzer
        self.swapper = swapper
        self.options = options or FrameOptions()
        self.occluders = list(occluders or [])
        self.enhancer = enhancer
        self.profiler: Profiler = NULL
        base = swapper.input_size
        self.swap_size = max(base, base * (int(self.options.swap_size) // base))
        self.work_size = self.options.work_size or max(512 if enhancer else 256, self.swap_size)
        self._box = box_mask(self.work_size, self.options.mask_blur, self.options.mask_padding)
        if enhancer is not None:
            self._to_enh = crop_to_crop(swapper.template, self.work_size, enhancer.template, enhancer.input_size)
            self._from_enh = cv2.invertAffineTransform(self._to_enh)

    def detect(self, frame: np.ndarray) -> list[Face]:
        return self.analyzer.detect(to_detection_image(frame))

    def face_mask(self, work: np.ndarray) -> np.ndarray:
        mask = self._box.copy()
        for occ in self.occluders:
            mask *= occ.mask(work)
        return mask

    def _enhance(self, crop: np.ndarray) -> np.ndarray:
        s = self.enhancer.input_size
        enh_in = cv2.warpAffine(crop, self._to_enh, (s, s), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        enh = self.enhancer.enhance(np.clip(enh_in, 0, 1))
        w = self.work_size
        back = cv2.warpAffine(enh, self._from_enh, (w, w), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
        b = self.options.enhance_blend
        return np.clip(crop * (1 - b) + back * b, 0.0, 1.0)

    def swap_face(self, frame: np.ndarray, face: Face, source_embedding: np.ndarray, matte: np.ndarray) -> None:
        w = self.work_size
        prof = self.profiler
        with prof("align"):
            matrix = estimate_alignment(face.kps, w, self.swapper.template)
            work = to_float(warp_crop(frame[..., :3], matrix, w))
        with prof("swap"):
            base, size = self.swapper.input_size, self.swap_size
            if size == w:
                boost_in = work
            elif size < w:
                boost_in = _resize(work, size)
            else:
                boost_in = to_float(warp_crop(frame[..., :3], estimate_alignment(face.kps, size, self.swapper.template), size))
            swapped = pixel_boost(lambda t: self.swapper.swap(t, source_embedding), boost_in, base)
            swapped = _resize(swapped, w)
        if self.enhancer is not None:
            with prof("enhance"):
                swapped = self._enhance(swapped)
        with prof("mask"):
            mask = self.face_mask(work)
        with prof("color"):
            swapped = match_color(swapped, work, mask, self.options.color_strength)
        with prof("paste"):
            paste_back(frame, swapped, mask, matrix, matte)

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
