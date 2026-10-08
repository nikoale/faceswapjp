"""Occlusion / region masks so hands, hair and props in front of the face stay on top.

- xseg: DeepFaceLab-style XSeg occlusion segmentation (face minus occluders)
- region: BiSeNet face parsing (CelebAMask-HQ classes); keeps only selected face regions
Both are non-commercial (see docs/MODEL_LICENSES.md).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..runtime import ProviderSpec, create_session


def _soften(mask: np.ndarray, sigma: float) -> np.ndarray:
    """Blur then re-threshold around 0.5: removes speckle while keeping a soft edge."""
    mask = cv2.GaussianBlur(np.clip(mask, 0, 1), (0, 0), sigma)
    return (np.clip(mask, 0.5, 1.0) - 0.5) * 2.0


class XSegOccluder:
    name = "xseg"

    def __init__(self, model_path: Path, providers: list[ProviderSpec]):
        self._session = create_session(model_path, providers)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = int(inp.shape[1])  # NHWC

    def mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        w = crop_bgr.shape[0]
        x = cv2.resize(crop_bgr, (self._size, self._size), interpolation=cv2.INTER_AREA)[None].astype(np.float32)
        out = self._session.run(None, {self._input: x})[0][0, ..., 0]
        out = _soften(out, sigma=self._size / 50)
        return cv2.resize(out, (w, w), interpolation=cv2.INTER_LINEAR).astype(np.float32)


# CelebAMask-HQ label ids as output by the BiSeNet face-parsing model
REGIONS = {
    "skin": 1, "left-eyebrow": 2, "right-eyebrow": 3, "left-eye": 4, "right-eye": 5, "glasses": 6,
    "left-ear": 7, "right-ear": 8, "earring": 9, "nose": 10, "mouth": 11, "upper-lip": 12, "lower-lip": 13,
    "neck": 14, "necklace": 15, "cloth": 16, "hair": 17, "hat": 18,
}
DEFAULT_REGIONS = ("skin", "left-eyebrow", "right-eyebrow", "left-eye", "right-eye", "glasses", "nose", "mouth", "upper-lip", "lower-lip")


class RegionOccluder:
    name = "region"
    _mean = np.array([0.485, 0.456, 0.406], np.float32)
    _std = np.array([0.229, 0.224, 0.225], np.float32)

    def __init__(self, model_path: Path, providers: list[ProviderSpec], regions: tuple[str, ...] = DEFAULT_REGIONS):
        unknown = set(regions) - set(REGIONS)
        if unknown:
            raise ValueError(f"unknown regions {sorted(unknown)}; choose from {', '.join(REGIONS)}")
        self._session = create_session(model_path, providers)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self._size = int(inp.shape[2])  # NCHW
        self._ids = [REGIONS[r] for r in regions]

    def mask(self, crop_bgr: np.ndarray) -> np.ndarray:
        w = crop_bgr.shape[0]
        x = cv2.resize(crop_bgr, (self._size, self._size), interpolation=cv2.INTER_AREA)[..., ::-1]
        x = ((x - self._mean) / self._std).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self._session.run(None, {self._input: x})[0][0]
        keep = np.isin(logits.argmax(0), self._ids).astype(np.float32)
        keep = _soften(keep, sigma=self._size / 100)
        return cv2.resize(keep, (w, w), interpolation=cv2.INTER_LINEAR).astype(np.float32)
