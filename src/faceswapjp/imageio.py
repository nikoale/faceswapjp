"""Still image read/write that keeps bit depth and alpha, and handles non-ASCII paths."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .compositing.blend import from_float, to_float

log = logging.getLogger(__name__)

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp"}


def read_image(path: str | Path) -> np.ndarray:
    """Return uint8/uint16 BGR or BGRA."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"cannot read image: {path}")
    if img.dtype not in (np.uint8, np.uint16):
        raise ValueError(f"unsupported image depth {img.dtype}: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 2:  # gray + alpha
        img = np.dstack([img[..., 0]] * 3 + [img[..., 1]])
    return img


def encode_image(img: np.ndarray, ext: str, jpeg_quality: int = 95) -> bytes:
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        if img.dtype != np.uint8:
            log.warning("JPEG is 8-bit only; reducing bit depth")
            img = from_float(to_float(img), np.uint8)
        if img.ndim == 3 and img.shape[2] == 4:
            log.warning("JPEG has no alpha; dropping alpha channel")
            img = img[..., :3]
        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
    else:
        params = []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise ValueError(f"cannot encode image as {ext}")
    return buf.tobytes()
