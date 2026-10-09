"""Paste an aligned crop back into the frame, touching only pixels inside the mask."""

from __future__ import annotations

import cv2
import numpy as np


def dtype_scale(dtype: np.dtype) -> float:
    if dtype == np.uint8:
        return 255.0
    if dtype == np.uint16:
        return 65535.0
    if dtype in (np.float32, np.float64):
        return 1.0
    raise ValueError(f"unsupported image dtype {dtype}")


def to_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32) / dtype_scale(img.dtype)


def from_float(img: np.ndarray, dtype: np.dtype) -> np.ndarray:
    scale = dtype_scale(dtype)
    if scale == 1.0:
        return img.astype(dtype)
    return np.clip(np.rint(img * scale), 0, scale).astype(dtype)


def crop_roi(matrix: np.ndarray, size: int, frame_shape: tuple[int, ...], pad: int = 2) -> tuple[int, int, int, int] | None:
    """Frame-space bounding box (x0, y0, x1, y1) of the crop square under the inverse transform."""
    inv = cv2.invertAffineTransform(matrix)
    corners = np.array([[0, 0, 1], [size, 0, 1], [0, size, 1], [size, size, 1]], dtype=np.float32)
    pts = corners @ inv.T
    h, w = frame_shape[:2]
    x0 = max(0, int(np.floor(pts[:, 0].min())) - pad)
    y0 = max(0, int(np.floor(pts[:, 1].min())) - pad)
    x1 = min(w, int(np.ceil(pts[:, 0].max())) + pad)
    y1 = min(h, int(np.ceil(pts[:, 1].max())) + pad)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def paste_back(
    frame: np.ndarray,
    crop: np.ndarray,
    crop_mask: np.ndarray,
    matrix: np.ndarray,
    matte: np.ndarray | None = None,
    texture: float = 0.0,
    texture_sigma: float = 0.0,
) -> np.ndarray:
    """Composite `crop` (float32 [0,1]) into `frame` in place and return it.

    matrix maps frame -> crop. Only pixels with mask > 0 are rewritten, so the rest of the
    frame stays bit-identical. If `matte` (float32 HxW) is given, the warped mask is max-merged
    into it for alpha/matte output.

    texture > 0 adds back the frame's own fine detail (skin texture, sensor grain) finer than
    `texture_sigma` frame pixels, i.e. the band the upscaled swap cannot produce. The detail is
    soft-limited so strong edges of the original face (eyes, lips) do not ghost through.
    """
    size = crop.shape[0]
    roi = crop_roi(matrix, size, frame.shape)
    if roi is None:
        return frame
    x0, y0, x1, y1 = roi
    inv = cv2.invertAffineTransform(matrix)
    inv[:, 2] -= (x0, y0)
    dsize = (x1 - x0, y1 - y0)
    face = cv2.warpAffine(crop, inv, dsize, flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.warpAffine(crop_mask, inv, dsize, flags=cv2.INTER_LINEAR, borderValue=0.0)
    alpha = np.clip(alpha, 0.0, 1.0)
    touched = alpha > 0
    if not touched.any():
        return frame

    region = frame[y0:y1, x0:x1]
    color = region[..., :3]
    orig = to_float(color)
    if texture > 0 and texture_sigma > 0:
        high = orig - cv2.GaussianBlur(orig, (0, 0), texture_sigma)
        limit = 0.04
        high = limit * np.tanh(high / limit)
        face = face + texture * high
    blended = orig * (1.0 - alpha[..., None]) + np.clip(face, 0.0, 1.0) * alpha[..., None]
    color[touched] = from_float(blended, frame.dtype)[touched]
    if matte is not None:
        np.maximum(matte[y0:y1, x0:x1], alpha, out=matte[y0:y1, x0:x1])
    return frame
