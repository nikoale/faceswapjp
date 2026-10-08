"""Masks in aligned-crop space. They are warped back to the frame together with the face."""

from __future__ import annotations

import cv2
import numpy as np

_TAIL = 0.01


def box_mask(
    size: int,
    blur: float = 0.12,
    padding: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> np.ndarray:
    """Rectangular mask with feathered edges that reach exactly zero at the crop border.

    blur: feather width as a fraction of the crop size.
    padding: extra inset (top, right, bottom, left) as fractions of the crop size.
    """
    margin = max(1, int(round(blur * size)))
    top, right, bottom, left = (int(round(p * size)) for p in padding)
    mask = np.zeros((size, size), dtype=np.float32)
    y0, y1 = margin + top, size - margin - bottom
    x0, x1 = margin + left, size - margin - right
    if y1 > y0 and x1 > x0:
        mask[y0:y1, x0:x1] = 1.0
    if margin > 1:
        # 3 sigma fits inside the margin; the small remaining tail is cut so the mask is
        # exactly zero at the crop border (otherwise a faint square seam shows up).
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=margin / 3.0, borderType=cv2.BORDER_CONSTANT)
        mask = (mask - _TAIL) / (1.0 - _TAIL)
    return np.clip(mask, 0.0, 1.0)
