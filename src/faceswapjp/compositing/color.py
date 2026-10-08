"""Color matching of the swapped face to the original plate."""

from __future__ import annotations

import cv2
import numpy as np


def match_color(
    swapped: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.5,
    eps: float = 1e-4,
) -> np.ndarray:
    """Reinhard transfer in Lab using statistics inside `mask` (> 0.5).

    swapped, reference: float32 BGR [0,1], same shape. strength 0 = off, 1 = full transfer.
    """
    if strength <= 0:
        return swapped
    src = cv2.cvtColor(swapped.astype(np.float32), cv2.COLOR_BGR2LAB)
    # Statistics don't need full resolution: measure them on a small copy (much faster).
    small = (64, 64) if swapped.shape[0] > 64 else swapped.shape[1::-1]
    sel = cv2.resize(mask, small, interpolation=cv2.INTER_AREA) > 0.5
    if sel.sum() < 16:
        return swapped
    src_s = cv2.resize(src, small, interpolation=cv2.INTER_AREA)[sel]
    ref_s = cv2.cvtColor(cv2.resize(reference.astype(np.float32), small, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2LAB)[sel]
    s_mu, s_sd = src_s.mean(0), src_s.std(0) + eps
    r_mu, r_sd = ref_s.mean(0), ref_s.std(0) + eps
    transferred = (src - s_mu) / s_sd * r_sd + r_mu
    out = src + strength * (transferred - src)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0.0, 1.0)
