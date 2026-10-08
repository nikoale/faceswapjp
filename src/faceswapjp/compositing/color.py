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
    sel = mask > 0.5
    if sel.sum() < 16:
        return swapped
    src = cv2.cvtColor(swapped.astype(np.float32), cv2.COLOR_BGR2LAB)
    ref = cv2.cvtColor(reference.astype(np.float32), cv2.COLOR_BGR2LAB)
    s_mu, s_sd = src[sel].mean(0), src[sel].std(0) + eps
    r_mu, r_sd = ref[sel].mean(0), ref[sel].std(0) + eps
    transferred = (src - s_mu) / s_sd * r_sd + r_mu
    out = src + strength * (transferred - src)
    return np.clip(cv2.cvtColor(out, cv2.COLOR_LAB2BGR), 0.0, 1.0)
