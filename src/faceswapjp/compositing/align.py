"""Similarity alignment of a face to a canonical template."""

from __future__ import annotations

import cv2
import numpy as np

# ArcFace 5-point template for a 112x112 crop.
ARCFACE_112 = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)
# FFHQ alignment used by GFPGAN / CodeFormer (normalized to a unit square).
FFHQ_NORM = np.array(
    [[0.37691676, 0.46864664], [0.62285697, 0.46912813], [0.50123859, 0.61331904], [0.39308822, 0.72541100], [0.61150205, 0.72490465]],
    dtype=np.float32,
)


def template_points(template: str, size: int) -> np.ndarray:
    """Destination landmarks for a crop of `size` pixels.

    "arcface_128" reproduces InsightFace's estimate_norm() for 128-multiple crops
    (scale by size/128 and shift x by 8*scale), which inswapper was trained with.
    """
    if template == "arcface_128":
        ratio = size / 128.0
        dst = ARCFACE_112 * ratio
        dst[:, 0] += 8.0 * ratio
        return dst
    if template == "ffhq_512":
        return FFHQ_NORM * size
    if template == "arcface_112":
        return ARCFACE_112 * (size / 112.0)
    raise ValueError(f"unknown template {template!r}")


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, translation) src -> dst. Returns 2x3."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var_s = (sc**2).sum() / n
    scale = (s * d).sum() / var_s
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def crop_to_crop(src_template: str, src_size: int, dst_template: str, dst_size: int) -> np.ndarray:
    """Transform between two aligned crops of the same face (2x3, src crop -> dst crop)."""
    return umeyama(template_points(src_template, src_size), template_points(dst_template, dst_size))


def estimate_alignment(kps: np.ndarray, size: int, template: str = "arcface_128") -> np.ndarray:
    return umeyama(kps, template_points(template, size))


def warp_crop(image: np.ndarray, matrix: np.ndarray, size: int) -> np.ndarray:
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
