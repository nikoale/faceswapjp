import cv2
import numpy as np

from faceswapjp.compositing.align import estimate_alignment, template_points, umeyama
from faceswapjp.compositing.blend import crop_roi, from_float, paste_back, to_float
from faceswapjp.compositing.color import match_color
from faceswapjp.compositing.mask import box_mask


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 100, (5, 2)).astype(np.float32)
    angle, s, t = np.deg2rad(23), 1.7, np.array([12.0, -5.0])
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    dst = (s * src @ R.T + t).astype(np.float32)
    M = umeyama(src, dst)
    np.testing.assert_allclose(M[:, :2], s * R, atol=1e-4)
    np.testing.assert_allclose(M[:, 2], t, atol=1e-3)


def test_arcface_128_template_matches_insightface_convention():
    # insightface estimate_norm(size=128): arcface_dst * (128/128) with x shifted by 8
    pts = template_points("arcface_128", 128)
    np.testing.assert_allclose(pts[0], [38.2946 + 8.0, 51.6963], atol=1e-4)
    np.testing.assert_allclose(template_points("arcface_128", 256), pts * 2, atol=1e-4)


def test_alignment_maps_landmarks_onto_template():
    kps = template_points("arcface_128", 128) * 2.5 + 300
    M = estimate_alignment(kps, 128)
    mapped = np.hstack([kps, np.ones((5, 1))]) @ M.T
    np.testing.assert_allclose(mapped, template_points("arcface_128", 128), atol=1e-3)


def test_box_mask_is_zero_at_border_and_one_inside():
    m = box_mask(128, blur=0.1)
    assert m.dtype == np.float32
    assert m[0].max() == 0 and m[-1].max() == 0 and m[:, 0].max() == 0 and m[:, -1].max() == 0
    assert m[64, 64] > 0.99


def test_box_mask_padding_shrinks_top():
    plain, padded = box_mask(128, 0.1), box_mask(128, 0.1, padding=(0.2, 0, 0, 0))
    assert padded[30, 64] < plain[30, 64]


def test_paste_back_only_touches_masked_pixels_16bit():
    rng = np.random.default_rng(1)
    frame = rng.integers(0, 65535, (240, 320, 3), dtype=np.uint16)
    original = frame.copy()
    kps = template_points("arcface_128", 128) + np.array([100, 50], np.float32)
    M = estimate_alignment(kps, 128)
    crop = np.full((128, 128, 3), 0.5, np.float32)
    mask = box_mask(128, 0.1)
    matte = np.zeros(frame.shape[:2], np.float32)
    paste_back(frame, crop, mask, M, matte)
    changed = np.any(frame != original, axis=2)
    assert changed.any()
    assert not (changed & (matte == 0)).any(), "pixels outside the matte were modified"
    assert frame.dtype == np.uint16
    cy, cx = 50 + 64, 100 + 64
    assert abs(int(frame[cy, cx, 0]) - 32768) < 2  # full-alpha center becomes the crop value


def test_paste_back_keeps_alpha_channel():
    frame = np.zeros((200, 200, 4), np.uint8)
    frame[..., 3] = 77
    M = estimate_alignment(template_points("arcface_128", 128) + 30, 128)
    paste_back(frame, np.ones((128, 128, 3), np.float32), box_mask(128), M)
    assert (frame[..., 3] == 77).all()
    assert frame[..., :3].max() == 255


def test_crop_roi_outside_frame_is_none():
    M = estimate_alignment(template_points("arcface_128", 128) + 5000, 128)
    assert crop_roi(M, 128, (100, 100, 3)) is None


def test_float_roundtrip():
    x = np.arange(0, 65536, 257, dtype=np.uint16).reshape(-1, 1, 1).repeat(3, 2)
    assert (from_float(to_float(x), np.uint16) == x).all()


def test_match_color_full_strength_matches_lab_mean():
    rng = np.random.default_rng(2)
    swapped = rng.uniform(0.2, 0.4, (64, 64, 3)).astype(np.float32)
    reference = rng.uniform(0.5, 0.8, (64, 64, 3)).astype(np.float32)
    mask = np.ones((64, 64), np.float32)
    out = match_color(swapped, reference, mask, strength=1.0)
    lab = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(0)  # noqa: E731
    np.testing.assert_allclose(lab(out), lab(reference), atol=1.0)
    assert match_color(swapped, reference, mask, strength=0.0) is swapped
