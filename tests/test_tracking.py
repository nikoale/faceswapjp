import numpy as np

from faceswapjp.analysis.face import Face
from faceswapjp.tracking.filters import LandmarkSmoother, OneEuroFilter
from faceswapjp.tracking.tracker import FaceTracker, iou

KPS = np.array([[30, 40], [60, 40], [45, 55], [33, 70], [57, 70]], np.float32)


def _face(dx=0.0, emb=None, noise=None):
    k = KPS + [dx, 0] + (noise if noise is not None else 0)
    x0, y0 = k.min(0) - 15
    x1, y1 = k.max(0) + 15
    return Face(np.array([x0, y0, x1, y1], np.float32), k.astype(np.float32), 0.9, emb)


def jitter(seq):
    seq = np.asarray(seq)
    return float(np.abs(np.diff(seq, n=2, axis=0)).mean())


def test_smoother_reduces_jitter_on_static_face():
    rng = np.random.default_rng(0)
    noisy = [KPS + rng.normal(0, 1.0, KPS.shape) for _ in range(120)]
    sm = LandmarkSmoother(strength=0.5)
    smoothed = [sm(k, 1 / 24) for k in noisy]
    assert jitter(smoothed[10:]) < 0.3 * jitter(noisy[10:])


def test_smoother_follows_fast_motion_with_small_lag():
    sm = LandmarkSmoother(strength=0.5)
    out = None
    for i in range(48):
        out = sm(KPS + [i * 10.0, 0], 1 / 24)  # 10 px/frame ~ 8 face widths per second
    assert abs(out[0, 0] - (KPS[0, 0] + 470)) < 25


def test_smoothing_off_is_identity():
    sm = LandmarkSmoother(strength=0)
    assert np.array_equal(sm(KPS, 1 / 24), KPS)


def test_one_euro_first_value_passthrough():
    f = OneEuroFilter()
    assert np.allclose(f(np.array([1.0, 2.0]), 0.04), [1.0, 2.0])


def test_tracker_keeps_ids_and_separates_people():
    a, b = np.eye(512, dtype=np.float32)[0], np.eye(512, dtype=np.float32)[1]
    tr = FaceTracker(smoothing=0)
    ids = []
    for i in range(10):
        pairs = tr.update([_face(i * 2, a), _face(150 - i * 2, b)], i, 1 / 24)
        ids.append(tuple(sorted(t.id for t, _ in pairs)))
    assert len(set(ids)) == 1 and len(tr.tracks) == 2


def test_tracker_reacquires_after_dropout_by_identity():
    a = np.eye(512, dtype=np.float32)[0]
    tr = FaceTracker(smoothing=0, max_misses=5)
    first = tr.update([_face(0, a)], 0, 1 / 24)[0][0].id
    tr.update([], 1, 1 / 24)
    tr.update([], 2, 1 / 24)
    again = tr.update([_face(120, a)], 3, 1 / 24)[0][0].id  # jumped far: no IoU, same identity
    assert again == first


def test_tracker_drops_stale_tracks():
    tr = FaceTracker(max_misses=2)
    tr.update([_face(0)], 0, 1 / 24)
    for i in range(1, 5):
        tr.update([], i, 1 / 24)
    assert tr.tracks == []


def test_needs_embedding_skips_settled_tracks():
    a = np.eye(512, dtype=np.float32)[0]
    tr = FaceTracker()
    for i in range(1, 7):
        tr.update([_face(0, a)], i, 1 / 24)
    assert not tr.needs_embedding(_face(1), 7)
    assert tr.needs_embedding(_face(1), 12)  # periodic refresh
    assert tr.needs_embedding(_face(500), 7)  # new face


def test_iou():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0
