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
    tr.update([_face(0, a)], 1, 1 / 24)
    assert tr.needs_embedding(_face(1), 2)  # only one embedding so far
    tr.update([_face(0, a)], 2, 1 / 24)
    assert not tr.needs_embedding(_face(1), 3)
    track_id = tr.tracks[0].id
    refresh_frame = next(i for i in range(4, 40) if (i + track_id * 7) % 24 == 0)
    assert tr.needs_embedding(_face(1), refresh_frame)  # periodic refresh
    assert tr.needs_embedding(_face(500), 3)  # new face


def test_iou():
    assert iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    assert iou(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) == 0.0


def test_interval_detection_uses_roi_and_recovers(solid_swapper):
    from faceswapjp.pipeline.frame import FrameOptions, FrameProcessor
    from faceswapjp.pipeline.video import TrackedProcessor, TrackingOptions

    a = np.eye(512, dtype=np.float32)[0]

    class CountingAnalyzer:
        def __init__(self):
            self.full = self.roi = 0
            self.lose = set()

        def detect(self, image, with_embedding=True):
            self.full += 1
            return [_face(0, a if with_embedding else None)]

        def detect_roi(self, image, bbox, window=None):
            self.roi += 1
            return None if self.frame in self.lose else _face(0)

        def embed(self, image, face):
            return a

    an = CountingAnalyzer()
    tracked = TrackedProcessor(FrameProcessor(an, solid_swapper, FrameOptions(color_strength=0)), a, 24,
                               TrackingOptions(detect_every=3))
    frame = np.zeros((160, 200, 3), np.uint8)
    for i in range(9):
        an.frame = i
        res = tracked.process(frame, i)
        assert len(res.faces) == 1
    # frames 0, 3, 6 full (landmarks refined in a crop); the other 6 frames crop-only
    assert an.full == 3 and an.roi == 9

    an.lose = {10}
    an.full = an.roi = 0
    for i in range(9, 12):
        an.frame = i
        tracked.process(frame, i)
    assert an.full == 2  # frame 9 scheduled + frame 10 forced because the target was lost


def test_detect_every_one_is_full_detection(solid_swapper):
    from faceswapjp.pipeline.frame import FrameProcessor
    from faceswapjp.pipeline.video import TrackedProcessor, TrackingOptions

    a = np.eye(512, dtype=np.float32)[0]

    class A:
        full = 0

        def detect(self, image, with_embedding=True):
            A.full += 1
            return [_face(0, a if with_embedding else None)]

        def detect_roi(self, image, bbox, window=None):
            raise AssertionError("ROI detection must not run with detect_every=1")

        def embed(self, image, face):
            return a

    tracked = TrackedProcessor(FrameProcessor(A(), solid_swapper), a, 24, TrackingOptions(detect_every=1))
    for i in range(4):
        tracked.process(np.zeros((160, 200, 3), np.uint8), i)
    assert A.full == 4
