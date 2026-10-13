import numpy as np

from faceswapjp.analysis.face import Face
from faceswapjp.scan import classify, flag_occlusion, merge, sample_stride, yaw_asymmetry


def _face(x=100, size=120, turn=0.0, emb=None):
    # eyes at +-20 px around the center, nose shifted sideways by `turn`
    cx, cy = x + size / 2, 100.0
    kps = np.array([[cx - 20, cy], [cx + 20, cy], [cx + turn, cy + 20], [cx - 15, cy + 40], [cx + 15, cy + 40]],
                   np.float32)
    f = Face(np.array([x, 40, x + size, 40 + size], np.float32), kps, 0.9)
    f.embedding = emb
    return f


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_yaw_asymmetry_grows_with_head_turn():
    assert abs(yaw_asymmetry(_face().kps)) < 0.01
    assert abs(yaw_asymmetry(_face(turn=15).kps)) > 0.3


def test_classify_flags_profile_small_and_ambiguous():
    ref = _unit([1, 0, 0])
    target = _face(size=40, turn=16, emb=_unit([1, 0.1, 0]))  # small and turned
    lookalike = _face(x=300, emb=_unit([0.38, 1, 0]))  # similarity ~0.36: just under the 0.4 threshold
    kinds, found, _ = classify([target, lookalike], ref[None], 0.4, occluded=None)
    assert found and kinds == {"profile", "small", "ambiguous"}
    kinds, found, _ = classify([lookalike], ref[None], 0.4, occluded=None)
    assert not found


def test_occlusion_is_judged_against_the_videos_typical_value():
    samples = [(i * 8, set(), True) for i in range(6)]
    flag_occlusion(samples, [0.3, 0.32, 0.31, 0.6, 0.29, 0.33])  # this person always scores ~0.3
    assert [bool(k) for _, k, _ in samples] == [False, False, False, True, False, False]


def test_merge_keeps_short_losses_between_sightings_only():
    fps, stride = 24.0, 8
    found = [True, True, False, False, True, False]  # gap in the middle, then gone at the end
    samples = [(i * stride, set(), f) for i, f in enumerate(found)]
    issues = merge(samples, stride, fps, last_frame=47)
    assert [(x.kind, x.start, x.end) for x in issues] == [("lost", 16, 31)]


def test_sample_stride_caps_long_videos():
    assert sample_stride(24, 240) == 8
    assert sample_stride(24, 24 * 60 * 30) >= 24 * 60 * 30 // 600
