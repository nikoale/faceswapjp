import numpy as np

from faceswapjp.analysis.face import Face
from faceswapjp.photo_quality import assess_photo, assess_set


def _face(size=200, turn=0.0, emb=None, score=0.9):
    x, y = 50.0, 50.0
    cx, cy = x + size / 2, y + size * 0.45
    e = size * 0.18
    kps = np.array([[cx - e, cy], [cx + e, cy], [cx + turn * e, cy + e], [cx - e * 0.8, cy + 2 * e],
                    [cx + e * 0.8, cy + 2 * e]], np.float32)
    f = Face(np.array([x, y, x + size, y + size], np.float32), kps, score)
    f.embedding = emb
    return f


def _textured(h=400, w=400, level=128):
    rng = np.random.default_rng(0)
    return np.clip(level + rng.normal(0, 40, (h, w, 3)), 0, 255).astype(np.uint8)


def _unit(v):
    v = np.asarray(v, np.float32)
    return v / np.linalg.norm(v)


def test_sharp_frontal_photo_is_good():
    report, face = assess_photo(_textured(), [_face()])
    assert report.grade == "good" and not report.notes and face is not None


def test_problems_are_named():
    flat = np.full((400, 400, 3), 128, np.uint8)  # no detail at all = blurry
    report, _ = assess_photo(flat, [_face(size=60, turn=1.5)])
    assert report.grade == "bad"
    joined = " ".join(report.notes)
    assert "ぼやけ" in joined and "小さい" in joined and "横を向いて" in joined
    assert assess_photo(_textured(), [])[0].notes == ["顔が見つかりません"]


def test_a_different_person_in_the_set_is_flagged():
    a, b = _unit([1, 0.05, 0]), _unit([1, -0.05, 0.02])
    stranger = _unit([0, 1, 0])
    img = _textured()
    result = assess_set([img, img, img], [[_face(emb=a)], [_face(emb=b)], [_face(emb=stranger)]])
    grades = [p["grade"] for p in result["photos"]]
    assert grades[:2] == ["good", "good"] and grades[2] == "bad"
    assert "別人" in result["photos"][2]["notes"][0]
