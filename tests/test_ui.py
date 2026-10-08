import numpy as np
import pytest

gr = pytest.importorskip("gradio")

from faceswapjp.ui import app  # noqa: E402


def test_ui_builds_and_creates_default_project(tmp_path):
    state = app.UIState(tmp_path, "cpu")
    assert app.build_ui(state) is not None
    assert state.projects() == [app.DEFAULT_PROJECT]


def _session(**kw):
    s = app.Session(identity="taro", target="clip.mov", fps=24.0, frames=240)
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def test_build_job_requires_choices():
    with pytest.raises(ValueError, match="使う顔"):
        app.build_job(app.Session(), True, False, 0.12, 0.5, 0.4, 0.5)
    with pytest.raises(ValueError, match="置き換える人"):
        app.build_job(_session(), True, False, 0.12, 0.5, 0.4, 0.5)


def test_build_job_maps_plain_language_options():
    emb = np.eye(512, dtype=np.float32)[:2]
    s = _session(picked=[app.Picked(emb[0], None), app.Picked(emb[1], None)])
    job = app.build_job(s, True, True, 0.1, 0.3, 0.45, 0.6, app.USE_EDIT, app.MATTE_LUMA, True, 1.0, 2.0)
    assert job.masks == "box,occlusion,region" and job.enhancer == "gfpgan"
    assert job.render.format == "prores422hq" and job.render.matte == "luma" and job.render.watermark is not None
    assert (job.render.start, job.render.end) == (24, 48)
    assert job.tracking.select == "reference" and job.reference_embeddings.shape == (2, 512)

    everyone = app.build_job(_session(who=app.WHO_ALL), False, False, 0.1, 0.3, 0.4, 0.5)
    assert everyone.tracking.select == "all" and everyone.reference_embeddings is None
    assert everyone.render.format == "h264" and everyone.masks == "box" and everyone.render.end is None


def test_picker_toggle_by_identity():
    e = np.eye(512, dtype=np.float32)
    s = _session(picked=[app.Picked(e[0], None)])
    assert s.is_picked(e[0]) == 0 and s.is_picked(e[1]) is None


def test_friendly_messages():
    from faceswapjp.models.registry import ModelError
    from faceswapjp.safety.nsfw import NSFWContentError

    assert "ダウンロード" in app.friendly(ModelError("x"))
    assert "処理できません" in app.friendly(NSFWContentError("x"))
    assert "顔が見つかりません" in app.friendly(ValueError("no target face found in a.png"))
