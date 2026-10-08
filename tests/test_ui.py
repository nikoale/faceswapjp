import pytest

gr = pytest.importorskip("gradio")


def test_ui_builds(tmp_path):
    from faceswapjp.ui.app import UIState, build_ui

    assert build_ui(UIState(tmp_path, "cpu")) is not None


def test_build_job_requires_reference_images():
    from faceswapjp.ui.app import SELECT_LABELS, build_job

    with pytest.raises(ValueError):
        build_job("id", SELECT_LABELS["reference"], [], 0.4, [], False, 0.8, 0.12, 0.5, 0.5, False, "", "bottom-right", "cpu")
    job = build_job("id", SELECT_LABELS["all"], [], 0.4, ["オクルージョン（手・小道具）"], True, 0.8, 0.12, 0.5, 0.5,
                    True, "AI", "top-left", "cpu", fmt="prores422hq", matte_label="フィル＋アルファ（ProRes 4444）")
    assert job.masks == "box,occlusion" and job.enhancer == "gfpgan" and job.render.matte == "alpha"
