"""jobs.* with a fake engine: safety gate, consent logging, batch error handling."""

import numpy as np
import pytest

from faceswapjp import jobs
from faceswapjp.engine import Engine
from faceswapjp.identity import validate_consent
from faceswapjp.jobs import SwapJob
from faceswapjp.safety.nsfw import NSFWContentError, SafetyGate
from faceswapjp.safety.watermark import Watermark


class FakeClassifier:
    def __init__(self, score=0.01):
        self.value = score

    def score(self, image):
        return self.value


@pytest.fixture
def engine(monkeypatch, fake_analyzer, solid_swapper):
    eng = Engine(fake_analyzer, solid_swapper, ["CPUExecutionProvider"], "buffalo_l", "inswapper_128",
                 SafetyGate(FakeClassifier()), model_names=["buffalo_l", "inswapper_128"])
    monkeypatch.setattr(jobs, "get_engine", lambda *a, **k: eng)
    return eng


@pytest.fixture
def project(tmp_path, engine):
    import cv2

    from faceswapjp.project import Project

    proj = Project.init(tmp_path / "proj")
    doc = tmp_path / "consent.pdf"
    doc.write_bytes(b"signed")
    ref = tmp_path / "ref.png"
    cv2.imwrite(str(ref), np.full((200, 220, 3), 90, np.uint8))
    jobs.add_identity(proj, "Taro", [ref], validate_consent("Taro", "2026-01-01", doc, "self"), identity_id="taro")
    return proj


def _img(path, value=100):
    import cv2

    cv2.imwrite(str(path), np.full((200, 220, 3), value, np.uint8))
    return path


def test_swap_still_logs_render(tmp_path, project):
    out = tmp_path / "o.png"
    res = jobs.swap_still(project, SwapJob("taro", render=jobs.RenderSettings(watermark=Watermark())),
                          _img(tmp_path / "t.png"), out)
    assert len(res.faces) == 1 and out.exists()
    last = project.consent_log.entries()[-1]
    assert last["event"] == "render" and last["data"]["identity_id"] == "taro"
    assert last["data"]["settings"]["watermark"] is True
    assert project.consent_log.verify() == []


def test_nsfw_stops_processing(tmp_path, project, engine):
    engine.safety.classifier.value = 0.95
    out = tmp_path / "o.png"
    with pytest.raises(NSFWContentError):
        jobs.swap_still(project, SwapJob("taro"), _img(tmp_path / "t.png"), out)
    assert not out.exists()
    assert project.consent_log.entries()[-1]["event"] != "render"


def test_nsfw_checked_on_registration(tmp_path, project, engine):
    engine.safety.classifier.value = 0.95
    doc = tmp_path / "consent.pdf"
    with pytest.raises(NSFWContentError):
        jobs.add_identity(project, "X", [_img(tmp_path / "r2.png")], validate_consent("X", "2026-01-01", doc, "self"))


def test_batch_skips_failures(tmp_path, project):
    src = tmp_path / "in"
    src.mkdir()
    _img(src / "a.png")
    (src / "b.mov").write_bytes(b"not a video")
    report = jobs.batch(project, SwapJob("taro"), src, tmp_path / "out")
    assert [r.ok for r in report] == [True, False]
    assert (tmp_path / "out" / "batch_report.json").exists()


def test_reference_job_switches_mode():
    job = SwapJob("x", reference_images=["r.png"])
    assert job.tracking.select == "reference"


def test_picked_embeddings_select_only_that_person(tmp_path, project, engine):
    other = np.eye(512, dtype=np.float32)[5]
    job = SwapJob("taro", reference_embeddings=other[None])
    assert job.tracking.select == "reference"
    with pytest.raises(ValueError, match="no target face"):
        jobs.swap_still(project, job, _img(tmp_path / "t.png"), tmp_path / "o.png")
    same = SwapJob("taro", reference_embeddings=engine.analyzer.embedding[None])
    assert len(jobs.swap_still(project, same, _img(tmp_path / "t.png"), tmp_path / "o.png").faces) == 1


def test_find_faces(tmp_path, project):
    frame, faces = jobs.find_faces(project, _img(tmp_path / "t.png"))
    assert frame.shape == (200, 220, 3) and len(faces) == 1
