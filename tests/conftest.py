import numpy as np
import pytest

from faceswapjp.analysis.face import Face
from faceswapjp.compositing.align import template_points


class FakeAnalyzer:
    """Returns one face whose landmarks match the arcface template scaled into the frame."""

    def __init__(self, center=(100, 80), scale=1.0, embedding=None):
        self.center, self.scale = center, scale
        self.embedding = embedding if embedding is not None else np.eye(512, dtype=np.float32)[0]

    def detect(self, image):
        kps = (template_points("arcface_128", 128) - 64) * self.scale + np.array(self.center, dtype=np.float32)
        x0, y0 = kps.min(0) - 20
        x1, y1 = kps.max(0) + 20
        return [Face(np.array([x0, y0, x1, y1], np.float32), kps.astype(np.float32), 0.9, self.embedding)]


class SolidSwapper:
    """Returns a constant color so the paste-back region is easy to measure."""

    name = "solid"
    template = "arcface_128"
    input_size = 128

    def __init__(self, bgr=(0.0, 0.0, 1.0)):
        self.bgr = np.array(bgr, np.float32)

    def swap(self, aligned, embedding):
        return np.broadcast_to(self.bgr, aligned.shape).astype(np.float32).copy()


@pytest.fixture
def fake_analyzer():
    return FakeAnalyzer()


@pytest.fixture
def solid_swapper():
    return SolidSwapper()
