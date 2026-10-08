"""NSFW gate (Yahoo open_nsfw, ONNX). Sexual content stops processing; there is no override switch."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..runtime import ProviderSpec, create_session

# open_nsfw scores above this are treated as sexual content. Fixed on purpose.
NSFW_THRESHOLD = 0.8


class NSFWContentError(RuntimeError):
    pass


@dataclass
class NSFWHit:
    source: str
    frame: int | None
    score: float


class NSFWClassifier:
    name = "open_nsfw"
    _mean = np.array([104.0, 117.0, 123.0], np.float32)  # BGR, Caffe convention

    def __init__(self, model_path: Path, providers: list[ProviderSpec]):
        self._session = create_session(model_path, providers)
        self._input = self._session.get_inputs()[0].name

    def score(self, image_bgr8: np.ndarray) -> float:
        """Probability of NSFW content for a uint8 BGR image."""
        x = cv2.resize(image_bgr8[..., :3], (256, 256), interpolation=cv2.INTER_LINEAR)[16:240, 16:240]
        x = (x.astype(np.float32) - self._mean)[None]
        return float(self._session.run(None, {self._input: x})[0][0][1])


class SafetyGate:
    def __init__(self, classifier: NSFWClassifier, threshold: float = NSFW_THRESHOLD):
        self.classifier = classifier
        self.threshold = threshold

    def check_frames(self, frames: Iterable[tuple[int | None, np.ndarray]], source: str) -> None:
        for index, frame in frames:
            s = self.classifier.score(frame)
            if s >= self.threshold:
                where = f" (frame {index})" if index is not None else ""
                raise NSFWContentError(f"sexual content detected in {source}{where}, score {s:.2f}; processing stopped")

    def check_images(self, paths: Iterable[Path]) -> None:
        from ..imageio import read_image
        from ..pipeline.frame import to_detection_image

        for p in paths:
            self.check_frames([(None, to_detection_image(read_image(p)))], str(p))

    def check_video(self, info, start: int = 0, count: int | None = None, every_seconds: float = 1.0) -> int:
        """Check one frame per `every_seconds` plus scene cuts. Returns the number of frames checked."""
        from ..media.ffpipe import FrameReader

        step = max(1, round(float(info.fps) * every_seconds))
        reader = FrameReader(info, start=start, count=count, depth=8, select=f"not(mod(n\\,{step}))+gt(scene\\,0.4)")
        checked = 0

        def frames():
            nonlocal checked
            for i, f in enumerate(reader):
                checked += 1
                yield None, f  # select drops frames, so the index is approximate; report the source only

        self.check_frames(frames(), str(info.path))
        return checked
