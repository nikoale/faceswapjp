"""End-to-end with real models. Skipped unless the model files are present.

Set FACESWAPJP_TEST_SOURCE and FACESWAPJP_TEST_TARGET to images you have the right to use
(e.g. your own photos or AI-generated faces).
"""

import os
from pathlib import Path

import numpy as np
import pytest

from faceswapjp.models import registry

pytestmark = pytest.mark.models

SOURCE = os.environ.get("FACESWAPJP_TEST_SOURCE")
TARGET = os.environ.get("FACESWAPJP_TEST_TARGET")


def _models_ready():
    return not any(registry.verify(registry.get_spec(n)) for n in ("buffalo_l", "inswapper_128"))


@pytest.mark.skipif(not (SOURCE and TARGET), reason="FACESWAPJP_TEST_SOURCE/TARGET not set")
@pytest.mark.skipif(not _models_ready(), reason="models not downloaded")
def test_identity_transfers(tmp_path):
    from faceswapjp.engine import build_engine
    from faceswapjp.identity import embed_references
    from faceswapjp.imageio import read_image
    from faceswapjp.pipeline.frame import FrameOptions, FrameProcessor, to_detection_image
    from faceswapjp.pipeline.image import swap_image

    engine = build_engine(device="cpu")
    src_emb, _ = embed_references(engine.analyzer, [Path(SOURCE)])
    out = tmp_path / "out.png"
    proc = FrameProcessor(engine.analyzer, engine.swapper, FrameOptions(faces="largest"))
    result = swap_image(proc, src_emb, Path(TARGET), out)
    before = float(result.faces[0].embedding @ src_emb)
    faces = engine.analyzer.detect(to_detection_image(read_image(out)))
    after = max(float(f.embedding @ src_emb) for f in faces)
    assert after > before + 0.3 and after > 0.5
