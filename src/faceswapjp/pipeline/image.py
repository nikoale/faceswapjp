"""Still image swap."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..analysis.face import Face
from ..imageio import encode_image, read_image
from ..safety.provenance import provenance_record, tag_image_bytes
from ..safety.watermark import Watermark
from .frame import FrameProcessor, FrameResult


def swap_image(
    processor: FrameProcessor,
    source_embedding: np.ndarray,
    target: Path | np.ndarray,
    output: Path,
    provenance: dict[str, object] | None = None,
    matte_output: Path | None = None,
    faces: list[Face] | None = None,
    watermark: Watermark | None = None,
) -> FrameResult:
    frame = read_image(target) if not isinstance(target, np.ndarray) else target
    result = processor.process(frame, source_embedding, faces=faces)
    if not result.faces:
        raise ValueError(f"no target face found in {target if not isinstance(target, np.ndarray) else 'image'}")
    out = watermark.apply(result.frame.copy()) if watermark is not None else result.frame
    record = provenance_record(**(provenance or {}))
    ext = Path(output).suffix.lower()
    data = tag_image_bytes(encode_image(out, ext), ext, record)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_bytes(data)
    if matte_output is not None:
        matte16 = np.clip(np.rint(result.matte * 65535), 0, 65535).astype(np.uint16)
        Path(matte_output).write_bytes(encode_image(matte16, Path(matte_output).suffix.lower() or ".png"))
    return result
