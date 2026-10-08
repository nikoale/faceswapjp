"""GFPGAN v1.4 face restoration (ONNX). License: see docs/MODEL_LICENSES.md (grey area)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime import ProviderSpec, create_session


class GFPGANEnhancer:
    name = "gfpgan_1.4"
    template = "ffhq_512"

    def __init__(self, model_path: Path, providers: list[ProviderSpec]):
        self._session = create_session(model_path, providers)
        inp = self._session.get_inputs()[0]
        self._input = inp.name
        self.input_size = int(inp.shape[2])

    def enhance(self, aligned_bgr: np.ndarray) -> np.ndarray:
        x = (aligned_bgr[..., ::-1] * 2.0 - 1.0).transpose(2, 0, 1)[None].astype(np.float32)
        y = self._session.run(None, {self._input: x})[0][0]
        return np.clip((y.transpose(1, 2, 0)[..., ::-1] + 1.0) / 2.0, 0.0, 1.0).astype(np.float32)
