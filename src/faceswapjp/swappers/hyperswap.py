"""FaceFusion HyperSwap 256 (ResearchRAIL, non-commercial; see docs/MODEL_LICENSES.md).

Same alignment as inswapper (arcface_128) but generates the face at 256 px directly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime import ProviderSpec, create_session


class HyperSwap:
    template = "arcface_128"

    def __init__(self, model_path: Path, providers: list[ProviderSpec], name: str = "hyperswap"):
        self.name = name
        self._session = create_session(model_path, providers)
        names = {i.name: i for i in self._session.get_inputs()}
        self.input_size = int(names["target"].shape[2])
        self._outputs = [o.name for o in self._session.get_outputs()]

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        emb = embedding.reshape(1, -1).astype(np.float32)
        return emb / np.linalg.norm(emb)

    def swap(self, aligned_bgr: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        blob = (aligned_bgr[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) - 0.5) / 0.5  # RGB, [-1,1]
        pred = self._session.run(self._outputs[:1], {"target": np.ascontiguousarray(blob),
                                                     "source": self.latent(source_embedding)})[0]
        return np.clip(pred[0].transpose(1, 2, 0)[..., ::-1] * 0.5 + 0.5, 0.0, 1.0).astype(np.float32)
