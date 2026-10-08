"""InsightFace inswapper_128 (non-commercial license; see docs/MODEL_LICENSES.md)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..runtime import ProviderSpec, create_session


class InSwapper:
    name = "inswapper_128"
    template = "arcface_128"

    def __init__(self, model_path: Path, providers: list[ProviderSpec]):
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(model_path))
        # The last initializer maps ArcFace embeddings into the swapper's latent space.
        self._emap = numpy_helper.to_array(model.graph.initializer[-1]).astype(np.float32)
        del model
        self._session = create_session(model_path, providers)
        inputs = self._session.get_inputs()
        self._target_name, self._source_name = inputs[0].name, inputs[1].name
        self._output_name = self._session.get_outputs()[0].name
        self.input_size = int(inputs[0].shape[2])

    def latent(self, embedding: np.ndarray) -> np.ndarray:
        lat = embedding.reshape(1, -1).astype(np.float32) @ self._emap
        return lat / np.linalg.norm(lat)

    def swap(self, aligned_bgr: np.ndarray, source_embedding: np.ndarray) -> np.ndarray:
        blob = aligned_bgr[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32)  # BGR->RGB, NCHW, [0,1]
        pred = self._session.run(
            [self._output_name],
            {self._target_name: np.ascontiguousarray(blob), self._source_name: self.latent(source_embedding)},
        )[0]
        return np.clip(pred[0].transpose(1, 2, 0)[..., ::-1], 0.0, 1.0).astype(np.float32)
