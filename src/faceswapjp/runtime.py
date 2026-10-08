"""onnxruntime execution provider selection (CoreML -> CUDA -> CPU)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import onnxruntime as ort

log = logging.getLogger(__name__)

DEVICE_PROVIDERS = {
    "coreml": "CoreMLExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
AUTO_ORDER = ("coreml", "cuda", "cpu")

ProviderSpec = str | tuple[str, dict[str, Any]]


def _with_options(name: str) -> ProviderSpec:
    if name == "CoreMLExecutionProvider":
        return (name, {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"})
    return name


def select_providers(device: str = "auto", available: list[str] | None = None) -> list[ProviderSpec]:
    """Return providers in priority order, always ending with CPU as fallback."""
    available = list(available if available is not None else ort.get_available_providers())
    if device == "auto":
        wanted = [DEVICE_PROVIDERS[d] for d in AUTO_ORDER]
    elif device in DEVICE_PROVIDERS:
        wanted = [DEVICE_PROVIDERS[device], "CPUExecutionProvider"]
    else:
        raise ValueError(f"unknown device {device!r}; choose from auto, {', '.join(DEVICE_PROVIDERS)}")
    chosen = [p for p in dict.fromkeys(wanted) if p in available]
    if device not in ("auto", "cpu") and DEVICE_PROVIDERS[device] not in available:
        log.warning("%s is not available; falling back to CPU", DEVICE_PROVIDERS[device])
    if "CPUExecutionProvider" not in chosen:
        chosen.append("CPUExecutionProvider")
    return [_with_options(p) for p in chosen]


def provider_names(providers: list[ProviderSpec]) -> list[str]:
    return [p if isinstance(p, str) else p[0] for p in providers]


def _coreml_model(path: Path) -> bytes | None:
    """Model bytes prepared for CoreML: symbolic batch dimensions pinned to 1 (CoreML only
    accelerates static shapes) and a COREML_CACHE_KEY so compiled models can be cached
    even though the model is passed as bytes. Returns None if the model can't be read."""
    import hashlib

    try:
        import onnx

        model = onnx.load(str(path))
    except Exception:  # noqa: BLE001
        return None
    for value in list(model.graph.input) + list(model.graph.output):
        dims = value.type.tensor_type.shape.dim
        if dims and dims[0].dim_param:
            dims[0].ClearField("dim_param")
            dims[0].dim_value = 1
    st = Path(path).stat()
    key = hashlib.sha1(f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}:static1".encode()).hexdigest()[:32]
    props = {p.key: p for p in model.metadata_props}
    if "COREML_CACHE_KEY" in props:
        props["COREML_CACHE_KEY"].value = key
    else:
        entry = model.metadata_props.add()
        entry.key, entry.value = "COREML_CACHE_KEY", key
    return model.SerializeToString()


def coreml_cache_dir(path: Path) -> Path:
    from .config import home_dir

    st = Path(path).stat()
    key = f"{Path(path).stem}-{st.st_size}-{int(st.st_mtime)}-ort{ort.__version__}"
    d = home_dir() / "coreml-cache" / key
    d.mkdir(parents=True, exist_ok=True)
    return d


def _attempts(path: Path, providers: list[ProviderSpec]) -> list[tuple[str, list[ProviderSpec]]]:
    """Provider configurations to try, best first."""
    names = provider_names(providers)
    if "CoreMLExecutionProvider" not in names:
        return [("default", providers)]
    rest = [p for p in providers if (p if isinstance(p, str) else p[0]) != "CoreMLExecutionProvider"]
    base = {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL", "RequireStaticInputShapes": "1"}
    attempts = []
    try:
        cached = {**base, "ModelCacheDirectory": str(coreml_cache_dir(path))}
        attempts.append(("coreml+cache", [("CoreMLExecutionProvider", cached), *rest]))
    except OSError:
        pass
    attempts.append(("coreml", [("CoreMLExecutionProvider", base), *rest]))
    attempts.append(("coreml-gpu", [("CoreMLExecutionProvider", {**base, "MLComputeUnits": "CPUAndGPU"}), *rest]))
    attempts.append(("cpu", ["CPUExecutionProvider"]))
    return attempts


def create_session(path: str | Path, providers: list[ProviderSpec]) -> ort.InferenceSession:
    """Create a session, trying the fastest provider configuration first.

    For CoreML the batch dimension is pinned to 1 and compiled models are cached on disk
    (so the second launch skips compilation). Any failure falls through to the next attempt,
    ending on CPU, so a missing feature never stops the tool from working.
    """
    path = Path(path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    use_coreml = "CoreMLExecutionProvider" in provider_names(providers)
    model: str | bytes = str(path)
    if use_coreml:
        model = _coreml_model(path) or str(path)
    last_exc: Exception | None = None
    for label, provs in _attempts(path, providers):
        try:
            session = ort.InferenceSession(model, sess_options=opts, providers=provs)
            log.info("%s: %s -> %s", path.name, label, session.get_providers()[0])
            return session
        except Exception as exc:  # noqa: BLE001 - provider-specific failures vary
            last_exc = exc
            log.warning("%s: session creation failed with %s (%s)", path.name, label, exc)
    raise RuntimeError(f"could not load {path.name}: {last_exc}")


def session_provider(session: ort.InferenceSession) -> str:
    return session.get_providers()[0].replace("ExecutionProvider", "")
