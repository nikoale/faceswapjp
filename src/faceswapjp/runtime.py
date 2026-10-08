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


def create_session(path: str | Path, providers: list[ProviderSpec]) -> ort.InferenceSession:
    """Create a session; if an accelerated provider fails to compile the model, retry on CPU."""
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    try:
        return ort.InferenceSession(str(path), sess_options=opts, providers=providers)
    except Exception as exc:  # noqa: BLE001 - provider-specific failures vary
        if provider_names(providers) == ["CPUExecutionProvider"]:
            raise
        log.warning("session creation failed with %s (%s); retrying on CPU", provider_names(providers), exc)
        return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
