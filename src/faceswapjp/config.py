"""Paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path


def home_dir() -> Path:
    return Path(os.environ.get("FACESWAPJP_HOME", "~/.faceswapjp")).expanduser()


def models_dir() -> Path:
    env = os.environ.get("FACESWAPJP_MODELS_DIR")
    return Path(env).expanduser() if env else home_dir() / "models"


DEFAULT_ANALYZER = "buffalo_l"
DEFAULT_SWAPPER = "inswapper_128"
DEFAULT_DET_SIZE = (640, 640)
