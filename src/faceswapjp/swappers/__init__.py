"""Swapper plugins: name -> factory(model_path, providers)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from ..models.interfaces import Swapper
from ..runtime import ProviderSpec

SwapperFactory = Callable[[Path, list[ProviderSpec]], Swapper]


def _inswapper(path: Path, providers: list[ProviderSpec]) -> Swapper:
    from .inswapper import InSwapper

    return InSwapper(path, providers)


SWAPPERS: dict[str, SwapperFactory] = {"inswapper_128": _inswapper}


def register(name: str, factory: SwapperFactory) -> None:
    SWAPPERS[name] = factory


def create(name: str, path: Path, providers: list[ProviderSpec]) -> Swapper:
    if name not in SWAPPERS:
        raise KeyError(f"unknown swapper {name!r}; available: {', '.join(SWAPPERS)}")
    return SWAPPERS[name](path, providers)
