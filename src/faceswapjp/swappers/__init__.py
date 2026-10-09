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


def _hyperswap(name: str) -> SwapperFactory:
    def factory(path: Path, providers: list[ProviderSpec]) -> Swapper:
        from .hyperswap import HyperSwap

        return HyperSwap(path, providers, name)

    return factory


SWAPPERS: dict[str, SwapperFactory] = {
    "inswapper_128": _inswapper,
    **{f"hyperswap_{v}_256": _hyperswap(f"hyperswap_{v}_256") for v in ("1a", "1b", "1c")},
}


def register(name: str, factory: SwapperFactory) -> None:
    SWAPPERS[name] = factory


def create(name: str, path: Path, providers: list[ProviderSpec]) -> Swapper:
    if name not in SWAPPERS:
        raise KeyError(f"unknown swapper {name!r}; available: {', '.join(SWAPPERS)}")
    return SWAPPERS[name](path, providers)
