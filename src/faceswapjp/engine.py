"""Build analyzer + swapper from model names, enforcing project license policy."""

from __future__ import annotations

from dataclasses import dataclass

from . import config, swappers
from .analysis.insightface_backend import InsightFaceAnalyzer
from .models import registry
from .models.interfaces import FaceAnalyzer, Swapper
from .project import Project
from .runtime import ProviderSpec, provider_names, select_providers


@dataclass
class Engine:
    analyzer: FaceAnalyzer
    swapper: Swapper | None
    providers: list[ProviderSpec]
    analyzer_name: str
    swapper_name: str | None

    @property
    def provider_names(self) -> list[str]:
        return provider_names(self.providers)


def check_project_licenses(project: Project | None, model_names: list[str]) -> None:
    if project is None:
        return
    for name in model_names:
        registry.check_license(
            registry.get_spec(name),
            commercial=project.settings.commercial,
            license_override=project.settings.license_overrides.get(name),
        )


def build_engine(
    project: Project | None = None,
    device: str = "auto",
    analyzer_name: str = config.DEFAULT_ANALYZER,
    swapper_name: str | None = config.DEFAULT_SWAPPER,
    det_size: tuple[int, int] = config.DEFAULT_DET_SIZE,
) -> Engine:
    names = [analyzer_name] + ([swapper_name] if swapper_name else [])
    check_project_licenses(project, names)
    providers = select_providers(device)
    analyzer = InsightFaceAnalyzer(registry.ensure(analyzer_name), providers, det_size=det_size)
    swapper = swappers.create(swapper_name, registry.ensure(swapper_name), providers) if swapper_name else None
    return Engine(analyzer, swapper, providers, analyzer_name, swapper_name)
