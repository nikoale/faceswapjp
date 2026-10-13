"""Build analyzer / swapper / occluders / enhancer / safety gate from model names,
enforcing the project's license policy."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config, swappers
from .analysis.insightface_backend import InsightFaceAnalyzer
from .models import registry
from .models.interfaces import Enhancer, FaceAnalyzer, Occluder, Swapper
from .pipeline.frame import FrameOptions, FrameProcessor
from .project import Project
from .runtime import ProviderSpec, provider_names, select_providers
from .safety.nsfw import NSFWClassifier, SafetyGate

# "mouth" keeps the inside of the mouth (teeth, tongue) from the original footage.
OCCLUDER_MODELS = {"occlusion": "xseg_1", "region": "bisenet_resnet_34", "mouth": "bisenet_resnet_34"}
ENHANCER_MODELS = {"gfpgan": "gfpgan_1_4"}
NSFW_MODEL = "open_nsfw"


@dataclass
class Engine:
    analyzer: FaceAnalyzer
    swapper: Swapper | None
    providers: list[ProviderSpec]
    analyzer_name: str
    swapper_name: str | None
    safety: SafetyGate
    occluders: list[Occluder] = field(default_factory=list)
    enhancer: Enhancer | None = None
    model_names: list[str] = field(default_factory=list)

    @property
    def provider_names(self) -> list[str]:
        return provider_names(self.providers)

    def frame_processor(self, options: FrameOptions | None = None) -> FrameProcessor:
        if self.swapper is None:
            raise RuntimeError("engine was built without a swapper")
        return FrameProcessor(self.analyzer, self.swapper, options, self.occluders, self.enhancer)


def check_project_licenses(project: Project | None, model_names: list[str]) -> None:
    if project is None:
        return
    for name in model_names:
        registry.check_license(
            registry.get_spec(name),
            commercial=project.settings.commercial,
            license_override=project.settings.license_overrides.get(name),
        )


def parse_masks(masks: str | list[str] | None) -> list[str]:
    """'box,occlusion,region' -> ['occlusion', 'region'] (box is always applied)."""
    if not masks:
        return []
    items = masks.split(",") if isinstance(masks, str) else list(masks)
    out = []
    for m in (x.strip() for x in items):
        if m in ("", "box"):
            continue
        if m not in OCCLUDER_MODELS:
            raise ValueError(f"unknown mask {m!r}; choose from box, {', '.join(OCCLUDER_MODELS)}")
        out.append(m)
    return out


def build_engine(
    project: Project | None = None,
    device: str = "auto",
    analyzer_name: str = config.DEFAULT_ANALYZER,
    swapper_name: str | None = config.DEFAULT_SWAPPER,
    det_size: tuple[int, int] = config.DEFAULT_DET_SIZE,
    masks: str | list[str] | None = None,
    enhancer: str | None = None,
    regions: tuple[str, ...] | None = None,
) -> Engine:
    mask_kinds = parse_masks(masks)
    if enhancer and enhancer not in ENHANCER_MODELS:
        raise ValueError(f"unknown enhancer {enhancer!r}; choose from {', '.join(ENHANCER_MODELS)}")
    names = [analyzer_name, NSFW_MODEL] + ([swapper_name] if swapper_name else [])
    names += [OCCLUDER_MODELS[m] for m in mask_kinds] + ([ENHANCER_MODELS[enhancer]] if enhancer else [])
    names = list(dict.fromkeys(names))
    check_project_licenses(project, names)
    paths = {n: registry.ensure(n) for n in names}  # fail early if anything is missing

    providers = select_providers(device)
    analyzer = InsightFaceAnalyzer(paths[analyzer_name], providers, det_size=det_size)
    swapper = swappers.create(swapper_name, paths[swapper_name], providers) if swapper_name else None
    occluders: list[Occluder] = []
    for m in mask_kinds:
        if m == "occlusion":
            from .occluders.onnx_occluders import XSegOccluder

            occluders.append(XSegOccluder(paths[OCCLUDER_MODELS[m]], providers))
        elif m == "region":
            from .occluders.onnx_occluders import DEFAULT_REGIONS, RegionOccluder

            keep = regions or DEFAULT_REGIONS
            if "mouth" in mask_kinds:  # one parser pass does both: just leave the mouth out
                keep = tuple(r for r in keep if r != "mouth")
            occluders.append(RegionOccluder(paths[OCCLUDER_MODELS[m]], providers, keep))
        elif m == "mouth" and "region" not in mask_kinds:
            from .occluders.onnx_occluders import RegionOccluder

            occluders.append(RegionOccluder(paths[OCCLUDER_MODELS[m]], providers, ("mouth",), invert=True))
    enh = None
    if enhancer:
        from .enhancers.gfpgan import GFPGANEnhancer

        enh = GFPGANEnhancer(paths[ENHANCER_MODELS[enhancer]], providers)
    safety = SafetyGate(NSFWClassifier(paths[NSFW_MODEL], providers))
    return Engine(analyzer, swapper, providers, analyzer_name, swapper_name, safety, occluders, enh, names)
