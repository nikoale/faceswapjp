"""High-level operations shared by the CLI and the Gradio UI.

Every entry point runs the same mandatory steps: license check, NSFW gate,
provenance metadata and a consent-log entry.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .analysis.face import Face, best_similarity
from .engine import Engine, build_engine
from .identity import ConsentInfo, Identity, embed_references, load_identity, register_identity
from .imageio import SUPPORTED_EXT as IMAGE_EXT
from .imageio import read_image
from .media.probe import probe
from .models.registry import sha256_file
from .pipeline.frame import FrameOptions, FrameResult, to_detection_image
from .pipeline.image import swap_image
from .pipeline.video import RenderResult, RenderSettings, TrackedProcessor, TrackingOptions, preview_frame, render_video
from .project import Project

log = logging.getLogger(__name__)

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".mxf", ".mkv", ".avi", ".webm", ".mts", ".m2ts"}


@dataclass
class SwapJob:
    identity_id: str
    frame: FrameOptions = field(default_factory=FrameOptions)
    tracking: TrackingOptions = field(default_factory=TrackingOptions)
    render: RenderSettings = field(default_factory=RenderSettings)
    reference_images: list[Path] = field(default_factory=list)  # "only replace this person"
    reference_embeddings: np.ndarray | None = None  # (N, 512): people picked in the UI
    masks: str = "box"
    enhancer: str | None = None
    device: str = "auto"

    def __post_init__(self):
        if (self.reference_images or self.reference_embeddings is not None) and self.tracking.select != "reference":
            self.tracking.select = "reference"

    def settings_record(self) -> dict:
        r = asdict(self.render)
        r["reference_people"] = 0 if self.reference_embeddings is None else len(np.atleast_2d(self.reference_embeddings))
        r.pop("watermark", None)
        return {
            "frame": asdict(self.frame), "tracking": asdict(self.tracking), "render": r,
            "masks": self.masks, "enhancer": self.enhancer, "watermark": self.render.watermark is not None,
        }


_ENGINES: dict[tuple, Engine] = {}
_ENGINE_LOCK = threading.Lock()


def get_engine(project: Project | None, device: str = "auto", masks: str = "box", enhancer: str | None = None,
               with_swapper: bool = True) -> Engine:
    """Engines are expensive to load, so they are cached per configuration (license check runs every time)."""
    from .engine import check_project_licenses

    key = (device, masks, enhancer, with_swapper)
    with _ENGINE_LOCK:
        if key not in _ENGINES:
            _ENGINES[key] = build_engine(project, device=device, masks=masks, enhancer=enhancer,
                                         swapper_name=None if not with_swapper else "inswapper_128")
        engine = _ENGINES[key]
    check_project_licenses(project, engine.model_names)
    return engine


def add_identity(project: Project, label: str, images: list[Path], consent: ConsentInfo, device: str = "auto",
                 identity_id: str | None = None) -> Identity:
    engine = get_engine(project, device, with_swapper=False)
    engine.safety.check_images(images)
    return register_identity(project, engine.analyzer, label, images, consent, engine.analyzer_name, identity_id)


def _reference(engine: Engine, job: SwapJob) -> np.ndarray | None:
    """(N, 512) reference embeddings from picked faces and/or reference images."""
    refs = []
    if job.reference_embeddings is not None:
        refs.append(np.atleast_2d(job.reference_embeddings))
    if job.reference_images:
        engine.safety.check_images(job.reference_images)
        emb, _ = embed_references(engine.analyzer, job.reference_images)
        refs.append(emb[None])
    return np.vstack(refs) if refs else None


def _models(engine: Engine) -> list[str]:
    return list(engine.model_names)


def is_video(path: Path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXT


def select_still_faces(processor, frame: np.ndarray, job: SwapJob, reference: np.ndarray | None):
    faces = [f for f in processor.detect(frame) if f.det_score >= job.frame.min_det_score]
    if reference is not None:
        faces = [f for f in faces if best_similarity(f.embedding, reference) >= job.tracking.reference_threshold]
    elif job.tracking.select == "largest" or job.frame.faces == "largest":
        faces = sorted(faces, key=lambda f: f.area, reverse=True)[:1]
    return faces


def swap_still(project: Project, job: SwapJob, target: Path, output: Path, matte: Path | None = None) -> FrameResult:
    engine = get_engine(project, job.device, job.masks, job.enhancer)
    identity, embedding = load_identity(project, job.identity_id)
    frame = read_image(target)
    engine.safety.check_frames([(None, to_detection_image(frame))], str(target))
    processor = engine.frame_processor(job.frame)
    faces = select_still_faces(processor, frame, job, _reference(engine, job))
    if not faces:
        raise ValueError(f"no target face found in {target}")
    result = swap_image(processor, embedding, frame, output,
                        provenance={"models": _models(engine), "identity": identity.id, "source": str(target)},
                        matte_output=matte, faces=faces, watermark=job.render.watermark)
    _log_render(project, "image", identity, engine, job, target, output, matte, faces=len(result.faces))
    return result


def preview(project: Project, job: SwapJob, target: Path, frame_index: int = 0) -> tuple[np.ndarray, FrameResult]:
    """Swap one frame (or a still) without writing an output. Returns (original, result)."""
    engine = get_engine(project, job.device, job.masks, job.enhancer)
    _, embedding = load_identity(project, job.identity_id)
    processor = engine.frame_processor(job.frame)
    reference = _reference(engine, job)
    if is_video(target):
        info = probe(target)
        original, result = preview_frame(processor, embedding, info, frame_index, job.tracking.select, reference,
                                         job.tracking.reference_threshold)
    else:
        original = read_image(target)
        result = processor.process(original, embedding, faces=select_still_faces(processor, original, job, reference))
    engine.safety.check_frames([(frame_index, to_detection_image(original))], str(target))
    if job.render.watermark is not None:
        job.render.watermark.apply(result.frame)
    return original, result


def find_faces(project: Project | None, target: Path, frame_index: int = 0, device: str = "auto") -> tuple[np.ndarray, list[Face]]:
    """Read one frame (or a still) and detect faces with embeddings, for picking who to replace."""
    engine = get_engine(project, device, with_swapper=False)
    if is_video(target):
        from .media.ffpipe import FrameReader

        info = probe(target)
        frame = FrameReader(info, start=min(max(0, frame_index), max(0, info.nb_frames - 1)), count=1).read_one()
    else:
        frame = read_image(target)
    image = to_detection_image(frame)
    engine.safety.check_frames([(frame_index, image)], str(target))
    faces = [f for f in engine.analyzer.detect(image) if f.det_score >= 0.5]
    faces.sort(key=lambda f: f.bbox[0])
    return frame, faces


def swap_video(project: Project, job: SwapJob, target: Path, output: Path, progress=None,
               cancel: threading.Event | None = None) -> RenderResult:
    engine = get_engine(project, job.device, job.masks, job.enhancer)
    identity, embedding = load_identity(project, job.identity_id)
    info = probe(target)
    end = job.render.end if job.render.end is not None else info.nb_frames
    checked = engine.safety.check_video(info, start=job.render.start, count=max(1, end - job.render.start))
    log.info("NSFW gate: %d frames checked", checked)
    reference = _reference(engine, job)
    tracked = TrackedProcessor(engine.frame_processor(job.frame), embedding, float(info.fps), job.tracking, reference)
    result = render_video(tracked, target, output, job.render,
                          provenance={"models": _models(engine), "identity": identity.id},
                          progress=progress, cancel=cancel, info=info)
    _log_render(project, "video", identity, engine, job, target, output, result.matte_output,
                frames=result.frames, frames_with_faces=result.frames_with_faces, warnings=result.warnings,
                nsfw_frames_checked=checked)
    return result


@dataclass
class BatchItem:
    source: str
    output: str | None
    ok: bool
    error: str | None = None
    frames: int | None = None
    warnings: list[str] = field(default_factory=list)


def batch(project: Project, job: SwapJob, input_dir: Path, output_dir: Path, progress=None,
          cancel: threading.Event | None = None, include_images: bool = True) -> list[BatchItem]:
    """Process every clip (and still) in a folder. Failures are recorded and skipped."""
    from .media.presets import get_preset

    files = sorted(p for p in Path(input_dir).iterdir()
                   if p.is_file() and (is_video(p) or (include_images and p.suffix.lower() in IMAGE_EXT)))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[BatchItem] = []
    for n, src in enumerate(files):
        if cancel is not None and cancel.is_set():
            break
        try:
            if is_video(src):
                out = output_dir / f"{src.stem}_swap{get_preset(job.render.format).ext}"
                res = swap_video(project, job, src, out,
                                 progress=(lambda d, t, n=n: progress(n, len(files), d, t)) if progress else None,
                                 cancel=cancel)
                report.append(BatchItem(str(src), str(out), True, frames=res.frames, warnings=res.warnings))
            else:
                out = output_dir / f"{src.stem}_swap.png"
                matte = output_dir / f"{src.stem}_swap_matte.png" if job.render.matte else None
                swap_still(project, job, src, out, matte)
                report.append(BatchItem(str(src), str(out), True, frames=1))
        except Exception as exc:  # noqa: BLE001 - one bad clip must not stop the batch
            if isinstance(exc, InterruptedError):
                raise
            log.error("batch: %s failed: %s", src, exc)
            report.append(BatchItem(str(src), None, False, error=f"{type(exc).__name__}: {exc}"))
    (output_dir / "batch_report.json").write_text(
        json.dumps([asdict(r) for r in report], ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _log_render(project: Project, kind: str, identity: Identity, engine: Engine, job: SwapJob, target: Path,
                output: Path, matte: Path | None, **extra) -> None:
    project.consent_log.append(
        "render",
        kind=kind,
        identity_id=identity.id,
        person_name=identity.consent.person_name,
        target=str(Path(target).resolve()),
        target_sha256=sha256_file(Path(target)),
        output=str(Path(output).resolve()),
        output_sha256=sha256_file(Path(output)),
        matte=str(Path(matte).resolve()) if matte else None,
        models=_models(engine),
        providers=engine.provider_names,
        settings=job.settings_record(),
        **extra,
    )
