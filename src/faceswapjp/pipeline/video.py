"""Video swap: tracking + person selection + smoothing, encoded with source audio/timecode."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.face import Face, best_similarity
from ..media.ffpipe import FrameReader, FrameWriter
from ..media.presets import get_preset
from ..media.probe import FFmpegError, VideoInfo, offset_timecode, probe
from ..safety.provenance import AI_TAG, provenance_record
from ..safety.watermark import Watermark
from ..tracking.tracker import FaceTracker, Track
from .frame import FrameProcessor, FrameResult, to_detection_image

log = logging.getLogger(__name__)

SELECT_MODES = ("all", "largest", "reference")


@dataclass
class TrackingOptions:
    select: str = "all"  # all | largest | reference
    reference_threshold: float = 0.4
    smoothing: float = 0.5  # 0 = off, 1 = strongest
    coast_frames: int = 2  # keep swapping a target through short detection dropouts
    min_det_score: float = 0.5


class TrackedProcessor:
    """Frame-by-frame swapping with face tracks, so identity decisions and smoothing are temporal."""

    def __init__(
        self,
        processor: FrameProcessor,
        source_embedding: np.ndarray,
        fps: float,
        options: TrackingOptions | None = None,
        reference_embedding: np.ndarray | None = None,
    ):
        self.processor = processor
        self.source_embedding = source_embedding
        self.options = options or TrackingOptions()
        if self.options.select not in SELECT_MODES:
            raise ValueError(f"select must be one of {', '.join(SELECT_MODES)}")
        if self.options.select == "reference" and reference_embedding is None:
            raise ValueError("select='reference' needs a reference embedding (--only-person)")
        self.reference = reference_embedding
        self.dt = 1.0 / fps
        self.tracker = FaceTracker(smoothing=self.options.smoothing)

    def reset(self) -> None:
        self.tracker = FaceTracker(smoothing=self.options.smoothing)

    def _is_target(self, track: Track, pairs: list[tuple[Track, Face]]) -> bool:
        mode = self.options.select
        if mode == "all":
            return True
        if mode == "largest":
            return track is max(pairs, key=lambda p: p[1].area)[0]
        track.is_target = best_similarity(track.embedding, self.reference) >= self.options.reference_threshold
        return track.is_target

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        image = to_detection_image(frame)
        analyzer = self.processor.analyzer
        faces = [f for f in analyzer.detect(image, with_embedding=False) if f.det_score >= self.options.min_det_score]
        for face in faces:
            if self.tracker.needs_embedding(face, index):
                face.embedding = analyzer.embed(image, face)
        pairs = self.tracker.update(faces, index, self.dt)
        targets = [face for track, face in pairs if self._is_target(track, pairs)]
        if self.options.select != "largest":
            for track in self.tracker.coasting(self.options.coast_frames):
                if self.options.select == "all" or track.is_target:
                    targets.append(track.face)
        return self.processor.process(frame, self.source_embedding, faces=targets)


@dataclass
class RenderSettings:
    format: str = "h264"  # see media.presets.PRESETS
    encoder: str | None = None
    matte: str | None = None  # None | "luma" (ProRes 422 HQ) | "alpha" (ProRes 4444 fill + alpha)
    start: int = 0
    end: int | None = None  # exclusive
    watermark: Watermark | None = None


@dataclass
class RenderResult:
    output: Path
    matte_output: Path | None
    frames: int
    frames_with_faces: int
    seconds: float
    info: VideoInfo
    warnings: list[str] = field(default_factory=list)


ProgressFn = Callable[[int, int], None]


def matte_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_matte.mov")


def video_warnings(info: VideoInfo) -> list[str]:
    warnings = []
    if info.is_vfr:
        warnings.append(
            f"variable frame rate detected ({float(info.avg_fps):.3f} avg vs {float(info.fps):.3f} nominal); "
            "output is constant frame rate - conform the clip to CFR first for exact sync"
        )
    if info.is_interlaced:
        warnings.append(f"interlaced source ({info.field_order}); deinterlace before swapping for best results")
    if info.has_alpha:
        warnings.append("source alpha channel is not carried over")
    return warnings


def render_video(
    tracked: TrackedProcessor,
    source: Path,
    output: Path,
    settings: RenderSettings,
    provenance: dict[str, object] | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    info: VideoInfo | None = None,
) -> RenderResult:
    info = info or probe(source)
    preset = get_preset(settings.format)
    if Path(output).suffix.lower() != preset.ext:
        raise ValueError(f"{settings.format} output must use {preset.ext}")
    if preset.name == "h264" and (info.width % 2 or info.height % 2):
        raise ValueError("H.264 (4:2:0) needs even width/height; use prores422hq for this clip")
    warnings = video_warnings(info)
    for w in warnings:
        log.warning(w)

    start = max(0, settings.start)
    end = info.nb_frames if settings.end is None else min(settings.end, info.nb_frames)
    if end <= start:
        raise ValueError(f"empty frame range {start}..{end}")
    count = end - start
    fps = info.fps
    tc = offset_timecode(info.timecode, fps, start)
    record = provenance_record(**(provenance or {}), source=str(source), frame_range=[start, end])
    meta = {"comment": AI_TAG, "description": AI_TAG, "faceswapjp": json.dumps(record, ensure_ascii=False)}

    reader = FrameReader(info, start=start, count=count)
    pix = "bgr48le" if reader.depth == 16 else "bgr24"
    partial = start > 0 or end < info.nb_frames
    main = FrameWriter(
        output, info, preset, pix, encoder=settings.encoder, timecode=tc, audio_source=source,
        audio_start=float(start / fps) if partial else 0.0,
        audio_duration=float(count / fps) if partial else None,
        metadata=meta,
    )
    matte_out = None
    matte_writer = None
    if settings.matte:
        if settings.matte not in ("luma", "alpha"):
            raise ValueError("matte must be 'luma' or 'alpha'")
        matte_out = matte_path(Path(output))
        if settings.matte == "luma":
            matte_writer = FrameWriter(matte_out, info, get_preset("prores422hq"), "gray16le", timecode=tc,
                                       metadata=meta, full_range_out=False)
        else:
            matte_writer = FrameWriter(matte_out, info, get_preset("prores4444"), "bgra64le", timecode=tc,
                                       metadata=meta)

    tracked.reset()
    t0 = time.time()
    done = with_faces = 0
    with main:
        if matte_writer:
            matte_writer.__enter__()
        try:
            for i, frame in enumerate(reader):
                if cancel is not None and cancel.is_set():
                    raise InterruptedError("render cancelled")
                result = tracked.process(frame, start + i)
                if result.faces:
                    with_faces += 1
                out = result.frame
                if matte_writer:
                    matte16 = np.clip(np.rint(result.matte * 65535), 0, 65535).astype(np.uint16)
                    if settings.matte == "luma":
                        matte_writer.write(matte16)
                    else:
                        fill = out.astype(np.uint16) * 257 if out.dtype == np.uint8 else out
                        matte_writer.write(np.dstack([fill, matte16]))
                if settings.watermark is not None:
                    out = settings.watermark.apply(out)
                main.write(out)
                done += 1
                if progress:
                    progress(done, count)
        except BaseException as exc:
            if matte_writer:
                matte_writer.__exit__(type(exc), exc, None)
            raise
        if matte_writer:
            matte_writer.__exit__(None, None, None)

    if done != count:
        warnings.append(f"decoded {done} frames, expected {count}")
    out_info = probe(output)
    if out_info.nb_frames != done:
        warnings.append(f"output has {out_info.nb_frames} frames, wrote {done}")
    for w in warnings[len(video_warnings(info)):]:
        log.warning(w)
    return RenderResult(Path(output), matte_out, done, with_faces, time.time() - t0, info, warnings)


def preview_frame(
    processor: FrameProcessor,
    source_embedding: np.ndarray,
    info: VideoInfo,
    index: int,
    select: str = "all",
    reference: np.ndarray | None = None,
    reference_threshold: float = 0.4,
) -> tuple[np.ndarray, FrameResult]:
    """Process a single frame (no temporal smoothing). Returns (original, result)."""
    if not 0 <= index < max(1, info.nb_frames):
        raise ValueError(f"frame {index} is outside 0..{info.nb_frames - 1}")
    frame = FrameReader(info, start=index, count=1).read_one()
    faces = [f for f in processor.detect(frame) if f.det_score >= processor.options.min_det_score]
    if select == "largest":
        faces = sorted(faces, key=lambda f: f.area, reverse=True)[:1]
    elif select == "reference":
        if reference is None:
            raise ValueError("select='reference' needs a reference embedding")
        faces = [f for f in faces if best_similarity(f.embedding, reference) >= reference_threshold]
    return frame, processor.process(frame, source_embedding, faces=faces)


__all__ = [
    "FFmpegError", "RenderResult", "RenderSettings", "TrackedProcessor", "TrackingOptions",
    "matte_path", "preview_frame", "render_video",
]
