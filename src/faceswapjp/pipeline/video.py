"""Video swap: tracking + person selection + smoothing, encoded with source audio/timecode."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.face import Face, best_similarity
from ..media.ffpipe import FrameReader, FrameWriter, concat_segments
from ..media.presets import get_preset
from ..media.probe import FFmpegError, VideoInfo, offset_timecode, probe
from ..safety.provenance import AI_TAG, provenance_record
from ..safety.watermark import Watermark
from ..tracking.tracker import FaceTracker, Track, iou
from ..profiling import NULL, Profiler
from .frame import FrameProcessor, FrameResult, to_detection_image

log = logging.getLogger(__name__)

SELECT_MODES = ("all", "largest", "reference")

# A multi-person cast: for each person, (reference embeddings (N, 512), source embedding to use).
Cast = list[tuple[np.ndarray, np.ndarray]]


def cast_reference(cast: Cast) -> np.ndarray:
    return np.vstack([np.atleast_2d(refs) for refs, _ in cast])


def cast_member(embedding: np.ndarray | None, cast: Cast, threshold: float) -> int | None:
    """Index of the cast member this face is (most similar one above threshold), else None."""
    if embedding is None:
        return None
    sims = [best_similarity(embedding, refs) for refs, _ in cast]
    best = int(np.argmax(sims))
    return best if sims[best] >= threshold else None


@dataclass
class TrackingOptions:
    select: str = "all"  # all | largest | reference
    reference_threshold: float = 0.4
    smoothing: float = 0.5  # 0 = off, 1 = strongest
    coast_frames: int = 2  # keep swapping a target through short detection dropouts
    min_det_score: float = 0.5
    # Full-frame face detection every N frames; in between, known faces are re-detected in a
    # small crop around their last position (much cheaper). A target that is lost forces a
    # full detection right away. 1 = full detection on every frame.
    detect_every: int = 3


class TrackedProcessor:
    """Frame-by-frame swapping with face tracks, so identity decisions and smoothing are temporal."""

    def __init__(
        self,
        processor: FrameProcessor,
        source_embedding: np.ndarray,
        fps: float,
        options: TrackingOptions | None = None,
        reference_embedding: np.ndarray | None = None,
        cast: Cast | None = None,
    ):
        self.processor = processor
        self.cast = cast
        if cast:
            reference_embedding = cast_reference(cast)
        self.source_embedding = source_embedding
        self.options = options or TrackingOptions()
        if self.options.select not in SELECT_MODES:
            raise ValueError(f"select must be one of {', '.join(SELECT_MODES)}")
        if self.options.select == "reference" and reference_embedding is None:
            raise ValueError("select='reference' needs a reference embedding (--only-person)")
        self.reference = reference_embedding
        self.dt = 1.0 / fps
        self.profiler: Profiler = NULL
        self.reset()

    def reset(self) -> None:
        self.tracker = FaceTracker(smoothing=self.options.smoothing)
        self._last_full = -(10**9)
        self._windows: dict[int, tuple[int, int, int, int]] = {}  # track id -> sticky crop window
        self.processor.reset_history()

    def _window_for(self, track_id: int | None, bbox: np.ndarray, shape) -> tuple[int, int, int, int] | None:
        """Keep the crop still while the face stays comfortably inside it; re-center otherwise."""
        analyzer = self.processor.analyzer
        make = getattr(analyzer, "roi_window", None)
        if make is None:
            return None
        win = self._windows.get(track_id) if track_id is not None else None
        if win is not None:
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
            wx0, wy0, wx1, wy1 = win
            h, w = shape[:2]
            side, face = max(wx1 - wx0, wy1 - wy0), max(x1 - x0, y1 - y0)
            m = 0.18 * side
            # sides clamped to the image border need no margin (the face can't move past them)
            inside = ((wx0 == 0 or x0 > wx0 + m) and (wy0 == 0 or y0 > wy0 + m)
                      and (wx1 == w or x1 < wx1 - m) and (wy1 == h or y1 < wy1 - m))
            if inside and 0.3 < face / max(side, 1) < 0.55:
                return win
        win = make(bbox, shape)
        if track_id is not None:
            self._windows[track_id] = win
        return win

    def _track_at(self, bbox: np.ndarray) -> Track | None:
        best = max(self.tracker.tracks, key=lambda t: iou(t.face.bbox, bbox), default=None)
        return best if best is not None and iou(best.face.bbox, bbox) >= 0.3 else None

    def _wanted(self, track: Track) -> bool:
        return self.options.select != "reference" or bool(track.is_target)

    def _detect(self, image: np.ndarray, index: int) -> list[Face]:
        """Full-frame detection every `detect_every` frames. In between, only faces we are
        replacing are re-detected in a crop around their last position; if there are none,
        detection is skipped until the next scheduled full pass."""
        analyzer = self.processor.analyzer
        prof = self.profiler
        roi = getattr(analyzer, "detect_roi", None)
        scheduled = index - self._last_full >= max(1, self.options.detect_every)
        if roi is not None and not scheduled:
            wanted = [t for t in self.tracker.tracks if t.misses <= self.options.coast_frames and self._wanted(t)]
            faces = []
            with prof("detect_roi"):
                for track in wanted:
                    face = roi(image, track.face.bbox, self._window_for(track.id, track.face.bbox, image.shape))
                    if face is None:
                        break  # a face we are replacing slipped away: do a full pass now
                    faces.append(face)
                else:
                    return faces
        with prof("detect"):
            faces = analyzer.detect(image, with_embedding=False)
        self._last_full = index
        if roi is not None and self.options.detect_every > 1:
            # Landmarks from the crop detector differ slightly from full-frame ones; mixing the two
            # would make faces jitter every few frames. Faces we replace therefore always take
            # their landmarks from the crop detector, also on full-detection frames.
            with prof("detect_refine"):
                for face in faces:
                    if self._refine_wanted(face):
                        track = self._track_at(face.bbox)
                        refined = roi(image, face.bbox,
                                      self._window_for(track.id if track else None, face.bbox, image.shape))
                        if refined is not None:
                            face.bbox, face.kps = refined.bbox, refined.kps
        return faces

    def _refine_wanted(self, face: Face) -> bool:
        if self.options.select != "reference":
            return True
        track = self._track_at(face.bbox)
        return track is not None and bool(track.is_target)

    def _is_target(self, track: Track, pairs: list[tuple[Track, Face]]) -> bool:
        mode = self.options.select
        if mode == "all":
            return True
        if mode == "largest":
            return track is max(pairs, key=lambda p: p[1].area)[0]
        if self.cast:
            track.cast_index = cast_member(track.embedding, self.cast, self.options.reference_threshold)
            track.is_target = track.cast_index is not None
        else:
            track.is_target = best_similarity(track.embedding, self.reference) >= self.options.reference_threshold
        return track.is_target

    def process(self, frame: np.ndarray, index: int) -> FrameResult:
        prof = self.profiler
        self.processor.profiler = prof
        with prof("to_8bit"):
            image = to_detection_image(frame)
        analyzer = self.processor.analyzer
        faces = [f for f in self._detect(image, index) if f.det_score >= self.options.min_det_score]
        with prof("embed"):
            # identity only matters for "reference"; other modes just need it to re-find lost tracks
            settle, refresh = (2, 24) if self.options.select == "reference" else (1, 96)
            for face in faces:
                if self.tracker.needs_embedding(face, index, settle, refresh):
                    face.embedding = analyzer.embed(image, face)
        pairs = self.tracker.update(faces, index, self.dt)
        chosen = [(track, face) for track, face in pairs if self._is_target(track, pairs)]
        if self.options.select != "largest":
            for track in self.tracker.coasting(self.options.coast_frames):
                if self.options.select == "all" or track.is_target:
                    chosen.append((track, track.face))
        with prof("swap_total"):
            sources = [self.cast[t.cast_index][1] for t, _ in chosen] if self.cast else None
            return self.processor.process(frame, self.source_embedding, faces=[f for _, f in chosen],
                                          keys=[t.id for t, _ in chosen], sources=sources)


@dataclass
class RenderSettings:
    format: str = "h264"  # see media.presets.OUTPUT_FORMATS
    encoder: str | None = None  # None/auto, software, hardware, or an ffmpeg encoder name
    quality: str = "standard"  # high | standard | light (H.264 / H.265 only)
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
    timings: dict[str, dict[str, float]] = field(default_factory=dict)
    resumed_frames: int = 0  # frames taken over from an interrupted render

    @property
    def fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0


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


SEGMENT_SECONDS = 10.0


def segment_dir(output: Path) -> Path:
    """Where a resumable render keeps its finished segments (removed once the output is complete)."""
    return output.with_name(f".{output.name}.parts")


class _DirectSink:
    """Encode straight into the final files (one pass, not resumable)."""

    resumed_frames = 0
    resumed_with_faces = 0

    def __init__(self, main: FrameWriter, matte: FrameWriter | None):
        self.main, self.matte = main, matte

    def open(self) -> None:
        self.main.__enter__()
        if self.matte:
            self.matte.__enter__()

    def write(self, frame: np.ndarray, matte: np.ndarray | None, has_face: bool) -> None:
        self.main.write(frame)
        if self.matte is not None and matte is not None:
            self.matte.write(matte)

    def abort(self, exc: BaseException) -> None:
        if self.matte:
            self.matte.__exit__(type(exc), exc, None)
        self.main.__exit__(type(exc), exc, None)

    def finish(self, *_args) -> None:
        if self.matte:
            self.matte.__exit__(None, None, None)
        self.main.__exit__(None, None, None)


class _SegmentSink:
    """Encode into numbered segments; finished ones are recorded in state.json and survive a crash,
    cancel or sleep. finish() joins them (stream copy) and adds audio, timecode and metadata."""

    def __init__(self, parts: Path, key: str, seg_frames: int, make_writers):
        self.parts, self.key, self.seg_frames, self.make_writers = parts, key, seg_frames, make_writers
        self.state = self._load()
        self.cur: tuple[FrameWriter, FrameWriter | None] | None = None
        self.n = self.faces = 0

    def _load(self) -> dict:
        try:
            state = json.loads((self.parts / "state.json").read_text(encoding="utf-8"))
            if state.get("key") == self.key and state.get("seg_frames") == self.seg_frames:
                ok = all((self.parts / f"v{i:05d}.mov").is_file() for i in range(len(state["segments"])))
                if ok:
                    return state
        except (OSError, ValueError, KeyError, TypeError):
            pass
        shutil.rmtree(self.parts, ignore_errors=True)
        return {"key": self.key, "seg_frames": self.seg_frames, "segments": []}

    @property
    def resumed_frames(self) -> int:
        return sum(seg["frames"] for seg in self.state["segments"])

    @property
    def resumed_with_faces(self) -> int:
        return sum(seg["with_faces"] for seg in self.state["segments"])

    def _save(self) -> None:
        tmp = self.parts / "state.json.tmp"
        tmp.write_text(json.dumps(self.state), encoding="utf-8")
        tmp.replace(self.parts / "state.json")

    def open(self) -> None:
        self.parts.mkdir(parents=True, exist_ok=True)
        self._save()

    def write(self, frame: np.ndarray, matte: np.ndarray | None, has_face: bool) -> None:
        if self.cur is None:
            v, m = self.make_writers(len(self.state["segments"]), self.parts)
            v.__enter__()
            if m:
                m.__enter__()
            self.cur, self.n, self.faces = (v, m), 0, 0
        v, m = self.cur
        v.write(frame)
        if m is not None and matte is not None:
            m.write(matte)
        self.n += 1
        self.faces += int(has_face)
        if self.n >= self.seg_frames:
            self._close()

    def _close(self) -> None:
        v, m = self.cur
        self.cur = None
        v.__exit__(None, None, None)
        if m:
            m.__exit__(None, None, None)
        self.state["segments"].append({"frames": self.n, "with_faces": self.faces})
        self._save()

    def abort(self, exc: BaseException) -> None:
        """Drop the unfinished segment; finished ones stay for the next attempt."""
        if self.cur is not None:
            v, m = self.cur
            self.cur = None
            for w in (m, v):
                if w is not None:
                    try:
                        w.__exit__(type(exc), exc, None)
                    except Exception:  # noqa: BLE001 - already failing
                        pass
                    w.output.unlink(missing_ok=True)

    def finish(self, output: Path, matte_out: Path | None, info: VideoInfo, preset, matte_preset, tc, source,
               audio_start: float, audio_duration: float | None, meta: dict[str, str]) -> None:
        if self.cur is not None:
            self._close()
        n = len(self.state["segments"])
        concat_segments([self.parts / f"v{i:05d}.mov" for i in range(n)], output, info, preset, tc,
                        audio_source=source, audio_start=audio_start, audio_duration=audio_duration, metadata=meta)
        if matte_out is not None:
            concat_segments([self.parts / f"m{i:05d}.mov" for i in range(n)], matte_out, info, matte_preset, tc,
                            metadata=meta)
        shutil.rmtree(self.parts, ignore_errors=True)


def render_video(
    tracked: TrackedProcessor,
    source: Path,
    output: Path,
    settings: RenderSettings,
    provenance: dict[str, object] | None = None,
    progress: ProgressFn | None = None,
    cancel: threading.Event | None = None,
    info: VideoInfo | None = None,
    on_frame: Callable[[int, np.ndarray], None] | None = None,
    resume_key: str | None = None,
    on_resume: Callable[[int, int], None] | None = None,
) -> RenderResult:
    """on_frame(index, frame) receives each finished frame (e.g. for a live preview); keep it cheap.

    resume_key: render in ~10 s segments next to the output and keep finished ones, so an
    interrupted render with the same key (same source and settings) continues where it stopped.
    on_resume(frames_already_done, total) is called when that happens."""
    info = info or probe(source)
    preset = get_preset(settings.format)
    if Path(output).suffix.lower() != preset.ext:
        raise ValueError(f"{settings.format} output must use {preset.ext}")
    if preset.group == "internal":
        raise ValueError(f"{settings.format} is not an output format")
    if preset.even_dims and (info.width % 2 or info.height % 2):
        raise ValueError(f"{settings.format} (4:2:0) needs even width/height; use a ProRes or DNxHR format for this clip")
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

    probe_depth = FrameReader(info, start=start, count=count).depth
    pix = "bgr48le" if probe_depth == 16 else "bgr24"
    partial = start > 0 or end < info.nb_frames
    audio_start = float(start / fps) if partial else 0.0
    audio_duration = float(count / fps) if partial else None
    matte_out = None
    matte_preset = None
    if settings.matte:
        if settings.matte not in ("luma", "alpha"):
            raise ValueError("matte must be 'luma' or 'alpha'")
        matte_out = matte_path(Path(output))
        matte_preset = get_preset("prores422hq" if settings.matte == "luma" else "prores4444_alpha")
    matte_pix = "gray16le" if settings.matte == "luma" else "bgra64le"
    matte_range = False if settings.matte == "luma" else None

    if resume_key is None:
        main = FrameWriter(output, info, preset, pix, encoder=settings.encoder, quality=settings.quality, timecode=tc,
                           audio_source=source, audio_start=audio_start, audio_duration=audio_duration,
                           metadata=meta)
        matte_writer = (FrameWriter(matte_out, info, matte_preset, matte_pix, timecode=tc, metadata=meta,
                                    full_range_out=matte_range) if settings.matte else None)
        sink: _DirectSink | _SegmentSink = _DirectSink(main, matte_writer)
    else:
        def make_writers(index: int, parts: Path):
            v = FrameWriter(parts / f"v{index:05d}.mov", info, preset, pix, encoder=settings.encoder,
                            quality=settings.quality)
            m = (FrameWriter(parts / f"m{index:05d}.mov", info, matte_preset, matte_pix, full_range_out=matte_range)
                 if settings.matte else None)
            return v, m

        seg_frames = max(1, int(round(float(fps) * SEGMENT_SECONDS)))
        key = f"{resume_key}|{settings.format}|{settings.encoder}|{settings.quality}|{settings.matte}|{start}|{end}"
        sink = _SegmentSink(segment_dir(Path(output)), key, seg_frames, make_writers)

    resumed = sink.resumed_frames
    if resumed:
        log.info("resuming %s at frame %d of %d", output, resumed, count)
        if on_resume is not None:
            on_resume(resumed, count)
    tracked.reset()
    prof = Profiler()
    tracked.profiler = prof
    t0 = time.time()
    done, with_faces = resumed, sink.resumed_with_faces
    remaining = count - resumed
    reader = FrameReader(info, start=start + resumed, count=remaining) if remaining > 0 else None

    # Encoding runs on its own thread so ffmpeg I/O overlaps with inference.
    out_q: queue.Queue = queue.Queue(maxsize=6)
    writer_error: list[BaseException] = []

    def writer_loop() -> None:
        try:
            while (item := out_q.get()) is not None:
                sink.write(*item)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            writer_error.append(exc)
            while out_q.get() is not None:  # keep draining so the producer never blocks
                pass

    def enqueue(item) -> None:
        t_put = time.perf_counter()
        while True:
            if writer_error:
                raise writer_error[0]
            try:
                out_q.put(item, timeout=0.5)
                break
            except queue.Full:
                continue
        prof.add("wait_encoder", time.perf_counter() - t_put)

    sink.open()
    writer = threading.Thread(target=writer_loop, name="fsj-encode", daemon=True)
    writer.start()
    try:
        frames = iter(reader) if reader is not None else iter(())
        i = resumed
        while True:
            t_read = time.perf_counter()
            frame = next(frames, None)
            prof.add("wait_decoder", time.perf_counter() - t_read)
            if frame is None:
                break
            if cancel is not None and cancel.is_set():
                raise InterruptedError("render cancelled")
            result = tracked.process(frame, start + i)
            if result.faces:
                with_faces += 1
            out = result.frame
            matte_frame = None
            if settings.matte:
                with prof("matte"):
                    matte16 = np.clip(np.rint(result.matte * 65535), 0, 65535).astype(np.uint16)
                    if settings.matte == "luma":
                        matte_frame = matte16
                    else:
                        fill = out.astype(np.uint16) * 257 if out.dtype == np.uint8 else out
                        matte_frame = np.dstack([fill, matte16])
            if settings.watermark is not None:
                out = settings.watermark.apply(out)
            if on_frame is not None:
                on_frame(start + i, out)
            enqueue((out, matte_frame, bool(result.faces)))
            done += 1
            i += 1
            if progress:
                progress(done, count)
        out_q.put(None)
        writer.join()
        if writer_error:
            raise writer_error[0]
    except BaseException as exc:
        if writer.is_alive():
            try:
                out_q.put_nowait(None)
            except queue.Full:
                pass
            writer.join(5)
        sink.abort(exc)
        raise
    sink.finish(output, matte_out, info, preset, matte_preset, tc, source, audio_start, audio_duration, meta)

    if done != count:
        warnings.append(f"decoded {done} frames, expected {count}")
    out_info = probe(output)
    if out_info.nb_frames != done:
        warnings.append(f"output has {out_info.nb_frames} frames, wrote {done}")
    for w in warnings[len(video_warnings(info)):]:
        log.warning(w)
    return RenderResult(Path(output), matte_out, done, with_faces, time.time() - t0, info, warnings,
                        prof.report(max(1, done - resumed)), resumed_frames=resumed)


def preview_frame(
    processor: FrameProcessor,
    source_embedding: np.ndarray,
    info: VideoInfo,
    index: int,
    select: str = "all",
    reference: np.ndarray | None = None,
    reference_threshold: float = 0.4,
    cast: Cast | None = None,
) -> tuple[np.ndarray, FrameResult]:
    """Process a single frame (no temporal smoothing). Returns (original, result)."""
    if not 0 <= index < max(1, info.nb_frames):
        raise ValueError(f"frame {index} is outside 0..{info.nb_frames - 1}")
    frame = FrameReader(info, start=index, count=1).read_one()
    faces = [f for f in processor.detect(frame) if f.det_score >= processor.options.min_det_score]
    if select == "largest":
        faces = sorted(faces, key=lambda f: f.area, reverse=True)[:1]
    elif cast:
        members = [(f, cast_member(f.embedding, cast, reference_threshold)) for f in faces]
        members = [(f, m) for f, m in members if m is not None]
        return frame, processor.process(frame, source_embedding, faces=[f for f, _ in members],
                                        sources=[cast[m][1] for _, m in members])
    elif select == "reference":
        if reference is None:
            raise ValueError("select='reference' needs a reference embedding")
        faces = [f for f in faces if best_similarity(f.embedding, reference) >= reference_threshold]
    return frame, processor.process(frame, source_embedding, faces=faces)


__all__ = [
    "FFmpegError", "RenderResult", "RenderSettings", "TrackedProcessor", "TrackingOptions",
    "matte_path", "preview_frame", "render_video",
]
