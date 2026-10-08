"""Raw frame pipes to/from ffmpeg."""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np

from .presets import QUALITIES, Preset, choose_encoder
from .probe import FFmpegError, VideoInfo, ffmpeg_bin

log = logging.getLogger(__name__)

SWS_FLAGS = "spline+accurate_rnd+full_chroma_int+full_chroma_inp"


def _range(full: bool) -> str:
    return "pc" if full else "tv"


class _StderrDrain(threading.Thread):
    def __init__(self, stream):
        super().__init__(daemon=True)
        self.stream, self.lines = stream, []

    def run(self):
        for line in iter(self.stream.readline, b""):
            self.lines.append(line.decode("utf-8", "replace").rstrip())
            del self.lines[:-50]

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


class FrameReader:
    """Decode frames as BGR uint8 (8-bit sources) or uint16 (deeper sources)."""

    def __init__(
        self,
        info: VideoInfo,
        start: int = 0,
        count: int | None = None,
        depth: int | None = None,
        prefetch: int = 4,
        select: str | None = None,
    ):
        self.info = info
        self.select = select
        self.start = start
        self.count = count
        self.depth = depth or (16 if info.bit_depth > 8 else 8)
        self.dtype = np.uint16 if self.depth == 16 else np.uint8
        self.pix_fmt = "bgr48le" if self.depth == 16 else "bgr24"
        self.prefetch = prefetch

    def command(self) -> list[str]:
        info = self.info
        cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error"]
        if self.start:
            cmd += ["-ss", f"{float(Fraction(self.start) / info.fps):.6f}"]
        cmd += ["-i", str(info.path), "-map", "0:v:0", "-an", "-sn", "-dn", "-fps_mode", "passthrough"]
        vf = f"scale=in_color_matrix={info.matrix}:in_range={_range(info.is_full_range)}"
        if self.count is not None:
            # limit by input frame count (before any select), so ranges stay exact
            vf = f"trim=end_frame={self.count}," + vf
        if self.select:
            vf = f"{vf},select='{self.select}'"
        cmd += ["-vf", vf, "-sws_flags", SWS_FLAGS]
        cmd += ["-pix_fmt", self.pix_fmt, "-f", "rawvideo", "-"]
        return cmd

    def __iter__(self) -> Iterator[np.ndarray]:
        h, w = self.info.height, self.info.width
        frame_bytes = h * w * 3 * (2 if self.depth == 16 else 1)
        proc = subprocess.Popen(self.command(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=frame_bytes)
        drain = _StderrDrain(proc.stderr)
        drain.start()
        q: queue.Queue = queue.Queue(maxsize=self.prefetch)
        stop = threading.Event()

        def pump():
            try:
                while not stop.is_set():
                    buf = proc.stdout.read(frame_bytes)
                    if len(buf) < frame_bytes:
                        break
                    q.put(np.frombuffer(buf, dtype=self.dtype).reshape(h, w, 3).copy())
            finally:
                q.put(None)

        t = threading.Thread(target=pump, daemon=True)
        t.start()
        finished = False
        try:
            while (frame := q.get()) is not None:
                yield frame
            finished = True
        finally:
            stop.set()
            while t.is_alive():  # unblock the pump if the consumer stopped early
                try:
                    q.get_nowait()
                except queue.Empty:
                    t.join(0.05)
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            drain.join(1)
            if finished and proc.returncode != 0:
                raise FFmpegError(f"decoding failed: {drain.text}")

    def read_one(self) -> np.ndarray:
        for frame in FrameReader(self.info, self.start, 1, self.depth, prefetch=1):
            return frame
        raise FFmpegError(f"could not read frame {self.start} of {self.info.path}")


class FrameWriter:
    """Encode raw frames, muxing audio/timecode/color tags/metadata from the source."""

    def __init__(
        self,
        output: Path,
        info: VideoInfo,
        preset: Preset,
        input_pix_fmt: str,
        encoder: str | None = None,
        quality: str = "standard",
        timecode: str | None = None,
        audio_source: Path | None = None,
        audio_start: float = 0.0,
        audio_duration: float | None = None,
        metadata: dict[str, str] | None = None,
        full_range_out: bool | None = None,
    ):
        self.output = Path(output)
        self.info = info
        self.preset = preset
        self.input_pix_fmt = input_pix_fmt
        self.encoder = choose_encoder(preset, encoder)
        if quality not in QUALITIES:
            raise ValueError(f"quality must be one of {', '.join(QUALITIES)}")
        self.quality = quality
        self.timecode = timecode
        self.audio_source = audio_source
        self.audio_start = audio_start
        self.audio_duration = audio_duration
        self.metadata = metadata or {}
        self.full_range_out = info.is_full_range if full_range_out is None else full_range_out
        self._proc: subprocess.Popen | None = None
        self._drain: _StderrDrain | None = None
        self.frames_written = 0

    def command(self) -> list[str]:
        info, preset = self.info, self.preset
        cmd = [ffmpeg_bin(), "-nostdin", "-hide_banner", "-v", "error", "-y"]
        cmd += ["-f", "rawvideo", "-pix_fmt", self.input_pix_fmt, "-s", f"{info.width}x{info.height}"]
        cmd += ["-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "-"]
        audio = self.audio_source is not None and info.has_audio
        partial = self.audio_start > 0 or self.audio_duration is not None
        if audio:
            if info.video_start:
                cmd += ["-itsoffset", f"{-info.video_start:.6f}"]
            if self.audio_start:
                cmd += ["-ss", f"{self.audio_start:.6f}"]
            if self.audio_duration is not None:
                cmd += ["-t", f"{self.audio_duration:.6f}"]
            cmd += ["-i", str(self.audio_source)]
        cmd += ["-map", "0:v:0"]
        if audio:
            cmd += ["-map", "1:a?"]
        out_range = _range(self.full_range_out)
        cmd += ["-vf", f"scale=out_color_matrix={info.matrix}:out_range={out_range}", "-sws_flags", SWS_FLAGS]
        enc = self.encoder
        pix_fmt = preset.pix_fmt_for(info.bit_depth if self.input_pix_fmt in ("bgr48le", "bgra64le") else 8)
        cmd += ["-c:v", enc.name, *enc.args, *enc.quality.get(self.quality, ()), "-pix_fmt", pix_fmt]
        colorspace = {"bt709": "bt709", "bt601": "smpte170m", "bt2020": "bt2020nc"}[info.matrix]
        cmd += ["-colorspace", info.color_space or colorspace, "-color_range", out_range]
        if info.color_primaries and info.color_primaries != "unknown":
            cmd += ["-color_primaries", info.color_primaries]
        if info.color_trc and info.color_trc != "unknown":
            cmd += ["-color_trc", info.color_trc]
        if info.sar and info.sar not in ("0:1", "1:1"):
            cmd += ["-aspect", f"{Fraction(info.width, info.height) * Fraction(info.sar.replace(':', '/'))}"]
        if audio:
            if not partial and all(c in preset.audio_copy for c in info.audio_codecs):
                cmd += ["-c:a", "copy"]
            else:
                cmd += list(preset.audio_fallback)
        if self.timecode:
            cmd += ["-timecode", self.timecode]
        for key, value in self.metadata.items():
            cmd += ["-metadata", f"{key}={value}"]
        if self.output.suffix.lower() in (".mov", ".mp4", ".m4v"):
            cmd += ["-movflags", "+use_metadata_tags" + ("+faststart" if self.output.suffix.lower() == ".mp4" else "")]
        cmd += [str(self.output)]
        return cmd

    def __enter__(self) -> FrameWriter:
        self.output.parent.mkdir(parents=True, exist_ok=True)
        log.debug("ffmpeg: %s", " ".join(self.command()))
        self._proc = subprocess.Popen(self.command(), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self._drain = _StderrDrain(self._proc.stderr)
        self._drain.start()
        return self

    def write(self, frame: np.ndarray) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            self._proc.wait()
            raise FFmpegError(f"encoder exited: {self._drain.text}") from exc
        self.frames_written += 1

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._proc is None:
            return
        try:
            self._proc.stdin.close()
        except BrokenPipeError:
            pass
        if exc_type is not None:
            self._proc.kill()
        code = self._proc.wait()
        self._drain.join(1)
        if exc_type is None and code != 0:
            raise FFmpegError(f"encoding failed ({code}): {self._drain.text}")
