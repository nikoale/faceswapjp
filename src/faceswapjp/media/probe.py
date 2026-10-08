"""ffprobe wrapper: everything the writer needs to match the source."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path


class FFmpegError(RuntimeError):
    pass


def ffmpeg_bin(name: str = "ffmpeg") -> str:
    path = shutil.which(name)
    if path is None:
        raise FFmpegError(f"{name} not found; install it (macOS: brew install ffmpeg)")
    return path


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction  # nominal frame rate (r_frame_rate)
    avg_fps: Fraction
    nb_frames: int  # exact if the container reports it, else estimated from duration
    duration: float
    codec: str
    pix_fmt: str
    bit_depth: int
    has_alpha: bool
    color_primaries: str | None
    color_trc: str | None
    color_space: str | None
    color_range: str | None
    field_order: str | None
    sar: str | None
    timecode: str | None
    audio_codecs: list[str] = field(default_factory=list)
    format_name: str = ""
    video_start: float = 0.0  # video stream start relative to the container start (seconds)

    @property
    def is_vfr(self) -> bool:
        if not self.avg_fps:
            return False
        return abs(float(self.fps) - float(self.avg_fps)) / float(self.fps) > 0.002

    @property
    def is_interlaced(self) -> bool:
        return self.field_order not in (None, "progressive", "unknown")

    @property
    def has_audio(self) -> bool:
        return bool(self.audio_codecs)

    @property
    def matrix(self) -> str:
        """Color matrix to use for YUV<->RGB (tagged value, else a resolution-based guess)."""
        tagged = {"bt709": "bt709", "bt470bg": "bt601", "smpte170m": "bt601", "bt2020nc": "bt2020", "bt2020c": "bt2020"}
        if self.color_space in tagged:
            return tagged[self.color_space]
        return "bt709" if self.height >= 720 else "bt601"

    @property
    def is_full_range(self) -> bool:
        return self.color_range in ("pc", "jpeg")


def _fraction(text: str | None) -> Fraction:
    try:
        f = Fraction(text or "0")
    except (ValueError, ZeroDivisionError):
        return Fraction(0)
    return f


def _bit_depth(pix_fmt: str) -> int:
    for depth in (16, 12, 10):
        if str(depth) in pix_fmt:
            return depth
    return 8


def probe(path: str | Path) -> VideoInfo:
    cmd = [ffmpeg_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe failed for {path}: {proc.stderr.strip()}")
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise FFmpegError(f"no video stream in {path}")
    fmt = data.get("format", {})
    fps = _fraction(video.get("r_frame_rate"))
    avg = _fraction(video.get("avg_frame_rate"))
    duration = float(video.get("duration") or fmt.get("duration") or 0.0)
    nb = video.get("nb_frames")
    nb_frames = int(nb) if nb and nb.isdigit() else int(round(duration * float(avg or fps)))
    timecode = (
        video.get("tags", {}).get("timecode")
        or next((s.get("tags", {}).get("timecode") for s in streams if s.get("codec_type") == "data" and s.get("tags", {}).get("timecode")), None)
        or fmt.get("tags", {}).get("timecode")
    )
    pix_fmt = video.get("pix_fmt", "")
    return VideoInfo(
        path=Path(path),
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        avg_fps=avg,
        nb_frames=nb_frames,
        duration=duration,
        codec=video.get("codec_name", ""),
        pix_fmt=pix_fmt,
        bit_depth=_bit_depth(pix_fmt),
        has_alpha=any(t in pix_fmt for t in ("yuva", "rgba", "bgra", "argb", "gbrap", "ya")),
        color_primaries=video.get("color_primaries"),
        color_trc=video.get("color_transfer"),
        color_space=video.get("color_space"),
        color_range=video.get("color_range"),
        field_order=video.get("field_order"),
        sar=video.get("sample_aspect_ratio"),
        timecode=timecode,
        audio_codecs=[s.get("codec_name", "") for s in streams if s.get("codec_type") == "audio"],
        format_name=fmt.get("format_name", ""),
        video_start=float(video.get("start_time") or 0.0) - float(fmt.get("start_time") or 0.0),
    )


def tc_to_frames(tc: str, fps: Fraction) -> int:
    """SMPTE timecode -> frame count. Drop-frame (';') supported for 29.97/59.94."""
    drop = ";" in tc or "," in tc
    h, m, s, f = (int(x) for x in tc.replace(";", ":").replace(",", ":").replace(".", ":").split(":"))
    nominal = round(float(fps))
    if not drop:
        return ((h * 3600 + m * 60 + s) * nominal) + f
    dropped = 2 if nominal == 30 else 4
    total_minutes = h * 60 + m
    frames = ((h * 3600 + m * 60 + s) * nominal) + f
    return frames - dropped * (total_minutes - total_minutes // 10)


def frames_to_tc(frames: int, fps: Fraction, drop: bool = False) -> str:
    nominal = round(float(fps))
    if drop:
        dropped = 2 if nominal == 30 else 4
        per_10min = nominal * 600 - dropped * 9
        per_min = nominal * 60 - dropped
        d, mod = divmod(frames, per_10min)
        if mod > dropped:
            frames += dropped * 9 * d + dropped * ((mod - dropped) // per_min)
        else:
            frames += dropped * 9 * d
    f = frames % nominal
    s = (frames // nominal) % 60
    m = (frames // (nominal * 60)) % 60
    h = frames // (nominal * 3600) % 24
    sep = ";" if drop else ":"
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{f:02d}"


def offset_timecode(tc: str | None, fps: Fraction, offset_frames: int) -> str | None:
    """Timecode of frame `offset_frames` of a clip that starts at `tc` (for partial renders)."""
    if not tc:
        return None
    if offset_frames == 0:
        return tc
    drop = ";" in tc or "," in tc
    return frames_to_tc(tc_to_frames(tc, fps) + offset_frames, fps, drop)
