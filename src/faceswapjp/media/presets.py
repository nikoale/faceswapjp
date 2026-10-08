"""Encoder presets for the main output and the matte."""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field

from .probe import FFmpegError, ffmpeg_bin


@dataclass(frozen=True)
class Preset:
    name: str
    ext: str
    pix_fmt: str
    encoders: tuple[str, ...]  # candidates in priority order
    args: dict[str, tuple[str, ...]] = field(default_factory=dict)  # encoder -> codec args
    audio_copy: tuple[str, ...] = ()  # source audio codecs that can be stream-copied
    audio_fallback: tuple[str, ...] = ("-c:a", "aac", "-b:a", "320k")
    alpha: bool = False


PRORES_AUDIO_OK = ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "pcm_s24be", "aac", "alac")
MP4_AUDIO_OK = ("aac", "alac", "mp3", "ac3", "eac3", "opus")

PRESETS: dict[str, Preset] = {
    "prores422hq": Preset(
        name="prores422hq",
        ext=".mov",
        pix_fmt="yuv422p10le",
        encoders=("prores_ks", "prores_videotoolbox"),
        args={
            "prores_ks": ("-profile:v", "3", "-vendor", "apl0"),
            "prores_videotoolbox": ("-profile:v", "hq"),
        },
        audio_copy=PRORES_AUDIO_OK,
        audio_fallback=("-c:a", "pcm_s24le"),
    ),
    "prores4444": Preset(
        name="prores4444",
        ext=".mov",
        pix_fmt="yuva444p10le",
        encoders=("prores_ks",),
        args={"prores_ks": ("-profile:v", "4", "-vendor", "apl0", "-alpha_bits", "16")},
        audio_copy=PRORES_AUDIO_OK,
        audio_fallback=("-c:a", "pcm_s24le"),
        alpha=True,
    ),
    "h264": Preset(
        name="h264",
        ext=".mp4",
        pix_fmt="yuv420p",
        encoders=("libx264", "h264_videotoolbox", "h264_nvenc"),
        args={
            "libx264": ("-crf", "16", "-preset", "slow", "-profile:v", "high"),
            "h264_videotoolbox": ("-q:v", "70", "-profile:v", "high"),
            "h264_nvenc": ("-preset", "p6", "-rc", "vbr", "-cq", "18", "-b:v", "0", "-profile:v", "high"),
        },
        audio_copy=MP4_AUDIO_OK,
    ),
}


@functools.lru_cache(maxsize=1)
def available_encoders() -> frozenset[str]:
    out = subprocess.run([ffmpeg_bin(), "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6 and parts[0][0] in "VAS":
            names.add(parts[1])
    return frozenset(names)


def get_preset(name: str) -> Preset:
    if name not in PRESETS:
        raise ValueError(f"unknown format {name!r}; choose from {', '.join(PRESETS)}")
    return PRESETS[name]


def choose_encoder(preset: Preset, requested: str | None = None) -> str:
    available = available_encoders()
    if requested:
        if requested not in preset.args:
            raise ValueError(f"encoder {requested!r} is not supported for {preset.name}; use one of {', '.join(preset.encoders)}")
        if requested not in available:
            raise FFmpegError(f"encoder {requested!r} is not available in this ffmpeg build")
        return requested
    for enc in preset.encoders:
        if enc in available:
            return enc
    raise FFmpegError(f"no encoder for {preset.name} in this ffmpeg build (tried {', '.join(preset.encoders)})")
