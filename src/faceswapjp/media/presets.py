"""Output codecs. Each preset lists encoders in priority order (software first, then hardware)."""

from __future__ import annotations

import functools
import subprocess
from dataclasses import dataclass, field

from .probe import FFmpegError, ffmpeg_bin

QUALITIES = ("high", "standard", "light")
ENCODER_MODES = ("auto", "software", "hardware")


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hardware: bool
    args: tuple[str, ...] = ()
    quality: dict[str, tuple[str, ...]] = field(default_factory=dict)  # quality level -> extra args


@dataclass(frozen=True)
class Preset:
    name: str
    label: str  # shown in the UI
    group: str  # "delivery" | "intermediate" | "internal"
    ext: str
    pix_fmt: str
    encoders: tuple[EncoderSpec, ...]
    pix_fmt_high_depth: str | None = None  # used when the source is deeper than 8 bit
    audio_copy: tuple[str, ...] = ()
    audio_fallback: tuple[str, ...] = ("-c:a", "aac", "-b:a", "320k")
    alpha: bool = False
    even_dims: bool = False  # 4:2:0 needs even width/height

    @property
    def has_quality(self) -> bool:
        return any(e.quality for e in self.encoders)

    def pix_fmt_for(self, bit_depth: int) -> str:
        return self.pix_fmt_high_depth if bit_depth > 8 and self.pix_fmt_high_depth else self.pix_fmt


MOV_AUDIO_OK = ("pcm_s16le", "pcm_s24le", "pcm_s32le", "pcm_s16be", "pcm_s24be", "aac", "alac")
MP4_AUDIO_OK = ("aac", "alac", "mp3", "ac3", "eac3", "opus")
PCM = ("-c:a", "pcm_s24le")


def _q(flag: str, high: str, standard: str, light: str) -> dict[str, tuple[str, ...]]:
    return {"high": (flag, high), "standard": (flag, standard), "light": (flag, light)}


def _prores(name: str, label: str, ks_profile: str, vt_profile: str, pix_fmt: str) -> Preset:
    return Preset(
        name=name, label=label, group="intermediate", ext=".mov", pix_fmt=pix_fmt,
        encoders=(
            EncoderSpec("prores_ks", False, ("-profile:v", ks_profile, "-vendor", "apl0")),
            EncoderSpec("prores_videotoolbox", True, ("-profile:v", vt_profile)),
        ),
        audio_copy=MOV_AUDIO_OK, audio_fallback=PCM,
    )


def _dnxhr(name: str, label: str, profile: str, pix_fmt: str) -> Preset:
    return Preset(
        name=name, label=label, group="intermediate", ext=".mov", pix_fmt=pix_fmt,
        encoders=(EncoderSpec("dnxhd", False, ("-profile:v", profile)),),
        audio_copy=MOV_AUDIO_OK, audio_fallback=PCM,
    )


_PRESET_LIST = [
    Preset(
        name="h264", label="H.264（.mp4）— 確認・共有用。どこでも再生できる", group="delivery",
        ext=".mp4", pix_fmt="yuv420p", even_dims=True, audio_copy=MP4_AUDIO_OK,
        encoders=(
            EncoderSpec("libx264", False, ("-preset", "slow", "-profile:v", "high"), _q("-crf", "14", "18", "23")),
            EncoderSpec("h264_videotoolbox", True, ("-profile:v", "high"), _q("-q:v", "80", "65", "50")),
            EncoderSpec("h264_nvenc", True, ("-preset", "p6", "-rc", "vbr", "-b:v", "0", "-profile:v", "high"),
                        _q("-cq", "16", "20", "26")),
        ),
    ),
    Preset(
        name="h265", label="H.265 / HEVC（.mp4）— H.264 より小さいファイルで高画質", group="delivery",
        ext=".mp4", pix_fmt="yuv420p", pix_fmt_high_depth="yuv420p10le", even_dims=True, audio_copy=MP4_AUDIO_OK,
        encoders=(
            EncoderSpec("libx265", False, ("-preset", "medium", "-tag:v", "hvc1", "-x265-params", "log-level=error"),
                        _q("-crf", "16", "20", "26")),
            EncoderSpec("hevc_videotoolbox", True, ("-tag:v", "hvc1"), _q("-q:v", "80", "65", "50")),
            EncoderSpec("hevc_nvenc", True, ("-preset", "p6", "-rc", "vbr", "-b:v", "0", "-tag:v", "hvc1"),
                        _q("-cq", "18", "22", "28")),
        ),
    ),
    _prores("prores_proxy", "ProRes 422 Proxy（.mov）— オフライン編集用の軽いファイル", "0", "proxy", "yuv422p10le"),
    _prores("prores_lt", "ProRes 422 LT（.mov）", "1", "lt", "yuv422p10le"),
    _prores("prores422", "ProRes 422（.mov）", "2", "standard", "yuv422p10le"),
    _prores("prores422hq", "ProRes 422 HQ（.mov）— 編集用の定番", "3", "hq", "yuv422p10le"),
    _prores("prores4444", "ProRes 4444（.mov）— 合成・カラー作業向け", "4", "4444", "yuv444p10le"),
    _prores("prores4444xq", "ProRes 4444 XQ（.mov）— 最高画質・ファイルが大きい", "5", "xq", "yuv444p10le"),
    _dnxhr("dnxhr_lb", "DNxHR LB（.mov）— Avid 向け・軽量", "dnxhr_lb", "yuv422p"),
    _dnxhr("dnxhr_sq", "DNxHR SQ（.mov）", "dnxhr_sq", "yuv422p"),
    _dnxhr("dnxhr_hq", "DNxHR HQ（.mov）— Avid / Resolve 向けの定番", "dnxhr_hq", "yuv422p"),
    _dnxhr("dnxhr_hqx", "DNxHR HQX（.mov）— 10bit", "dnxhr_hqx", "yuv422p10le"),
    _dnxhr("dnxhr_444", "DNxHR 444（.mov）— 10bit 4:4:4", "dnxhr_444", "yuv444p10le"),
    # used for the alpha matte (fill + alpha); not offered as a main output format
    Preset(
        name="prores4444_alpha", label="ProRes 4444 + alpha", group="internal", ext=".mov",
        pix_fmt="yuva444p10le", alpha=True, audio_copy=MOV_AUDIO_OK, audio_fallback=PCM,
        encoders=(EncoderSpec("prores_ks", False, ("-profile:v", "4", "-vendor", "apl0", "-alpha_bits", "16")),),
    ),
]
PRESETS: dict[str, Preset] = {p.name: p for p in _PRESET_LIST}
OUTPUT_FORMATS = [p.name for p in _PRESET_LIST if p.group != "internal"]


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
        raise ValueError(f"unknown format {name!r}; choose from {', '.join(OUTPUT_FORMATS)}")
    return PRESETS[name]


def choose_encoder(preset: Preset, requested: str | None = None) -> EncoderSpec:
    """requested: None/'auto' (best available, software first), 'software', 'hardware', or an encoder name."""
    available = available_encoders()
    mode = requested or "auto"
    if mode in ENCODER_MODES:
        candidates = [e for e in preset.encoders
                      if mode == "auto" or e.hardware == (mode == "hardware")]
        for enc in candidates:
            if enc.name in available:
                return enc
        if mode == "hardware":
            raise FFmpegError(f"no hardware encoder for {preset.name} is available on this machine")
        raise FFmpegError(f"no encoder for {preset.name} in this ffmpeg build "
                          f"(tried {', '.join(e.name for e in candidates)})")
    by_name = {e.name: e for e in preset.encoders}
    if mode not in by_name:
        raise ValueError(f"encoder {mode!r} is not supported for {preset.name}; "
                         f"use one of {', '.join(by_name)} or auto/software/hardware")
    if mode not in available:
        raise FFmpegError(f"encoder {mode!r} is not available in this ffmpeg build")
    return by_name[mode]


def hardware_available(preset: Preset) -> bool:
    return any(e.hardware and e.name in available_encoders() for e in preset.encoders)
