"""Speed measurement shared by `faceswapjp bench` and the studio's 速度チェック panel."""

from __future__ import annotations

import platform
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .media.ffpipe import FrameReader
from .media.presets import get_preset
from .media.probe import probe
from .pipeline.frame import FrameOptions, to_detection_image
from .pipeline.video import RenderSettings, TrackedProcessor, TrackingOptions, render_video
from .runtime import session_provider

STAGE_LABELS = {
    "swap": "顔の差し替え（AI）", "detect": "顔の検出（全体）", "detect_roi": "顔の検出（顔の周り）",
    "detect_refine": "ランドマーク補正", "embed": "人物の判定", "mask": "手・髪のマスク", "enhance": "顔の補正",
    "color": "色合わせ", "paste": "貼り戻し", "align": "位置合わせ", "to_8bit": "検出用の変換",
    "wait_decoder": "動画の読み込み待ち", "wait_encoder": "書き出し待ち", "matte": "マスク動画の作成",
}


def run_bench(
    target: Path,
    frames: int = 48,
    start: int = 0,
    masks: str = "box",
    enhancer: str | None = None,
    detect_every: int = 3,
    fmt: str = "h264",
    encoder: str | None = None,
    device: str = "auto",
    progress: Callable[[int, int], None] | None = None,
    cancel: threading.Event | None = None,
) -> dict[str, Any]:
    """Swap the largest face of the first frame with itself and time every stage.

    Nothing is kept: the output goes to a temporary folder that is deleted afterwards.
    """
    from .jobs import get_engine

    t0 = time.time()
    engine = get_engine(None, device, masks, enhancer)
    load_s = time.time() - t0
    info = probe(target)
    start = max(0, min(start, info.nb_frames - 1))
    first = FrameReader(info, start=start, count=1).read_one()
    image = to_detection_image(first)
    engine.safety.check_frames([(start, image)], str(target))
    faces = engine.analyzer.detect(image)
    if not faces:
        raise ValueError("測定する場面に顔が映っていません。顔が映っている素材・場面を選んでください。")
    source = max(faces, key=lambda f: f.area).embedding

    sessions = {"swapper": getattr(engine.swapper, "_session", None)}
    for occ in engine.occluders:
        sessions[occ.name] = getattr(occ, "_session", None)
    if engine.enhancer is not None:
        sessions[engine.enhancer.name] = getattr(engine.enhancer, "_session", None)
    runs_on = {name: session_provider(s) for name, s in sessions.items() if s is not None}

    tracked = TrackedProcessor(engine.frame_processor(FrameOptions()), source, float(info.fps),
                               TrackingOptions(select="reference", detect_every=detect_every),
                               reference_embedding=source[None])
    end = min(info.nb_frames, start + frames)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / ("bench" + get_preset(fmt).ext)
        res = render_video(tracked, target, out, RenderSettings(format=fmt, encoder=encoder, start=start, end=end),
                           progress=progress, cancel=cancel, info=info)
    fps = res.fps
    src_fps = float(info.fps) or 24.0
    return {
        "machine": f"{platform.system()} {platform.machine()} · {platform.processor() or ''}".strip(" ·"),
        "providers": engine.provider_names,
        "runs_on": runs_on,
        "load_s": round(load_s, 1),
        "frames": res.frames,
        "seconds": round(res.seconds, 1),
        "fps": round(fps, 2),
        "resolution": f"{info.width}x{info.height}",
        "source_fps": round(src_fps, 3),
        "minutes_per_minute": round(src_fps / fps, 1) if fps else None,  # processing minutes for 1 min of footage
        "settings": {"masks": masks, "enhancer": enhancer, "detect_every": detect_every, "format": fmt},
        "timings": res.timings,
        "labels": STAGE_LABELS,
    }


def summary_text(r: dict[str, Any]) -> str:
    """Plain-text report, easy to paste into a chat."""
    lines = [
        "faceswapjp 速度チェック",
        f"マシン: {r['machine']}",
        f"実行環境: {', '.join(r['providers'])}",
        "モデル: " + ", ".join(f"{k}={v}" for k, v in r["runs_on"].items()),
        f"素材: {r['resolution']} {r['source_fps']}fps / {r['frames']} フレーム",
        f"速度: {r['fps']} fps（1 分の素材に約 {r['minutes_per_minute']} 分）/ モデル読み込み {r['load_s']} 秒",
        f"設定: {r['settings']}",
        "段階ごと (ms/フレーム):",
    ]
    for stage, t in r["timings"].items():
        lines.append(f"  {stage:14} {t['ms_per_frame']:>8} ms   ({t['calls']} 回, 1 回 {t['ms_per_call']} ms)")
    return "\n".join(lines)
