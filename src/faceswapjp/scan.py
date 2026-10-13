"""Find the scenes a face swap is likely to struggle with, so they can be checked before rendering.

Samples the video a few times per second and flags, for the faces that will be replaced:
lost (the person vanishes for a moment while still on screen), profile (strongly turned face),
occluded (hands / props over the face), small (too few pixels) and ambiguous (the "who to
replace" decision is close to the threshold, so the wrong person might be swapped).
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from .analysis.face import Face, best_similarity
from .compositing.align import estimate_alignment, warp_crop
from .compositing.blend import to_float

LABELS = {
    "lost": "置き換える人を見失っている",
    "profile": "横顔",
    "occluded": "顔が手や物で隠れている",
    "small": "顔が小さい",
    "ambiguous": "別の人と見分けにくい",
}
HINTS = {
    "lost": "この間は差し替わりません。「人物判定の厳しさ」をゆるめると拾えることがあります",
    "profile": "横顔は AI が苦手です。仕上がりを確認してください",
    "occluded": "「手や髪を顔の前に残す」をオンにして、手の周りを確認してください",
    "small": "細部が出にくい大きさです。気になる場合は「最高画質」を試してください",
    "ambiguous": "別の人まで置き換わっていないか確認してください。「人物判定の厳しさ」で調整できます",
}

PROFILE_ASYM = 0.30  # |left-right nose/eye distance asymmetry| above this = strongly turned
SMALL_PX = 64  # face width in pixels
# Share of the inner face the occlusion model calls "not face". Even clean faces score 0.1-0.4
# (beards, glasses, shadows), so a sample is flagged against the video's own typical value.
OCCLUDED_ABOVE_TYPICAL = 0.2
OCCLUDED_ALWAYS = 0.55
AMBIGUOUS_MARGIN = (0.07, 0.05)  # similarity within (threshold - a, threshold + b)
MAX_SAMPLES = 600
LOST_MAX_SECONDS = 5.0  # longer absences are someone leaving the shot, not a detection miss


@dataclass
class Issue:
    kind: str
    label: str
    hint: str
    start: int  # first frame
    end: int  # last frame (inclusive)
    frame: int  # a representative frame to preview


def yaw_asymmetry(kps: np.ndarray) -> float:
    """0 for a frontal face, toward +-1 as the head turns (nose closer to one eye)."""
    left, right, nose = kps[0], kps[1], kps[2]
    dl, dr = np.linalg.norm(nose - left), np.linalg.norm(nose - right)
    return float((dl - dr) / max(dl + dr, 1e-6))


def occluded_fraction(occluder, image: np.ndarray, face: Face, template: str = "arcface_128") -> float:
    size = 256
    crop = to_float(warp_crop(image, estimate_alignment(face.kps, size, template), size))
    mask = occluder.mask(crop)
    inner = mask[int(size * 0.30):int(size * 0.85), int(size * 0.22):int(size * 0.78)]
    return float((inner < 0.5).mean())


def sample_stride(fps: float, frames: int, per_second: float = 3.0) -> int:
    stride = max(1, int(round(fps / per_second)))
    return max(stride, int(np.ceil(frames / MAX_SAMPLES)))


def classify(faces: list[Face], reference: np.ndarray | None, threshold: float,
             occluded: Callable[[Face], float] | None) -> tuple[set[str], bool, float]:
    """Issues in one sampled frame, whether a face to replace was found, and the highest
    occlusion score among those faces (judged later against the whole video)."""
    kinds: set[str] = set()
    if reference is not None:
        sims = [best_similarity(f.embedding, reference) for f in faces]  # same rule as the renderer
        targets = [f for f, s in zip(faces, sims) if s >= threshold]
        lo, hi = threshold - AMBIGUOUS_MARGIN[0], threshold + AMBIGUOUS_MARGIN[1]
        if any(lo <= s < hi for s in sims) and len(faces) > 1:
            kinds.add("ambiguous")
    else:
        targets = faces
    for f in targets:
        if abs(yaw_asymmetry(f.kps)) > PROFILE_ASYM:
            kinds.add("profile")
        if f.bbox[2] - f.bbox[0] < SMALL_PX:
            kinds.add("small")
    occ = max((occluded(f) for f in targets), default=0.0) if occluded is not None else 0.0
    return kinds, bool(targets), occ


def flag_occlusion(samples: list[tuple[int, set[str], bool]], scores: list[float]) -> None:
    seen = [sc for (_, _, found), sc in zip(samples, scores) if found]
    if not seen:
        return
    typical = float(np.median(seen))
    for (_, kinds, found), sc in zip(samples, scores):
        if found and (sc > typical + OCCLUDED_ABOVE_TYPICAL or sc > OCCLUDED_ALWAYS):
            kinds.add("occluded")


def merge(samples: list[tuple[int, set[str], bool]], stride: int, fps: float, last_frame: int) -> list[Issue]:
    """Turn per-sample flags into frame ranges."""
    runs: list[Issue] = []
    for kind in LABELS:
        if kind == "lost":
            flags = [not found for _, _, found in samples]
        else:
            flags = [kind in ks for _, ks, _ in samples]
        i = 0
        while i < len(samples):
            if not flags[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(samples) and flags[j + 1]:
                j += 1
            start, end = samples[i][0], min(last_frame, samples[j][0] + stride - 1)
            keep = True
            if kind == "lost":  # only a short gap between sightings is a detection problem
                keep = i > 0 and j < len(samples) - 1 and (end - start + 1) / fps <= LOST_MAX_SECONDS
            if keep:
                mid = samples[(i + j) // 2][0]
                runs.append(Issue(kind, LABELS[kind], HINTS[kind], start, end, mid))
            i = j + 1
    runs.sort(key=lambda r: (r.start, r.kind))
    return runs


def scan_video(engine, target, reference: np.ndarray | None = None, threshold: float = 0.4,
               start: int = 0, end: int | None = None,
               progress: Callable[[int, int], None] | None = None,
               cancel: threading.Event | None = None) -> dict:
    from .media.ffpipe import FrameReader
    from .media.probe import probe
    from .pipeline.frame import to_detection_image

    info = probe(target)
    end = info.nb_frames if end is None else min(end, info.nb_frames)
    fps = float(info.fps)
    stride = sample_stride(fps, end - start)
    total = (end - start + stride - 1) // stride
    occluder = next((o for o in engine.occluders if getattr(o, "name", "") == "xseg"), None)
    samples: list[tuple[int, set[str], bool]] = []
    scores: list[float] = []
    reader = FrameReader(info, start=start, count=end - start, depth=8, select=f"not(mod(n\\,{stride}))")
    for i, frame in enumerate(reader):
        if cancel is not None and cancel.is_set():
            break
        index = start + i * stride
        image = to_detection_image(frame)
        faces = [f for f in engine.analyzer.detect(image, with_embedding=reference is not None)
                 if f.det_score >= 0.5]
        occ = (lambda f: occluded_fraction(occluder, image, f)) if occluder is not None else None
        kinds, found, score = classify(faces, reference, threshold, occ)
        samples.append((index, kinds, found))
        scores.append(score)
        if progress:
            progress(i + 1, total)
    flag_occlusion(samples, scores)
    issues = merge(samples, stride, fps, end - 1)
    return {"issues": [asdict(x) for x in issues], "samples": len(samples), "stride": stride,
            "flagged_seconds": round(sum(x.end - x.start + 1 for x in issues) / fps, 1)}
