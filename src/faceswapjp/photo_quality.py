"""Grade the photos used to register a face, so weak ones can be swapped out before registering.

Each photo gets a 0-100 score and plain-language notes. A set of photos is also checked for
consistency: a photo whose face does not match the others is probably a different person.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from .analysis.face import Face
from .compositing.align import estimate_alignment, warp_crop
from .scan import yaw_asymmetry

SMALL_PX, TINY_PX = 140, 80  # face width in the photo
SOFT, BLURRY = 120.0, 50.0  # Laplacian variance of the aligned 112 px face
TURNED, PROFILE = 0.18, 0.30  # yaw asymmetry (see scan.yaw_asymmetry)
DARK, BRIGHT = 60.0, 210.0  # mean luma of the face
OTHER_PERSON = 0.45  # cosine similarity to the rest of the set


@dataclass
class PhotoReport:
    score: int
    grade: str  # good | ok | bad
    notes: list[str] = field(default_factory=list)
    face_px: int = 0


def _sharpness(image: np.ndarray, face: Face) -> float:
    crop = warp_crop(image, estimate_alignment(face.kps, 112, "arcface_128"), 112)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)[20:100, 20:100]
    gray = (gray - gray.mean()) / (gray.std() + 1.0) * 48.0  # contrast-normalized: dark != blurry
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _brightness(image: np.ndarray, face: Face) -> float:
    crop = warp_crop(image, estimate_alignment(face.kps, 112, "arcface_128"), 112)
    return float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[25:95, 25:95].mean())


def assess_photo(image: np.ndarray, faces: list[Face]) -> tuple[PhotoReport, Face | None]:
    """image: uint8 BGR. faces: detections in it. Returns the report and the face that would be used."""
    if not faces:
        return PhotoReport(0, "bad", ["顔が見つかりません"]), None
    face = max(faces, key=lambda f: f.area)
    score, notes = 100, []

    def hit(points: int, note: str):
        nonlocal score
        score -= points
        notes.append(note)

    if len(faces) > 1:
        hit(10, "ほかの人も写っています（一番大きい顔を使います）")
    width = int(face.bbox[2] - face.bbox[0])
    if width < TINY_PX:
        hit(50, "顔がとても小さい写真です")
    elif width < SMALL_PX:
        hit(15, "顔が少し小さい写真です")
    sharp = _sharpness(image, face)
    if sharp < BLURRY:
        hit(50, "ぼやけています")
    elif sharp < SOFT:
        hit(15, "少しぼやけています")
    yaw = abs(yaw_asymmetry(face.kps))
    if yaw > PROFILE:
        hit(35, "横を向いています")
    elif yaw > TURNED:
        hit(10, "少し横を向いています（正面の写真もあると安定します）")
    light = _brightness(image, face)
    if light < DARK:
        hit(30, "顔が暗く写っています")
    elif light > BRIGHT:
        hit(30, "顔が明るすぎます（白飛び）")
    if face.det_score < 0.7:
        hit(15, "顔がはっきり写っていません")
    score = max(0, score)
    return PhotoReport(score, _grade(score), notes, width), face


def _grade(score: int) -> str:
    return "good" if score >= 80 else "ok" if score >= 55 else "bad"


def assess_set(images: list[np.ndarray], detections: list[list[Face]]) -> dict:
    reports, used = [], []
    for image, faces in zip(images, detections):
        report, face = assess_photo(image, faces)
        reports.append(report)
        used.append(face)
    embs = [f.embedding / np.linalg.norm(f.embedding) if f is not None and f.embedding is not None else None
            for f in used]
    valid = [e for e in embs if e is not None]
    if len(valid) >= 2:
        for report, e in zip(reports, embs):
            if e is None:
                continue
            others = [o for o in valid if o is not e]
            mean = np.mean(others, axis=0)
            if float(e @ (mean / np.linalg.norm(mean))) < OTHER_PERSON:
                report.score = max(0, report.score - 60)
                report.notes.insert(0, "他の写真と別人のように見えます")
                report.grade = _grade(report.score)
    usable = sum(r.grade != "bad" for r in reports)
    if not reports:
        summary = ""
    elif usable == 0:
        summary = "使える写真がありません。正面を向いた、明るくはっきりした写真を選んでください"
    elif any(r.grade == "bad" for r in reports):
        summary = "✕ の写真は外すか、別の写真に替えるのがおすすめです"
    elif len(reports) < 3:
        summary = "角度や表情の違う写真をあと数枚足すと、仕上がりが安定します"
    else:
        summary = "良い写真がそろっています"
    return {"photos": [asdict(r) for r in reports], "summary": summary}
