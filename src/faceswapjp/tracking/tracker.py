"""Associate per-frame detections into face tracks (IoU + identity embedding)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..analysis.face import Face
from .filters import LandmarkSmoother


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


@dataclass
class Track:
    id: int
    face: Face  # latest detection (kps already smoothed when used for swapping)
    smoother: LandmarkSmoother
    embedding_sum: np.ndarray | None = None
    n_embeddings: int = 0
    hits: int = 1
    misses: int = 0
    first_frame: int = 0
    is_target: bool | None = None
    history: list[int] = field(default_factory=list)

    @property
    def embedding(self) -> np.ndarray | None:
        if self.embedding_sum is None:
            return None
        n = np.linalg.norm(self.embedding_sum)
        return self.embedding_sum / n if n > 0 else None

    def add_embedding(self, emb: np.ndarray | None) -> None:
        if emb is None:
            return
        self.embedding_sum = emb.copy() if self.embedding_sum is None else self.embedding_sum + emb
        self.n_embeddings += 1


class FaceTracker:
    def __init__(
        self,
        smoothing: float = 0.5,
        max_misses: int = 5,
        min_iou: float = 0.2,
        min_similarity: float = 0.35,
    ):
        self.smoothing = smoothing
        self.max_misses = max_misses
        self.min_iou = min_iou
        self.min_similarity = min_similarity
        self.tracks: list[Track] = []
        self._next_id = 0

    def _score(self, track: Track, face: Face) -> float:
        """Higher is better; -inf means 'cannot be the same face'."""
        overlap = iou(track.face.bbox, face.bbox)
        sim = None
        if track.embedding is not None and face.embedding is not None:
            sim = float(track.embedding @ face.embedding)
        if overlap >= self.min_iou:
            # spatially consistent; reject only if the identity clearly differs
            if sim is not None and sim < 0.1:
                return -np.inf
            return overlap + (sim or 0.0)
        if sim is not None and sim >= self.min_similarity and track.misses > 0:
            return sim  # re-acquire after a short loss (fast motion, occlusion)
        return -np.inf

    def update(self, faces: list[Face], frame_index: int, dt: float) -> list[tuple[Track, Face]]:
        """Return (track, face-with-smoothed-kps) for every detection in this frame."""
        pairs = []
        candidates = [(self._score(t, f), ti, fi) for ti, t in enumerate(self.tracks) for fi, f in enumerate(faces)]
        candidates = sorted((c for c in candidates if np.isfinite(c[0])), reverse=True)
        used_t, used_f, matched = set(), set(), {}
        for _, ti, fi in candidates:
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            matched[fi] = self.tracks[ti]
        for fi, face in enumerate(faces):
            track = matched.get(fi)
            if track is None:
                track = Track(self._next_id, face, LandmarkSmoother(self.smoothing), first_frame=frame_index)
                self._next_id += 1
                self.tracks.append(track)
            else:
                track.hits += 1
                track.misses = 0
            track.add_embedding(face.embedding)
            smoothed = Face(face.bbox, track.smoother(face.kps, dt), face.det_score, face.embedding, face.landmarks_106)
            track.face = smoothed
            track.history.append(frame_index)
            pairs.append((track, smoothed))
        for track in self.tracks:
            if track not in matched.values() and track.history[-1] != frame_index:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]
        return pairs

    def needs_embedding(self, face: Face, frame_index: int, settle: int = 2, refresh: int = 24) -> bool:
        """Identity embeddings are expensive (~100 ms each on CPU), so they are computed only for
        new or unsettled tracks, plus a periodic refresh that catches identity swaps when people
        cross. Refreshes are staggered per track so they don't all land on the same frame.
        """
        best = max(self.tracks, key=lambda t: iou(t.face.bbox, face.bbox), default=None)
        if best is None or iou(best.face.bbox, face.bbox) < 0.5:
            return True
        if best.n_embeddings < settle:
            return True
        return (frame_index + best.id * 7) % refresh == 0

    def coasting(self, max_coast: int) -> list[Track]:
        """Tracks missed in this frame but recently seen (to bridge 1-2 frame detection dropouts)."""
        return [t for t in self.tracks if 0 < t.misses <= max_coast]
