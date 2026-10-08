"""Source identities: embeddings built from reference images, with mandatory consent info."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .imageio import read_image
from .models.interfaces import FaceAnalyzer
from .models.registry import sha256_file
from .pipeline.frame import to_detection_image
from .project import Project

log = logging.getLogger(__name__)

SOURCE_TYPES = ("consented_person", "self", "synthetic")
# Reference images whose embedding is this far from the mean probably show someone else.
CONSISTENCY_WARN = 0.45


class ConsentError(ValueError):
    pass


@dataclass
class ConsentInfo:
    person_name: str
    consent_date: str  # ISO 8601 date
    consent_document: str  # absolute path at registration time
    consent_document_sha256: str
    source_type: str


@dataclass
class Identity:
    id: str
    label: str
    consent: ConsentInfo
    reference_images: list[str]  # relative to the identity folder
    embedding_model: str
    created_at: str
    consistency: list[float]  # cosine similarity of each reference to the mean embedding

    @property
    def folder_name(self) -> str:
        return self.id


def validate_consent(person_name: str, consent_date: str, consent_document: Path, source_type: str) -> ConsentInfo:
    if not person_name or not person_name.strip():
        raise ConsentError("person name is required")
    try:
        d = date.fromisoformat(consent_date)
    except ValueError as exc:
        raise ConsentError(f"consent date must be YYYY-MM-DD, got {consent_date!r}") from exc
    if d > date.today():
        raise ConsentError(f"consent date is in the future: {consent_date}")
    doc = Path(consent_document).expanduser()
    if not doc.is_file():
        raise ConsentError(f"consent document not found: {doc}")
    if source_type not in SOURCE_TYPES:
        raise ConsentError(f"source type must be one of {', '.join(SOURCE_TYPES)}")
    return ConsentInfo(
        person_name=person_name.strip(),
        consent_date=d.isoformat(),
        consent_document=str(doc.resolve()),
        consent_document_sha256=sha256_file(doc),
        source_type=source_type,
    )


def make_id(label: str, existing: set[str]) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    if not slug:  # e.g. Japanese-only names
        slug = "id-" + hashlib.sha1(label.encode("utf-8")).hexdigest()[:8]
    candidate, n = slug, 2
    while candidate in existing:
        candidate, n = f"{slug}-{n}", n + 1
    return candidate


def embed_references(analyzer: FaceAnalyzer, images: list[Path]) -> tuple[np.ndarray, list[np.ndarray]]:
    embeddings = []
    for path in images:
        faces = analyzer.detect(to_detection_image(read_image(path)))
        faces = [f for f in faces if f.embedding is not None]
        if not faces:
            raise ValueError(f"no face found in reference image: {path}")
        if len(faces) > 1:
            log.warning("%s contains %d faces; using the largest", path, len(faces))
        embeddings.append(max(faces, key=lambda f: f.area).embedding)
    mean = np.mean(embeddings, axis=0)
    return (mean / np.linalg.norm(mean)).astype(np.float32), embeddings


def register_identity(
    project: Project,
    analyzer: FaceAnalyzer,
    label: str,
    images: list[Path],
    consent: ConsentInfo,
    embedding_model: str,
    identity_id: str | None = None,
) -> Identity:
    if not images:
        raise ValueError("at least one reference image is required")
    existing = {p.name for p in project.identities_dir.iterdir()} if project.identities_dir.exists() else set()
    if identity_id and identity_id in existing:
        raise FileExistsError(f"identity already exists: {identity_id}")
    iid = identity_id or make_id(label, existing)

    embedding, per_image = embed_references(analyzer, [Path(p) for p in images])
    consistency = [float(np.dot(e, embedding)) for e in per_image]
    for path, sim in zip(images, consistency):
        if sim < CONSISTENCY_WARN:
            log.warning("reference %s looks different from the others (similarity %.2f)", path, sim)

    folder = project.identities_dir / iid
    (folder / "refs").mkdir(parents=True)
    refs = []
    for i, src in enumerate(images):
        dst = folder / "refs" / f"{i:02d}{Path(src).suffix.lower()}"
        shutil.copy2(src, dst)
        refs.append(str(dst.relative_to(folder)))
    np.save(folder / "embedding.npy", embedding)
    identity = Identity(
        id=iid,
        label=label,
        consent=consent,
        reference_images=refs,
        embedding_model=embedding_model,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        consistency=consistency,
    )
    (folder / "identity.json").write_text(json.dumps(asdict(identity), ensure_ascii=False, indent=2), encoding="utf-8")
    project.consent_log.append(
        "identity_registered",
        identity_id=iid,
        label=label,
        **asdict(consent),
        reference_images_sha256=[sha256_file(folder / r) for r in refs],
    )
    return identity


def load_identity(project: Project, identity_id: str) -> tuple[Identity, np.ndarray]:
    folder = project.identities_dir / identity_id
    meta = folder / "identity.json"
    if not meta.exists():
        raise FileNotFoundError(f"identity not found: {identity_id}")
    data = json.loads(meta.read_text(encoding="utf-8"))
    data["consent"] = ConsentInfo(**data["consent"])
    return Identity(**data), np.load(folder / "embedding.npy")


def list_identities(project: Project) -> list[Identity]:
    if not project.identities_dir.exists():
        return []
    return [load_identity(project, p.name)[0] for p in sorted(project.identities_dir.iterdir()) if p.is_dir()]
