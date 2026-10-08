"""Project folder: identities, consent log, renders."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .safety.consent import ConsentLog

PROJECT_FILE = "project.json"


@dataclass
class ProjectSettings:
    name: str
    created_at: str
    commercial: bool = False
    # model name -> commercial license reference (contract id etc.)
    license_overrides: dict[str, str] = field(default_factory=dict)
    version: int = 1


class Project:
    def __init__(self, root: Path, settings: ProjectSettings):
        self.root = Path(root)
        self.settings = settings

    @property
    def identities_dir(self) -> Path:
        return self.root / "identities"

    @property
    def renders_dir(self) -> Path:
        return self.root / "renders"

    @property
    def previews_dir(self) -> Path:
        return self.root / "previews"

    @property
    def consent_log(self) -> ConsentLog:
        return ConsentLog(self.root / "consent_log.json")

    @classmethod
    def init(cls, root: Path, name: str | None = None, commercial: bool = False) -> Project:
        root = Path(root)
        if (root / PROJECT_FILE).exists():
            raise FileExistsError(f"project already exists: {root}")
        settings = ProjectSettings(
            name=name or root.resolve().name,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            commercial=commercial,
        )
        project = cls(root, settings)
        for d in (root, project.identities_dir, project.renders_dir, project.previews_dir):
            d.mkdir(parents=True, exist_ok=True)
        project.save()
        project.consent_log.append("project_created", name=settings.name, commercial=commercial)
        return project

    @classmethod
    def load(cls, root: Path) -> Project:
        path = Path(root) / PROJECT_FILE
        if not path.exists():
            raise FileNotFoundError(f"not a faceswapjp project (missing {PROJECT_FILE}): {root}")
        return cls(Path(root), ProjectSettings(**json.loads(path.read_text(encoding="utf-8"))))

    def save(self) -> None:
        (self.root / PROJECT_FILE).write_text(
            json.dumps(asdict(self.settings), ensure_ascii=False, indent=2), encoding="utf-8"
        )
