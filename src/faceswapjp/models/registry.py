"""Model manifest: locate, download, verify and check licenses of model files."""

from __future__ import annotations

import hashlib
import logging
import shutil
import tomllib
import urllib.request
import zipfile
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)


class ModelError(RuntimeError):
    pass


class LicenseError(ModelError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str
    url: str
    dest: str
    license: str
    commercial_use: bool | None
    license_url: str = ""
    archive: str | None = None
    files: dict[str, str] = field(default_factory=dict)
    optional: bool = False  # not fetched by a plain `models download`; only when asked for by name

    def path(self, root: Path | None = None) -> Path:
        return (root or config.models_dir()) / self.dest

    def file_paths(self, root: Path | None = None) -> dict[str, Path]:
        base = self.path(root)
        if self.archive is None:
            return {self.dest: base}
        return {name: base / name for name in self.files}


def load_manifest(text: str | None = None) -> dict[str, ModelSpec]:
    if text is None:
        text = resources.files(__package__).joinpath("manifest.toml").read_text(encoding="utf-8")
    data = tomllib.loads(text)
    return {name: ModelSpec(name=name, **entry) for name, entry in data.get("models", {}).items()}


def get_spec(name: str) -> ModelSpec:
    manifest = load_manifest()
    if name not in manifest:
        raise ModelError(f"unknown model {name!r}; known: {', '.join(manifest)}")
    return manifest[name]


_VERIFIED: dict[tuple[str, int, int], str] = {}


def _cached_sha256(path: Path) -> str:
    """Model files are large (up to ~550 MB); hash each one once per process unless it changes."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _VERIFIED:
        _VERIFIED[key] = sha256_file(path)
    return _VERIFIED[key]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def verify(spec: ModelSpec, root: Path | None = None) -> list[str]:
    """Return a list of problems (empty if all files are present and match)."""
    problems = []
    for name, path in spec.file_paths(root).items():
        expected = spec.files.get(name)
        if not path.is_file():
            problems.append(f"missing: {path}")
        elif expected and _cached_sha256(path) != expected:
            problems.append(f"sha256 mismatch: {path}")
    return problems


def present(spec: ModelSpec, root: Path | None = None) -> bool:
    """Cheap check (files exist); use verify() for integrity."""
    return all(p.is_file() for p in spec.file_paths(root).values())


def check_license(spec: ModelSpec, commercial: bool, license_override: str | None = None) -> None:
    """Refuse non-commercial models in a commercial project unless a license is recorded."""
    if commercial and spec.commercial_use is not True and not license_override:
        raise LicenseError(
            f"model {spec.name!r} is not licensed for commercial use ({spec.license}). "
            "Obtain a commercial license and record it as license_override, or use another model."
        )


def ensure(name: str, root: Path | None = None, download: bool = False) -> Path:
    """Return the model path, optionally downloading it. Raises if missing or corrupt."""
    spec = get_spec(name)
    problems = verify(spec, root)
    if problems and download:
        _download(spec, root)
        problems = verify(spec, root)
    if problems:
        hint = "" if download else f" Run `faceswapjp models download {name}` or place the files manually."
        raise ModelError(f"model {name!r} is not ready: {'; '.join(problems)}.{hint}")
    return spec.path(root)


def _download(spec: ModelSpec, root: Path | None) -> None:
    target = spec.path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.parent / (Path(spec.url).name + ".part")
    log.info("downloading %s from %s", spec.name, spec.url)
    with urllib.request.urlopen(spec.url) as resp, open(tmp, "wb") as out:  # noqa: S310 - fixed https URLs
        shutil.copyfileobj(resp, out, 1 << 20)
    if spec.archive == "zip":
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tmp) as zf:
            for member in zf.infolist():
                fname = Path(member.filename).name  # flatten; archives may contain a top-level folder
                if fname in spec.files:
                    with zf.open(member) as src, open(target / fname, "wb") as dst:
                        shutil.copyfileobj(src, dst)
        tmp.unlink()
    else:
        tmp.replace(target)
