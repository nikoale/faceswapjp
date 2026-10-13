"""In-app update for installs made from the GitHub ZIP (macos/1_セットアップ.command).

Only runs when the user presses the update button. It talks to GitHub alone (version check
and the source ZIP); no footage, project data or identity information is sent anywhere.
"""

from __future__ import annotations

import io
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import tomllib
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from packaging.version import InvalidVersion, Version

from . import __version__

REPO = os.environ.get("FACESWAPJP_UPDATE_REPO", "nikoale/faceswapjp")
BRANCH = os.environ.get("FACESWAPJP_UPDATE_BRANCH", "claude/determined-feynman-t3b2na")
# Never touched by an update: the Python environment, downloaded models and anything git-managed.
KEEP = {".venv", "models", ".git", "projects"}
TIMEOUT = 30


class UpdateError(Exception):
    pass


@dataclass
class UpdateInfo:
    current: str
    latest: str
    available: bool


def install_root() -> Path:
    """Folder the app was installed into (the one holding pyproject.toml)."""
    return Path(__file__).resolve().parents[2]


def can_update(root: Path | None = None) -> str | None:
    """None if this install can update itself, else the reason (shown to the user)."""
    root = root or install_root()
    if (root / ".git").exists():
        return "git で取得したフォルダです。git pull で更新してください"
    if not (root / "pyproject.toml").is_file() or not (root / ".venv").is_dir():
        return "セットアップで作ったフォルダではないため、自動更新できません"
    return None


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": f"faceswapjp/{__version__}"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            return r.read()
    except OSError as exc:
        raise UpdateError("GitHub に接続できませんでした。インターネット接続を確認してください") from exc


def parse_version(init_py: str) -> str:
    m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_py, re.M)
    if not m:
        raise UpdateError("最新版のバージョンを読み取れませんでした")
    return m.group(1)


def is_newer(latest: str, current: str) -> bool:
    try:
        return Version(latest) > Version(current)
    except InvalidVersion:
        return False


def check(fetch: Callable[[str], bytes] = _get) -> UpdateInfo:
    url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/src/faceswapjp/__init__.py"
    latest = parse_version(fetch(url).decode("utf-8"))
    return UpdateInfo(__version__, latest, is_newer(latest, __version__))


def _dependencies(pyproject: Path) -> dict:
    data = tomllib.loads(pyproject.read_text(encoding="utf-8")).get("project", {})
    return {"deps": data.get("dependencies", []), "extras": data.get("optional-dependencies", {})}


def unpack(archive: bytes, dest: Path) -> Path:
    """Extract a GitHub source ZIP safely and return its top-level folder."""
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        names = z.namelist()
        tops = {n.split("/", 1)[0] for n in names}
        if len(tops) != 1:
            raise UpdateError("ダウンロードしたファイルの形式が想定と違います")
        dest_r = dest.resolve()
        for n in names:
            target = (dest / n).resolve()
            if dest_r not in target.parents and target != dest_r:
                raise UpdateError("ダウンロードしたファイルに不正なパスが含まれています")
        z.extractall(dest)
    top = dest / tops.pop()
    if not (top / "pyproject.toml").is_file() or not (top / "src" / "faceswapjp" / "__init__.py").is_file():
        raise UpdateError("ダウンロードしたファイルに faceswapjp が含まれていません")
    return top


def sync_tree(new: Path, root: Path) -> None:
    """Replace the installed files with `new`, keeping KEEP entries. Files removed upstream are removed."""
    for item in root.iterdir():
        if item.name in KEEP or item.name.startswith("."):
            continue
        if item.is_dir() and not item.is_symlink():
            shutil.rmtree(item)
        else:
            item.unlink()
    for item in new.iterdir():
        if item.name in KEEP:
            continue
        dst = root / item.name
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        elif dst.exists():
            dst.unlink()
        if item.is_dir():
            shutil.copytree(item, dst)
        else:
            shutil.copy2(item, dst)
    for script in (root / "macos").glob("*.command"):
        script.chmod(0o755)


def _uv() -> str | None:
    extra = [str(Path.home() / ".local/bin"), "/opt/homebrew/bin", "/usr/local/bin"]
    return shutil.which("uv", path=os.pathsep.join([os.environ.get("PATH", ""), *extra]))


def apply(root: Path | None = None, progress: Callable[[str], None] = lambda _: None,
          fetch: Callable[[str], bytes] = _get, run: Callable[..., object] = subprocess.run) -> str:
    """Download the latest source and install it over `root`. Returns the new version."""
    root = root or install_root()
    reason = can_update(root)
    if reason:
        raise UpdateError(reason)
    progress("最新版をダウンロードしています…")
    archive = fetch(f"https://codeload.github.com/{REPO}/zip/refs/heads/{BRANCH}")
    with tempfile.TemporaryDirectory(prefix="faceswapjp-update-") as tmp:
        new = unpack(archive, Path(tmp))
        version = parse_version((new / "src" / "faceswapjp" / "__init__.py").read_text(encoding="utf-8"))
        deps_changed = _dependencies(new / "pyproject.toml") != _dependencies(root / "pyproject.toml")
        progress("ファイルを入れ替えています…")
        sync_tree(new, root)
    python = root / ".venv" / "bin" / "python"
    if deps_changed:
        progress("必要な部品を更新しています…")
        uv = _uv()
        if not uv:
            raise UpdateError("部品の更新に必要な uv が見つかりません。「1_セットアップ.command」を実行してください")
        run([uv, "pip", "install", "-q", "--python", str(python), "-e", ".[mac,ui]" if sys.platform == "darwin"
             else ".[cpu,ui]"], cwd=root, check=True)
    progress("AI モデルを確認しています…")
    run([str(root / ".venv" / "bin" / "faceswapjp"), "models", "download"], cwd=root, check=True)
    return version


def restart_later(argv: list[str], port: int) -> None:
    """Start `argv` once `port` is free (after this server exits), detached from this process."""
    script = (
        "import socket,subprocess,sys,time\n"
        "port=int(sys.argv[1])\n"
        "for _ in range(120):\n"
        "    s=socket.socket()\n"
        "    s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)\n"  # like uvicorn: TIME_WAIT is fine
        "    try:\n"
        "        s.bind(('127.0.0.1',port)); s.close(); break\n"
        "    except OSError:\n"
        "        s.close(); time.sleep(0.5)\n"
        "subprocess.Popen(sys.argv[2:])\n"
    )
    subprocess.Popen([sys.executable, "-c", script, str(port), *argv], start_new_session=True,
                     stdin=subprocess.DEVNULL)


def relaunch_argv() -> list[str]:
    """The command that started this server, without opening another browser tab."""
    args = [a for a in sys.argv[1:] if a != "--no-browser"]
    return [sys.argv[0], *args, "--no-browser"]

