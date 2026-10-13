import io
import zipfile

import pytest

from faceswapjp import updater


def _zip(files: dict[str, str], top: str = "faceswapjp-main") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, text in files.items():
            z.writestr(f"{top}/{name}" if top else name, text)
    return buf.getvalue()


def _source(version: str, deps: str = '"numpy"') -> dict[str, str]:
    return {
        "pyproject.toml": f'[project]\nname = "faceswapjp"\ndependencies = [{deps}]\n',
        "src/faceswapjp/__init__.py": f'__version__ = "{version}"\n',
        "src/faceswapjp/new_module.py": "x = 1\n",
        "macos/1_セットアップ.command": "#!/bin/bash\n",
        "README.md": "new\n",
    }


def _install(root, version="0.1.0b3", deps='"numpy"'):
    for name, text in _source(version, deps).items():
        if name.endswith("new_module.py"):
            continue
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (root / "src/faceswapjp/stale.py").write_text("old\n")
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").write_text("venv")
    (root / "models").mkdir()
    (root / "models/big.onnx").write_text("weights")


def test_version_check_compares_pep440():
    assert updater.is_newer("0.1.0b4", "0.1.0b3")
    assert not updater.is_newer("0.1.0b3", "0.1.0b3")
    assert updater.is_newer("0.1.0", "0.1.0b9")
    info = updater.check(fetch=lambda url: b'__version__ = "9.0.0"\n')
    assert info.available and info.latest == "9.0.0"


def test_git_checkout_is_not_self_updated(tmp_path):
    _install(tmp_path)
    (tmp_path / ".git").mkdir()
    assert "git" in updater.can_update(tmp_path)
    with pytest.raises(updater.UpdateError):
        updater.apply(tmp_path, fetch=lambda url: b"", run=lambda *a, **k: None)


def test_unpack_rejects_path_traversal(tmp_path):
    bad = _zip({"faceswapjp-main/../../evil.txt": "x"}, top="")
    with pytest.raises(updater.UpdateError):
        updater.unpack(bad, tmp_path)
    assert not (tmp_path.parent / "evil.txt").exists()


def test_apply_replaces_code_but_keeps_env_and_models(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    _install(root)
    calls = []
    version = updater.apply(root, fetch=lambda url: _zip(_source("0.1.0b4")),
                            run=lambda cmd, **kw: calls.append(cmd))
    assert version == "0.1.0b4"
    assert (root / "src/faceswapjp/__init__.py").read_text() == '__version__ = "0.1.0b4"\n'
    assert (root / "src/faceswapjp/new_module.py").exists()
    assert not (root / "src/faceswapjp/stale.py").exists()  # removed upstream -> removed here
    assert (root / ".venv/bin/python").read_text() == "venv"
    assert (root / "models/big.onnx").read_text() == "weights"
    assert (root / "macos/1_セットアップ.command").stat().st_mode & 0o111
    # dependencies unchanged: no reinstall, only the model check
    assert len(calls) == 1 and calls[0][1:] == ["models", "download"]


def test_apply_reinstalls_when_dependencies_change(tmp_path, monkeypatch):
    root = tmp_path / "app"
    root.mkdir()
    _install(root)
    monkeypatch.setattr(updater, "_uv", lambda: "/usr/bin/uv")
    calls = []
    updater.apply(root, fetch=lambda url: _zip(_source("0.1.0b4", deps='"numpy", "packaging"')),
                  run=lambda cmd, **kw: calls.append(cmd))
    assert calls[0][:3] == ["/usr/bin/uv", "pip", "install"]
