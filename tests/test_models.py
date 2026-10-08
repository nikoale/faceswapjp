import pytest

from faceswapjp import runtime
from faceswapjp.models import registry
from faceswapjp.project import Project


def test_select_providers_auto_prefers_coreml_then_cuda():
    avail = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    assert runtime.provider_names(runtime.select_providers("auto", avail)) == ["CoreMLExecutionProvider", "CPUExecutionProvider"]
    avail = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.provider_names(runtime.select_providers("auto", avail)) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.provider_names(runtime.select_providers("coreml", ["CPUExecutionProvider"])) == ["CPUExecutionProvider"]
    with pytest.raises(ValueError):
        runtime.select_providers("tpu", [])


def test_manifest_lists_non_commercial_defaults():
    m = registry.load_manifest()
    assert m["inswapper_128"].commercial_use is False
    assert m["buffalo_l"].commercial_use is False
    assert all(len(h) == 64 for spec in m.values() for h in spec.files.values())


def test_license_gate():
    spec = registry.get_spec("inswapper_128")
    registry.check_license(spec, commercial=False)
    with pytest.raises(registry.LicenseError):
        registry.check_license(spec, commercial=True)
    registry.check_license(spec, commercial=True, license_override="CONTRACT-123")


def test_commercial_project_refuses_default_models(tmp_path):
    from faceswapjp.engine import check_project_licenses

    project = Project.init(tmp_path / "p", commercial=True)
    with pytest.raises(registry.LicenseError):
        check_project_licenses(project, ["buffalo_l", "inswapper_128"])


def test_verify_detects_missing_and_corrupt(tmp_path):
    spec = registry.ModelSpec(
        name="t", kind="swapper", url="", dest="t.onnx", license="", commercial_use=None,
        files={"t.onnx": "0" * 64},
    )
    assert registry.verify(spec, tmp_path)[0].startswith("missing")
    (tmp_path / "t.onnx").write_bytes(b"x")
    assert "mismatch" in registry.verify(spec, tmp_path)[0]
    with pytest.raises(registry.ModelError):
        registry.ensure("does-not-exist")
