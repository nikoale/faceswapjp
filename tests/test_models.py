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


def test_hyperswap_is_default_and_only_1c_is_fetched_by_default():
    from faceswapjp import config

    m = registry.load_manifest()
    assert config.DEFAULT_SWAPPER == "hyperswap_1c_256"
    assert m["hyperswap_1c_256"].commercial_use is False and not m["hyperswap_1c_256"].optional
    assert m["hyperswap_1a_256"].optional and m["hyperswap_1b_256"].optional


def test_hyperswap_normalization_roundtrip(tmp_path):
    """A stand-in graph with HyperSwap's I/O (identity on target) must give back the input crop."""
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    from faceswapjp.swappers.hyperswap import HyperSwap

    graph = helper.make_graph(
        [helper.make_node("Identity", ["target"], ["output"]),
         helper.make_node("ReduceMean", ["target"], ["mask"], axes=[1], keepdims=1),
         helper.make_node("Identity", ["source"], ["unused"])],
        "hs",
        [helper.make_tensor_value_info("source", TensorProto.FLOAT, [1, 512]),
         helper.make_tensor_value_info("target", TensorProto.FLOAT, [1, 3, 256, 256])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 256, 256]),
         helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, 256, 256]),
         helper.make_tensor_value_info("unused", TensorProto.FLOAT, [1, 512])],
    )
    path = tmp_path / "hs.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8), str(path))
    sw = HyperSwap(path, [("CPUExecutionProvider", {})], "hyperswap_test")
    crop = np.random.default_rng(0).random((256, 256, 3), dtype=np.float32)
    assert sw.input_size == 256
    np.testing.assert_allclose(sw.swap(crop, np.ones(512, np.float32)), crop, atol=1e-6)
