from __future__ import annotations

import copy

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto  # noqa: E402
from onnx import helper  # noqa: E402

from openpi.exporting.onnx_artifacts import onnx_model_identity  # noqa: E402
from openpi.exporting.onnx_artifacts import require_validated_models  # noqa: E402


def _write_model(path, *, dynamic=False):
    dimension = "batch" if dynamic else 1
    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"], name="identity")],
        "fixed",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [dimension, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [dimension, 2])],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), path)


def test_identity_records_fixed_contract_and_validation_detects_mutation(tmp_path):
    model_path = tmp_path / "encode-prefix.bf16.onnx"
    _write_model(model_path)
    identity = onnx_model_identity(model_path)
    assert identity["inputs"] == {"x": {"shape": [1, 2], "dtype": "FLOAT"}}

    report = {
        "passes": True,
        "precision": "bf16",
        "models": {"encode-prefix": identity},
        "provenance": {"track": "libero"},
    }
    require_validated_models(
        report,
        precision="bf16",
        models={"encode-prefix": model_path},
        provenance={"track": "libero"},
    )

    changed = copy.deepcopy(onnx.load(model_path))
    changed.graph.node[0].name = "changed"
    onnx.save(changed, model_path)
    with pytest.raises(ValueError, match="identity no longer matches"):
        require_validated_models(report, precision="bf16", models={"encode-prefix": model_path})


def test_identity_rejects_dynamic_shapes(tmp_path):
    model_path = tmp_path / "dynamic.onnx"
    _write_model(model_path, dynamic=True)
    with pytest.raises(ValueError, match="dynamic"):
        onnx_model_identity(model_path)
