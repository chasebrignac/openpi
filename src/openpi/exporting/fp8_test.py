from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from openpi.exporting.fp8 import prepare_fp32_calibration_model
from openpi.exporting.fp8 import quantization_audit
from openpi.exporting.fp8 import select_transformer_mlp_nodes

onnx = pytest.importorskip("onnx")
from onnx import TensorProto  # noqa: E402
from onnx import helper  # noqa: E402
from onnx import numpy_helper  # noqa: E402


@dataclasses.dataclass
class _Node:
    name: str
    op_type: str = "MatMul"


def test_selects_only_named_mlp_projections():
    nodes = [
        _Node("/gemma/layers.0/mlp/gate_proj/MatMul"),
        _Node("/gemma/layers.0/mlp/down_proj/MatMul"),
        _Node("/gemma/layers.0/self_attn/q_proj/MatMul"),
        _Node("/gemma/layers.0/input_layernorm/Mul", "Mul"),
    ]
    assert select_transformer_mlp_nodes(nodes) == [
        "/gemma/layers.0/mlp/gate_proj/MatMul",
        "/gemma/layers.0/mlp/down_proj/MatMul",
    ]


def test_selector_fails_closed_without_semantic_names():
    with pytest.raises(ValueError, match="No semantically named"):
        select_transformer_mlp_nodes([_Node("node_123")])


def test_override_still_excludes_attention():
    nodes = [_Node("expert_keep"), _Node("self_attn_keep")]
    assert select_transformer_mlp_nodes(nodes, include_regex="keep") == ["expert_keep"]


def test_prepare_fp32_calibration_model_removes_bf16_surfaces(tmp_path):
    weight = helper.make_tensor(
        "weight",
        TensorProto.BFLOAT16,
        [1, 1],
        np.asarray([0x4000], dtype="<u2").tobytes(),
        raw=True,
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["x", "weight"], ["y"], name="/mlp/up_proj/MatMul")],
        "bf16",
        [helper.make_tensor_value_info("x", TensorProto.BFLOAT16, [1, 1])],
        [helper.make_tensor_value_info("y", TensorProto.BFLOAT16, [1, 1])],
        [weight],
    )
    source = tmp_path / "model.bf16.onnx"
    output = tmp_path / "model.fp32.onnx"
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)]), source)

    stats = prepare_fp32_calibration_model(source, output)
    converted = onnx.load(output, load_external_data=True)

    assert stats["initializers"] == 1
    assert stats["inputs"] == 1
    assert stats["outputs"] == 1
    assert converted.graph.initializer[0].data_type == TensorProto.FLOAT
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    np.testing.assert_array_equal(numpy_helper.to_array(converted.graph.initializer[0]), [[2.0]])


def test_quantization_audit_requires_exact_selected_matmul_set():
    selected = "/model/layer/mlp/up_proj/MatMul"
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [], [1.0])
    zero_point = helper.make_tensor("zero_point", TensorProto.FLOAT8E4M3FN, [], b"\x00", raw=True)
    graph = helper.make_graph(
        [
            helper.make_node("QuantizeLinear", ["w", "scale", "zero_point"], ["w_q"], name="quantize"),
            helper.make_node("DequantizeLinear", ["w_q", "scale", "zero_point"], ["w_dq"], name="dequantize"),
            helper.make_node("MatMul", ["x", "w_dq"], ["y"], name=selected),
        ],
        "quantized",
        [],
        [],
        [scale, zero_point],
    )
    model = helper.make_model(graph)
    audit = quantization_audit(model, [selected])
    assert audit["verified_quantized_nodes"] == [selected]

    with pytest.raises(ValueError, match="missing_selected"):
        quantization_audit(model, [selected, "/model/layer/mlp/down_proj/MatMul"])
