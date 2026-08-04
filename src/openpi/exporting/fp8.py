# ruff: noqa: PLC0415
"""Selective ModelOpt FP8 node selection and graph auditing."""

from __future__ import annotations

from collections.abc import Iterable
import pathlib
import re
from typing import Any

import numpy as np

_MLP_COMPONENT = re.compile(r"(?:^|[./_])(mlp|feed_forward|feedforward|ffn)(?:[./_]|$)", re.IGNORECASE)
_MLP_PROJECTION = re.compile(
    r"(?:gate_proj|up_proj|down_proj|fc1|fc2|dense_h_to_4h|dense_4h_to_h)",
    re.IGNORECASE,
)
_SENSITIVE = re.compile(r"(?:self_attn|attention|attn|softmax|norm|rotary)", re.IGNORECASE)


def select_transformer_mlp_nodes(nodes: Iterable[Any], *, include_regex: str | None = None) -> list[str]:
    """Return named MatMul/Gemm nodes belonging only to transformer MLPs.

    PyTorch's ONNX exporter normally preserves module paths in node names.  We
    fail closed when it does not: a human can inspect the graph and pass a
    narrower ``include_regex`` rather than accidentally quantizing attention.
    """

    override = re.compile(include_regex) if include_regex else None
    selected: list[str] = []
    unnamed = 0
    for node in nodes:
        if getattr(node, "op_type", None) not in {"MatMul", "Gemm"}:
            continue
        name = str(getattr(node, "name", ""))
        if not name:
            unnamed += 1
            continue
        semantic_match = bool(_MLP_COMPONENT.search(name) and _MLP_PROJECTION.search(name))
        if override is not None:
            semantic_match = bool(override.search(name))
        if semantic_match and not _SENSITIVE.search(name):
            selected.append(name)
    if not selected:
        suffix = f" ({unnamed} unnamed MatMul/Gemm nodes)" if unnamed else ""
        raise ValueError(
            "No semantically named transformer MLP MatMul/Gemm nodes were found" + suffix + "; inspect the ONNX graph"
        )
    if len(selected) != len(set(selected)):
        raise ValueError("ONNX graph contains duplicate selected node names")
    return selected


def prepare_fp32_calibration_model(source: pathlib.Path, output: pathlib.Path) -> dict[str, int]:
    """Convert BF16 graph surfaces to FP32 for ModelOpt 0.45 calibration.

    ModelOpt 0.45's ONNX calibrator selects only FLOAT and FLOAT16 activation
    tensors. The exported model is BF16, so calibrating it directly silently
    yields no ranges for the requested MLPs. This temporary bridge retains the
    exact BF16-rounded weights as FP32 values; ModelOpt converts unquantized
    operations back to BF16 when it emits the final FP8 graph.
    """

    import onnx
    from onnx import AttributeProto
    from onnx import TensorProto
    from onnx import numpy_helper

    source = source.resolve()
    output = output.resolve()
    model = onnx.load(source, load_external_data=True)
    stats = {
        "initializers": 0,
        "attribute_tensors": 0,
        "casts": 0,
        "value_info": 0,
        "inputs": 0,
        "outputs": 0,
    }

    def convert_tensor(tensor) -> bool:
        if tensor.data_type != TensorProto.BFLOAT16:
            return False
        if tensor.raw_data:
            bits = np.frombuffer(tensor.raw_data, dtype="<u2")
        elif tensor.int32_data:
            bits = np.asarray(tensor.int32_data, dtype="<u2")
        else:
            # Older ONNX releases expose BF16 as a structured uint16 dtype.
            bits = np.asarray(numpy_helper.to_array(tensor)).view("<u2").reshape(-1)
        array = (bits.astype("<u4") << 16).view("<f4").reshape(tuple(tensor.dims))
        replacement = numpy_helper.from_array(array, name=tensor.name)
        replacement.doc_string = tensor.doc_string
        tensor.CopyFrom(replacement)
        return True

    def convert_value(value, category: str) -> None:
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            value.type.tensor_type.elem_type = TensorProto.FLOAT
            stats[category] += 1

    def convert_graph(graph) -> None:
        for initializer in graph.initializer:
            stats["initializers"] += int(convert_tensor(initializer))
        for value in graph.value_info:
            convert_value(value, "value_info")
        for value in graph.input:
            convert_value(value, "inputs")
        for value in graph.output:
            convert_value(value, "outputs")
        for node in graph.node:
            if node.op_type == "Cast":
                for attribute in node.attribute:
                    if attribute.name == "to" and attribute.i == TensorProto.BFLOAT16:
                        attribute.i = TensorProto.FLOAT
                        stats["casts"] += 1
            for attribute in node.attribute:
                if attribute.type == AttributeProto.TENSOR:
                    stats["attribute_tensors"] += int(convert_tensor(attribute.t))
                elif attribute.type == AttributeProto.TENSORS:
                    for tensor in attribute.tensors:
                        stats["attribute_tensors"] += int(convert_tensor(tensor))
                elif attribute.type == AttributeProto.GRAPH:
                    convert_graph(attribute.g)
                elif attribute.type == AttributeProto.GRAPHS:
                    for nested in attribute.graphs:
                        convert_graph(nested)

    convert_graph(model.graph)
    converted = sum(stats.values())
    if converted == 0:
        raise ValueError(f"Expected a BF16 export, but no BF16 graph surfaces were found: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=output.name + ".data",
        size_threshold=0,
        convert_attribute=False,
    )
    onnx.checker.check_model(str(output))
    reloaded = onnx.load(output, load_external_data=False)
    remaining = [
        value.name
        for value in (*reloaded.graph.input, *reloaded.graph.output, *reloaded.graph.value_info)
        if value.type.HasField("tensor_type") and value.type.tensor_type.elem_type == TensorProto.BFLOAT16
    ]
    remaining.extend(
        initializer.name for initializer in reloaded.graph.initializer if initializer.data_type == TensorProto.BFLOAT16
    )
    if remaining:
        raise RuntimeError(f"BF16 calibration bridge is incomplete: {remaining[:10]}")
    return stats


def quantization_audit(model: Any, selected_nodes: list[str]) -> dict[str, Any]:
    nodes = list(model.graph.node)
    q_nodes = [node for node in nodes if node.op_type == "QuantizeLinear"]
    dq_nodes = [node for node in nodes if node.op_type == "DequantizeLinear"]
    if not q_nodes or not dq_nodes:
        raise ValueError("FP8 graph has no explicit QuantizeLinear/DequantizeLinear nodes")
    import onnx

    initializer_types = {initializer.name: initializer.data_type for initializer in model.graph.initializer}
    non_fp8_q_nodes = [
        node.name
        for node in q_nodes
        if len(node.input) < 3 or initializer_types.get(node.input[2]) != onnx.TensorProto.FLOAT8E4M3FN
    ]
    if non_fp8_q_nodes:
        raise ValueError(f"QuantizeLinear nodes are not explicitly FP8 E4M3: {non_fp8_q_nodes[:10]}")
    producers = {output: node for node in nodes for output in node.output}
    quantized_nodes = {
        node.name
        for node in nodes
        if node.op_type in {"MatMul", "Gemm"}
        and any(
            input_name in producers and producers[input_name].op_type == "DequantizeLinear" for input_name in node.input
        )
    }
    selected = set(selected_nodes)
    missing = sorted(selected - quantized_nodes)
    unexpected = sorted(quantized_nodes - selected)
    sensitive = sorted(name for name in quantized_nodes if _SENSITIVE.search(name))
    if missing or unexpected or sensitive:
        raise ValueError(
            "Selective FP8 audit failed: "
            f"missing_selected={missing[:10]}, unexpected_quantized={unexpected[:10]}, "
            f"sensitive_quantized={sensitive[:10]}"
        )
    return {
        "selected_mlp_nodes": selected_nodes,
        "selected_mlp_node_count": len(selected_nodes),
        "verified_quantized_nodes": sorted(quantized_nodes),
        "verified_quantized_node_count": len(quantized_nodes),
        "quantize_linear_count": len(q_nodes),
        "fp8_e4m3_quantize_linear_count": len(q_nodes),
        "dequantize_linear_count": len(dq_nodes),
        "sensitive_node_policy": "attention, softmax, normalization and rotary nodes excluded by allow-list",
    }
