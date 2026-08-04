# ruff: noqa: PLC0415
"""Portable identities and fixed-shape contracts for ONNX artifacts."""

from __future__ import annotations

from collections.abc import Mapping
import pathlib
from typing import Any

from openpi.exporting.artifacts import sha256_file


def file_identity(path: pathlib.Path, *, name: str | None = None) -> dict[str, Any]:
    """Return a relocatable content identity rather than an absolute path."""

    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "name": path.name if name is None else name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _fixed_tensor_contract(values, *, surface: str) -> dict[str, Any]:
    import onnx

    result: dict[str, Any] = {}
    for value in values:
        if value.name in result:
            raise ValueError(f"Duplicate ONNX {surface} name: {value.name!r}")
        tensor_type = value.type.tensor_type
        if not value.type.HasField("tensor_type"):
            raise TypeError(f"ONNX {surface} {value.name!r} is not a tensor")
        shape = []
        for dimension in tensor_type.shape.dim:
            if not dimension.HasField("dim_value"):
                label = dimension.dim_param or "unknown"
                raise ValueError(f"ONNX {surface} {value.name!r} is dynamic: {label}")
            shape.append(int(dimension.dim_value))
        result[value.name] = {
            "shape": shape,
            "dtype": onnx.TensorProto.DataType.Name(tensor_type.elem_type),
        }
    return result


def _external_locations(model) -> list[str]:
    locations = {
        field.value
        for initializer in model.graph.initializer
        for field in initializer.external_data
        if field.key == "location"
    }
    return sorted(locations)


def onnx_model_identity(model_path: pathlib.Path) -> dict[str, Any]:
    """Hash an ONNX graph and external weights and reject dynamic I/O."""

    import onnx

    model_path = model_path.resolve()
    model = onnx.load(model_path, load_external_data=False)
    root = model_path.parent.resolve()
    external_data = []
    for location in _external_locations(model):
        external_path = (root / location).resolve()
        try:
            relative = external_path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"ONNX external data escapes the artifact directory: {location!r}") from exc
        external_data.append(file_identity(external_path, name=relative.as_posix()))
    return {
        "model": file_identity(model_path),
        "external_data": external_data,
        "opsets": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "inputs": _fixed_tensor_contract(model.graph.input, surface="input"),
        "outputs": _fixed_tensor_contract(model.graph.output, surface="output"),
    }


def require_validated_models(
    report: Mapping[str, Any],
    *,
    precision: str,
    models: Mapping[str, pathlib.Path],
    provenance: Mapping[str, str] | None = None,
) -> None:
    """Tie a passing numerical report to the exact graphs being consumed."""

    if report.get("passes") is not True:
        raise RuntimeError("Refusing an ONNX artifact without a passing validation report")
    if report.get("precision") != precision:
        raise ValueError(f"Validation precision {report.get('precision')!r} does not match requested {precision!r}")
    recorded_models = report.get("models")
    if not isinstance(recorded_models, Mapping) or set(recorded_models) != set(models):
        raise ValueError("Validation report does not identify the complete split ONNX model set")
    for name, path in models.items():
        actual = onnx_model_identity(path)
        if recorded_models[name] != actual:
            raise ValueError(f"Validated ONNX identity no longer matches {name}: {path}")
    if provenance is not None:
        recorded_provenance = report.get("provenance")
        for key, expected in provenance.items():
            if not isinstance(recorded_provenance, Mapping) or recorded_provenance.get(key) != expected:
                raise ValueError(
                    f"Validation provenance mismatch for {key}: "
                    f"recorded={None if not isinstance(recorded_provenance, Mapping) else recorded_provenance.get(key)!r}, "
                    f"expected={expected!r}"
                )
