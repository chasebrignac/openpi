"""Fail-closed TensorRT policy adapter for split, one-step pi0.5 engines.

The adapter preserves OpenPI's ``BasePolicy.infer`` interface. It deliberately
keeps environment/model transforms on the host and only replaces the model
forward pass with the fixed-shape ``encode-prefix`` and ``decode-denoise``
TensorRT engines.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import dataclasses
import json
import pathlib
import re
import time
from typing import Any, Protocol

import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi.exporting.artifacts import sha256_file

IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
PREFIX_INPUTS = (
    "image_0",
    "image_1",
    "image_2",
    "image_mask_0",
    "image_mask_1",
    "image_mask_2",
    "lang_tokens",
    "lang_mask",
)
DECODER_BASE_INPUTS = ("state", "x_t", "timestep", "target_time")


@dataclasses.dataclass(frozen=True)
class RuntimeIdentity:
    image_digest: str
    instance_type: str
    instance_id: str
    gpu_inventory: tuple[str, ...]
    tensorrt_version: str

    @property
    def manifest_runtime(self) -> dict[str, str]:
        return {
            "image_digest": self.image_digest,
            "instance_type": self.instance_type,
            "instance_id": self.instance_id,
        }


@dataclasses.dataclass(frozen=True)
class ArtifactBundle:
    root: pathlib.Path
    precision: str
    track: str
    config: str
    checkpoint: Mapping[str, Any]
    prefix_plan: pathlib.Path
    decoder_plan: pathlib.Path
    build_manifest_path: pathlib.Path
    validation_report_path: pathlib.Path
    build_manifest: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: str


@dataclasses.dataclass(frozen=True)
class PolicyShape:
    action_horizon: int
    action_dim: int
    max_token_len: int


class Engine(Protocol):
    @property
    def inputs(self) -> Mapping[str, TensorSpec]: ...

    @property
    def outputs(self) -> Mapping[str, TensorSpec]: ...

    def execute(self, values: Mapping[str, Any]) -> Mapping[str, Any]: ...


def gpu_inventory() -> tuple[str, ...]:
    from openpi.exporting.runtime_identity import query_gpu_inventory

    return query_gpu_inventory()


def _version_major(version: str) -> int:
    dotted = re.search(r"(?:TensorRT(?:\s+version)?[^0-9]*)?([0-9]+)\.[0-9]+", version, re.IGNORECASE)
    if dotted:
        return int(dotted.group(1))
    compact = re.search(r"TensorRT\s+v([0-9]{6})", version, re.IGNORECASE)
    if compact:
        return int(compact.group(1)[:-4])
    raise ValueError(f"Cannot parse TensorRT version: {version!r}")


def _load_json(path: pathlib.Path, *, label: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load {label}: {path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} must contain a JSON object: {path}")
    return payload


def _portable_basename(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = pathlib.PurePath(value)
    if path.name != value and pathlib.PurePosixPath(value).name != value:
        # Stage manifests use absolute paths, but the deployment reference must
        # collapse to one unambiguous file inside the artifact directory.
        value = path.name
    if not value or value in {".", ".."}:
        raise ValueError(f"{label} has an invalid artifact name")
    return value


def _manifest_record(manifest: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    records = [
        record
        for record in manifest.get("artifacts", ())
        if isinstance(record, Mapping) and pathlib.PurePath(str(record.get("path", ""))).name == name
    ]
    if len(records) != 1:
        raise ValueError(f"Build manifest must identify exactly one artifact named {name!r}")
    record = records[0]
    if set(record) != {"path", "bytes", "sha256"}:
        raise ValueError(f"Build-manifest artifact {name!r} has an invalid record")
    return record


def _require_artifact(manifest: Mapping[str, Any], path: pathlib.Path) -> None:
    record = _manifest_record(manifest, path.name)
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != record.get("bytes")
        or sha256_file(path) != record.get("sha256")
    ):
        raise ValueError(f"Artifact no longer matches the engine build manifest: {path}")


def _require_source_manifests(build_manifest: Mapping[str, Any], root: pathlib.Path, identities: Any) -> None:
    if not isinstance(identities, list) or not identities:
        raise ValueError("TensorRT policy contract has no sealed source manifests")
    names: set[str] = set()
    for identity in identities:
        if not isinstance(identity, Mapping) or set(identity) != {"name", "bytes", "sha256"}:
            raise ValueError("TensorRT source-manifest identity is malformed")
        name = _portable_basename(identity["name"], label="source manifest")
        if name in names:
            raise ValueError(f"TensorRT source-manifest identity is duplicated: {name}")
        names.add(name)
        path = root / name
        _require_artifact(build_manifest, path)
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"Source manifest differs from the policy contract: {name}")


def load_artifact_bundle(
    artifact_dir: pathlib.Path,
    *,
    precision: str,
    track: str,
    dataset: str,
    dataset_revision: str,
    runtime: RuntimeIdentity,
) -> ArtifactBundle:
    """Load and verify the complete build/validation/runtime identity chain."""

    if precision not in {"bf16", "fp8"}:
        raise ValueError(f"Unsupported TensorRT precision: {precision!r}")
    if track not in {"libero", "droid"}:
        raise ValueError(f"Unsupported policy track: {track!r}")
    root = artifact_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"TensorRT artifact directory is unavailable: {root}")
    manifest_path = root / f"tensorrt-manifest.{precision}.json"
    manifest = _load_json(manifest_path, label="TensorRT build manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported TensorRT build-manifest schema")
    if manifest.get("stage") != f"tensorrt-build-{precision}" or manifest.get("track") != track:
        raise ValueError("TensorRT build stage/track differs from the requested policy")
    if manifest.get("source", {}).get("dirty") is not False:
        raise ValueError("TensorRT engine must be built from a clean source tree")
    if manifest.get("dataset") != {"name": dataset, "revision": dataset_revision}:
        raise ValueError("TensorRT build dataset identity differs from the requested policy")
    if manifest.get("runtime") != runtime.manifest_runtime:
        raise ValueError("TensorRT engine must run on its exact build instance and pinned image")

    details = manifest.get("details")
    if not isinstance(details, Mapping):
        raise ValueError("TensorRT build manifest has no details object")
    if details.get("strongly_typed") is not True:
        raise ValueError("TensorRT policy requires a strongly typed engine build")
    if details.get("precision_source") != "explicit ONNX tensor types and Q/DQ nodes":
        raise ValueError("TensorRT engine precision provenance differs")
    build_inventory = details.get("gpu_inventory")
    from openpi.exporting.runtime_identity import validate_gpu_inventory

    if not isinstance(build_inventory, list):
        raise ValueError("TensorRT build manifest has no valid GPU inventory")
    validated_build_inventory = validate_gpu_inventory(build_inventory)
    validated_runtime_inventory = validate_gpu_inventory(runtime.gpu_inventory)
    if validated_build_inventory != validated_runtime_inventory:
        raise ValueError("TensorRT engine GPU/driver inventory differs from its build GPU")
    build_version = details.get("tensorrt_version")
    if not isinstance(build_version, str) or _version_major(build_version) != 11:
        raise ValueError("TensorRT build is not pinned to major version 11")
    if _version_major(runtime.tensorrt_version) != 11:
        raise ValueError("TensorRT serving runtime is not pinned to major version 11")

    contract = details.get("policy_contract")
    if not isinstance(contract, Mapping) or set(contract) != {
        "schema_version",
        "protocol",
        "config",
        "checkpoint",
        "precision",
        "num_denoise_steps",
        "source_manifests",
        "export_runtime",
    }:
        raise ValueError("TensorRT build has no complete policy contract")
    expected_config = f"pi05_{track}_l09_snapflow"
    if contract.get("schema_version") != 1 or contract.get("protocol") != "openpi-policy-websocket-v1":
        raise ValueError("TensorRT policy protocol contract differs")
    if contract.get("config") != expected_config or contract.get("precision") != precision:
        raise ValueError("TensorRT policy config/precision contract differs")
    if contract.get("num_denoise_steps") != 1:
        raise ValueError("TensorRT closed-loop serving requires exactly one denoising step")
    checkpoint = contract.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("TensorRT policy checkpoint contract is missing")
    export_runtime = contract.get("export_runtime")
    if (
        not isinstance(export_runtime, Mapping)
        or set(export_runtime) != {"image_digest", "instance_type", "instance_id"}
        or export_runtime.get("image_digest") != runtime.image_digest
        or export_runtime.get("instance_type") != runtime.instance_type
        or export_runtime.get("instance_id") != runtime.instance_id
    ):
        raise ValueError("TensorRT policy export runtime differs from its build/serving runtime")
    _require_source_manifests(manifest, root, contract.get("source_manifests"))

    validation_name = _portable_basename(details.get("validation_report"), label="validation report")
    validation_path = root / validation_name
    prefix_plan = root / f"encode-prefix.{precision}.plan"
    decoder_plan = root / f"decode-denoise.{precision}.plan"
    for path in (manifest_path, validation_path, prefix_plan, decoder_plan):
        if path == manifest_path:
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
        else:
            _require_artifact(manifest, path)

    validation = _load_json(validation_path, label="ONNX validation report")
    if validation.get("schema_version") != 1 or validation.get("precision") != precision:
        raise ValueError("TensorRT validation schema/precision differs")
    if validation.get("passes") is not True:
        raise RuntimeError("Refusing to serve an engine without a passing numerical validation")
    expected_provenance = {
        "track": track,
        "dataset": dataset,
        "dataset_revision": dataset_revision,
        **runtime.manifest_runtime,
    }
    for key, expected in expected_provenance.items():
        if validation.get("provenance", {}).get(key) != expected:
            raise ValueError(f"TensorRT validation provenance differs for {key}")
    action_gate = validation.get("end_to_end_actions")
    if (
        not isinstance(action_gate, Mapping)
        or action_gate.get("bias_passes") is not True
        or action_gate.get("action_limits_pass") is not True
        or action_gate.get("action_gate_kind") != "corpus_envelope_not_hardware_safety"
    ):
        raise ValueError("TensorRT validation has no passing end-to-end action gate")

    return ArtifactBundle(
        root=root,
        precision=precision,
        track=track,
        config=expected_config,
        checkpoint=checkpoint,
        prefix_plan=prefix_plan,
        decoder_plan=decoder_plan,
        build_manifest_path=manifest_path,
        validation_report_path=validation_path,
        build_manifest=manifest,
    )


def _torch_dtype(trt: Any, torch: Any, dtype: Any) -> tuple[Any, str]:
    pairs = (
        (trt.float32, torch.float32, "float32"),
        (trt.float16, torch.float16, "float16"),
        (trt.bfloat16, torch.bfloat16, "bfloat16"),
        (trt.int64, torch.int64, "int64"),
        (trt.int32, torch.int32, "int32"),
        (trt.int8, torch.int8, "int8"),
        (trt.uint8, torch.uint8, "uint8"),
        (trt.bool, torch.bool, "bool"),
    )
    for trt_dtype, torch_dtype, name in pairs:
        if dtype == trt_dtype:
            return torch_dtype, name
    raise TypeError(f"Unsupported TensorRT policy I/O dtype: {dtype}")


class TensorRTEngine:
    """Fixed-shape TensorRT 11 engine using Torch-owned CUDA buffers."""

    def __init__(self, path: pathlib.Path) -> None:
        import tensorrt as trt
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT policy serving")
        self._trt = trt
        self._torch = torch
        self._path = path
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not deserialize engine: {path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"TensorRT could not create an execution context: {path}")
        inputs: dict[str, TensorSpec] = {}
        outputs: dict[str, TensorSpec] = {}
        self._torch_dtypes: dict[str, Any] = {}
        self._output_buffers: dict[str, Any] = {}
        self._input_buffers: dict[str, Any] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(dim) for dim in self._engine.get_tensor_shape(name))
            if not shape or any(dim < 0 for dim in shape):
                raise ValueError(f"TensorRT policy engine must be fixed-shape: {name}={shape}")
            torch_dtype, dtype_name = _torch_dtype(trt, torch, self._engine.get_tensor_dtype(name))
            self._torch_dtypes[name] = torch_dtype
            spec = TensorSpec(shape=shape, dtype=dtype_name)
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                inputs[name] = spec
            else:
                outputs[name] = spec
                tensor = torch.empty(shape, dtype=torch_dtype, device="cuda")
                self._output_buffers[name] = tensor
                self._context.set_tensor_address(name, tensor.data_ptr())
        self._inputs = inputs
        self._outputs = outputs

    @property
    def inputs(self) -> Mapping[str, TensorSpec]:
        return self._inputs

    @property
    def outputs(self) -> Mapping[str, TensorSpec]:
        return self._outputs

    def execute(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        if set(values) != set(self._inputs):
            raise ValueError(f"TensorRT input names differ: got={sorted(values)}, expected={sorted(self._inputs)}")
        for name, spec in self._inputs.items():
            value = values[name]
            if isinstance(value, self._torch.Tensor):
                tensor = value.to(device="cuda", dtype=self._torch_dtypes[name]).contiguous()
            else:
                tensor = (
                    self._torch.as_tensor(np.asarray(value), device="cuda")
                    .to(dtype=self._torch_dtypes[name])
                    .contiguous()
                )
            if tuple(tensor.shape) != spec.shape:
                raise ValueError(f"TensorRT input shape differs for {name}: {tuple(tensor.shape)} != {spec.shape}")
            self._input_buffers[name] = tensor
            self._context.set_tensor_address(name, tensor.data_ptr())
        stream = self._torch.cuda.current_stream()
        if not self._context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT policy execution failed: {self._path}")
        return self._output_buffers


def _cache_names(names: Sequence[str]) -> tuple[str, ...]:
    cache_names = tuple(sorted(name for name in names if name.startswith("cache_")))
    if not cache_names or len(cache_names) % 2:
        raise ValueError("TensorRT prefix must expose a non-empty even KV-cache boundary")
    expected = tuple(
        item
        for index in range(len(cache_names) // 2)
        for item in (f"cache_key_{index:02d}", f"cache_value_{index:02d}")
    )
    if cache_names != tuple(sorted(expected)):
        raise ValueError(f"TensorRT KV-cache names are not contiguous key/value pairs: {cache_names}")
    return cache_names


class TensorRTPipeline:
    def __init__(self, prefix: Engine, decoder: Engine, *, shape: PolicyShape) -> None:
        if set(prefix.inputs) != set(PREFIX_INPUTS):
            raise ValueError(f"Prefix input contract differs: {sorted(prefix.inputs)}")
        cache_names = _cache_names(tuple(prefix.outputs))
        if set(prefix.outputs) != {"prefix_pad_masks", *cache_names}:
            raise ValueError(f"Prefix output contract differs: {sorted(prefix.outputs)}")
        expected_decoder = {*DECODER_BASE_INPUTS, "prefix_pad_masks", *cache_names}
        if set(decoder.inputs) != expected_decoder or set(decoder.outputs) != {"actions"}:
            raise ValueError("Decoder input/output contract differs from the split one-step graph")
        for name in ("prefix_pad_masks", *cache_names):
            if prefix.outputs[name] != decoder.inputs[name]:
                raise ValueError(f"Prefix/decoder tensor boundary differs for {name}")
        expected_images = (1, 3, 224, 224)
        if any(prefix.inputs[f"image_{index}"].shape != expected_images for index in range(3)):
            raise ValueError("TensorRT policy images must use fixed batch-one 224x224 CHW inputs")
        if any(prefix.inputs[f"image_mask_{index}"].shape != (1,) for index in range(3)):
            raise ValueError("TensorRT image masks must have fixed batch-one shape")
        if any(prefix.inputs[f"image_{index}"].dtype != "float32" for index in range(3)):
            raise ValueError("TensorRT image inputs must remain float32")
        if any(prefix.inputs[f"image_mask_{index}"].dtype != "bool" for index in range(3)):
            raise ValueError("TensorRT image-mask inputs must remain boolean")
        if prefix.inputs["lang_tokens"].shape != (1, shape.max_token_len):
            raise ValueError("TensorRT prompt length differs from the policy config")
        if prefix.inputs["lang_mask"].shape != prefix.inputs["lang_tokens"].shape:
            raise ValueError("TensorRT prompt token/mask shapes differ")
        if prefix.inputs["lang_tokens"].dtype != "int64" or prefix.inputs["lang_mask"].dtype != "bool":
            raise ValueError("TensorRT prompt dtypes differ from the export contract")
        if decoder.inputs["state"].shape != (1, shape.action_dim):
            raise ValueError("TensorRT state dimension differs from the policy config")
        expected_actions = (1, shape.action_horizon, shape.action_dim)
        if decoder.inputs["x_t"].shape != expected_actions or decoder.outputs["actions"].shape != expected_actions:
            raise ValueError("TensorRT action shape differs from the policy config")
        if any(decoder.inputs[name].dtype != "float32" for name in DECODER_BASE_INPUTS):
            raise ValueError("TensorRT state/noise/time inputs must remain float32")
        if decoder.outputs["actions"].dtype != "float32":
            raise ValueError("TensorRT action output must remain float32")
        if decoder.inputs["timestep"].shape != (1,) or decoder.inputs["target_time"].shape != (1,):
            raise ValueError("TensorRT one-step time inputs must have fixed batch-one shape")
        self.prefix = prefix
        self.decoder = decoder
        self.shape = shape

    def execute(self, prefix_inputs: Mapping[str, Any], *, state: Any, noise: Any) -> Any:
        prefix_outputs = self.prefix.execute(prefix_inputs)
        decoder_inputs = {
            "state": state,
            "x_t": noise,
            "timestep": np.ones((1,), dtype=np.float32),
            "target_time": np.zeros((1,), dtype=np.float32),
            **prefix_outputs,
        }
        return self.decoder.execute(decoder_inputs)["actions"]


def _copy_structure(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _copy_structure(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_structure(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_structure(item) for item in value)
    return value


def _canonical_image(value: Any, *, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.shape == (224, 224, 3):
        image = np.transpose(image, (2, 0, 1))
    elif image.shape != (3, 224, 224):
        raise ValueError(f"Transformed image {name!r} has the wrong shape: {image.shape}")
    if image.dtype == np.uint8:
        image = image.astype(np.float32) / 255.0 * 2.0 - 1.0
    else:
        image = image.astype(np.float32, copy=False)
    if not np.all(np.isfinite(image)) or (image.size and (image.min() < -1.001 or image.max() > 1.001)):
        raise ValueError(f"Transformed image {name!r} is non-finite or outside [-1, 1]")
    return np.ascontiguousarray(image[None, ...])


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
        value = value.float().cpu() if value.is_cuda else value.float()
    return np.asarray(value)


class TensorRTPolicy(_base_policy.BasePolicy):
    def __init__(
        self,
        pipeline: TensorRTPipeline,
        *,
        input_transform: Callable[[dict], dict],
        output_transform: Callable[[dict], dict],
        metadata: Mapping[str, Any] | None = None,
        noise_source: Callable[[tuple[int, ...]], Any] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._input_transform = input_transform
        self._output_transform = output_transform
        self._metadata = dict(metadata or {})
        if noise_source is None:
            import torch

            def torch_noise(shape: tuple[int, ...]) -> Any:
                return torch.randn(shape, dtype=torch.float32, device="cuda")

            noise_source = torch_noise
        self._noise_source = noise_source

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        transformed = self._input_transform(_copy_structure(obs))
        if not isinstance(transformed, Mapping):
            raise TypeError("TensorRT input transforms must return a mapping")
        images = transformed.get("image")
        masks = transformed.get("image_mask")
        if not isinstance(images, Mapping) or tuple(images) != IMAGE_KEYS:
            raise ValueError(f"TensorRT policy requires canonical camera order {IMAGE_KEYS}")
        if not isinstance(masks, Mapping) or tuple(masks) != IMAGE_KEYS:
            raise ValueError(f"TensorRT policy requires canonical camera masks {IMAGE_KEYS}")
        prefix_inputs: dict[str, Any] = {}
        for index, name in enumerate(IMAGE_KEYS):
            prefix_inputs[f"image_{index}"] = _canonical_image(images[name], name=name)
            mask = np.asarray(masks[name], dtype=np.bool_)
            if mask.shape != ():
                raise ValueError(f"Transformed image mask {name!r} must be scalar, got {mask.shape}")
            prefix_inputs[f"image_mask_{index}"] = mask.reshape(1)
        tokens = np.asarray(transformed.get("tokenized_prompt"), dtype=np.int64)
        token_mask = np.asarray(transformed.get("tokenized_prompt_mask"), dtype=np.bool_)
        if tokens.shape != (self._pipeline.shape.max_token_len,) or token_mask.shape != tokens.shape:
            raise ValueError("Transformed prompt tensors differ from the fixed TensorRT contract")
        prefix_inputs["lang_tokens"] = tokens[None, ...]
        prefix_inputs["lang_mask"] = token_mask[None, ...]
        state = np.asarray(transformed.get("state"), dtype=np.float32)
        if state.shape != (self._pipeline.shape.action_dim,) or not np.all(np.isfinite(state)):
            raise ValueError("Transformed state differs from the fixed TensorRT contract")
        state_batch = np.ascontiguousarray(state[None, ...])

        action_shape = (
            1,
            self._pipeline.shape.action_horizon,
            self._pipeline.shape.action_dim,
        )
        if noise is None:
            noise_value = self._noise_source(action_shape)
        else:
            noise_value = np.asarray(noise, dtype=np.float32)
            if noise_value.shape == action_shape[1:]:
                noise_value = noise_value[None, ...]
            if noise_value.shape != action_shape or not np.all(np.isfinite(noise_value)):
                raise ValueError(f"Explicit noise must have shape {action_shape[1:]} or {action_shape}")

        start = time.monotonic()
        actions = _to_numpy(self._pipeline.execute(prefix_inputs, state=state_batch, noise=noise_value))
        model_time = time.monotonic() - start
        if actions.shape != action_shape or not np.all(np.isfinite(actions)):
            raise RuntimeError(f"TensorRT engine returned invalid actions: {actions.shape}")
        outputs = self._output_transform({"state": state, "actions": actions[0]})
        if not isinstance(outputs, dict) or "actions" not in outputs:
            raise TypeError("TensorRT output transforms must return an actions mapping")
        outputs["policy_timing"] = {"infer_ms": model_time * 1000}
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def _require_checkpoint(bundle: ArtifactBundle, checkpoint_dir: pathlib.Path) -> None:
    root = checkpoint_dir.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"Checkpoint directory is unavailable: {root}")
    checkpoint = bundle.checkpoint
    if set(checkpoint) != {"path", "sha256", "assets"}:
        raise ValueError("TensorRT checkpoint contract has an invalid schema")
    weights = root / pathlib.PurePath(str(checkpoint["path"])).name
    if weights.name != "model.safetensors" or weights.is_symlink() or not weights.is_file():
        raise FileNotFoundError(f"TensorRT source checkpoint weights are unavailable: {weights}")
    if sha256_file(weights) != checkpoint["sha256"]:
        raise ValueError("TensorRT source checkpoint hash differs from the export contract")
    identities = checkpoint["assets"]
    if not isinstance(identities, list) or not identities:
        raise ValueError("TensorRT checkpoint contract has no normalization assets")
    assets_root = root / "assets"
    entries = tuple(assets_root.rglob("*")) if assets_root.is_dir() and not assets_root.is_symlink() else ()
    if not entries or any(path.is_symlink() for path in entries):
        raise ValueError("TensorRT checkpoint assets are absent or contain symlinks")
    actual_files = {path.relative_to(root).as_posix(): path for path in entries if path.is_file()}
    expected_files: dict[str, Mapping[str, Any]] = {}
    for identity in identities:
        if not isinstance(identity, Mapping) or set(identity) != {"name", "bytes", "sha256"}:
            raise ValueError("TensorRT checkpoint asset identity is malformed")
        name = identity.get("name")
        path = pathlib.PurePosixPath(str(name))
        if path.is_absolute() or ".." in path.parts or not str(name).startswith("assets/"):
            raise ValueError("TensorRT checkpoint asset path is not relocatable")
        if str(name) in expected_files:
            raise ValueError(f"TensorRT checkpoint asset is duplicated: {name}")
        expected_files[str(name)] = identity
    if set(actual_files) != set(expected_files):
        raise ValueError("Checkpoint asset file set differs from the sealed export contract")
    for name, path in actual_files.items():
        identity = expected_files[name]
        if path.stat().st_size != identity["bytes"] or sha256_file(path) != identity["sha256"]:
            raise ValueError(f"Checkpoint asset differs from the sealed export contract: {name}")
    if sum(pathlib.PurePosixPath(name).name == "norm_stats.json" for name in expected_files) != 1:
        raise ValueError("TensorRT checkpoint must bind exactly one normalization asset")


def create_policy(
    bundle: ArtifactBundle,
    checkpoint_dir: pathlib.Path,
    *,
    default_prompt: str | None = None,
    engine_factory: Callable[[pathlib.Path], Engine] = TensorRTEngine,
    noise_source: Callable[[tuple[int, ...]], Any] | None = None,
) -> TensorRTPolicy:
    """Create an engine-backed policy with the training config's exact transforms."""

    import dataclasses as dc

    from openpi import transforms
    from openpi.models import pi0_config
    from openpi.training import config as training_config

    _require_checkpoint(bundle, checkpoint_dir)
    train_config = training_config.get_config(bundle.config)
    if not isinstance(train_config.model, pi0_config.SnapFlowPi0Config):
        raise ValueError("TensorRT policy config is not a nine-layer SnapFlow model")
    assets = dc.replace(train_config.data.assets, assets_dir=str(checkpoint_dir.resolve() / "assets"))
    data_factory = dc.replace(train_config.data, assets=assets)
    data_config = data_factory.create(train_config.assets_dirs, train_config.model)
    if data_config.asset_id is None or data_config.norm_stats is None:
        raise ValueError("TensorRT policy config did not load checkpoint-local normalization stats")
    input_transform = transforms.compose(
        [
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_transform = transforms.compose(
        [
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )
    shape = PolicyShape(
        action_horizon=train_config.model.action_horizon,
        action_dim=train_config.model.action_dim,
        max_token_len=train_config.model.max_token_len,
    )
    pipeline = TensorRTPipeline(
        engine_factory(bundle.prefix_plan),
        engine_factory(bundle.decoder_plan),
        shape=shape,
    )
    metadata = dict(train_config.policy_metadata or {})
    metadata["tensorrt_policy"] = {
        "precision": bundle.precision,
        "track": bundle.track,
        "config": bundle.config,
        "build_manifest_sha256": sha256_file(bundle.build_manifest_path),
    }
    return TensorRTPolicy(
        pipeline,
        input_transform=input_transform,
        output_transform=output_transform,
        metadata=metadata,
        noise_source=noise_source,
    )
