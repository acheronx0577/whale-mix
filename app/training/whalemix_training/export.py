"""ONNX export, validation and the inference bundle manifest consumed by the Rust engine.

Bundle kinds:
- `fixture-untrained`: a randomly initialised network exported only to test the export, parity and
  runtime plumbing. It is written under `<output_dir>/fixture-bundles/`, carries no thresholds that mean
  anything, and the engine refuses it unless the developer passes `--allow-fixture-bundle`.
- `selected`: the exported selected checkpoint of a qualified training lineage (future run).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

from whalemix_training.hashing import canonical_hash, sha256_file
from whalemix_training.io_utils import atomic_write_json
from whalemix_training.labels import LabelMap, load_profiles
from whalemix_training.model import AcousticNet, ProbabilityHead
from whalemix_training.preprocess import PreprocessingSpec

BUNDLE_SCHEMA = "whalemix-model-bundle/1"
OPSET = 18
INPUT_NAME = "spectrogram"
OUTPUT_NAME = "probabilities"
MODEL_FILE = "model.onnx"


VALIDATION_BATCHES = (1, 3, 16)


def export_onnx(model: AcousticNet, spec: PreprocessingSpec, path: Path) -> None:
    """Export with a symbolic batch axis.

    The TorchScript exporter with `dynamic_axes` is used because the dynamo exporter in PyTorch 2.7
    specialises a batch dimension traced at size 1 to a static `1` (even with `dynamic_shapes`), which
    the engine's batched inference cannot feed. `ensure_dynamic_batch` refuses any static result.
    """
    model.eval()
    dummy = torch.zeros(2, model.in_channels, spec.n_bins, spec.window_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            ProbabilityHead(model).eval(),
            (dummy,),
            str(path),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
            verbose=False,
        )
    ensure_dynamic_batch(path)


def ensure_dynamic_batch(path: Path) -> None:
    """Raise unless the graph's input and output batch axes are symbolic."""
    graph = onnx.load(str(path)).graph
    for value in (graph.input[0], graph.output[0]):
        dim0 = value.type.tensor_type.shape.dim[0]
        if dim0.HasField("dim_value") or not dim0.dim_param:
            raise RuntimeError(f"ONNX export left a static batch axis on '{value.name}'")


def ort_session(path: Path, threads: int = 2) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def validate_onnx(
    model: AcousticNet, spec: PreprocessingSpec, path: Path, tolerance: float = 1e-4
) -> dict[str, Any]:
    onnx.checker.check_model(onnx.load(str(path)))
    session = ort_session(path)
    inputs, outputs = session.get_inputs(), session.get_outputs()
    report: dict[str, Any] = {
        "input": {"name": inputs[0].name, "shape": [str(d) for d in inputs[0].shape]},
        "output": {"name": outputs[0].name, "shape": [str(d) for d in outputs[0].shape]},
        "opset": OPSET,
        "onnxruntime": ort.__version__,
    }
    gen = torch.Generator().manual_seed(1234)
    model.eval()
    batch_diffs: dict[str, float] = {}
    finite = True
    width_ok = True
    largest = max(VALIDATION_BATCHES)
    full = torch.randn(largest, model.in_channels, spec.n_bins, spec.window_frames, generator=gen)
    with torch.no_grad():
        ref_full = torch.sigmoid(model(full)).numpy()
    got_full = None
    for size in VALIDATION_BATCHES:
        batch = full[:size]
        try:
            got = session.run([OUTPUT_NAME], {INPUT_NAME: batch.numpy()})[0]
        except Exception as exc:  # onnxruntime raises on a shape the graph cannot accept
            report.setdefault("errors", {})[str(size)] = str(exc).splitlines()[0][:300]
            width_ok = False
            batch_diffs[str(size)] = float("inf")
            continue
        if got.shape != (size, len(model.label_map.outputs)):
            width_ok = False
            batch_diffs[str(size)] = float("inf")
            continue
        finite = finite and bool(np.isfinite(got).all())
        batch_diffs[str(size)] = float(np.max(np.abs(ref_full[:size] - got)))
        if size == largest:
            got_full = got
    alone_diff = float("inf")
    if got_full is not None:
        alone = session.run([OUTPUT_NAME], {INPUT_NAME: full[1:2].numpy()})[0]
        alone_diff = float(np.max(np.abs(alone[0] - got_full[1])))
    report["batch_sizes"] = list(VALIDATION_BATCHES)
    report["max_abs_diff_by_batch"] = batch_diffs
    report["max_abs_diff_vs_torch"] = max(batch_diffs.values())
    report["max_abs_diff_alone_vs_batched"] = alone_diff
    report["finite"] = finite
    report["names_ok"] = inputs[0].name == INPUT_NAME and outputs[0].name == OUTPUT_NAME
    report["dynamic_batch"] = not str(inputs[0].shape[0]).isdigit() and not str(outputs[0].shape[0]).isdigit()
    report["output_width"] = len(model.label_map.outputs) if width_ok else -1
    report["passed"] = bool(
        report["names_ok"]
        and report["dynamic_batch"]
        and width_ok
        and finite
        and report["max_abs_diff_vs_torch"] <= tolerance
        and report["max_abs_diff_alone_vs_batched"] <= tolerance
    )
    return report


def write_bundle(
    model: AcousticNet,
    spec: PreprocessingSpec,
    label_map: LabelMap,
    bundle_dir: Path,
    kind: str,
    lineage: dict[str, Any],
    thresholds: dict[str, float] | None = None,
    qualification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export and validate a bundle. Only the `qualify` command passes a qualification block; everything
    else is written `unqualified` (or `not-applicable-fixture`), which the engine refuses."""
    if kind not in ("fixture-untrained", "selected"):
        raise ValueError("bundle kind must be fixture-untrained or selected")
    if qualification is not None and (kind != "selected" or qualification.get("status") != "qualified"):
        raise ValueError("only a selected checkpoint can carry a qualified status")
    if kind == "selected" and lineage.get("initialization") != "random":
        raise ValueError("a selected bundle must record its random-initialised training lineage")
    model_path = bundle_dir / MODEL_FILE
    export_onnx(model, spec, model_path)
    validation = validate_onnx(model, spec, model_path)
    if not validation["passed"]:
        raise RuntimeError(f"ONNX validation failed: {validation}")
    profiles = load_profiles()
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "artifact_kind": kind,
        "model": {
            "file": MODEL_FILE,
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
            "format": "onnx",
            "opset": OPSET,
            "precision": "fp32",
            "input": {
                "name": INPUT_NAME,
                "shape": ["batch", model.in_channels, spec.n_bins, spec.window_frames],
            },
            "output": {
                "name": OUTPUT_NAME,
                "shape": ["batch", len(label_map.outputs)],
                "activation": "sigmoid",
            },
        },
        "label_map": label_map.to_json(),
        "label_map_hash": label_map.hash,
        "profiles": profiles,
        "preprocessing": spec.to_json(),
        "preprocessing_hash": spec.hash,
        "thresholds": thresholds or {},
        "threshold_status": "uncalibrated" if kind == "fixture-untrained" else "validation-calibrated",
        "lineage": lineage,
        "runtime": {"onnxruntime_min": "1.20", "intra_op_threads": 2, "execution_mode": "sequential"},
        "qualification": qualification
        or {"status": "not-applicable-fixture" if kind == "fixture-untrained" else "unqualified"},
        "validation": validation,
        "created_unix": int(time.time()),
    }
    manifest["manifest_hash"] = canonical_hash({k: v for k, v in manifest.items() if k != "created_unix"})
    atomic_write_json(bundle_dir / "bundle.json", manifest)
    return manifest


def parity_reference(
    samples_by_channel: list[np.ndarray], src_sr: int, spec: PreprocessingSpec, session: ort.InferenceSession
) -> dict[str, Any]:
    """Python reference windows and probabilities for a WAV, for the Rust parity check (kept out of Git)."""
    from whalemix_training.preprocess import analyze_channel

    out = []
    for channel, samples in enumerate(samples_by_channel):
        for w in analyze_channel(samples, src_sr, spec):
            probs = session.run([OUTPUT_NAME], {INPUT_NAME: w["tensor"][None]})[0][0]
            out.append(
                {
                    "channel": channel,
                    "start_s": w["start_s"],
                    "end_s": w["end_s"],
                    "tensor_sum": float(np.sum(w["tensor"], dtype=np.float64)),
                    "tensor": w["tensor"].ravel().tolist(),
                    "probabilities": probs.tolist(),
                }
            )
    return {"preprocessing_hash": spec.hash, "windows": out}


def load_bundle(bundle_dir: Path) -> dict[str, Any]:
    manifest = json.loads((bundle_dir / "bundle.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != BUNDLE_SCHEMA:
        raise ValueError("unsupported bundle schema")
    if sha256_file(bundle_dir / manifest["model"]["file"]) != manifest["model"]["sha256"]:
        raise ValueError("bundle model hash mismatch")
    return manifest
