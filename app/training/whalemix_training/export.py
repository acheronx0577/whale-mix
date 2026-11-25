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


def export_onnx(model: AcousticNet, spec: PreprocessingSpec, path: Path) -> None:
    model.eval()
    dummy = torch.zeros(1, model.in_channels, spec.n_bins, spec.window_frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    program = torch.onnx.export(
        ProbabilityHead(model).eval(),
        (dummy,),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_shapes={"spectrogram": {0: torch.export.Dim("batch", min=1, max=1024)}},
        opset_version=OPSET,
        dynamo=True,
        external_data=False,
        optimize=True,
        verbose=False,
    )
    if program is None:
        raise RuntimeError("ONNX export returned no program")
    program.save(str(path), external_data=False)


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
    batch = torch.randn(3, model.in_channels, spec.n_bins, spec.window_frames, generator=gen)
    model.eval()
    with torch.no_grad():
        ref = torch.sigmoid(model(batch)).numpy()
    got = session.run([OUTPUT_NAME], {INPUT_NAME: batch.numpy()})[0]
    alone = session.run([OUTPUT_NAME], {INPUT_NAME: batch[1:2].numpy()})[0]
    report["max_abs_diff_vs_torch"] = float(np.max(np.abs(ref - got)))
    report["max_abs_diff_alone_vs_batched"] = float(np.max(np.abs(alone[0] - got[1])))
    report["finite"] = bool(np.isfinite(got).all())
    report["names_ok"] = inputs[0].name == INPUT_NAME and outputs[0].name == OUTPUT_NAME
    report["output_width"] = int(got.shape[1])
    report["passed"] = bool(
        report["names_ok"]
        and report["finite"]
        and report["max_abs_diff_vs_torch"] <= tolerance
        and report["max_abs_diff_alone_vs_batched"] <= tolerance
        and got.shape[1] == len(model.label_map.outputs)
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
) -> dict[str, Any]:
    if kind not in ("fixture-untrained", "selected"):
        raise ValueError("bundle kind must be fixture-untrained or selected")
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
        "qualification": {
            "status": "not-applicable-fixture" if kind == "fixture-untrained" else "unqualified"
        },
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
