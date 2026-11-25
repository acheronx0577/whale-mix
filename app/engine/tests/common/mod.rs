#![allow(dead_code)]
use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use whale_mix_engine::bundle::hex;
use whale_mix_engine::preprocess::Spec;

pub fn data_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

pub fn parity() -> Value {
    let text = std::fs::read_to_string(data_dir().join("preprocess-parity.json")).unwrap();
    serde_json::from_str(&text).unwrap()
}

pub fn default_preprocessing() -> Value {
    parity()["cases"][0]["preprocessing"].clone()
}

pub fn spec_of(v: &Value) -> Spec {
    serde_json::from_value(v.clone()).unwrap()
}

/// Same formula as tests/data/make_parity_fixture.py.
pub fn signal(rate: u32, seconds: f64) -> Vec<f64> {
    let n = (rate as f64 * seconds).round() as usize;
    (0..n)
        .map(|i| {
            let t = i as f64 / rate as f64;
            let tau = std::f64::consts::TAU;
            0.3 * (tau * 17.0 * t).sin()
                + 0.2 * (tau * 400.0 * t).sin()
                + 0.1 * (tau * (60.0 * t + 15.0 * t * t)).sin()
        })
        .collect()
}

pub const WHALES: [&str; 4] = ["blue_a", "blue_b", "fin_20hz_pulse", "humpback_song"];
pub const CONTEXT: [&str; 7] = [
    "vessel",
    "other_human",
    "fish",
    "invertebrate",
    "other_marine_mammal",
    "weather_water",
    "recorder_artifact",
];

/// Write a synthetic bundle with an arbitrary (non-ONNX) model payload for manifest validation tests.
pub fn write_bundle(dir: &Path, kind: &str, qualification: &str, model_bytes: &[u8]) -> Value {
    use sha2::{Digest, Sha256};
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("model.onnx"), model_bytes).unwrap();
    let profiles: Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../training/whalemix_training/profiles.json"),
        )
        .unwrap(),
    )
    .unwrap();
    let outputs: Vec<&str> = WHALES.iter().chain(CONTEXT.iter()).copied().collect();
    let manifest = json!({
        "schema": "whalemix-model-bundle/1",
        "artifact_kind": kind,
        "model": {"file": "model.onnx", "sha256": hex(&Sha256::digest(model_bytes)),
                  "size_bytes": model_bytes.len(), "format": "onnx", "opset": 18, "precision": "fp32",
                  "input": {"name": "spectrogram", "shape": ["batch", 1, 257, 100]},
                  "output": {"name": "probabilities", "shape": ["batch", 11], "activation": "sigmoid"}},
        "label_map": {"version": "acoustic-v1", "whale_targets": WHALES, "context_targets": CONTEXT,
                      "outputs": outputs},
        "label_map_hash": "test", "profiles": profiles, "preprocessing": default_preprocessing(),
        "preprocessing_hash": "test", "thresholds": {}, "threshold_status": "uncalibrated",
        "qualification": {"status": qualification}, "manifest_hash": "test",
    });
    std::fs::write(
        dir.join("bundle.json"),
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    manifest
}

/// Fixture bundle exported by `python -m whalemix_training export --fixture` (kept out of Git).
pub fn fixture_bundle() -> Option<PathBuf> {
    let dir = PathBuf::from(std::env::var_os("WHALEMIX_FIXTURE_BUNDLE")?);
    dir.join("bundle.json").is_file().then_some(dir)
}

pub fn write_wav(path: &Path, rate: u32, channels: u16, samples: &[f64]) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for s in samples {
        for _ in 0..channels {
            w.write_sample((s * 32767.0).round() as i16).unwrap();
        }
    }
    w.finalize().unwrap();
}
