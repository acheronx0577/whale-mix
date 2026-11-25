//! Model bundle discovery and validation. The engine never trains, downloads or substitutes weights.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::preprocess::Spec;

pub const BUNDLE_SCHEMA: &str = "whalemix-model-bundle/1";

#[derive(Debug, Clone, Deserialize)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelDesc {
    pub file: String,
    pub sha256: String,
    pub size_bytes: u64,
    pub format: String,
    pub opset: u32,
    pub input: TensorDesc,
    pub output: TensorDesc,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LabelMap {
    pub version: String,
    pub whale_targets: Vec<String>,
    pub context_targets: Vec<String>,
    pub outputs: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Profile {
    pub id: String,
    pub status: String,
    pub outputs: Vec<String>,
    pub required_max_hz: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Profiles {
    pub profiles_version: String,
    pub profiles: Vec<Profile>,
    pub context_required_max_hz: std::collections::BTreeMap<String, Option<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub schema: String,
    pub artifact_kind: String,
    pub model: ModelDesc,
    pub label_map: LabelMap,
    pub label_map_hash: String,
    pub profiles: Profiles,
    pub preprocessing: Spec,
    pub preprocessing_hash: String,
    pub thresholds: std::collections::BTreeMap<String, f64>,
    pub threshold_status: String,
    pub qualification: Value,
    pub manifest_hash: String,
}

#[derive(Debug, Clone)]
pub struct Bundle {
    pub dir: PathBuf,
    pub manifest: Manifest,
    pub model_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BundleError {
    Missing(String),
    Invalid(String),
    HashMismatch,
    NotQualified(String),
    FixtureRefused,
}

impl std::fmt::Display for BundleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BundleError::Missing(m) => write!(f, "model bundle missing: {m}"),
            BundleError::Invalid(m) => write!(f, "model bundle invalid: {m}"),
            BundleError::HashMismatch => {
                write!(f, "model file hash does not match the bundle manifest")
            }
            BundleError::NotQualified(m) => {
                write!(f, "model bundle is not a qualified release: {m}")
            }
            BundleError::FixtureRefused => {
                write!(
                    f,
                    "bundle is an untrained fixture; refused without --allow-fixture-bundle"
                )
            }
        }
    }
}

impl BundleError {
    pub fn code(&self) -> &'static str {
        match self {
            BundleError::Missing(_) => "missing",
            BundleError::Invalid(_) => "invalid",
            BundleError::HashMismatch => "hash_mismatch",
            BundleError::NotQualified(_) => "not_qualified",
            BundleError::FixtureRefused => "fixture_refused",
        }
    }
}

pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

/// Lowercase hex encoding.
pub fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write as _;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn dim(v: &Value) -> Option<u64> {
    v.as_u64()
}

/// Validate a bundle directory. Fixture bundles are accepted only when `allow_fixture` is set.
pub fn load(dir: &Path, allow_fixture: bool) -> Result<Bundle, BundleError> {
    let manifest_path = dir.join("bundle.json");
    let text = std::fs::read_to_string(&manifest_path)
        .map_err(|_| BundleError::Missing(format!("{} not found", manifest_path.display())))?;
    let raw: Value =
        serde_json::from_str(&text).map_err(|e| BundleError::Invalid(e.to_string()))?;
    if raw.get("schema").and_then(Value::as_str) != Some(BUNDLE_SCHEMA) {
        return Err(BundleError::Invalid("unsupported bundle schema".into()));
    }
    let manifest: Manifest =
        serde_json::from_value(raw).map_err(|e| BundleError::Invalid(e.to_string()))?;
    match manifest.artifact_kind.as_str() {
        "fixture-untrained" if !allow_fixture => return Err(BundleError::FixtureRefused),
        "fixture-untrained" => {}
        "selected" => {
            let status = manifest
                .qualification
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("");
            if status != "qualified" {
                return Err(BundleError::NotQualified(format!(
                    "qualification status '{status}'"
                )));
            }
        }
        other => {
            return Err(BundleError::Invalid(format!(
                "unknown artifact_kind '{other}'"
            )))
        }
    }
    if manifest.model.format != "onnx" || manifest.model.file.contains(['/', '\\']) {
        return Err(BundleError::Invalid(
            "model must be an ONNX file inside the bundle directory".into(),
        ));
    }
    manifest
        .preprocessing
        .validate()
        .map_err(BundleError::Invalid)?;
    let lm = &manifest.label_map;
    let expected: Vec<String> = lm
        .whale_targets
        .iter()
        .chain(lm.context_targets.iter())
        .cloned()
        .collect();
    if lm.outputs != expected || lm.outputs.is_empty() {
        return Err(BundleError::Invalid(
            "label map output order is inconsistent".into(),
        ));
    }
    let shape = &manifest.model.input.shape;
    let spec = &manifest.preprocessing;
    if shape.len() != 4
        || dim(&shape[1]) != Some(1)
        || dim(&shape[2]) != Some(spec.n_bins() as u64)
        || dim(&shape[3]) != Some(spec.window_frames as u64)
    {
        return Err(BundleError::Invalid(
            "model input shape does not match the preprocessing contract".into(),
        ));
    }
    if manifest.model.output.shape.len() != 2
        || dim(&manifest.model.output.shape[1]) != Some(lm.outputs.len() as u64)
    {
        return Err(BundleError::Invalid(
            "model output width does not match the label map".into(),
        ));
    }
    for profile in &manifest.profiles.profiles {
        if profile
            .outputs
            .iter()
            .any(|o| !lm.whale_targets.contains(o))
        {
            return Err(BundleError::Invalid(format!(
                "profile {} names an unknown output",
                profile.id
            )));
        }
    }
    let model_path = dir.join(&manifest.model.file);
    if !model_path.is_file() {
        return Err(BundleError::Missing(format!(
            "{} not found",
            model_path.display()
        )));
    }
    let actual = sha256_file(&model_path).map_err(|e| BundleError::Missing(e.to_string()))?;
    if actual != manifest.model.sha256 {
        return Err(BundleError::HashMismatch);
    }
    Ok(Bundle {
        dir: dir.to_path_buf(),
        manifest,
        model_path,
    })
}

impl Bundle {
    /// Outputs that the recording bandwidth can assess, restricted to the requested profiles
    /// (`auto` = every required profile). Context outputs are included when assessable.
    pub fn assessable_outputs(&self, profiles: &[String], source_rate: u32) -> Vec<String> {
        let nyquist = source_rate as f64 / 2.0;
        let auto = profiles.iter().any(|p| p == "auto");
        let mut out = Vec::new();
        for p in &self.manifest.profiles.profiles {
            if p.status != "required" || !(auto || profiles.contains(&p.id)) {
                continue;
            }
            if p.required_max_hz.is_none_or(|need| nyquist >= need) {
                out.extend(p.outputs.iter().cloned());
            }
        }
        for (label, need) in &self.manifest.profiles.context_required_max_hz {
            if need.is_none_or(|n| nyquist >= n) {
                out.push(label.clone());
            }
        }
        out
    }

    pub fn known_profile(&self, id: &str) -> bool {
        id == "auto"
            || self
                .manifest
                .profiles
                .profiles
                .iter()
                .any(|p| p.id == id && p.status == "required")
    }
}
