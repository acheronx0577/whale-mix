//! One analysis run: model load, a bounded decode/preprocess producer, one ONNX Runtime session on
//! the calling thread, durable incremental results and atomic final artifacts. The default build
//! uses the CPU execution provider. `--features cuda` uses CUDA when a GPU is available.

use std::collections::BTreeMap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::bundle::{sha256_file, Bundle};
use crate::preprocess::{ChannelPipeline, Window};
use crate::protocol::Emitter;
use crate::wav::{ChunkedWav, DecodeError};

pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
const CHUNK_FRAMES: usize = 1 << 16;
const MAX_THREADS: usize = 2;

#[derive(Debug, Clone)]
pub struct RunConfig {
    pub inputs: PathBuf,
    pub out: PathBuf,
    pub bundle_dir: PathBuf,
    pub profiles: Vec<String>,
    pub threads: usize,
    pub batch: usize,
    pub allow_fixture: bool,
    /// Requested compute mode: "auto", "hybrid", "gpu", "cuda" or "cpu".
    pub device: String,
    pub run_id: String,
}

/// Accepted `--device` values.
pub const DEVICES: [&str; 5] = ["auto", "hybrid", "gpu", "cuda", "cpu"];

/// The backend that will run the model, and why.
///
/// `cpu` always stays on CPU. `auto`, `hybrid`, `gpu` and `cuda` use the CUDA execution provider
/// when this binary is built with `--features cuda` and a CUDA GPU can be initialized. Otherwise
/// they stay on CPU and the note says why. The default build has no CUDA provider, so every
/// non-CPU request falls back without touching ONNX Runtime's CUDA API.
pub fn resolve_device(requested: &str) -> (&'static str, String) {
    if requested == "cpu" {
        return ("cpu", "CPU requested".to_string());
    }
    match cuda_plan(requested) {
        None => (
            "cuda",
            match requested {
                "cuda" => "CUDA execution provider selected".to_string(),
                "gpu" => "GPU requested; CUDA execution provider selected".to_string(),
                "hybrid" => "hybrid requested; CUDA execution provider selected".to_string(),
                _ => "auto selected the CUDA execution provider".to_string(),
            },
        ),
        Some(note) => ("cpu", note),
    }
}

/// `None` selects CUDA. `Some` is the reason the run stays on CPU.
fn cuda_plan(requested: &str) -> Option<String> {
    #[cfg(not(feature = "cuda"))]
    {
        let _ = requested;
        Some("no GPU execution provider in this build; running on CPU".to_string())
    }
    #[cfg(feature = "cuda")]
    {
        match probe_cuda(requested) {
            Ok(()) => None,
            Err(reason) => Some(format!("{reason}; running on CPU")),
        }
    }
}

/// A CUDA GPU the driver is exposing. `CUDA_VISIBLE_DEVICES` empty, `none` or `-1` hides it.
#[cfg(feature = "cuda")]
fn nvidia_gpu_visible() -> bool {
    if let Ok(visible) = std::env::var("CUDA_VISIBLE_DEVICES") {
        let visible = visible.trim();
        if visible.is_empty() || visible == "-1" || visible.eq_ignore_ascii_case("none") {
            return false;
        }
    }
    nvidia_device_node_present()
}

#[cfg(all(feature = "cuda", target_os = "linux"))]
fn nvidia_device_node_present() -> bool {
    if Path::new("/dev/nvidia0").exists() || Path::new("/dev/nvidiactl").exists() {
        return true;
    }
    fs::read_dir("/proc/driver/nvidia/gpus")
        .ok()
        .and_then(|entries| entries.flatten().next())
        .is_some()
}

#[cfg(all(feature = "cuda", target_os = "windows"))]
fn nvidia_device_node_present() -> bool {
    Path::new(r"C:\Windows\System32\nvcuda.dll").exists()
}

#[cfg(all(feature = "cuda", not(any(target_os = "linux", target_os = "windows"))))]
fn nvidia_device_node_present() -> bool {
    false
}

#[cfg(feature = "cuda")]
fn probe_cuda(requested: &str) -> Result<(), String> {
    use ort::execution_providers::ExecutionProvider;

    let ep = ort::execution_providers::CUDAExecutionProvider::default();
    if !ep.supported_by_platform() {
        return Err("CUDA execution provider is not supported on this platform".into());
    }
    // `auto` and `hybrid` stay on CPU when no device node is visible, without loading the
    // provider. An explicit `cuda` or `gpu` request still tries, so a missed device node can
    // succeed and a real initialization failure is reported.
    if !nvidia_gpu_visible() && !matches!(requested, "cuda" | "gpu") {
        return Err("no CUDA GPU is visible".into());
    }
    match ep.is_available() {
        Ok(true) => {}
        Ok(false) => {
            return Err(
                "CUDA execution provider is not available in this ONNX Runtime build (need the CUDA 12 and cuDNN 9 runtimes on the loader path; a custom 1.22.x library is selected at build time with ORT_LIB_LOCATION)"
                    .into(),
            );
        }
        Err(error) => return Err(format!("CUDA execution provider check failed: {error}")),
    }
    let builder = ort::session::Session::builder().map_err(|error| error.to_string())?;
    builder
        .with_execution_providers([ort::execution_providers::CUDAExecutionProvider::default()
            .with_device_id(0)
            .build()
            .error_on_failure()])
        .map(|_| ())
        .map_err(|error| format!("CUDA execution provider failed to register: {error}"))
}

struct OpenedSession {
    session: ort::session::Session,
    on_cuda: bool,
    fallback_note: Option<String>,
}

fn configure_session(threads: usize) -> Result<ort::session::builder::SessionBuilder, RunError> {
    ort::session::Session::builder()
        .and_then(|builder| builder.with_intra_threads(threads))
        .and_then(|builder| builder.with_inter_threads(1))
        .and_then(|builder| builder.with_parallel_execution(false))
        .map_err(|error| RunError::Model(error.to_string()))
}

fn open_session(model: &Path, threads: usize, use_cuda: bool) -> Result<OpenedSession, RunError> {
    let open_cpu = || -> Result<ort::session::Session, RunError> {
        configure_session(threads)?
            .commit_from_file(model)
            .map_err(|error| RunError::Model(error.to_string()))
    };
    if use_cuda {
        #[cfg(feature = "cuda")]
        {
            let cuda = configure_session(threads).and_then(|builder| {
                builder
                    .with_execution_providers([
                        ort::execution_providers::CUDAExecutionProvider::default()
                            .with_device_id(0)
                            .build()
                            .error_on_failure(),
                    ])
                    .and_then(|builder| builder.commit_from_file(model))
                    .map_err(|error| RunError::Model(error.to_string()))
            });
            return match cuda {
                Ok(session) => Ok(OpenedSession {
                    session,
                    on_cuda: true,
                    fallback_note: None,
                }),
                Err(error) => match open_cpu() {
                    Ok(session) => Ok(OpenedSession {
                        session,
                        on_cuda: false,
                        fallback_note: Some(format!("{error}; running on CPU")),
                    }),
                    Err(cpu_error) => Err(RunError::Model(format!(
                        "{error}; CPU fallback also failed: {cpu_error}"
                    ))),
                },
            };
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Ok(OpenedSession {
                session: open_cpu()?,
                on_cuda: false,
                fallback_note: Some(
                    "no GPU execution provider in this build; running on CPU".to_string(),
                ),
            });
        }
    }
    Ok(OpenedSession {
        session: open_cpu()?,
        on_cuda: false,
        fallback_note: None,
    })
}

#[derive(Default)]
pub struct Control {
    pub cancel: AtomicBool,
    pub pause: AtomicBool,
}

impl Control {
    /// Apply one control line from the app (`pause`, `resume`, `cancel`). Unknown lines are ignored.
    pub fn apply(&self, line: &str) -> Option<&'static str> {
        match line.trim() {
            "cancel" => {
                self.cancel.store(true, Ordering::SeqCst);
                Some("cancel_requested")
            }
            "pause" => {
                self.pause.store(true, Ordering::SeqCst);
                Some("paused")
            }
            "resume" => {
                self.pause.store(false, Ordering::SeqCst);
                Some("resumed")
            }
            _ => None,
        }
    }

    fn wait_while_paused(&self) {
        while self.pause.load(Ordering::SeqCst) && !self.cancel.load(Ordering::SeqCst) {
            thread::sleep(Duration::from_millis(20));
        }
    }

    fn canceled(&self) -> bool {
        self.cancel.load(Ordering::SeqCst)
    }
}

#[derive(Debug, Deserialize)]
struct InputLine {
    source_id: Option<String>,
    path: String,
}

#[derive(Debug)]
pub enum Outcome {
    Completed { rows: u64 },
    Canceled { rows: u64 },
}

#[derive(Debug)]
pub enum RunError {
    Config(String),
    Model(String),
    Io(String),
}

impl std::fmt::Display for RunError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RunError::Config(m) => write!(f, "invalid configuration: {m}"),
            RunError::Model(m) => write!(f, "model error: {m}"),
            RunError::Io(m) => write!(f, "I/O error: {m}"),
        }
    }
}

fn io_err(e: impl std::fmt::Display) -> RunError {
    RunError::Io(e.to_string())
}

/// Messages from the producer thread. A bounded channel of capacity 1 means at most one prepared
/// batch waits while the consumer runs inference; the producer blocks when the consumer falls behind.
enum Work {
    FileStart {
        index: usize,
        source_id: String,
        path: String,
        sha256: String,
        rate: u32,
        channels: usize,
        frames: u64,
    },
    Batch {
        index: usize,
        channel: usize,
        windows: Vec<Window>,
    },
    FileDone {
        index: usize,
        windows: u64,
    },
    FileError {
        index: usize,
        source_id: String,
        path: String,
        kind: &'static str,
        message: String,
    },
}

pub fn count_inputs(path: &Path) -> Result<usize, RunError> {
    let file = File::open(path).map_err(|e| RunError::Config(format!("inputs manifest: {e}")))?;
    Ok(BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .count())
}

fn producer(
    cfg: RunConfig,
    spec: crate::preprocess::Spec,
    control: Arc<Control>,
    tx: SyncSender<Work>,
) {
    let Ok(file) = File::open(&cfg.inputs) else {
        return;
    };
    for (index, line) in BufReader::new(file)
        .lines()
        .map_while(Result::ok)
        .filter(|l| !l.trim().is_empty())
        .enumerate()
    {
        if control.canceled() {
            return;
        }
        control.wait_while_paused();
        let parsed: Result<InputLine, _> = serde_json::from_str(&line);
        let Ok(input) = parsed else {
            let _ = tx.send(Work::FileError {
                index,
                source_id: format!("line-{}", index + 1),
                path: String::new(),
                kind: "invalid_input",
                message: "inputs manifest line is not valid JSON".into(),
            });
            continue;
        };
        let source_id = input
            .source_id
            .clone()
            .unwrap_or_else(|| format!("src-{}", index + 1));
        let path = PathBuf::from(&input.path);
        let mut wav = match ChunkedWav::open(&path) {
            Ok(w) => w,
            Err(e) => {
                let kind = match e {
                    DecodeError::Unsupported(_) => "unsupported",
                    DecodeError::Unreadable(_) => "unreadable",
                };
                let _ = tx.send(Work::FileError {
                    index,
                    source_id,
                    path: input.path,
                    kind,
                    message: e.to_string(),
                });
                continue;
            }
        };
        let sha = sha256_file(&path).unwrap_or_default();
        let info = wav.info.clone();
        if tx
            .send(Work::FileStart {
                index,
                source_id: source_id.clone(),
                path: input.path.clone(),
                sha256: sha,
                rate: info.sample_rate,
                channels: info.channels,
                frames: info.frames,
            })
            .is_err()
        {
            return;
        }
        let mut pipes: Vec<ChannelPipeline> = (0..info.channels)
            .map(|_| ChannelPipeline::new(&spec, info.sample_rate))
            .collect();
        let mut pending: Vec<Vec<Window>> = vec![Vec::new(); info.channels];
        let mut total = 0u64;
        let mut failed = false;
        loop {
            if control.canceled() {
                return;
            }
            control.wait_while_paused();
            let chunk = match wav.read_chunk(CHUNK_FRAMES) {
                Ok(c) => c,
                Err(e) => {
                    let _ = tx.send(Work::FileError {
                        index,
                        source_id: source_id.clone(),
                        path: input.path.clone(),
                        kind: "decode_error",
                        message: e.to_string(),
                    });
                    failed = true;
                    break;
                }
            };
            let eof = chunk[0].is_empty();
            for (c, samples) in chunk.iter().enumerate() {
                let new = if eof {
                    pipes[c].finish()
                } else {
                    pipes[c].push(samples)
                };
                pending[c].extend(new);
                while pending[c].len() >= cfg.batch || (eof && !pending[c].is_empty()) {
                    let take = pending[c].len().min(cfg.batch);
                    let windows: Vec<Window> = pending[c].drain(..take).collect();
                    total += windows.len() as u64;
                    if tx
                        .send(Work::Batch {
                            index,
                            channel: c,
                            windows,
                        })
                        .is_err()
                    {
                        return;
                    }
                }
            }
            if eof {
                break;
            }
        }
        if !failed
            && tx
                .send(Work::FileDone {
                    index,
                    windows: total,
                })
                .is_err()
        {
            return;
        }
    }
}

struct FileState {
    source_id: String,
    path: String,
    sha256: String,
    assessable: Vec<String>,
}

fn peak_rss_mb() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    let kb: f64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb / 1024.0)
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), RunError> {
    let tmp = path.with_extension("tmp");
    {
        let mut f = File::create(&tmp).map_err(io_err)?;
        f.write_all(bytes).map_err(io_err)?;
        f.sync_all().map_err(io_err)?;
    }
    fs::rename(&tmp, path).map_err(io_err)
}

fn automatic_label(bundle: &Bundle, scores: &BTreeMap<String, Option<f32>>) -> (String, String) {
    let m = &bundle.manifest;
    if m.threshold_status != "validation-calibrated" || m.thresholds.is_empty() {
        return (
            "review".into(),
            format!("thresholds {}; scores only", m.threshold_status),
        );
    }
    let mut hits = Vec::new();
    let mut any_scored = false;
    for target in &m.label_map.whale_targets {
        if let (Some(Some(score)), Some(thr)) = (scores.get(target), m.thresholds.get(target)) {
            any_scored = true;
            if f64::from(*score) >= *thr {
                hits.push(target.clone());
            }
        }
    }
    if !any_scored {
        (
            "review".into(),
            "no assessable whale profile for this recording".into(),
        )
    } else if hits.is_empty() {
        (
            "not-call".into(),
            "all assessable whale scores below threshold".into(),
        )
    } else {
        ("call".into(), hits.join("+"))
    }
}

pub fn run(
    cfg: &RunConfig,
    bundle: &Bundle,
    control: Arc<Control>,
    emitter: &Emitter,
) -> Result<Outcome, RunError> {
    let started = Instant::now();
    if cfg.threads == 0 || cfg.batch == 0 {
        return Err(RunError::Config(
            "threads and batch must be positive".into(),
        ));
    }
    for p in &cfg.profiles {
        if !bundle.known_profile(p) {
            return Err(RunError::Config(format!(
                "unknown or non-operational profile '{p}'"
            )));
        }
    }
    let threads = cfg.threads.min(MAX_THREADS);
    let total_files = count_inputs(&cfg.inputs)?;
    fs::create_dir_all(&cfg.out).map_err(io_err)?;
    let partial_path = cfg.out.join("results.partial.jsonl");
    let files_path = cfg.out.join("files.partial.jsonl");
    let mut partial = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&partial_path)
            .map_err(io_err)?,
    );
    let mut files_log = BufWriter::new(File::create(&files_path).map_err(io_err)?);
    let m = &bundle.manifest;
    let fixture = m.artifact_kind == "fixture-untrained";
    let (mut device_active, mut device_note) = resolve_device(&cfg.device);
    // Open a CUDA session before `started` so the reported device is the one that will run.
    // CPU sessions stay lazy and are created on the first batch, as before.
    let mut session: Option<ort::session::Session> = None;
    let mut model_load_ms: Option<f64> = None;
    if device_active == "cuda" {
        let load_started = Instant::now();
        let opened = open_session(&bundle.model_path, threads, true)?;
        model_load_ms = Some(load_started.elapsed().as_secs_f64() * 1000.0);
        if !opened.on_cuda {
            if let Some(note) = opened.fallback_note {
                device_active = "cpu";
                device_note = note;
            }
        }
        session = Some(opened.session);
    }
    emitter.emit("started", json!({
        "engine_version": ENGINE_VERSION, "model_sha256": m.model.sha256, "artifact_kind": m.artifact_kind,
        "fixture": fixture, "threads": threads, "batch": cfg.batch, "files_total": total_files,
        "profiles": cfg.profiles, "device_requested": cfg.device, "device_active": device_active,
        "device_note": device_note,
    }));
    emitter.emit(
        "dataset_added",
        json!({"dataset_id": format!("{}-results", cfg.run_id), "kind": "acoustic_results"}),
    );
    let columns: Vec<String> = m
        .label_map
        .outputs
        .iter()
        .map(|o| format!("score_{o}"))
        .collect();
    emitter.emit(
        "schema_updated",
        json!({"dataset_id": format!("{}-results", cfg.run_id), "score_columns": columns}),
    );
    let first_progress_ms = started.elapsed().as_secs_f64() * 1000.0;
    emitter.emit("progress", json!({"files_done": 0, "files_total": total_files, "windows_done": 0, "phase": "discovery"}));

    let (tx, rx): (SyncSender<Work>, Receiver<Work>) = sync_channel(1);
    let producer_cfg = cfg.clone();
    let spec = m.preprocessing.clone();
    let producer_control = Arc::clone(&control);
    let handle = thread::spawn(move || producer(producer_cfg, spec, producer_control, tx));

    let mut files: BTreeMap<usize, FileState> = BTreeMap::new();
    let (mut rows_total, mut windows_done, mut files_done, mut files_failed) =
        (0u64, 0u64, 0u64, 0u64);
    let mut audio_s_done = 0.0f64;
    let mut inference_s = 0.0f64;
    let spec = &m.preprocessing;
    let width = spec.n_bins() * spec.window_frames;

    for work in rx.iter() {
        if control.canceled() {
            break;
        }
        match work {
            Work::FileStart {
                index,
                source_id,
                path,
                sha256,
                rate,
                channels,
                frames,
            } => {
                let assessable = bundle.assessable_outputs(&cfg.profiles, rate);
                audio_s_done += frames as f64 / rate as f64;
                let _ = channels;
                files.insert(
                    index,
                    FileState {
                        source_id,
                        path,
                        sha256,
                        assessable,
                    },
                );
            }
            Work::Batch {
                index,
                channel,
                windows,
            } => {
                control.wait_while_paused();
                if session.is_none() {
                    let load_started = Instant::now();
                    let opened = open_session(&bundle.model_path, threads, false)?;
                    model_load_ms = Some(load_started.elapsed().as_secs_f64() * 1000.0);
                    session = Some(opened.session);
                }
                let sess = session.as_mut().expect("session");
                let n = windows.len();
                let mut data = Vec::with_capacity(n * width);
                for w in &windows {
                    data.extend_from_slice(&w.tensor);
                }
                let t = Instant::now();
                let input = ort::value::TensorRef::from_array_view((
                    [n, 1, spec.n_bins(), spec.window_frames],
                    &data[..],
                ))
                .map_err(|e| RunError::Model(e.to_string()))?;
                let outputs = sess
                    .run(ort::inputs![m.model.input.name.as_str() => input])
                    .map_err(|e| RunError::Model(e.to_string()))?;
                let (shape, probs) = outputs[m.model.output.name.as_str()]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| RunError::Model(e.to_string()))?;
                inference_s += t.elapsed().as_secs_f64();
                let n_out = m.label_map.outputs.len();
                if shape.len() != 2 || shape[0] as usize != n || shape[1] as usize != n_out {
                    return Err(RunError::Model(format!(
                        "unexpected output shape {shape:?}"
                    )));
                }
                if probs.iter().any(|p| !p.is_finite() || *p < 0.0 || *p > 1.0) {
                    return Err(RunError::Model(
                        "non-finite or out-of-range probabilities".into(),
                    ));
                }
                let file = files
                    .get(&index)
                    .ok_or_else(|| RunError::Io("batch for unknown file".into()))?;
                let mut ids = Vec::with_capacity(n);
                for (i, w) in windows.iter().enumerate() {
                    let mut scores = BTreeMap::new();
                    for (j, label) in m.label_map.outputs.iter().enumerate() {
                        let available = file.assessable.contains(label);
                        scores.insert(
                            label.clone(),
                            if available {
                                Some(probs[i * n_out + j])
                            } else {
                                None
                            },
                        );
                    }
                    let (label, reason) = automatic_label(bundle, &scores);
                    let mut h = Sha256::new();
                    h.update(
                        format!(
                            "{}:{}:{}:{}",
                            file.sha256, channel, w.start_frame, m.model.sha256
                        )
                        .as_bytes(),
                    );
                    let event_id = crate::bundle::hex(&h.finalize()[..8]);
                    let row = json!({
                        "event_id": event_id, "source_id": file.source_id, "source_sha256": file.sha256,
                        "path": file.path, "channel": channel, "start_s": w.start_s, "end_s": w.end_s,
                        "audio_rms": w.audio_rms, "band_10_110": w.band_10_110, "band_all": w.band_all,
                        "granularity": "coarse_window", "scores": scores, "automatic_label": label,
                        "automatic_reason": reason, "model_sha256": m.model.sha256,
                        "artifact_kind": m.artifact_kind, "fixture": fixture, "provisional": false,
                    });
                    writeln!(partial, "{row}").map_err(io_err)?;
                    ids.push(event_id);
                }
                partial.flush().map_err(io_err)?;
                partial.get_ref().sync_data().map_err(io_err)?;
                rows_total += n as u64;
                windows_done += n as u64;
                emitter.emit("results_batch", json!({
                    "dataset_id": format!("{}-results", cfg.run_id), "source_id": file.source_id,
                    "inserted": ids, "committed_rows_total": rows_total,
                    "results_path": partial_path.display().to_string(),
                }));
                emitter.emit(
                    "progress",
                    json!({"files_done": files_done, "files_total": total_files,
                    "windows_done": windows_done, "phase": "analysis"}),
                );
            }
            Work::FileDone { index, windows } => {
                files_done += 1;
                if let Some(f) = files.get(&index) {
                    writeln!(files_log, "{}", json!({"source_id": f.source_id, "path": f.path, "sha256": f.sha256,
                        "status": "completed", "windows": windows, "zero_detections_is_negative": false})).map_err(io_err)?;
                }
                emitter.emit(
                    "progress",
                    json!({"files_done": files_done, "files_total": total_files,
                    "windows_done": windows_done, "phase": "analysis"}),
                );
            }
            Work::FileError {
                index,
                source_id,
                path,
                kind,
                message,
            } => {
                files_done += 1;
                files_failed += 1;
                files.remove(&index);
                writeln!(
                    files_log,
                    "{}",
                    json!({"source_id": source_id, "path": path, "status": "error",
                    "error_kind": kind, "message": message})
                )
                .map_err(io_err)?;
                emitter.emit(
                    "file_error",
                    json!({"source_id": source_id, "path": path, "kind": kind, "message": message}),
                );
            }
        }
    }
    drop(rx);
    let _ = handle.join();
    drop(session); // release the model session and its buffers as soon as the run ends
    files_log.flush().map_err(io_err)?;
    partial.flush().map_err(io_err)?;

    let wall_s = started.elapsed().as_secs_f64();
    let resources = json!({
        "measurement_kind": if fixture { "fixture" } else { "run" },
        "model_load_ms": model_load_ms, "first_progress_ms": first_progress_ms, "wall_s": wall_s,
        "audio_s": audio_s_done, "x_realtime": if wall_s > 0.0 { Some(audio_s_done / wall_s) } else { None },
        "inference_s": inference_s, "peak_rss_mb": peak_rss_mb(), "threads": threads, "batch": cfg.batch,
        "device_requested": cfg.device, "device_active": device_active,
    });
    let canceled = control.canceled();
    let status = if canceled { "canceled" } else { "completed" };
    let suffix = if canceled { ".partial" } else { "" };
    let csv_path = cfg.out.join(format!("results{suffix}.csv"));
    write_csv(&partial_path, &csv_path, &m.label_map.outputs)?;
    let run_json = json!({
        "protocol": crate::protocol::PROTOCOL, "engine_version": ENGINE_VERSION, "run_id": cfg.run_id,
        "status": status, "partial": canceled,
        "bundle": {"model_sha256": m.model.sha256, "manifest_hash": m.manifest_hash, "artifact_kind": m.artifact_kind,
                   "label_map_hash": m.label_map_hash, "preprocessing_hash": m.preprocessing_hash,
                   "threshold_status": m.threshold_status},
        "inputs_manifest_sha256": sha256_file(&cfg.inputs).unwrap_or_default(),
        "profiles": cfg.profiles, "files_total": total_files, "files_done": files_done, "files_failed": files_failed,
        "rows": rows_total, "files_log": files_path.display().to_string(), "results_csv": csv_path.display().to_string(),
        "resources": resources,
    });
    atomic_write(
        &cfg.out.join(format!("run{suffix}.json")),
        serde_json::to_string_pretty(&run_json)
            .map_err(io_err)?
            .as_bytes(),
    )?;
    validate_outputs(&csv_path, rows_total)?;
    if canceled {
        emitter.emit(
            "canceled",
            json!({"committed_rows": rows_total, "results_csv": csv_path.display().to_string()}),
        );
        Ok(Outcome::Canceled { rows: rows_total })
    } else {
        emitter.emit(
            "completed",
            json!({"rows": rows_total, "files_done": files_done, "files_failed": files_failed,
            "results_csv": csv_path.display().to_string(), "resources": resources}),
        );
        Ok(Outcome::Completed { rows: rows_total })
    }
}

fn write_csv(partial: &Path, csv_path: &Path, outputs: &[String]) -> Result<(), RunError> {
    let tmp = csv_path.with_extension("csv.tmp");
    {
        let mut w = BufWriter::new(File::create(&tmp).map_err(io_err)?);
        let mut header = vec![
            "event_id",
            "source_id",
            "path",
            "channel",
            "start_s",
            "end_s",
        ]
        .into_iter()
        .map(String::from)
        .collect::<Vec<_>>();
        header.extend(outputs.iter().map(|o| format!("score_{o}")));
        header.extend(
            [
                "automatic_label",
                "automatic_reason",
                "model_sha256",
                "artifact_kind",
            ]
            .map(String::from),
        );
        writeln!(w, "{}", header.join(",")).map_err(io_err)?;
        for line in BufReader::new(File::open(partial).map_err(io_err)?).lines() {
            let row: Value = serde_json::from_str(&line.map_err(io_err)?).map_err(io_err)?;
            let s = |k: &str| row[k].as_str().unwrap_or("").to_string();
            let mut fields = vec![
                s("event_id"),
                s("source_id"),
                s("path"),
                row["channel"].to_string(),
                format!("{:.6}", row["start_s"].as_f64().unwrap_or(f64::NAN)),
                format!("{:.6}", row["end_s"].as_f64().unwrap_or(f64::NAN)),
            ];
            for o in outputs {
                fields.push(
                    row["scores"][o]
                        .as_f64()
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_default(),
                );
            }
            fields.extend([
                s("automatic_label"),
                s("automatic_reason"),
                s("model_sha256"),
                s("artifact_kind"),
            ]);
            writeln!(
                w,
                "{}",
                fields
                    .iter()
                    .map(|f| csv_field(f))
                    .collect::<Vec<_>>()
                    .join(",")
            )
            .map_err(io_err)?;
        }
        w.flush().map_err(io_err)?;
        w.get_ref().sync_all().map_err(io_err)?;
    }
    fs::rename(&tmp, csv_path).map_err(io_err)
}

/// Final artifact validation: the CSV row count must equal the committed rows and every score is
/// finite and within [0, 1] (or empty when unavailable).
pub fn validate_outputs(csv_path: &Path, expected_rows: u64) -> Result<(), RunError> {
    let reader = BufReader::new(File::open(csv_path).map_err(io_err)?);
    let mut lines = reader.lines();
    let header = lines
        .next()
        .ok_or_else(|| RunError::Io("results CSV has no header".into()))?
        .map_err(io_err)?;
    let cols: Vec<&str> = header.split(',').collect();
    let score_idx: Vec<usize> = cols
        .iter()
        .enumerate()
        .filter(|(_, c)| c.starts_with("score_"))
        .map(|(i, _)| i)
        .collect();
    let mut n = 0u64;
    for line in lines {
        let line = line.map_err(io_err)?;
        let fields: Vec<&str> = line.split(',').collect();
        for &i in &score_idx {
            let f = fields.get(i).copied().unwrap_or("");
            if !f.is_empty() {
                let v: f64 = f
                    .parse()
                    .map_err(|_| RunError::Io(format!("bad score '{f}'")))?;
                if !(0.0..=1.0).contains(&v) {
                    return Err(RunError::Io(format!("score {v} out of range")));
                }
            }
        }
        n += 1;
    }
    if n != expected_rows {
        return Err(RunError::Io(format!(
            "results CSV has {n} rows, expected {expected_rows}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::resolve_device;

    #[test]
    fn cpu_request_stays_on_cpu() {
        let (active, note) = resolve_device("cpu");
        assert_eq!(active, "cpu");
        assert_eq!(note, "CPU requested");
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn gpu_requests_without_cuda_feature_fall_back() {
        for requested in ["auto", "hybrid", "gpu", "cuda"] {
            let (active, note) = resolve_device(requested);
            assert_eq!(active, "cpu", "{requested}");
            assert_eq!(
                note, "no GPU execution provider in this build; running on CPU",
                "{requested}"
            );
        }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn auto_without_a_visible_gpu_falls_back() {
        let (active, note) = resolve_device("auto");
        if active == "cuda" {
            assert!(note.to_ascii_lowercase().contains("cuda"));
            return;
        }
        assert_eq!(active, "cpu");
        assert!(
            note.contains("running on CPU"),
            "fallback note should say why: {note}"
        );
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn explicit_cuda_reports_cuda_or_a_cpu_reason() {
        let (active, note) = resolve_device("cuda");
        match active {
            "cuda" => assert!(note.to_ascii_lowercase().contains("cuda")),
            "cpu" => assert!(
                note.contains("running on CPU"),
                "fallback note should say why: {note}"
            ),
            other => panic!("unexpected device {other}"),
        }
    }
}
