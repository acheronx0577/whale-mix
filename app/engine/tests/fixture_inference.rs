//! End-to-end inference with an untrained fixture bundle (`WHALEMIX_FIXTURE_BUNDLE`). Skipped, with a
//! message, when the variable is unset; scores from a fixture are plumbing checks, never accuracy.
mod common;

use std::sync::Arc;

use whale_mix_engine::bundle::load;
use whale_mix_engine::engine::{run, Control, Outcome, RunConfig};
use whale_mix_engine::protocol::Emitter;

fn config(
    dir: &std::path::Path,
    bundle: &std::path::Path,
    wavs: &[std::path::PathBuf],
) -> RunConfig {
    let inputs = dir.join("inputs.jsonl");
    let lines: Vec<String> = wavs
        .iter()
        .enumerate()
        .map(|(i, p)| serde_json::json!({"source_id": format!("s{i}"), "path": p}).to_string())
        .collect();
    std::fs::write(&inputs, lines.join("\n") + "\n").unwrap();
    RunConfig {
        inputs,
        out: dir.join("out"),
        bundle_dir: bundle.to_path_buf(),
        profiles: vec!["auto".into()],
        threads: 2,
        batch: 8,
        allow_fixture: true,
        run_id: "test".into(),
    }
}

#[test]
fn fixture_bundle_end_to_end() {
    let Some(bundle_dir) = common::fixture_bundle() else {
        common::skip("WHALEMIX_FIXTURE_BUNDLE not set");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("tone.wav");
    common::write_wav(&wav, 8000, 2, &common::signal(8000, 23.0));
    let bad = dir.path().join("bad.wav");
    std::fs::write(&bad, b"RIFF0000WAVEjunk").unwrap();
    let cfg = config(dir.path(), &bundle_dir, &[wav, bad]);
    let bundle = load(&bundle_dir, true).unwrap();
    let events = dir.path().join("events.jsonl");
    let emitter = Emitter::new("test", Box::new(std::fs::File::create(&events).unwrap()));
    let outcome = run(&cfg, &bundle, Arc::new(Control::default()), &emitter).unwrap();
    drop(emitter);
    // 23 s at 5 s hop -> windows at 0,5,10 plus an end-aligned tail, per channel.
    let Outcome::Completed { rows } = outcome else {
        panic!("not completed")
    };
    assert_eq!(rows, 8);
    let csv = std::fs::read_to_string(cfg.out.join("results.csv")).unwrap();
    assert_eq!(csv.lines().count(), 9);
    assert!(csv.contains("fixture-untrained"));
    let run_json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(cfg.out.join("run.json")).unwrap()).unwrap();
    assert_eq!(run_json["files_failed"], 1);
    assert_eq!(run_json["resources"]["measurement_kind"], "fixture");
    let ev = std::fs::read_to_string(&events).unwrap();
    for kind in ["started", "results_batch", "file_error", "completed"] {
        assert!(
            ev.contains(&format!("\"type\":\"{kind}\"")),
            "missing {kind}"
        );
    }
}

#[test]
fn fixture_matches_python_parity_reference() {
    let Some(bundle_dir) = common::fixture_bundle() else {
        common::skip("WHALEMIX_FIXTURE_BUNDLE not set");
        return;
    };
    let reference = bundle_dir.with_file_name(format!(
        "{}-parity-reference.json",
        bundle_dir.file_name().unwrap().to_string_lossy()
    ));
    let Some(wav) = std::env::var_os("WHALEMIX_PARITY_WAV") else {
        common::skip("WHALEMIX_PARITY_WAV not set");
        return;
    };
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path(), &bundle_dir, &[wav.into()]);
    cfg.batch = 3;
    let bundle = load(&bundle_dir, true).unwrap();
    let emitter = Emitter::new("test", Box::new(std::io::sink()));
    run(&cfg, &bundle, Arc::new(Control::default()), &emitter).unwrap();
    let rows: Vec<serde_json::Value> =
        std::fs::read_to_string(cfg.out.join("results.partial.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
    let windows = reference["windows"].as_array().unwrap();
    assert_eq!(rows.len(), windows.len());
    let outputs = &bundle.manifest.label_map.outputs;
    let mut max_diff = 0.0f64;
    for (r, w) in rows.iter().zip(windows) {
        assert!((r["start_s"].as_f64().unwrap() - w["start_s"].as_f64().unwrap()).abs() < 1e-9);
        for (j, o) in outputs.iter().enumerate() {
            if let Some(v) = r["scores"][o].as_f64() {
                max_diff = max_diff.max((v - w["probabilities"][j].as_f64().unwrap()).abs());
            }
        }
    }
    eprintln!("fixture parity max |rust - python| probability = {max_diff:e}");
    assert!(max_diff < 1e-4, "probability parity {max_diff}");
}

/// Writer that requests cancellation as soon as the first durable results batch is announced,
/// mimicking the app's cancel button arriving mid-run.
struct CancelOnFirstBatch(Arc<Control>, std::sync::mpsc::Sender<std::time::Instant>);

impl std::io::Write for CancelOnFirstBatch {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        if buf.windows(15).any(|w| w == b"\"results_batch\"")
            && !self.0.cancel.load(std::sync::atomic::Ordering::SeqCst)
        {
            self.0.apply("cancel");
            let _ = self.1.send(std::time::Instant::now());
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn cancel_stops_at_a_chunk_boundary_with_valid_partial_output() {
    let Some(bundle_dir) = common::fixture_bundle() else {
        common::skip("WHALEMIX_FIXTURE_BUNDLE not set");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("long.wav");
    common::write_wav(&wav, 8000, 1, &common::signal(8000, 600.0));
    let cfg = config(dir.path(), &bundle_dir, &[wav]);
    let bundle = load(&bundle_dir, true).unwrap();
    let control = Arc::new(Control::default());
    let (tx, rx) = std::sync::mpsc::channel();
    let emitter = Emitter::new(
        "test",
        Box::new(CancelOnFirstBatch(Arc::clone(&control), tx)),
    );
    let outcome = run(&cfg, &bundle, control, &emitter).unwrap();
    let ack_ms = rx.recv().unwrap().elapsed().as_secs_f64() * 1000.0;
    eprintln!("fixture cancel: run returned {ack_ms:.1} ms after the cancel request");
    let Outcome::Canceled { rows } = outcome else {
        panic!("expected cancel")
    };
    assert!((8..119).contains(&rows), "rows {rows}");
    let csv = std::fs::read_to_string(cfg.out.join("results.partial.csv")).unwrap();
    assert_eq!(csv.lines().count() as u64, rows + 1);
    assert!(cfg.out.join("run.partial.json").is_file());
    assert!(!cfg.out.join("results.csv").exists());
}

#[test]
fn pause_holds_work_until_resume() {
    let Some(bundle_dir) = common::fixture_bundle() else {
        common::skip("WHALEMIX_FIXTURE_BUNDLE not set");
        return;
    };
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("tone.wav");
    common::write_wav(&wav, 8000, 1, &common::signal(8000, 30.0));
    let cfg = config(dir.path(), &bundle_dir, &[wav]);
    let bundle = load(&bundle_dir, true).unwrap();
    let control = Arc::new(Control::default());
    control.apply("pause");
    let c2 = Arc::clone(&control);
    let resumer = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(400));
        c2.apply("resume");
    });
    let t = std::time::Instant::now();
    let emitter = Emitter::new("test", Box::new(std::io::sink()));
    let outcome = run(&cfg, &bundle, control, &emitter).unwrap();
    resumer.join().unwrap();
    assert!(t.elapsed().as_millis() >= 400);
    assert!(matches!(outcome, Outcome::Completed { rows: 5 }));
}
