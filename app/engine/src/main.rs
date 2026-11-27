//! `whale-mix-engine --inputs <manifest.jsonl> --out <dir> --model-bundle <dir> [--profile <id>]...
//!                   [--threads N] [--batch N] [--device auto|hybrid|gpu|cuda|cpu] [--check-bundle]
//!                   [--allow-fixture-bundle]`
//!
//! Exit codes: 0 completed and validated, 1 run failed, 2 invalid arguments/configuration,
//! 3 model bundle unavailable or invalid, 4 canceled with partial results.

use std::io::BufRead;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use serde_json::json;
use whale_mix_engine::bundle;
use whale_mix_engine::engine::{self, Control, Outcome, RunConfig};
use whale_mix_engine::protocol::Emitter;

struct Args {
    cfg: RunConfig,
    check_only: bool,
}

fn parse(argv: &[String]) -> Result<Args, String> {
    let mut cfg = RunConfig {
        inputs: PathBuf::new(),
        out: PathBuf::new(),
        bundle_dir: PathBuf::new(),
        profiles: Vec::new(),
        threads: 2,
        batch: 8,
        allow_fixture: false,
        device: "auto".into(),
        run_id: format!("run-{}", std::process::id()),
    };
    let mut check_only = false;
    let mut it = argv.iter().skip(1);
    while let Some(arg) = it.next() {
        let mut value = |name: &str| {
            it.next()
                .cloned()
                .ok_or_else(|| format!("{name} needs a value"))
        };
        match arg.as_str() {
            "--inputs" => cfg.inputs = PathBuf::from(value("--inputs")?),
            "--out" => cfg.out = PathBuf::from(value("--out")?),
            "--model-bundle" => cfg.bundle_dir = PathBuf::from(value("--model-bundle")?),
            "--profile" => cfg.profiles.push(value("--profile")?),
            "--threads" => {
                cfg.threads = value("--threads")?
                    .parse()
                    .map_err(|_| "--threads must be a number")?
            }
            "--batch" => {
                cfg.batch = value("--batch")?
                    .parse()
                    .map_err(|_| "--batch must be a number")?
            }
            "--run-id" => cfg.run_id = value("--run-id")?,
            "--device" => {
                let d = value("--device")?.to_ascii_lowercase();
                if !engine::DEVICES.contains(&d.as_str()) {
                    return Err("--device must be auto, hybrid, gpu, cuda or cpu".into());
                }
                cfg.device = d;
            }
            "--allow-fixture-bundle" => cfg.allow_fixture = true,
            "--check-bundle" => check_only = true,
            other => return Err(format!("unknown argument {other}")),
        }
    }
    if cfg.bundle_dir.as_os_str().is_empty() {
        return Err("--model-bundle is required".into());
    }
    if !check_only && (cfg.inputs.as_os_str().is_empty() || cfg.out.as_os_str().is_empty()) {
        return Err("--inputs and --out are required".into());
    }
    if cfg.profiles.is_empty() {
        cfg.profiles.push("auto".into());
    }
    Ok(Args { cfg, check_only })
}

fn main() -> ExitCode {
    let argv: Vec<String> = std::env::args().collect();
    let args = match parse(&argv) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return ExitCode::from(2);
        }
    };
    let emitter = Arc::new(Emitter::stdout(&args.cfg.run_id));
    let bundle = match bundle::load(&args.cfg.bundle_dir, args.cfg.allow_fixture) {
        Ok(b) => b,
        Err(e) => {
            emitter.emit(
                "model_unavailable",
                json!({"code": e.code(), "reason": e.to_string()}),
            );
            eprintln!("error: {e}");
            return ExitCode::from(3);
        }
    };
    if args.check_only {
        let m = &bundle.manifest;
        emitter.emit(
            "model_ready",
            json!({"artifact_kind": m.artifact_kind, "model_sha256": m.model.sha256,
            "label_map": m.label_map.outputs, "preprocessing_hash": m.preprocessing_hash}),
        );
        return ExitCode::SUCCESS;
    }
    let control = Arc::new(Control::default());
    let stdin_control = Arc::clone(&control);
    let ack = Arc::clone(&emitter);
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines().map_while(Result::ok) {
            if let Some(kind) = stdin_control.apply(&line) {
                ack.emit(kind, json!({}));
            }
        }
    });
    match engine::run(&args.cfg, &bundle, control, &emitter) {
        Ok(Outcome::Completed { .. }) => ExitCode::SUCCESS,
        Ok(Outcome::Canceled { .. }) => ExitCode::from(4),
        Err(e) => {
            emitter.emit("failed", json!({"reason": e.to_string()}));
            eprintln!("error: {e}");
            match e {
                engine::RunError::Config(_) => ExitCode::from(2),
                _ => ExitCode::from(1),
            }
        }
    }
}
