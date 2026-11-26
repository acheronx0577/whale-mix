mod common;

use whale_mix_engine::preprocess::{ChannelPipeline, Window};

fn run_pipeline(
    spec_json: &serde_json::Value,
    rate: u32,
    samples: &[f64],
    chunk: usize,
) -> (Vec<Window>, usize) {
    let spec = common::spec_of(spec_json);
    let mut pipe = ChannelPipeline::new(&spec, rate);
    let mut out = Vec::new();
    let mut max_retained = 0;
    for c in samples.chunks(chunk) {
        out.extend(pipe.push(c));
        max_retained = max_retained.max(pipe.retained_frames());
    }
    out.extend(pipe.finish());
    (out, max_retained)
}

#[test]
fn rust_windows_match_python_reference() {
    let parity = common::parity();
    let stride = parity["stride"].as_u64().unwrap() as usize;
    for case in parity["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rate = case["source_rate"].as_u64().unwrap() as u32;
        let samples = common::signal(rate, case["seconds"].as_f64().unwrap());
        let (windows, _) = run_pipeline(&case["preprocessing"], rate, &samples, 4096);
        let expected = case["windows"].as_array().unwrap();
        assert_eq!(windows.len(), expected.len(), "{name}: window count");
        for (w, e) in windows.iter().zip(expected) {
            assert!(
                (w.start_s - e["start_s"].as_f64().unwrap()).abs() < 1e-9,
                "{name}: start"
            );
            assert!(
                (w.end_s - e["end_s"].as_f64().unwrap()).abs() < 1e-9,
                "{name}: end"
            );
            let sum: f64 = w.tensor.iter().map(|v| *v as f64).sum();
            let esum = e["tensor_sum"].as_f64().unwrap();
            assert!(
                (sum - esum).abs() <= 1e-4 * esum.abs().max(1.0),
                "{name}: sum {sum} vs {esum}"
            );
            let mut max_diff = 0.0f64;
            for (k, ev) in e["samples"].as_array().unwrap().iter().enumerate() {
                max_diff = max_diff.max((w.tensor[k * stride] as f64 - ev.as_f64().unwrap()).abs());
            }
            assert!(max_diff < 1e-3, "{name}: max sampled diff {max_diff}");
        }
    }
}

#[test]
fn chunk_size_does_not_change_windows() {
    let parity = common::parity();
    let case = &parity["cases"][1];
    let samples = common::signal(8000, 13.0);
    let (a, _) = run_pipeline(&case["preprocessing"], 8000, &samples, 777);
    let (b, _) = run_pipeline(&case["preprocessing"], 8000, &samples, 65536);
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(&b) {
        assert_eq!(x.start_frame, y.start_frame);
        let d = x
            .tensor
            .iter()
            .zip(&y.tensor)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        assert!(d < 1e-5, "chunking changed values by {d}");
    }
}

#[test]
fn retained_frames_stay_bounded_on_long_input() {
    let pre = common::default_preprocessing();
    let spec = common::spec_of(&pre);
    // 10 minutes of synthetic audio streamed in chunks: retained state must not grow with duration.
    let samples = common::signal(8000, 600.0);
    let (windows, max_retained) = run_pipeline(&pre, 8000, &samples, 65536);
    assert!(windows.len() > 100);
    assert!(
        max_retained <= spec.window_frames + 2 * 65536 / spec.hop_length + 4,
        "retained {max_retained} frames"
    );
    for w in &windows {
        assert_eq!(w.tensor.len(), spec.n_bins() * spec.window_frames);
        assert!(w.tensor.iter().all(|v| v.is_finite()));
    }
}

/// Tensor-level parity against the Python reference for a real sample WAV (reference kept out of Git).
#[test]
fn sample_wav_tensors_match_python_reference() {
    let (Some(bundle), Some(wav)) = (
        common::fixture_bundle(),
        std::env::var_os("WHALEMIX_PARITY_WAV"),
    ) else {
        common::skip("WHALEMIX_FIXTURE_BUNDLE / WHALEMIX_PARITY_WAV not set");
        return;
    };
    let reference = bundle.with_file_name(format!(
        "{}-parity-reference.json",
        bundle.file_name().unwrap().to_string_lossy()
    ));
    let reference: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(reference).unwrap()).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(bundle.join("bundle.json")).unwrap())
            .unwrap();
    let spec = common::spec_of(&manifest["preprocessing"]);
    let mut reader = whale_mix_engine::wav::ChunkedWav::open(std::path::Path::new(&wav)).unwrap();
    let mut pipe = ChannelPipeline::new(&spec, reader.info.sample_rate);
    let mut windows = Vec::new();
    loop {
        let chunk = reader.read_chunk(10_000).unwrap();
        if chunk[0].is_empty() {
            break;
        }
        windows.extend(pipe.push(&chunk[0]));
    }
    windows.extend(pipe.finish());
    let expected = reference["windows"].as_array().unwrap();
    assert_eq!(windows.len(), expected.len());
    let mut max_diff = 0.0f64;
    for (w, e) in windows.iter().zip(expected) {
        for (a, b) in w.tensor.iter().zip(e["tensor"].as_array().unwrap()) {
            max_diff = max_diff.max((*a as f64 - b.as_f64().unwrap()).abs());
        }
    }
    eprintln!("sample wav tensor parity max |rust - python| = {max_diff:e}");
    assert!(max_diff < 1e-3);
}
