//! Streaming mirror of the Python preprocessing contract v1 (`whalemix_training.preprocess`).
//!
//! Each source channel gets its own [`ChannelPipeline`]. Source samples are pushed in chunks; the
//! pipeline resamples with the formula-defined Kaiser-sinc polyphase filter, computes power frames on the
//! global grid, keeps only the frames that pending windows still need, and emits finished windows.
//! Overlapping windows reuse the same power frames, and memory stays bounded by one window plus one hop.

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, Fft, FftPlanner};
use serde::Deserialize;

pub const CONTRACT_VERSION: u32 = 1;

#[derive(Debug, Clone, Deserialize)]
pub struct Pcen {
    pub time_constant_s: f64,
    pub gain: f64,
    pub bias: f64,
    pub power: f64,
    pub eps: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub kind: String,
    pub mean: Option<f64>,
    pub std: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Spec {
    pub contract_version: u32,
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop_length: usize,
    pub window: String,
    pub compression: String,
    pub log_floor: f64,
    pub pcen: Pcen,
    pub window_frames: usize,
    pub window_hop_frames: usize,
    pub bin_start: usize,
    pub bin_end: usize,
    pub resample_half_taps_per_phase: usize,
    pub resample_kaiser_beta: f64,
    pub normalization: Normalization,
    pub channel_policy: String,
}

impl Spec {
    pub fn validate(&self) -> Result<(), String> {
        if self.contract_version != CONTRACT_VERSION {
            return Err(format!(
                "unsupported preprocessing contract {}",
                self.contract_version
            ));
        }
        if self.window != "hann_periodic" || self.channel_policy != "independent" {
            return Err("unsupported window or channel policy".into());
        }
        if self.compression != "log" && self.compression != "pcen" {
            return Err("unsupported compression".into());
        }
        if self.window_frames == 0 || self.window_hop_frames == 0 || self.hop_length == 0 {
            return Err("window and hop must be positive".into());
        }
        if self.bin_end <= self.bin_start || self.bin_end > self.n_fft / 2 + 1 {
            return Err("invalid frequency bin range".into());
        }
        match self.normalization.kind.as_str() {
            "none" => Ok(()),
            "global_standardize" => match (self.normalization.mean, self.normalization.std) {
                (Some(_), Some(s)) if s > 0.0 => Ok(()),
                _ => Err("global_standardize needs mean and a positive std".into()),
            },
            _ => Err("unsupported normalization".into()),
        }
    }

    pub fn n_bins(&self) -> usize {
        self.bin_end - self.bin_start
    }

    pub fn window_span_s(&self) -> f64 {
        ((self.window_frames - 1) * self.hop_length + self.n_fft) as f64 / self.sample_rate as f64
    }
}

fn bessel_i0(x: f64) -> f64 {
    let (mut total, mut term, mut k) = (1.0_f64, 1.0_f64, 1.0_f64);
    let half = x / 2.0;
    loop {
        term *= (half / k).powi(2);
        total += term;
        if term < 1e-17 * total {
            return total;
        }
        k += 1.0;
    }
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

pub fn resample_ratio(src: u32, dst: u32) -> (usize, usize) {
    let g = gcd(src as u64, dst as u64);
    ((dst as u64 / g) as usize, (src as u64 / g) as usize)
}

pub fn resample_filter(up: usize, down: usize, half_taps_per_phase: usize, beta: f64) -> Vec<f64> {
    let factor = up.max(down);
    let half = half_taps_per_phase * factor;
    let n = 2 * half + 1;
    let cutoff = 1.0 / factor as f64;
    let i0_beta = bessel_i0(beta);
    (0..n)
        .map(|k| {
            let t = k as f64 - half as f64;
            let x = cutoff * t;
            let sinc = if k == half {
                1.0
            } else {
                (PI * x).sin() / (PI * x)
            };
            let r = 2.0 * k as f64 / (n - 1) as f64 - 1.0;
            let win = bessel_i0(beta * (1.0 - r * r).max(0.0).sqrt()) / i0_beta;
            up as f64 * cutoff * sinc * win
        })
        .collect()
}

pub fn resampled_length(n_in: usize, up: usize, down: usize) -> usize {
    (n_in * up).div_ceil(down)
}

/// Streaming rational resampler; identical arithmetic to the Python `resample_segment`.
struct Resampler {
    up: usize,
    down: usize,
    taps: Vec<f64>,
    half: usize,
    src: VecDeque<f64>,
    src_base: usize,
    src_total: usize,
    next_out: usize,
}

impl Resampler {
    fn new(up: usize, down: usize, spec: &Spec) -> Self {
        let taps = if up == down {
            vec![1.0]
        } else {
            resample_filter(
                up,
                down,
                spec.resample_half_taps_per_phase,
                spec.resample_kaiser_beta,
            )
        };
        let half = (taps.len() - 1) / 2;
        Self {
            up,
            down,
            taps,
            half,
            src: VecDeque::new(),
            src_base: 0,
            src_total: 0,
            next_out: 0,
        }
    }

    fn push(&mut self, samples: &[f64], out: &mut Vec<f64>) {
        self.src.extend(samples.iter().copied());
        self.src_total += samples.len();
        self.drain(false, out);
    }

    fn finish(&mut self, out: &mut Vec<f64>) {
        self.drain(true, out);
    }

    fn value(&self, n: usize) -> f64 {
        if self.up == self.down {
            return self.src[n - self.src_base];
        }
        let base = n * self.down + self.half;
        let i_max = base / self.up;
        let k0 = base - i_max * self.up;
        let q_count = self.taps.len().div_ceil(self.up);
        let mut acc = 0.0;
        for q in 0..q_count {
            let k = k0 + q * self.up;
            if k >= self.taps.len() || q > i_max {
                continue;
            }
            let i = i_max - q;
            if i >= self.src_total || i < self.src_base {
                continue;
            }
            acc += self.taps[k] * self.src[i - self.src_base];
        }
        acc
    }

    fn drain(&mut self, eof: bool, out: &mut Vec<f64>) {
        let total_out = if eof {
            Some(resampled_length(self.src_total, self.up, self.down))
        } else {
            None
        };
        loop {
            let n = self.next_out;
            if let Some(t) = total_out {
                if n >= t {
                    break;
                }
            }
            let i_hi = (n * self.down + self.half) / self.up;
            if !eof && i_hi >= self.src_total {
                break;
            }
            out.push(self.value(n));
            self.next_out += 1;
        }
        // drop source samples no later output can use
        let lowest_needed =
            (self.next_out * self.down + self.half).saturating_sub(self.taps.len() - 1) / self.up;
        let keep_from = lowest_needed.saturating_sub(1);
        while self.src_base < keep_from && !self.src.is_empty() {
            self.src.pop_front();
            self.src_base += 1;
        }
    }
}

/// One model window: start frame on the global grid and its `[1, bins, frames]` tensor (bin-major).
#[derive(Debug, Clone)]
pub struct Window {
    pub start_frame: usize,
    pub start_s: f64,
    pub end_s: f64,
    pub tensor: Vec<f32>,
}

pub struct ChannelPipeline {
    spec: Spec,
    resampler: Resampler,
    res: VecDeque<f64>,
    res_base: usize,
    res_total: usize,
    frames: VecDeque<Vec<f64>>,
    frame_base: usize,
    frames_total: usize,
    next_window: usize,
    last_emitted: Option<usize>,
    fft: Arc<dyn Fft<f64>>,
    hann: Vec<f64>,
    scratch: Vec<Complex<f64>>,
}

impl ChannelPipeline {
    pub fn new(spec: &Spec, source_rate: u32) -> Self {
        let (up, down) = resample_ratio(source_rate, spec.sample_rate);
        let mut planner = FftPlanner::<f64>::new();
        let n = spec.n_fft;
        Self {
            spec: spec.clone(),
            resampler: Resampler::new(up, down, spec),
            res: VecDeque::new(),
            res_base: 0,
            res_total: 0,
            frames: VecDeque::new(),
            frame_base: 0,
            frames_total: 0,
            next_window: 0,
            last_emitted: None,
            fft: planner.plan_fft_forward(n),
            hann: (0..n)
                .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f64 / n as f64).cos())
                .collect(),
            scratch: vec![Complex::new(0.0, 0.0); n],
        }
    }

    pub fn push(&mut self, samples: &[f64]) -> Vec<Window> {
        let mut out = Vec::new();
        self.resampler.push(samples, &mut out);
        self.accept_resampled(&out);
        self.emit_regular()
    }

    pub fn finish(&mut self) -> Vec<Window> {
        let mut out = Vec::new();
        self.resampler.finish(&mut out);
        self.accept_resampled(&out);
        let need = self.spec.n_fft + (self.spec.window_frames - 1) * self.spec.hop_length;
        if self.res_total < need {
            let pad = vec![0.0; need - self.res_total];
            self.accept_resampled(&pad);
        }
        let mut windows = self.emit_regular();
        let width = self.spec.window_frames;
        let n = self.frames_total;
        let covered = self.last_emitted.map(|s| s + width).unwrap_or(0);
        if covered < n {
            let start = n - width;
            windows.push(self.make_window(start));
            self.last_emitted = Some(start);
        }
        windows
    }

    fn accept_resampled(&mut self, samples: &[f64]) {
        self.res.extend(samples.iter().copied());
        self.res_total += samples.len();
        let (n_fft, hop) = (self.spec.n_fft, self.spec.hop_length);
        while self.frames_total * hop + n_fft <= self.res_total {
            let start = self.frames_total * hop - self.res_base;
            for i in 0..n_fft {
                self.scratch[i] = Complex::new(self.res[start + i] * self.hann[i], 0.0);
            }
            self.fft.process(&mut self.scratch);
            let power = self.scratch[self.spec.bin_start..self.spec.bin_end]
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect();
            self.frames.push_back(power);
            self.frames_total += 1;
        }
        let keep_from = self.frames_total * hop;
        while self.res_base < keep_from && !self.res.is_empty() {
            self.res.pop_front();
            self.res_base += 1;
        }
    }

    fn emit_regular(&mut self) -> Vec<Window> {
        let width = self.spec.window_frames;
        let mut out = Vec::new();
        while self.next_window + width <= self.frames_total {
            out.push(self.make_window(self.next_window));
            self.last_emitted = Some(self.next_window);
            self.next_window += self.spec.window_hop_frames;
        }
        // keep frames a later regular or tail window may still need (bounded by one window + one hop)
        let keep_from = self
            .next_window
            .min(self.frames_total.saturating_sub(width));
        while self.frame_base < keep_from && !self.frames.is_empty() {
            self.frames.pop_front();
            self.frame_base += 1;
        }
        out
    }

    fn make_window(&self, start: usize) -> Window {
        let (width, bins) = (self.spec.window_frames, self.spec.n_bins());
        let mut block: Vec<Vec<f64>> = (start..start + width)
            .map(|f| self.frames[f - self.frame_base].clone())
            .collect();
        self.compress(&mut block);
        let mut tensor = vec![0.0_f32; bins * width];
        for (t, frame) in block.iter().enumerate() {
            for (b, v) in frame.iter().enumerate() {
                tensor[b * width + t] = *v as f32;
            }
        }
        let start_s = (start * self.spec.hop_length) as f64 / self.spec.sample_rate as f64;
        Window {
            start_frame: start,
            start_s,
            end_s: start_s + self.spec.window_span_s(),
            tensor,
        }
    }

    fn compress(&self, block: &mut [Vec<f64>]) {
        if self.spec.compression == "log" {
            for frame in block.iter_mut() {
                for v in frame.iter_mut() {
                    *v = (*v + self.spec.log_floor).ln();
                }
            }
        } else {
            let p = &self.spec.pcen;
            let t_frames =
                p.time_constant_s * self.spec.sample_rate as f64 / self.spec.hop_length as f64;
            let s = ((1.0 + 4.0 * t_frames * t_frames).sqrt() - 1.0) / (2.0 * t_frames * t_frames);
            let mut smooth = block[0].clone();
            let bias_r = p.bias.powf(p.power);
            for (t, frame) in block.iter_mut().enumerate() {
                if t > 0 {
                    for (m, x) in smooth.iter_mut().zip(frame.iter()) {
                        *m = (1.0 - s) * *m + s * *x;
                    }
                }
                for (v, m) in frame.iter_mut().zip(smooth.iter()) {
                    *v = (*v / (p.eps + *m).powf(p.gain) + p.bias).powf(p.power) - bias_r;
                }
            }
        }
        if self.spec.normalization.kind == "global_standardize" {
            let (mean, std) = (
                self.spec.normalization.mean.unwrap_or(0.0),
                self.spec.normalization.std.unwrap_or(1.0),
            );
            for frame in block.iter_mut() {
                for v in frame.iter_mut() {
                    *v = (*v - mean) / std;
                }
            }
        }
    }

    /// Frames currently retained (for bounded-memory tests).
    pub fn retained_frames(&self) -> usize {
        self.frames.len()
    }
}
