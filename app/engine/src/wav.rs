//! Chunked WAV decoding with per-channel output. Files are opened read-only and never modified.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use hound::{SampleFormat, WavReader};

#[derive(Debug, Clone)]
pub struct AudioInfo {
    pub sample_rate: u32,
    pub channels: usize,
    pub frames: u64,
    pub encoding: String,
}

#[derive(Debug)]
pub enum DecodeError {
    Unsupported(String),
    Unreadable(String),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Unsupported(m) => write!(f, "unsupported input: {m}"),
            DecodeError::Unreadable(m) => write!(f, "unreadable input: {m}"),
        }
    }
}

pub struct ChunkedWav {
    reader: WavReader<BufReader<File>>,
    pub info: AudioInfo,
    scale: f64,
    float: bool,
}

impl ChunkedWav {
    pub fn open(path: &Path) -> Result<Self, DecodeError> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext != "wav" {
            return Err(DecodeError::Unsupported(format!(
                "'.{ext}' files are not supported yet (WAV only)"
            )));
        }
        let reader = WavReader::open(path).map_err(|e| DecodeError::Unreadable(e.to_string()))?;
        let spec = reader.spec();
        let float = spec.sample_format == SampleFormat::Float;
        if float && spec.bits_per_sample != 32 {
            return Err(DecodeError::Unsupported(format!(
                "{}-bit float WAV",
                spec.bits_per_sample
            )));
        }
        if !float && !(8..=32).contains(&spec.bits_per_sample) {
            return Err(DecodeError::Unsupported(format!(
                "{}-bit PCM WAV",
                spec.bits_per_sample
            )));
        }
        if spec.channels == 0 || spec.sample_rate == 0 {
            return Err(DecodeError::Unreadable(
                "zero channels or sample rate".into(),
            ));
        }
        let info = AudioInfo {
            sample_rate: spec.sample_rate,
            channels: spec.channels as usize,
            frames: reader.duration() as u64,
            encoding: format!(
                "{}{}",
                if float { "float" } else { "pcm" },
                spec.bits_per_sample
            ),
        };
        let scale = if float {
            1.0
        } else {
            (1_u64 << (spec.bits_per_sample - 1)) as f64
        };
        Ok(Self {
            reader,
            info,
            scale,
            float,
        })
    }

    /// Read up to `max_frames` frames, de-interleaved into one buffer per channel. Empty at end of file.
    pub fn read_chunk(&mut self, max_frames: usize) -> Result<Vec<Vec<f64>>, DecodeError> {
        let ch = self.info.channels;
        let mut out: Vec<Vec<f64>> = (0..ch).map(|_| Vec::with_capacity(max_frames)).collect();
        let wanted = max_frames * ch;
        let mut idx = 0usize;
        if self.float {
            for s in self.reader.samples::<f32>().take(wanted) {
                let v = s.map_err(|e| DecodeError::Unreadable(e.to_string()))?;
                out[idx % ch].push(v as f64);
                idx += 1;
            }
        } else {
            for s in self.reader.samples::<i32>().take(wanted) {
                let v = s.map_err(|e| DecodeError::Unreadable(e.to_string()))?;
                out[idx % ch].push(v as f64 / self.scale);
                idx += 1;
            }
        }
        if idx % ch != 0 {
            return Err(DecodeError::Unreadable("truncated final frame".into()));
        }
        Ok(out)
    }
}
