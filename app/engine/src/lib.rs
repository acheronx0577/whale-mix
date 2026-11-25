//! WhaleMix Acoustic v1 inference engine.
//!
//! One background process per active analysis run. It validates an explicitly selected model bundle,
//! decodes WAV files in bounded chunks, reproduces the Python preprocessing contract, runs one ONNX
//! Runtime CPU session, and streams JSON Lines events on stdout.

pub mod bundle;
pub mod engine;
pub mod preprocess;
pub mod protocol;
pub mod wav;
