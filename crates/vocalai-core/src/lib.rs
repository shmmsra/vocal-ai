//! Inference runtime library for vocal-ai. See `docs/phase1-onnx-rust-cli-plan.md`
//! for the full architecture and `docs/issues.md` for milestone tracking.

pub mod audio;
pub mod campplus;
pub mod mel;
pub mod pipeline;
pub mod s3gen;
pub mod s3tokenizer;
pub mod session;
pub mod t3;
pub mod tokenizer;
pub mod voice_encoder;
pub mod watermark;
