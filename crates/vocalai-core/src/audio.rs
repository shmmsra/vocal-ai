//! Audio I/O. B.1 (Milestone 6, part B, `docs/issues.md` VAI-006) only needs WAV
//! *writing* -- the default-voice pipeline has no audio input to preprocess.
//! Reading/resampling reference audio for `--voice` zero-shot cloning is B.2 scope
//! (not yet implemented) and will be added to this module when that work starts,
//! rather than speculatively now.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6).

use std::path::Path;

/// Writes a mono `f32` waveform (samples in `[-1.0, 1.0]`, matching S3Gen/HiFiGAN's
/// output convention) to a 16-bit PCM WAV file at `sample_rate`.
pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<(), hound::Error> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)?;
    for &sample in samples {
        let clamped = sample.clamp(-1.0, 1.0);
        let quantized = (clamped * i16::MAX as f32).round() as i16;
        writer.write_sample(quantized)?;
    }
    writer.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_wav_round_trips_sample_values() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vocalai-audio-test-{}.wav", std::process::id()));
        let samples = [0.0_f32, 0.5, -0.5, 1.0, -1.0];

        write_wav(&path, &samples, 24_000).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        assert_eq!(reader.spec().sample_rate, 24_000);
        assert_eq!(reader.spec().channels, 1);
        let read_back: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(read_back.len(), samples.len());
        assert_eq!(read_back[0], 0);
        assert_eq!(read_back[3], i16::MAX);
        assert_eq!(read_back[4], -i16::MAX);
    }

    #[test]
    fn write_wav_clamps_out_of_range_samples() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vocalai-audio-clamp-test-{}.wav",
            std::process::id()
        ));
        write_wav(&path, &[2.0, -2.0], 24_000).unwrap();

        let mut reader = hound::WavReader::open(&path).unwrap();
        let read_back: Vec<i16> = reader.samples::<i16>().map(|s| s.unwrap()).collect();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(read_back[0], i16::MAX);
        assert_eq!(read_back[1], -i16::MAX);
    }
}
