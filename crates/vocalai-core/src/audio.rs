//! Audio I/O. B.1 (Milestone 6, part B, `docs/issues.md` VAI-006) only needed WAV
//! *writing* -- the default-voice pipeline has no audio input to preprocess.
//! B.2 (`--voice` zero-shot cloning) adds [`read_wav`] (mono `f32` PCM, WAV only
//! -- matching this pipeline's existing WAV-only convention, no mp3/flac
//! decoding) and [`resample`] (arbitrary sample-rate pairs, `Result`-returning).
//!
//! [`resample`] duplicates `watermark.rs`'s private `resample` rather than
//! sharing it: that one is hardcoded/`.expect()`-tuned for the one fixed
//! 32000<->24000 ratio it always sees, while this one must handle arbitrary
//! user-supplied reference-file sample rates and report failure via `Result`
//! instead of panicking -- an accepted small duplication (same call the plan
//! made for `mel.rs`'s DSP helpers vs. `watermark.rs`'s).
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6).

use std::path::Path;

use rubato::audioadapter_buffers::direct::SequentialSliceOfVecs;
use rubato::{Fft as FftResampler, FixedSync, Resampler};

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

/// Reads a WAV file as mono `f32` PCM in `[-1.0, 1.0]`, downmixing multi-channel
/// audio by averaging channels (matching `librosa.load`'s mono downmix).
/// Returns `(samples, sample_rate)`. WAV only, matching this pipeline's
/// existing input-format convention (no mp3/flac decoding).
pub fn read_wav(path: &Path) -> Result<(Vec<f32>, u32), hound::Error> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    let channels = spec.channels as usize;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, hound::Error>>()?,
        hound::SampleFormat::Int => {
            let scale = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / scale))
                .collect::<Result<Vec<f32>, hound::Error>>()?
        }
    };

    let mono = if channels > 1 {
        samples
            .chunks(channels)
            .map(|frame| frame.iter().sum::<f32>() / channels as f32)
            .collect()
    } else {
        samples
    };
    Ok((mono, spec.sample_rate))
}

/// Errors from [`resample`]: either the requested sample-rate pair can't
/// construct a resampler, or a well-formed resampler still fails to process
/// (both are `rubato` error types, boxed here rather than duplicated as an
/// enum since callers only ever need `Display`).
#[derive(Debug)]
pub struct ResampleError(String);

impl std::fmt::Display for ResampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ResampleError {}

impl From<rubato::ResamplerConstructionError> for ResampleError {
    fn from(e: rubato::ResamplerConstructionError) -> Self {
        ResampleError(e.to_string())
    }
}

impl From<rubato::ResampleError> for ResampleError {
    fn from(e: rubato::ResampleError) -> Self {
        ResampleError(e.to_string())
    }
}

/// Resamples `signal` from `from_hz` to `to_hz` (mono) via `rubato`'s FFT-based
/// synchronous resampler, for arbitrary sample-rate pairs (unlike
/// `watermark.rs`'s fixed-ratio twin -- see module docs). Not bit-exact with
/// librosa's default `soxr_hq` resampler -- correctness here is "reasonable
/// audio fidelity", not cross-language numerical parity (same accepted gap as
/// `watermark.rs`'s resampler, `docs/issues.md` VAI-005).
pub fn resample(signal: &[f32], from_hz: u32, to_hz: u32) -> Result<Vec<f32>, ResampleError> {
    if from_hz == to_hz || signal.is_empty() {
        return Ok(signal.to_vec());
    }
    const CHUNK_SIZE: usize = 1024;
    let mut resampler = FftResampler::<f32>::new(
        from_hz as usize,
        to_hz as usize,
        CHUNK_SIZE,
        1,
        FixedSync::Both,
    )?;

    let input_channels = vec![signal.to_vec()];
    let input = SequentialSliceOfVecs::new(&input_channels, 1, signal.len())
        .expect("input buffer sized exactly to signal.len()");
    let output = resampler.process_all(&input, signal.len(), None)?;
    Ok(output.take_data())
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

    #[test]
    fn read_wav_round_trips_what_write_wav_wrote() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vocalai-audio-readback-test-{}.wav",
            std::process::id()
        ));
        let samples = [0.0_f32, 0.5, -0.5, 1.0, -1.0];
        write_wav(&path, &samples, 22_050).unwrap();

        let (read_back, sample_rate) = read_wav(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(sample_rate, 22_050);
        assert_eq!(read_back.len(), samples.len());
        for (a, b) in read_back.iter().zip(samples.iter()) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
    }

    #[test]
    fn read_wav_downmixes_stereo_by_averaging_channels() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "vocalai-audio-stereo-test-{}.wav",
            std::process::id()
        ));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(&path, spec).unwrap();
        // Frame 0: left=1.0, right=-1.0 -> mono average 0.0.
        writer.write_sample(i16::MAX).unwrap();
        writer.write_sample(-i16::MAX).unwrap();
        writer.finalize().unwrap();

        let (mono, sample_rate) = read_wav(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(sample_rate, 16_000);
        assert_eq!(mono.len(), 1);
        assert!(mono[0].abs() < 1e-3);
    }

    #[test]
    fn resample_preserves_duration_for_an_arbitrary_ratio() {
        let signal: Vec<f32> = (0..44_100)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 44_100.0).sin())
            .collect();
        let resampled = resample(&signal, 44_100, 16_000).unwrap();
        assert!(
            (resampled.len() as i64 - 16_000).abs() < 64,
            "resampled.len()={}",
            resampled.len()
        );
    }

    #[test]
    fn resample_is_a_no_op_for_equal_rates() {
        let signal = vec![0.1_f32, 0.2, 0.3];
        let resampled = resample(&signal, 24_000, 24_000).unwrap();
        assert_eq!(resampled, signal);
    }
}
