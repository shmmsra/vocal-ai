//! S3Gen flow-matching Euler ODE loop, chained into the HiFiGAN vocoder.
//!
//! Reimplements `ConditionalCFM.solve_euler`'s CFG-doubled Euler loop
//! (`chatterbox/models/s3gen/flow_matching.py`) against the exported flow
//! estimator ONNX graph (`export/export_s3gen.py`). The loop math is generic
//! over the per-step estimator call so it can be unit-tested without an ONNX
//! session (see `tests` below); `run_estimator`/`generate_waveform` provide the
//! real `ort`-backed wiring into the estimator and (Milestone 2) HiFiGAN
//! sessions. Numerical parity against the PyTorch reference is validated by
//! `export/parity_check.py::check_s3gen`, which drives the identical loop.
//!
//! Milestone 6 (`docs/issues.md` VAI-006) adds the *upstream* half: turning S3
//! speech tokens (prompt + T3-generated) into the `mu`/`mask`/`cond` tensors the
//! Euler loop above consumes, against the bucketed flow-encoder export
//! (`export/export_s3gen_flow_encoder.py`, ADR-0009) plus the speaker-embedding
//! normalize+affine step (`CausalMaskedDiffWithXvec.inference`,
//! `chatterbox/models/s3gen/flow.py`). See [`select_bucket`], [`pad_tokens`],
//! [`slice_valid_prefix`], [`build_cond`], [`sample_noise`], [`embed_speaker`],
//! [`run_flow_encoder`].
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 3, Milestone 6).

use core::f32::consts::FRAC_PI_2;

use ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;
use rand::Rng;
use rand_distr::StandardNormal;

/// `S3GEN_SR` (`chatterbox/models/s3gen/const.py`) -- S3Gen's mel/output
/// sample rate. Reused by `pipeline.rs`'s `--voice` reference-audio
/// preprocessing (Milestone 6, part B.2) wherever the Python reference resamples
/// to this rate.
pub const S3GEN_SR: u32 = 24_000;

/// Fixed-step count for the Euler solver. The base (non-meanflow) Chatterbox
/// config never overrides this — `s3gen.inference()` calls `flow_inference()`
/// with no `n_cfm_timesteps` argument (see `chatterbox/tts.py::generate`).
pub const N_TIMESTEPS: usize = 10;

/// Classifier-free-guidance rate, fixed by `CFM_PARAMS` in
/// `chatterbox/models/s3gen/configs.py` (not a runtime/CLI-exposed parameter).
pub const INFERENCE_CFG_RATE: f32 = 0.7;

/// The CFG-doubled (`2*B` batch) per-step input to the flow estimator.
pub struct EstimatorStep<'a> {
    pub x: &'a Array3<f32>,
    pub mask: &'a Array3<f32>,
    pub mu: &'a Array3<f32>,
    pub t: &'a Array1<f32>,
    pub spks: &'a Array2<f32>,
    pub cond: &'a Array3<f32>,
}

/// Cosine `t_scheduler` schedule matching `ConditionalCFM.solve_euler` (the only
/// scheduler the base Chatterbox config uses). Returns `n_timesteps + 1` values.
pub fn cosine_t_span(n_timesteps: usize) -> Vec<f32> {
    (0..=n_timesteps)
        .map(|i| {
            let t = i as f32 / n_timesteps as f32;
            1.0 - (t * FRAC_PI_2).cos()
        })
        .collect()
}

/// Fixed-step CFG-doubled Euler ODE solver, matching `ConditionalCFM.solve_euler`:
/// each step assembles a `2*B`-batch input (real `mu`/`spks`/`cond` in the first
/// `B` rows, zeroed in the second `B` — the CFG "unconditional" branch), calls
/// `estimator_step` once, then combines
/// `dxdt = (1 + cfg_rate) * dxdt_cond - cfg_rate * dxdt_uncond` before the update
/// `x += dt * dxdt`.
///
/// Generic over the estimator call's error type so the loop itself carries no
/// `ort` dependency in its signature; `estimator_step` threads through `?`.
#[allow(clippy::too_many_arguments)]
pub fn solve_euler<E>(
    x0: Array3<f32>,
    mu: &Array3<f32>,
    mask: &Array3<f32>,
    spks: &Array2<f32>,
    cond: &Array3<f32>,
    t_span: &[f32],
    cfg_rate: f32,
    mut estimator_step: impl FnMut(EstimatorStep<'_>) -> Result<Array3<f32>, E>,
) -> Result<Array3<f32>, E> {
    let b = mu.shape()[0];
    let mut x = x0;
    let zeros_mu = Array3::zeros(mu.raw_dim());
    let zeros_spks = Array2::zeros(spks.raw_dim());
    let zeros_cond = Array3::zeros(cond.raw_dim());

    for window in t_span.windows(2) {
        let (t, r) = (window[0], window[1]);

        let x_in = ndarray::concatenate(Axis(0), &[x.view(), x.view()])
            .expect("same-shaped views always concatenate");
        let mask_in = ndarray::concatenate(Axis(0), &[mask.view(), mask.view()])
            .expect("same-shaped views always concatenate");
        let mu_in = ndarray::concatenate(Axis(0), &[mu.view(), zeros_mu.view()])
            .expect("same-shaped views always concatenate");
        let t_in = Array1::from_elem(2 * b, t);
        let spks_in = ndarray::concatenate(Axis(0), &[spks.view(), zeros_spks.view()])
            .expect("same-shaped views always concatenate");
        let cond_in = ndarray::concatenate(Axis(0), &[cond.view(), zeros_cond.view()])
            .expect("same-shaped views always concatenate");

        let dxdt = estimator_step(EstimatorStep {
            x: &x_in,
            mask: &mask_in,
            mu: &mu_in,
            t: &t_in,
            spks: &spks_in,
            cond: &cond_in,
        })?;

        let dxdt_cond = dxdt.slice(s![..b, .., ..]);
        let dxdt_uncond = dxdt.slice(s![b.., .., ..]);
        let dt = r - t;
        let combined =
            dxdt_cond.mapv(|v| dt * (1.0 + cfg_rate) * v) - dxdt_uncond.mapv(|v| dt * cfg_rate * v);
        x += &combined;
    }
    Ok(x)
}

/// Runs one CFG-doubled estimator step against a live ONNX Runtime session.
pub fn run_estimator(session: &mut Session, step: EstimatorStep<'_>) -> ort::Result<Array3<f32>> {
    let outputs = session.run(ort::inputs![
        "x" => Tensor::from_array(step.x.clone())?,
        "mask" => Tensor::from_array(step.mask.clone())?,
        "mu" => Tensor::from_array(step.mu.clone())?,
        "t" => Tensor::from_array(step.t.clone())?,
        "spks" => Tensor::from_array(step.spks.clone())?,
        "cond" => Tensor::from_array(step.cond.clone())?,
    ])?;
    let dxdt = outputs["dxdt"].try_extract_array::<f32>()?;
    Ok(dxdt
        .into_dimensionality::<Ix3>()
        .expect("dxdt is always rank-3")
        .to_owned())
}

/// Runs the (Milestone 2) HiFiGAN vocoder session on a mel-spectrogram.
pub fn mel_to_waveform(session: &mut Session, mel: &Array3<f32>) -> ort::Result<Array2<f32>> {
    let outputs = session.run(ort::inputs!["speech_feat" => Tensor::from_array(mel.clone())?])?;
    let waveform = outputs["waveform"].try_extract_array::<f32>()?;
    Ok(waveform
        .into_dimensionality::<Ix2>()
        .expect("waveform is always rank-2")
        .to_owned())
}

/// Full S3Gen decode chain: CFG-doubled Euler ODE loop (driving `estimator_session`)
/// producing a mel-spectrogram, then the HiFiGAN vocoder (`hifigan_session`)
/// producing the waveform.
#[allow(clippy::too_many_arguments)]
pub fn generate_waveform(
    estimator_session: &mut Session,
    hifigan_session: &mut Session,
    x0: Array3<f32>,
    mu: &Array3<f32>,
    mask: &Array3<f32>,
    spks: &Array2<f32>,
    cond: &Array3<f32>,
) -> ort::Result<Array2<f32>> {
    let t_span = cosine_t_span(N_TIMESTEPS);
    let mel = solve_euler(
        x0,
        mu,
        mask,
        spks,
        cond,
        &t_span,
        INFERENCE_CFG_RATE,
        |step| run_estimator(estimator_session, step),
    )?;
    mel_to_waveform(hifigan_session, &mel)
}

/// Fixed-length buckets the flow encoder was exported at (`export_s3gen_flow_encoder.py`,
/// ADR-0009). Real token counts (`prompt_token_len + generated_token_len`) range from
/// ~1 to ~1150; texts that would need more than the largest bucket are Milestone 6's
/// problem to handle (truncate), not this module's.
pub const TOKEN_BUCKETS: [usize; 6] = [200, 400, 600, 800, 1000, 1200];

/// Picks the smallest bucket `>= token_len`, or `None` if `token_len` exceeds every
/// bucket (caller must truncate the generated-token count and retry).
pub fn select_bucket(token_len: usize) -> Option<usize> {
    TOKEN_BUCKETS.iter().copied().find(|&b| b >= token_len)
}

/// Right-pads `tokens` (the host-assembled `concat(prompt_token, generated_token)`,
/// matching `flow.py`'s own `torch.concat` one level above `input_embedding`) to
/// `bucket` with `0` (any padding id works -- `make_pad_mask` zeroes masked positions
/// before the encoder ever sees them, see ADR-0009). Panics if `tokens.len() >
/// bucket`; callers must pick a bucket via [`select_bucket`] first.
pub fn pad_tokens(tokens: &[i64], bucket: usize) -> Array2<i64> {
    assert!(
        tokens.len() <= bucket,
        "tokens.len()={} exceeds bucket={bucket}",
        tokens.len()
    );
    let mut out = Array2::<i64>::zeros((1, bucket));
    for (i, &tok) in tokens.iter().enumerate() {
        out[[0, i]] = tok;
    }
    out
}

/// Slices the flow encoder's bucket-padded `mu` output down to the real valid
/// prefix (`2 * token_len` frames, `token_mel_ratio=2`). Per ADR-0009's
/// `check_s3gen_flow_encoder` padding-invariance property, only this prefix is
/// guaranteed to match the eager (non-bucketed) reference -- the padding-region
/// output is not meaningful and must be discarded, not merely ignored.
pub fn slice_valid_prefix(mu_padded: &Array3<f32>, token_len: usize) -> Array3<f32> {
    let valid_frames = 2 * token_len;
    mu_padded.slice(s![.., .., ..valid_frames]).to_owned()
}

/// Assembles the `cond` tensor `flow.py::CausalMaskedDiffWithXvec.inference` builds:
/// a zero tensor spanning the full (prompt + generated) mel length, with the
/// reference audio's own real mel (`prompt_feat`, shape `(1, mel_len1, mel_channels)`,
/// channel-last like S3Gen's mel extractor emits it) copied into the first
/// `mel_len1` frames. `total_mel_len` is `mu`'s (post-[`slice_valid_prefix`]) frame
/// count, i.e. `2 * token_len`.
pub fn build_cond(prompt_feat: &Array3<f32>, total_mel_len: usize) -> Array3<f32> {
    let mel_len1 = prompt_feat.shape()[1];
    let mel_channels = prompt_feat.shape()[2];
    assert!(
        mel_len1 <= total_mel_len,
        "prompt_feat's {mel_len1} frames exceed total_mel_len={total_mel_len}"
    );
    let mut cond = Array3::<f32>::zeros((1, mel_channels, total_mel_len));
    let prompt_feat_ct = prompt_feat.view().permuted_axes([0, 2, 1]); // (1, mel_len1, C) -> (1, C, mel_len1)
    cond.slice_mut(s![.., .., ..mel_len1])
        .assign(&prompt_feat_ct);
    cond
}

/// Standard-normal noise the same shape as `mu`, matching `CausalConditionalCFM`'s
/// `z = torch.randn_like(mu)` (temperature fixed at `1.0` -- never overridden by
/// `chatterbox/tts.py::generate`'s call chain, so not a CLI-exposed knob).
pub fn sample_noise(shape: (usize, usize, usize), rng: &mut impl Rng) -> Array3<f32> {
    Array3::from_shape_fn(shape, |_| rng.sample(StandardNormal))
}

/// `flow.py::CausalMaskedDiffWithXvec.inference`'s speaker-embedding prep:
/// L2-normalize the raw CAMPPlus x-vector (dim=1), then apply the
/// `spk_embed_affine_layer` (`Linear(192, 80)`, hand-rolled here rather than its own
/// ONNX session -- ADR-0009). `weight` is `(80, 192)` (PyTorch `nn.Linear` layout),
/// `bias` is `(80,)`.
pub fn embed_speaker(
    raw_embedding: &Array2<f32>,
    weight: &Array2<f32>,
    bias: &Array1<f32>,
) -> Array2<f32> {
    let norm = (raw_embedding
        .mapv(|v| v * v)
        .sum_axis(Axis(1))
        .mapv(f32::sqrt))
    .insert_axis(Axis(1));
    let normalized = raw_embedding / &norm;
    normalized.dot(&weight.t()) + bias
}

/// Runs the flow encoder for one bucket (`export_s3gen_flow_encoder.py`,
/// `models/s3gen_flow_encoder_{bucket}.onnx`). `token` must already be padded to the
/// session's bucket size (see [`pad_tokens`]); `token_len` is the true, unpadded
/// length.
pub fn run_flow_encoder(
    session: &mut Session,
    token: &Array2<i64>,
    token_len: i64,
) -> ort::Result<Array3<f32>> {
    let token_len_arr = Array1::from_elem(1, token_len);
    let outputs = session.run(ort::inputs![
        "token" => Tensor::from_array(token.clone())?,
        "token_len" => Tensor::from_array(token_len_arr)?,
    ])?;
    let mu = outputs["mu"].try_extract_array::<f32>()?;
    Ok(mu
        .into_dimensionality::<Ix3>()
        .expect("mu is always rank-3")
        .to_owned())
}

/// Full token -> waveform chain: flow-encoder bucket call, valid-prefix slice,
/// `cond` assembly, noise init, the Milestone-3 Euler loop (via `solve_euler`), the
/// mandatory `feat[:, :, mel_len1:]` slice (`flow.py`'s own output only ever
/// contains the *generated* mel, not the prompt-conditioning prefix), then HiFiGAN.
#[allow(clippy::too_many_arguments)]
pub fn token_to_waveform(
    flow_encoder_session: &mut Session,
    estimator_session: &mut Session,
    hifigan_session: &mut Session,
    token: &[i64],
    prompt_token_len: usize,
    prompt_feat: &Array3<f32>,
    spks: &Array2<f32>,
    rng: &mut impl Rng,
) -> ort::Result<Array2<f32>> {
    let token_len = token.len();
    let bucket =
        select_bucket(token_len).expect("caller must truncate token_len to fit TOKEN_BUCKETS");
    let padded = pad_tokens(token, bucket);
    let mu_padded = run_flow_encoder(flow_encoder_session, &padded, token_len as i64)?;
    let mu = slice_valid_prefix(&mu_padded, token_len);

    let total_mel_len = mu.shape()[2];
    let mel_len1 = 2 * prompt_token_len;
    let cond = build_cond(prompt_feat, total_mel_len);
    let mask = Array3::<f32>::ones((1, 1, total_mel_len));
    let mu_shape = (mu.shape()[0], mu.shape()[1], mu.shape()[2]);
    let x0 = sample_noise(mu_shape, rng);

    let mel = generate_waveform_mel(estimator_session, x0, &mu, &mask, spks, &cond)?;
    let generated_mel = mel.slice(s![.., .., mel_len1..]).to_owned();
    mel_to_waveform(hifigan_session, &generated_mel)
}

/// The Euler-loop half of `generate_waveform`, split out so [`token_to_waveform`]
/// can apply the `mel_len1:` slice before HiFiGAN (`generate_waveform` itself stays
/// as the Milestone-3 entry point other callers/tests already rely on).
fn generate_waveform_mel(
    estimator_session: &mut Session,
    x0: Array3<f32>,
    mu: &Array3<f32>,
    mask: &Array3<f32>,
    spks: &Array2<f32>,
    cond: &Array3<f32>,
) -> ort::Result<Array3<f32>> {
    let t_span = cosine_t_span(N_TIMESTEPS);
    solve_euler(
        x0,
        mu,
        mask,
        spks,
        cond,
        &t_span,
        INFERENCE_CFG_RATE,
        |step| run_estimator(estimator_session, step),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use rand::SeedableRng;

    #[test]
    fn cosine_t_span_has_n_plus_one_points_from_zero_to_one() {
        let span = cosine_t_span(10);
        assert_eq!(span.len(), 11);
        assert!((span[0] - 0.0).abs() < 1e-6);
        assert!((span[10] - 1.0).abs() < 1e-6);
        // Cosine scheduling front-loads small steps: strictly increasing.
        for pair in span.windows(2) {
            assert!(pair[1] > pair[0]);
        }
    }

    #[test]
    fn cosine_t_span_matches_hand_computed_values() {
        let span = cosine_t_span(2);
        let expected = [0.0_f32, 1.0 - (0.25 * core::f32::consts::PI).cos(), 1.0];
        for (got, want) in span.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "got={got} want={want}");
        }
    }

    /// A synthetic linear "estimator" (`dxdt = mu - x`, no real CFG-varying
    /// behavior — `cond`/`spks` are unused) whose CFG combination and Euler
    /// update can be checked against a hand-computed closed form, without any
    /// ONNX Runtime dependency.
    fn linear_step(step: EstimatorStep<'_>) -> Result<Array3<f32>, Infallible> {
        Ok(step.mu - step.x)
    }

    #[test]
    fn solve_euler_matches_hand_computed_single_step() {
        let mu = Array3::from_elem((1, 1, 1), 2.0_f32);
        let mask = Array3::from_elem((1, 1, 1), 1.0_f32);
        let spks = Array2::from_elem((1, 1), 0.0_f32);
        let cond = Array3::from_elem((1, 1, 1), 0.0_f32);
        let x0 = Array3::from_elem((1, 1, 1), 0.0_f32);
        let t_span = [0.0_f32, 1.0_f32];
        let cfg_rate = 0.0; // isolates the Euler update from the CFG combination

        let x1 = solve_euler(x0, &mu, &mask, &spks, &cond, &t_span, cfg_rate, linear_step).unwrap();
        // dxdt = mu - x0 = 2.0; dt = 1.0; x1 = x0 + dt * dxdt = 2.0
        assert!((x1[[0, 0, 0]] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn solve_euler_applies_cfg_combination() {
        // With `mu` real only in the first (conditional) half of the CFG-doubled
        // batch, `dxdt_uncond = 0 - x`, so a nonzero cfg_rate must pull the result
        // away from the cfg_rate=0 case.
        let mu = Array3::from_elem((1, 1, 1), 2.0_f32);
        let mask = Array3::from_elem((1, 1, 1), 1.0_f32);
        let spks = Array2::from_elem((1, 1), 0.0_f32);
        let cond = Array3::from_elem((1, 1, 1), 0.0_f32);
        let x0 = Array3::from_elem((1, 1, 1), 0.0_f32);
        let t_span = [0.0_f32, 1.0_f32];

        let without_cfg = solve_euler(
            x0.clone(),
            &mu,
            &mask,
            &spks,
            &cond,
            &t_span,
            0.0,
            linear_step,
        )
        .unwrap();
        let with_cfg =
            solve_euler(x0, &mu, &mask, &spks, &cond, &t_span, 0.7, linear_step).unwrap();
        assert_ne!(without_cfg[[0, 0, 0]], with_cfg[[0, 0, 0]]);
        // dxdt_cond = mu - x0 = 2.0; dxdt_uncond = 0 - x0 = 0.0 (x_in's second half
        // is also x0, mu_in's second half is zero) => combined = (1+0.7)*2.0 - 0.7*0.0 = 3.4
        assert!((with_cfg[[0, 0, 0]] - 3.4).abs() < 1e-6);
    }

    #[test]
    fn solve_euler_propagates_estimator_errors() {
        let mu = Array3::from_elem((1, 1, 1), 0.0_f32);
        let mask = Array3::from_elem((1, 1, 1), 1.0_f32);
        let spks = Array2::from_elem((1, 1), 0.0_f32);
        let cond = Array3::from_elem((1, 1, 1), 0.0_f32);
        let x0 = Array3::from_elem((1, 1, 1), 0.0_f32);
        let t_span = [0.0_f32, 1.0_f32];

        let result: Result<Array3<f32>, &'static str> =
            solve_euler(x0, &mu, &mask, &spks, &cond, &t_span, 0.7, |_step| {
                Err("estimator failed")
            });
        assert_eq!(result, Err("estimator failed"));
    }

    #[test]
    fn select_bucket_picks_smallest_bucket_at_or_above_len() {
        assert_eq!(select_bucket(1), Some(200));
        assert_eq!(select_bucket(200), Some(200));
        assert_eq!(select_bucket(201), Some(400));
        assert_eq!(select_bucket(1200), Some(1200));
    }

    #[test]
    fn select_bucket_returns_none_beyond_largest_bucket() {
        assert_eq!(select_bucket(1201), None);
    }

    #[test]
    fn pad_tokens_right_pads_with_zero() {
        let out = pad_tokens(&[1, 2, 3], 5);
        assert_eq!(out.shape(), &[1, 5]);
        assert_eq!(out.row(0).to_vec(), vec![1, 2, 3, 0, 0]);
    }

    #[test]
    #[should_panic(expected = "exceeds bucket")]
    fn pad_tokens_panics_if_tokens_longer_than_bucket() {
        pad_tokens(&[1, 2, 3], 2);
    }

    #[test]
    fn slice_valid_prefix_keeps_only_2x_token_len_frames() {
        let mut mu = Array3::<f32>::zeros((1, 2, 10));
        for t in 0..10 {
            mu[[0, 0, t]] = t as f32;
        }
        let sliced = slice_valid_prefix(&mu, 3);
        assert_eq!(sliced.shape(), &[1, 2, 6]);
        assert_eq!(
            sliced.slice(s![0, 0, ..]).to_vec(),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn build_cond_copies_prompt_feat_into_prefix_and_zeros_the_rest() {
        // prompt_feat: (1, mel_len1=2, channels=2), channel-last like S3Gen emits it.
        let prompt_feat = Array3::from_shape_vec((1, 2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let cond = build_cond(&prompt_feat, 5);
        assert_eq!(cond.shape(), &[1, 2, 5]);
        // channel 0: prompt frames [1.0, 3.0], then zeros
        assert_eq!(
            cond.slice(s![0, 0, ..]).to_vec(),
            vec![1.0, 3.0, 0.0, 0.0, 0.0]
        );
        // channel 1: prompt frames [2.0, 4.0], then zeros
        assert_eq!(
            cond.slice(s![0, 1, ..]).to_vec(),
            vec![2.0, 4.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn sample_noise_produces_requested_shape_and_is_not_constant() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let noise = sample_noise((1, 2, 3), &mut rng);
        assert_eq!(noise.shape(), &[1, 2, 3]);
        let first = noise[[0, 0, 0]];
        assert!(noise.iter().any(|&v| (v - first).abs() > 1e-9));
    }

    #[test]
    fn embed_speaker_l2_normalizes_then_applies_affine() {
        // raw_embedding = [3, 4] -> L2 norm 5 -> normalized [0.6, 0.8]
        let raw = Array2::from_shape_vec((1, 2), vec![3.0_f32, 4.0]).unwrap();
        // Identity-like weight (80->2 collapsed to 2->2 for this test) + zero bias.
        let weight = Array2::from_shape_vec((2, 2), vec![1.0_f32, 0.0, 0.0, 1.0]).unwrap();
        let bias = Array1::from_vec(vec![0.0_f32, 0.0]);
        let out = embed_speaker(&raw, &weight, &bias);
        assert!((out[[0, 0]] - 0.6).abs() < 1e-6);
        assert!((out[[0, 1]] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn embed_speaker_applies_bias() {
        let raw = Array2::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
        let weight = Array2::from_shape_vec((1, 1), vec![1.0_f32]).unwrap();
        let bias = Array1::from_vec(vec![2.5_f32]);
        let out = embed_speaker(&raw, &weight, &bias);
        assert!((out[[0, 0]] - 3.5).abs() < 1e-6);
    }
}
