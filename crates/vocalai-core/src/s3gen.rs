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
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 3).

use core::f32::consts::FRAC_PI_2;

use ndarray::{s, Array1, Array2, Array3, Axis, Ix2, Ix3};
use ort::session::Session;
use ort::value::Tensor;

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

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;

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
}
