//! T3 KV-cache autoregressive decode loop + sampling.
//!
//! Reimplements `T3.inference()`'s manual decode loop (`chatterbox/models/t3/t3.py`)
//! against the exported cond-prefill/decoder ONNX graphs (`export/export_t3.py`) plus
//! two raw embedding-table weight arrays (`t3_speech_emb.npy`/`t3_speech_pos_emb.npy` —
//! the per-step new-token embedding is a plain row lookup + add, not worth an extra
//! ONNX Runtime call per generated token; see docs/decisions/0005). As with
//! `s3gen::solve_euler` (ADR-0004), the loop's own math (CFG combine, repetition
//! penalty, temperature, min-p, top-p, sampling) is generic over the decoder-step
//! call so it can be unit-tested without an ONNX session; `run_decoder`/
//! `run_cond_prefill` provide the real `ort`-backed wiring.
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 4) and
//! docs/decisions/0005-t3-hand-rolled-decoder-export.md.

use std::collections::HashSet;
use std::path::Path;

use ndarray::{s, Array1, Array2, Array3, Array6, Axis};
use ort::session::Session;
use ort::value::Tensor;
use rand::Rng;

/// Fixed by `T3Config` (chatterbox/models/t3/modules/t3_config.py) for the base
/// English-only model.
pub const START_SPEECH_TOKEN: i64 = 6561;
pub const STOP_SPEECH_TOKEN: i64 = 6562;

/// Fixed by `LLAMA_520M_CONFIG_DICT` (chatterbox/models/t3/llama_configs.py).
pub const NUM_LAYERS: usize = 30;
pub const NUM_HEADS: usize = 16;
pub const HEAD_DIM: usize = 64;

/// Stacked KV cache: `(layers, k_or_v=2, batch, heads, seq, head_dim)`. See
/// docs/decisions/0005-t3-hand-rolled-decoder-export.md.
pub type KvCache = Array6<f32>;

/// Sampling hyperparameters, matching `T3.inference()`'s CLI-exposed knobs (plan §3).
#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub cfg_weight: f32,
}

pub fn empty_kv_cache(batch: usize) -> KvCache {
    Array6::zeros((NUM_LAYERS, 2, batch, NUM_HEADS, 0, HEAD_DIM))
}

/// Loads a raw embedding-table weight array dumped by `export/export_t3.py`
/// (`t3_speech_emb.npy` / `t3_speech_pos_emb.npy`) — see `embed_speech_token` and
/// docs/decisions/0005-t3-hand-rolled-decoder-export.md for why this bypasses ONNX.
pub fn load_embedding_table(path: &Path) -> Result<Array2<f32>, ndarray_npy::ReadNpyError> {
    ndarray_npy::read_npy(path)
}

/// `logits = cond + cfg_weight * (cond - uncond)` — `T3.inference()`'s CFG combine
/// (chatterbox/models/t3/t3.py). Always applied (see design decision in the plan's
/// approved Milestone-4 write-up): at `cfg_weight == 0.0` this reduces to `cond`.
pub fn combine_cfg_logits(
    cond: &Array1<f32>,
    uncond: &Array1<f32>,
    cfg_weight: f32,
) -> Array1<f32> {
    cond + &((cond - uncond).mapv(|v| v * cfg_weight))
}

/// `RepetitionPenaltyLogitsProcessor` (transformers `generation/logits_process.py`):
/// for each token id that has appeared in `generated_ids` (each id penalized once,
/// from its original logit value — not compounded across repeated occurrences),
/// `logit = logit * penalty` if `logit < 0` else `logit / penalty`.
pub fn apply_repetition_penalty(logits: &mut Array1<f32>, generated_ids: &[i64], penalty: f32) {
    let mut seen = HashSet::new();
    for &tok in generated_ids {
        if !seen.insert(tok) {
            continue;
        }
        let idx = tok as usize;
        let v = logits[idx];
        logits[idx] = if v < 0.0 { v * penalty } else { v / penalty };
    }
}

/// `TemperatureLogitsWarper`: `logits /= temperature` (skipped at `temperature == 1.0`,
/// matching `T3.inference()`'s explicit `if temperature != 1.0` guard).
pub fn apply_temperature(logits: &mut Array1<f32>, temperature: f32) {
    if temperature != 1.0 {
        logits.mapv_inplace(|v| v / temperature);
    }
}

/// Numerically-stable softmax.
pub fn softmax(logits: &Array1<f32>) -> Array1<f32> {
    let max = logits.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exps = logits.mapv(|v| (v - max).exp());
    let sum: f32 = exps.sum();
    exps.mapv(|v| v / sum)
}

/// `MinPLogitsWarper`: tokens whose probability is below `min_p * max_prob` are
/// masked to `-inf`, except the single highest-probability token (`min_tokens_to_keep
/// = 1` in the reference).
pub fn apply_min_p(logits: &mut Array1<f32>, min_p: f32) {
    let probs = softmax(logits);
    let (top_idx, top_prob) =
        probs
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &p)| {
                if p > acc.1 {
                    (i, p)
                } else {
                    acc
                }
            });
    let threshold = min_p * top_prob;
    for (i, &p) in probs.iter().enumerate() {
        if p < threshold && i != top_idx {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// `TopPLogitsWarper`: ascending-sort by logit, mask tokens whose cumulative
/// probability mass (from the low end) is still `<= 1 - top_p`, except the single
/// highest-probability token (`min_tokens_to_keep = 1`).
pub fn apply_top_p(logits: &mut Array1<f32>, top_p: f32) {
    let n = logits.len();
    if n == 0 {
        return;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| logits[a].partial_cmp(&logits[b]).unwrap());

    let sorted_max = logits[order[n - 1]];
    let exps: Vec<f32> = order
        .iter()
        .map(|&i| (logits[i] - sorted_max).exp())
        .collect();
    let sum: f32 = exps.iter().sum();

    let mut cum = 0.0f32;
    for (pos, &i) in order.iter().enumerate() {
        cum += exps[pos] / sum;
        let keep_last = pos == n - 1;
        if !keep_last && cum <= 1.0 - top_p {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// One step's full logits-processing chain, matching `T3.inference()`'s order:
/// CFG combine -> repetition penalty -> temperature -> min-p -> top-p.
#[allow(clippy::too_many_arguments)]
pub fn process_step_logits(
    cond_logits: &Array1<f32>,
    uncond_logits: &Array1<f32>,
    generated_ids: &[i64],
    config: &SamplingConfig,
) -> Array1<f32> {
    let mut logits = combine_cfg_logits(cond_logits, uncond_logits, config.cfg_weight);
    apply_repetition_penalty(&mut logits, generated_ids, config.repetition_penalty);
    apply_temperature(&mut logits, config.temperature);
    apply_min_p(&mut logits, config.min_p);
    apply_top_p(&mut logits, config.top_p);
    logits
}

/// Greedy (argmax) token selection — deterministic, no RNG. Used for parity
/// validation against the PyTorch reference (see `export/parity_check.py::check_t3`);
/// stochastic sampling can't be compared cross-language since PyTorch's and Rust's
/// RNGs are unrelated.
pub fn argmax_token(logits: &Array1<f32>) -> i64 {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
            if v > acc.1 {
                (i, v)
            } else {
                acc
            }
        })
        .0 as i64
}

/// Multinomial sample from `logits` (softmax'd internally), matching
/// `torch.multinomial(softmax(logits), 1)`.
pub fn sample_token(logits: &Array1<f32>, rng: &mut impl Rng) -> i64 {
    let probs = softmax(logits);
    let u: f32 = rng.gen();
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i as i64;
        }
    }
    (probs.len() - 1) as i64
}

/// Runs the KV-cache decode loop, generic over the decoder-step call (mirrors
/// `s3gen::solve_euler`'s estimator-call genericity, ADR-0004) and the next-token
/// selection strategy, so the logits-processing math can be unit-tested with a
/// synthetic decoder and both greedy and RNG-based tests without any `ort` session.
///
/// `cond_prefill_embeds` is `t3_cond_prefill.onnx`'s output (batch=2, len0, dim).
/// `embed_next_token(token_id, position)` embeds one new speech token (a plain
/// weight-table lookup, see `embed_speech_token` below) for position `position` in
/// the `speech_pos_emb` table. Returns the generated speech-token ids **including**
/// the EOS token if generation stopped that way — matching `T3.inference()`'s
/// `predicted` list, which appends `next_token` before checking for EOS. Filtering
/// out `>= start_speech_token` (which drops both BOS and EOS) happens one level up,
/// in `tts.py`'s `generate()` (`speech_tokens[speech_tokens < 6561]`) — Milestone 6
/// scope, not this loop.
pub fn generate_speech_tokens<E>(
    cond_prefill_embeds: Array3<f32>,
    config: &SamplingConfig,
    mut decoder_step: impl FnMut(&Array3<f32>, &KvCache) -> Result<(Array3<f32>, KvCache), E>,
    mut embed_next_token: impl FnMut(i64, usize) -> Array3<f32>,
    mut select_token: impl FnMut(&Array1<f32>) -> i64,
) -> Result<Vec<i64>, E> {
    let batch = cond_prefill_embeds.shape()[0];
    let mut past_kv = empty_kv_cache(batch);
    let mut generated_ids: Vec<i64> = vec![START_SPEECH_TOKEN];
    let mut predicted = Vec::new();

    let (logits, present_kv) = decoder_step(&cond_prefill_embeds, &past_kv)?;
    past_kv = present_kv;
    let mut last_logits = last_step_logits(&logits);

    for step in 0..config.max_new_tokens {
        let cond = last_logits.index_axis(Axis(0), 0).to_owned();
        let uncond = last_logits.index_axis(Axis(0), 1).to_owned();
        let processed = process_step_logits(&cond, &uncond, &generated_ids, config);
        let next = select_token(&processed);

        predicted.push(next);
        generated_ids.push(next);
        if next == STOP_SPEECH_TOKEN {
            break;
        }

        let next_embed = embed_next_token(next, step + 1);
        let (logits, present_kv) = decoder_step(&next_embed, &past_kv)?;
        past_kv = present_kv;
        last_logits = last_step_logits(&logits);
    }

    Ok(predicted)
}

fn last_step_logits(logits: &Array3<f32>) -> Array2<f32> {
    let last = logits.shape()[1] - 1;
    logits.index_axis(Axis(1), last).to_owned()
}

/// Embeds one new speech token: `speech_emb_table[token_id] + speech_pos_emb_table[position]`,
/// CFG-duplicated to batch 2 — a plain row lookup, matching `T3.inference()`'s
/// per-step `speech_emb(next_token) + speech_pos_emb.get_fixed_embedding(i + 1)`
/// (see docs/decisions/0005 for why this bypasses ONNX).
pub fn embed_speech_token(
    speech_emb_table: &Array2<f32>,
    speech_pos_emb_table: &Array2<f32>,
    token_id: i64,
    position: usize,
) -> Array3<f32> {
    let dim = speech_emb_table.shape()[1];
    let row = &speech_emb_table.slice(s![token_id as usize, ..])
        + &speech_pos_emb_table.slice(s![position, ..]);
    let mut out = Array3::zeros((2, 1, dim));
    out.index_axis_mut(Axis(0), 0)
        .index_axis_mut(Axis(0), 0)
        .assign(&row);
    out.index_axis_mut(Axis(0), 1)
        .index_axis_mut(Axis(0), 0)
        .assign(&row);
    out
}

/// Runs one decoder-with-past step against a live ONNX Runtime session
/// (`t3_decoder.onnx`, see `export/export_t3.py`).
pub fn run_decoder(
    session: &mut Session,
    inputs_embeds: &Array3<f32>,
    past_kv: &KvCache,
) -> ort::Result<(Array3<f32>, KvCache)> {
    let outputs = session.run(ort::inputs![
        "inputs_embeds" => Tensor::from_array(inputs_embeds.clone())?,
        "past_kv" => Tensor::from_array(past_kv.clone())?,
    ])?;
    let logits = outputs["logits"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix3>()
        .expect("logits is always rank-3")
        .to_owned();
    let present_kv = outputs["present_kv"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix6>()
        .expect("present_kv is always rank-6")
        .to_owned();
    Ok((logits, present_kv))
}

/// Runs `t3_cond_prefill.onnx` to build the initial `inputs_embeds` (see
/// `export/export_t3.py::T3CondPrefillExport`).
pub fn run_cond_prefill(
    session: &mut Session,
    speaker_emb: &Array2<f32>,
    cond_prompt_speech_tokens: &Array2<i64>,
    emotion_adv: &Array3<f32>,
    text_tokens: &Array2<i64>,
    cfg_uncond_mask: &Array3<f32>,
) -> ort::Result<Array3<f32>> {
    let outputs = session.run(ort::inputs![
        "speaker_emb" => Tensor::from_array(speaker_emb.clone())?,
        "cond_prompt_speech_tokens" => Tensor::from_array(cond_prompt_speech_tokens.clone())?,
        "emotion_adv" => Tensor::from_array(emotion_adv.clone())?,
        "text_tokens" => Tensor::from_array(text_tokens.clone())?,
        "cfg_uncond_mask" => Tensor::from_array(cfg_uncond_mask.clone())?,
    ])?;
    let inputs_embeds = outputs["inputs_embeds"]
        .try_extract_array::<f32>()?
        .into_dimensionality::<ndarray::Ix3>()
        .expect("inputs_embeds is always rank-3")
        .to_owned();
    Ok(inputs_embeds)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::convert::Infallible;
    use rand::SeedableRng;

    #[test]
    fn combine_cfg_logits_reduces_to_cond_at_zero_weight() {
        let cond = Array1::from(vec![1.0_f32, 2.0, 3.0]);
        let uncond = Array1::from(vec![4.0_f32, 0.0, -1.0]);
        let out = combine_cfg_logits(&cond, &uncond, 0.0);
        for (a, b) in out.iter().zip(cond.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn combine_cfg_logits_matches_hand_computed_value() {
        let cond = Array1::from(vec![2.0_f32]);
        let uncond = Array1::from(vec![0.0_f32]);
        let out = combine_cfg_logits(&cond, &uncond, 0.7);
        // 2.0 + 0.7 * (2.0 - 0.0) = 3.4
        assert!((out[0] - 3.4).abs() < 1e-6);
    }

    #[test]
    fn repetition_penalty_divides_positive_logits_seen_before() {
        let mut logits = Array1::from(vec![4.0_f32, -4.0, 1.0]);
        apply_repetition_penalty(&mut logits, &[0, 1], 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6); // 4/2
        assert!((logits[1] - (-8.0)).abs() < 1e-6); // -4*2
        assert!((logits[2] - 1.0).abs() < 1e-6); // untouched
    }

    #[test]
    fn repetition_penalty_does_not_compound_duplicate_ids() {
        let mut logits = Array1::from(vec![4.0_f32]);
        apply_repetition_penalty(&mut logits, &[0, 0, 0], 2.0);
        assert!((logits[0] - 2.0).abs() < 1e-6); // still 4/2, not 4/2/2/2
    }

    #[test]
    fn temperature_scales_logits_and_skips_at_one() {
        let mut logits = Array1::from(vec![2.0_f32, 4.0]);
        apply_temperature(&mut logits, 2.0);
        assert!((logits[0] - 1.0).abs() < 1e-6);
        assert!((logits[1] - 2.0).abs() < 1e-6);

        let mut unchanged = Array1::from(vec![2.0_f32, 4.0]);
        apply_temperature(&mut unchanged, 1.0);
        assert!((unchanged[0] - 2.0).abs() < 1e-6);
        assert!((unchanged[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one_and_is_shift_invariant() {
        let a = softmax(&Array1::from(vec![1.0_f32, 2.0, 3.0]));
        let b = softmax(&Array1::from(vec![101.0_f32, 102.0, 103.0]));
        assert!((a.sum() - 1.0).abs() < 1e-6);
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn min_p_masks_low_probability_tokens_but_keeps_the_top_one() {
        // logits far apart -> top token dominates softmax; min_p=0.5 should mask
        // everything else except the single highest (min_tokens_to_keep=1).
        let mut logits = Array1::from(vec![10.0_f32, 0.0, -10.0]);
        apply_min_p(&mut logits, 0.5);
        assert!(logits[0].is_finite());
        assert!(logits[1].is_infinite() && logits[1] < 0.0);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
    }

    #[test]
    fn min_p_zero_is_a_no_op() {
        let mut logits = Array1::from(vec![10.0_f32, 0.0, -10.0]);
        apply_min_p(&mut logits, 0.0);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn top_p_keeps_only_top_mass_but_never_drops_the_single_best() {
        let mut logits = Array1::from(vec![10.0_f32, 0.0, -10.0]);
        apply_top_p(&mut logits, 0.01); // very small kept mass
        assert!(logits[0].is_finite());
        assert!(logits[1].is_infinite());
        assert!(logits[2].is_infinite());
    }

    #[test]
    fn top_p_one_is_a_no_op() {
        let mut logits = Array1::from(vec![10.0_f32, 0.0, -10.0]);
        apply_top_p(&mut logits, 1.0);
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn argmax_token_picks_highest_logit() {
        let logits = Array1::from(vec![1.0_f32, 9.0, 3.0]);
        assert_eq!(argmax_token(&logits), 1);
    }

    #[test]
    fn sample_token_is_deterministic_for_a_one_hot_distribution() {
        let logits = Array1::from(vec![f32::NEG_INFINITY, 100.0, f32::NEG_INFINITY]);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for _ in 0..8 {
            assert_eq!(sample_token(&logits, &mut rng), 1);
        }
    }

    #[test]
    fn embed_speech_token_looks_up_and_adds_rows_and_duplicates_to_batch_two() {
        let speech_emb =
            Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0]).unwrap();
        let pos_emb = Array2::from_shape_vec((2, 3), vec![0.1, 0.1, 0.1, 0.2, 0.2, 0.2]).unwrap();
        let out = embed_speech_token(&speech_emb, &pos_emb, 1, 0);
        assert_eq!(out.shape(), &[2, 1, 3]);
        for b in 0..2 {
            let row = out.index_axis(Axis(0), b).index_axis(Axis(0), 0).to_owned();
            assert!((row[0] - 10.1).abs() < 1e-6);
            assert!((row[1] - 20.1).abs() < 1e-6);
            assert!((row[2] - 30.1).abs() < 1e-6);
        }
    }

    #[test]
    fn load_embedding_table_round_trips_through_npy() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vocalai-t3-test-{}.npy", std::process::id()));
        let table = Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        ndarray_npy::write_npy(&path, &table).unwrap();

        let loaded = load_embedding_table(&path).unwrap();
        std::fs::remove_file(&path).unwrap();

        assert_eq!(loaded.shape(), table.shape());
        for (a, b) in loaded.iter().zip(table.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn generate_speech_tokens_stops_at_eos_and_includes_it() {
        // Synthetic decoder: cond logits always pick STOP after one real token.
        let dim = 2usize;
        let cond_prefill_embeds = Array3::zeros((2, 1, dim));
        let vocab = (STOP_SPEECH_TOKEN + 1) as usize;
        let mut call_count = 0usize;

        let decoder_step = |_embeds: &Array3<f32>,
                            past_kv: &KvCache|
         -> Result<(Array3<f32>, KvCache), Infallible> {
            call_count += 1;
            let mut logits = Array3::from_elem((2, 1, vocab), -1.0_f32);
            let next_token = if call_count == 1 {
                5_i64
            } else {
                STOP_SPEECH_TOKEN
            };
            logits[[0, 0, next_token as usize]] = 100.0;
            logits[[1, 0, next_token as usize]] = 100.0;
            let past_len = past_kv.shape()[4];
            let mut present = Array6::zeros((NUM_LAYERS, 2, 2, NUM_HEADS, past_len + 1, HEAD_DIM));
            present
                .slice_mut(s![.., .., .., .., ..past_len, ..])
                .assign(past_kv);
            Ok((logits, present))
        };

        let config = SamplingConfig {
            max_new_tokens: 10,
            temperature: 1.0,
            top_p: 1.0,
            min_p: 0.0,
            repetition_penalty: 1.0,
            cfg_weight: 0.0,
        };

        let result = generate_speech_tokens(
            cond_prefill_embeds,
            &config,
            decoder_step,
            |_tok, _pos| Array3::zeros((2, 1, dim)),
            argmax_token,
        )
        .unwrap();

        assert_eq!(result, vec![5, STOP_SPEECH_TOKEN]);
    }
}
