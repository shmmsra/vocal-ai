"""Check numerical parity between exported ONNX graphs and the PyTorch reference.

Usage:
    python parity_check.py [--component hifigan|ve|s3tokenizer|s3gen] [--atol 1e-4] [--rtol 1e-3]

Exits non-zero if any checked component fails. Per docs/agents/CONVENTIONS.md §3: no
exported component may be wired into vocalai-core until this passes for it.
"""
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import torch
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    RepetitionPenaltyLogitsProcessor,
    TopPLogitsWarper,
)

import export_hifigan
import export_s3gen
import export_s3tokenizer
import export_t3
import export_ve
from _common import allclose_report, load_s3gen, load_t3, models_dir

DEFAULT_ATOL = 1e-4
DEFAULT_RTOL = 1e-3

S3GEN_N_TIMESTEPS = 10  # fixed by chatterbox/tts.py's s3gen.inference() call (no override, non-meanflow)

# T3 parity fixture / sampling knobs. A short `max_new_tokens` keeps the greedy
# free-running comparison (see check_t3) fast and low-risk: greedy argmax on two
# independently-computed (PyTorch vs ONNX) logit tensors can in principle diverge
# on a near-exact tie, and risk compounds with every extra step.
T3_MAX_NEW_TOKENS = 6
T3_TEMPERATURE = 0.8
T3_TOP_P = 1.0
T3_MIN_P = 0.05
T3_REPETITION_PENALTY = 1.2
T3_CFG_WEIGHT = 0.5


@dataclass
class ParityResult:
    component: str
    passed: bool
    max_abs_diff: float


def _run_onnx(onnx_path, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return session.run(None, feeds)


def check_hifigan(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    wrapper = export_hifigan.build_wrapper()
    onnx_path = models_dir() / "hifigan.onnx"
    if not onnx_path.exists():
        export_hifigan.export(onnx_path)

    speech_feat = torch.randn(1, export_hifigan.MEL_CHANNELS, 50)
    with torch.no_grad():
        torch_out = wrapper(speech_feat).numpy()

    (onnx_out,) = _run_onnx(onnx_path, {"speech_feat": speech_feat.numpy()})
    passed, diff = allclose_report(torch_out, onnx_out, atol, rtol)
    return ParityResult("hifigan", passed, diff)


def check_ve(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    ve = export_ve.build_module()
    onnx_path = models_dir() / "ve.onnx"
    if not onnx_path.exists():
        export_ve.export(onnx_path)

    mels = torch.rand(1, ve.hp.ve_partial_frames, ve.hp.num_mels)
    with torch.no_grad():
        torch_out = ve(mels).numpy()

    (onnx_out,) = _run_onnx(onnx_path, {"mels": mels.numpy()})
    passed, diff = allclose_report(torch_out, onnx_out, atol, rtol)
    return ParityResult("ve", passed, diff)


def check_s3tokenizer(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    wrapper = export_s3tokenizer.build_wrapper()
    onnx_path = models_dir() / "s3tokenizer.onnx"
    if not onnx_path.exists():
        export_s3tokenizer.export(onnx_path)

    mel = torch.randn(1, export_s3tokenizer.N_MELS, export_s3tokenizer.EXAMPLE_FRAMES)
    mel_len = torch.tensor([export_s3tokenizer.EXAMPLE_FRAMES], dtype=torch.long)
    with torch.no_grad():
        torch_code, _ = wrapper(mel, mel_len)
    torch_code = torch_code.numpy()

    onnx_code, _ = _run_onnx(onnx_path, {"mel": mel.numpy(), "mel_len": mel_len.numpy()})
    # Tokens are discrete integer codes: parity means an exact match, not a tolerance band.
    passed = bool(np.array_equal(torch_code, onnx_code))
    diff = allclose_report(torch_code.astype(np.float64), onnx_code.astype(np.float64), 0, 0)[1]
    return ParityResult("s3tokenizer", passed, diff)


def _cosine_t_span(n_timesteps: int) -> np.ndarray:
    """Matches ConditionalCFM.solve_euler's cosine `t_scheduler` (the only scheduler
    the base Chatterbox config uses; see chatterbox/models/s3gen/configs.py)."""
    t_span = np.linspace(0.0, 1.0, n_timesteps + 1, dtype=np.float32)
    return (1.0 - np.cos(t_span * 0.5 * np.pi)).astype(np.float32)


def _solve_euler_onnx(
    session: ort.InferenceSession,
    x0: np.ndarray,
    t_span: np.ndarray,
    mu: np.ndarray,
    mask: np.ndarray,
    spks: np.ndarray,
    cond: np.ndarray,
    cfg_rate: float,
) -> np.ndarray:
    """Python-side replica of `ConditionalCFM.solve_euler`'s CFG-doubled Euler loop
    (chatterbox/models/s3gen/flow_matching.py), driving the exported estimator ONNX
    graph instead of the PyTorch module. This is the exact loop `vocalai-core::s3gen`
    reimplements in Rust — see `crates/vocalai-core/src/s3gen.rs`.
    """
    b = mu.shape[0]
    x = x0.copy()
    zeros_mu, zeros_spks, zeros_cond = np.zeros_like(mu), np.zeros_like(spks), np.zeros_like(cond)
    for i in range(len(t_span) - 1):
        t, r = t_span[i : i + 1], t_span[i + 1 : i + 2]
        feeds = {
            "x": np.concatenate([x, x], axis=0),
            "mask": np.concatenate([mask, mask], axis=0),
            "mu": np.concatenate([mu, zeros_mu], axis=0),
            "t": np.concatenate([t, t]),
            "spks": np.concatenate([spks, zeros_spks], axis=0),
            "cond": np.concatenate([cond, zeros_cond], axis=0),
        }
        (dxdt,) = session.run(None, feeds)
        dxdt_cond, dxdt_uncond = dxdt[:b], dxdt[b:]
        dxdt_combined = (1.0 + cfg_rate) * dxdt_cond - cfg_rate * dxdt_uncond
        dt = float((r - t)[0])
        x = x + dt * dxdt_combined
    return x


def check_s3gen(atol: float, rtol: float) -> ParityResult:
    torch.manual_seed(0)
    s3gen = load_s3gen()
    decoder = s3gen.flow.decoder
    cfg_rate = float(decoder.inference_cfg_rate)

    onnx_path = models_dir() / "s3gen_estimator.onnx"
    if not onnx_path.exists():
        export_s3gen.export(onnx_path)
    hifigan_path = models_dir() / "hifigan.onnx"
    if not hifigan_path.exists():
        export_hifigan.export(hifigan_path)

    batch, frames = export_s3gen.EXAMPLE_BATCH, export_s3gen.EXAMPLE_FRAMES
    mu = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)
    mask = torch.ones(batch, 1, frames)
    spks = torch.randn(batch, export_s3gen.SPK_EMB_DIM)
    cond = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)
    x0 = torch.randn(batch, export_s3gen.MEL_CHANNELS, frames)

    t_span = torch.from_numpy(_cosine_t_span(S3GEN_N_TIMESTEPS))
    with torch.no_grad():
        mel_ref = decoder.solve_euler(x0.clone(), t_span, mu, mask, spks, cond, meanflow=False)
        wav_ref = export_hifigan.build_wrapper()(mel_ref).numpy()
    mel_ref = mel_ref.numpy()

    estimator_session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    mel_onnx = _solve_euler_onnx(
        estimator_session,
        x0.numpy(),
        _cosine_t_span(S3GEN_N_TIMESTEPS),
        mu.numpy(),
        mask.numpy(),
        spks.numpy(),
        cond.numpy(),
        cfg_rate,
    )
    (wav_onnx,) = _run_onnx(hifigan_path, {"speech_feat": mel_onnx.astype(np.float32)})

    mel_passed, mel_diff = allclose_report(mel_ref, mel_onnx, atol, rtol)
    wav_passed, wav_diff = allclose_report(wav_ref, wav_onnx, atol, rtol)
    return ParityResult("s3gen", mel_passed and wav_passed, max(mel_diff, wav_diff))


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


def _apply_repetition_penalty_np(logits: np.ndarray, generated_ids: list[int], penalty: float) -> np.ndarray:
    """Numpy replica of `RepetitionPenaltyLogitsProcessor`'s 2D-scores path (see
    `crates/vocalai-core/src/t3.rs::apply_repetition_penalty` for the Rust twin)."""
    logits = logits.copy()
    for tok in set(generated_ids):
        v = logits[tok]
        logits[tok] = v * penalty if v < 0 else v / penalty
    return logits


def _apply_min_p_np(logits: np.ndarray, min_p: float) -> np.ndarray:
    """Numpy replica of `MinPLogitsWarper` (see `t3.rs::apply_min_p`)."""
    probs = _softmax_np(logits)
    top_idx = int(np.argmax(probs))
    threshold = min_p * probs[top_idx]
    out = logits.copy()
    mask = (probs < threshold) & (np.arange(len(logits)) != top_idx)
    out[mask] = -np.inf
    return out


def _apply_top_p_np(logits: np.ndarray, top_p: float) -> np.ndarray:
    """Numpy replica of `TopPLogitsWarper` (see `t3.rs::apply_top_p`)."""
    order = np.argsort(logits)  # ascending
    sorted_logits = logits[order]
    probs = _softmax_np(sorted_logits)
    cumulative = np.cumsum(probs)
    remove = cumulative <= (1.0 - top_p)
    remove[-1] = False  # min_tokens_to_keep = 1
    out = logits.copy()
    out[order[remove]] = -np.inf
    return out


def _process_step_logits_np(
    cond: np.ndarray,
    uncond: np.ndarray,
    generated_ids: list[int],
    cfg_weight: float,
    repetition_penalty: float,
    temperature: float,
    min_p: float,
    top_p: float,
) -> np.ndarray:
    logits = cond + cfg_weight * (cond - uncond)
    logits = _apply_repetition_penalty_np(logits, generated_ids, repetition_penalty)
    if temperature != 1.0:
        logits = logits / temperature
    logits = _apply_min_p_np(logits, min_p)
    logits = _apply_top_p_np(logits, top_p)
    return logits


def _t3_fixture(t3):
    """Fixed-seed synthetic T3 conditioning + text tokens — analogous to
    `check_s3gen`'s random-tensor fixture (no real reference audio/text needed for
    a numerical-parity check; real audio preprocessing is Milestone 6 scope)."""
    torch.manual_seed(0)
    speaker_emb = torch.randn(1, export_t3.SPEAKER_EMBED_SIZE)
    cond_prompt_speech_tokens = torch.randint(0, 6561, (1, export_t3.SPEECH_COND_PROMPT_LEN))
    emotion_adv = torch.full((1, 1, 1), 0.5)

    sot, eot = t3.hp.start_text_token, t3.hp.stop_text_token
    text_tokens = torch.randint(1, 254, (1, export_t3.EXAMPLE_LEN_TEXT))
    text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # CFG-doubled, per tts.py::generate()
    text_tokens = torch.nn.functional.pad(text_tokens, (1, 0), value=sot)
    text_tokens = torch.nn.functional.pad(text_tokens, (0, 1), value=eot)

    from chatterbox.models.t3.modules.cond_enc import T3Cond

    t3_cond = T3Cond(
        speaker_emb=speaker_emb,
        cond_prompt_speech_tokens=cond_prompt_speech_tokens,
        emotion_adv=emotion_adv,
    )
    return t3_cond, text_tokens, speaker_emb, cond_prompt_speech_tokens, emotion_adv


def _greedy_reference_t3(t3, t3_cond, text_tokens, cfg_weight: float) -> tuple[list[int], list[np.ndarray]]:
    """Near-identical copy of `T3.inference()` (chatterbox/models/t3/t3.py), with
    greedy (argmax) token selection in place of `torch.multinomial`. PyTorch's and
    Rust's RNGs are unrelated, so a free-running *stochastic* comparison across
    languages is meaningless; greedy removes randomness from the comparison
    entirely while still exercising the real reference forward pass (real `Cache`,
    real RoPE, real weights) end to end. Returns (predicted_token_ids,
    per_step_processed_logits) — the latter is what's compared against the ONNX
    side for numerical parity.
    """
    from chatterbox.models.t3.inference.t3_hf_backend import T3HuggingfaceBackend

    text_tokens = torch.atleast_2d(text_tokens).to(dtype=torch.long, device=t3.device)
    initial_speech_tokens = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, _ = t3.prepare_input_embeds(
        t3_cond=t3_cond, text_tokens=text_tokens, speech_tokens=initial_speech_tokens, cfg_weight=cfg_weight
    )

    bos_token = torch.tensor([[t3.hp.start_speech_token]], dtype=torch.long, device=t3.device)
    bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)
    bos_embed = torch.cat([bos_embed, bos_embed])
    inputs_embeds = torch.cat([embeds, bos_embed], dim=1)

    patched_model = T3HuggingfaceBackend(
        config=t3.cfg, llama=t3.tfmr, speech_enc=t3.speech_emb, speech_head=t3.speech_head,
        alignment_stream_analyzer=None,
    )

    generated_ids = bos_token.clone()
    predicted: list[int] = []
    per_step_logits: list[np.ndarray] = []

    repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=T3_REPETITION_PENALTY)
    min_p_warper = MinPLogitsWarper(min_p=T3_MIN_P)
    top_p_warper = TopPLogitsWarper(top_p=T3_TOP_P)

    output = patched_model(
        inputs_embeds=inputs_embeds, past_key_values=None, use_cache=True,
        output_attentions=True, output_hidden_states=True, return_dict=True,
    )
    past = output.past_key_values

    for i in range(T3_MAX_NEW_TOKENS):
        logits_step = output.logits[:, -1, :]
        cond, uncond = logits_step[0:1, :], logits_step[1:2, :]
        cfg = torch.as_tensor(cfg_weight, device=cond.device, dtype=cond.dtype)
        logits = cond + cfg * (cond - uncond)

        ids_for_proc = generated_ids[:1, ...]
        logits = repetition_penalty_processor(ids_for_proc, logits)
        if T3_TEMPERATURE != 1.0:
            logits = logits / T3_TEMPERATURE
        logits = min_p_warper(ids_for_proc, logits)
        logits = top_p_warper(ids_for_proc, logits)
        per_step_logits.append(logits.detach().numpy().reshape(-1).copy())

        next_token = logits.argmax(dim=-1, keepdim=True)
        predicted.append(int(next_token.item()))
        generated_ids = torch.cat([generated_ids, next_token], dim=1)
        if next_token.view(-1) == t3.hp.stop_speech_token:
            break

        next_token_embed = t3.speech_emb(next_token) + t3.speech_pos_emb.get_fixed_embedding(i + 1)
        next_token_embed = torch.cat([next_token_embed, next_token_embed])
        output = patched_model(
            inputs_embeds=next_token_embed, past_key_values=past,
            output_attentions=True, output_hidden_states=True, return_dict=True,
        )
        past = output.past_key_values

    return predicted, per_step_logits


def _greedy_onnx_t3(
    cond_prefill_path,
    decoder_path,
    speech_emb_table: np.ndarray,
    speech_pos_emb_table: np.ndarray,
    speaker_emb: np.ndarray,
    cond_prompt_speech_tokens: np.ndarray,
    emotion_adv: np.ndarray,
    text_tokens: np.ndarray,
    cfg_weight: float,
) -> tuple[list[int], list[np.ndarray]]:
    """Free-running greedy decode loop driving the exported ONNX graphs — the exact
    Python-side twin of `crates/vocalai-core/src/t3.rs::generate_speech_tokens`."""
    cond_prefill_session = ort.InferenceSession(str(cond_prefill_path), providers=["CPUExecutionProvider"])
    decoder_session = ort.InferenceSession(str(decoder_path), providers=["CPUExecutionProvider"])

    cfg_uncond_mask = np.array([[[1.0]], [[0.0]]], dtype=np.float32) if cfg_weight > 0.0 else np.ones((2, 1, 1), dtype=np.float32)
    (inputs_embeds,) = cond_prefill_session.run(
        None,
        {
            "speaker_emb": speaker_emb.astype(np.float32),
            "cond_prompt_speech_tokens": cond_prompt_speech_tokens.astype(np.int64),
            "emotion_adv": emotion_adv.astype(np.float32),
            "text_tokens": text_tokens.astype(np.int64),
            "cfg_uncond_mask": cfg_uncond_mask,
        },
    )

    batch = inputs_embeds.shape[0]
    past_kv = np.zeros((export_t3.NUM_LAYERS, 2, batch, export_t3.NUM_HEADS, 0, export_t3.HEAD_DIM), dtype=np.float32)

    start_speech_token, stop_speech_token = 6561, 6562
    generated_ids = [start_speech_token]
    predicted: list[int] = []
    per_step_logits: list[np.ndarray] = []

    logits, past_kv = decoder_session.run(None, {"inputs_embeds": inputs_embeds.astype(np.float32), "past_kv": past_kv})

    for i in range(T3_MAX_NEW_TOKENS):
        last_logits = logits[:, -1, :]
        processed = _process_step_logits_np(
            last_logits[0], last_logits[1], generated_ids, cfg_weight,
            T3_REPETITION_PENALTY, T3_TEMPERATURE, T3_MIN_P, T3_TOP_P,
        )
        per_step_logits.append(processed.copy())

        next_token = int(np.argmax(processed))
        predicted.append(next_token)
        generated_ids.append(next_token)
        if next_token == stop_speech_token:
            break

        row = speech_emb_table[next_token] + speech_pos_emb_table[i + 1]
        next_embed = np.stack([row, row])[:, None, :].astype(np.float32)
        logits, past_kv = decoder_session.run(None, {"inputs_embeds": next_embed, "past_kv": past_kv})

    return predicted, per_step_logits


def check_t3(atol: float, rtol: float) -> ParityResult:
    t3 = load_t3()
    t3_cond, text_tokens, speaker_emb, cond_prompt_speech_tokens, emotion_adv = _t3_fixture(t3)

    cond_prefill_path = models_dir() / "t3_cond_prefill.onnx"
    decoder_path = models_dir() / "t3_decoder.onnx"
    speech_emb_path = models_dir() / "t3_speech_emb.npy"
    speech_pos_emb_path = models_dir() / "t3_speech_pos_emb.npy"
    if not (cond_prefill_path.exists() and decoder_path.exists() and speech_emb_path.exists()):
        export_t3.export(models_dir())

    ref_tokens, ref_logits = _greedy_reference_t3(t3, t3_cond, text_tokens, T3_CFG_WEIGHT)

    # Free the ~2GB PyTorch T3 model (and _common.load_t3's cached singleton)
    # before loading the ONNX Runtime sessions below. A GitHub Actions runner
    # doesn't have enough RAM to hold the live torch model, the in-memory ONNX
    # protobuf built during export, and a loaded onnxruntime session on the same
    # ~1.9GB graph all at once -- this OOM-killed CI before (see
    # docs/decisions/0006-split-ci-into-fast-and-parity-workflows.md).
    speaker_emb_np = speaker_emb.numpy()
    cond_prompt_speech_tokens_np = cond_prompt_speech_tokens.numpy()
    emotion_adv_np = emotion_adv.numpy()
    text_tokens_np = text_tokens.numpy()
    del t3, t3_cond, text_tokens, speaker_emb, cond_prompt_speech_tokens, emotion_adv
    load_t3.cache_clear()
    gc.collect()

    speech_emb_table = np.load(speech_emb_path)
    speech_pos_emb_table = np.load(speech_pos_emb_path)

    onnx_tokens, onnx_logits = _greedy_onnx_t3(
        cond_prefill_path,
        decoder_path,
        speech_emb_table,
        speech_pos_emb_table,
        speaker_emb_np,
        cond_prompt_speech_tokens_np,
        emotion_adv_np,
        text_tokens_np,
        T3_CFG_WEIGHT,
    )

    tokens_match = ref_tokens == onnx_tokens
    max_diff = 0.0
    logits_passed = True
    for ref, onnx in zip(ref_logits, onnx_logits):
        finite_mask = np.isfinite(ref) & np.isfinite(onnx)
        passed, diff = allclose_report(ref[finite_mask], onnx[finite_mask], atol, rtol)
        logits_passed = logits_passed and passed
        max_diff = max(max_diff, diff)
        # -inf entries (top-p/min-p masking) must land on the same token indices.
        if not np.array_equal(np.isfinite(ref), np.isfinite(onnx)):
            logits_passed = False

    return ParityResult("t3", tokens_match and logits_passed, max_diff)


CHECKS = {
    "hifigan": check_hifigan,
    "ve": check_ve,
    "s3tokenizer": check_s3tokenizer,
    "s3gen": check_s3gen,
    "t3": check_t3,
}


def run_checks(components: list[str], atol: float, rtol: float) -> list[ParityResult]:
    return [CHECKS[name](atol, rtol) for name in components]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=sorted(CHECKS), default=None)
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL)
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL)
    args = parser.parse_args()

    components = [args.component] if args.component else sorted(CHECKS)
    results = run_checks(components, args.atol, args.rtol)

    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.component}: max_abs_diff={result.max_abs_diff:.3e}")
        all_passed = all_passed and result.passed

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
