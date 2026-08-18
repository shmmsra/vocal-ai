//! Text tokenizer wrapper: `tokenizer.json` loading plus the exact host-side wrapping
//! `ChatterboxTTS.generate()` does around it (`chatterbox/tts.py`) before text tokens
//! ever reach T3 — `punc_norm`, space -> `[SPACE]` substitution, CFG-doubling, and
//! `start_text_token`/`stop_text_token` padding. None of this is ONNX-exported (it's
//! plain tensor/string prep, like T3's sampling math or S3Gen's Euler loop), so it's
//! reimplemented directly here and unit-tested without a live tokenizer file where
//! possible; [`TextTokenizer::encode_ids`] is the one piece that needs a real
//! `tokenizer.json` (loaded via the `tokenizers` crate).
//!
//! See docs/phase1-onnx-rust-cli-plan.md §4/§7 (Milestone 6).

use ndarray::Array2;

/// Fixed by `T3Config` (chatterbox/models/t3/modules/t3_config.py) for the base
/// English-only model.
pub const START_TEXT_TOKEN: i64 = 255;
pub const STOP_TEXT_TOKEN: i64 = 0;

/// `EnTokenizer`'s space placeholder (chatterbox/models/tokenizers/tokenizer.py).
const SPACE_TOKEN: &str = "[SPACE]";

/// Wraps a loaded `tokenizers::Tokenizer` (`tokenizer.json`), matching `EnTokenizer`.
pub struct TextTokenizer {
    tokenizer: tokenizers::Tokenizer,
}

impl TextTokenizer {
    pub fn from_file(
        path: &std::path::Path,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let tokenizer = tokenizers::Tokenizer::from_file(path)?;
        Ok(Self { tokenizer })
    }

    /// `EnTokenizer.encode`: replace literal spaces with `[SPACE]`, then run the
    /// underlying tokenizer. Returns raw token ids (pre CFG-doubling/sot-eot padding
    /// -- see [`build_text_tokens`]).
    pub fn encode_ids(
        &self,
        text: &str,
    ) -> Result<Vec<i64>, Box<dyn std::error::Error + Send + Sync>> {
        let replaced = text.replace(' ', SPACE_TOKEN);
        let encoding = self
            .tokenizer
            .encode(replaced, true)
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> { e })?;
        Ok(encoding.get_ids().iter().map(|&id| id as i64).collect())
    }
}

/// Punctuation/whitespace cleanup for LLM-ish input text, matching `punc_norm`
/// (chatterbox/tts.py) exactly: capitalize the first letter, collapse whitespace
/// runs, replace a fixed set of "uncommon" punctuation, and ensure the text ends
/// with sentence-ending punctuation.
pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }

    let mut chars: Vec<char> = text.chars().collect();
    if chars[0].is_lowercase() {
        chars[0] = chars[0].to_uppercase().next().unwrap_or(chars[0]);
    }
    let mut out: String = chars.into_iter().collect();

    out = out.split_whitespace().collect::<Vec<_>>().join(" ");

    const PUNC_TO_REPLACE: &[(&str, &str)] = &[
        ("...", ", "),
        ("\u{2026}", ", "), // …
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("\u{2014}", "-"), // —
        ("\u{2013}", "-"), // –
        (" ,", ","),
        ("\u{201c}", "\""), // "
        ("\u{201d}", "\""), // "
        ("\u{2018}", "'"),  // '
        ("\u{2019}", "'"),  // '
    ];
    for (old, new) in PUNC_TO_REPLACE {
        out = out.replace(old, new);
    }

    out = out.trim_end_matches(' ').to_string();
    const SENTENCE_ENDERS: &[char] = &['.', '!', '?', '-', ','];
    if !SENTENCE_ENDERS.iter().any(|&p| out.ends_with(p)) {
        out.push('.');
    }
    out
}

/// Builds the `(batch, len_text + 2)` `text_tokens` tensor `T3.inference()`'s caller
/// (`ChatterboxTTS.generate()`) assembles: CFG-double to batch 2 when `cfg_double` is
/// set, then pad with [`START_TEXT_TOKEN`] at the front and [`STOP_TEXT_TOKEN`] at
/// the back.
///
/// `chatterbox/tts.py::generate` only doubles when `cfg_weight > 0.0`, but
/// `crates/vocalai-core/src/t3.rs`'s decode loop (unchanged since VAI-004) always
/// indexes a 2-row `(cond, uncond)` batch -- so `vocalai-core`'s pipeline always
/// passes `cfg_double = true` regardless of `cfg_weight`, relying on
/// `combine_cfg_logits` to reduce to the conditional branch at `cfg_weight == 0.0`
/// rather than mirroring the Python reference's batch-size branching (see
/// `pipeline::synthesize`'s doc comment for the full rationale).
pub fn build_text_tokens(ids: &[i64], cfg_double: bool) -> Array2<i64> {
    let batch = if cfg_double { 2 } else { 1 };
    let len = ids.len() + 2;
    let mut out = Array2::<i64>::zeros((batch, len));
    for b in 0..batch {
        out[[b, 0]] = START_TEXT_TOKEN;
        for (i, &id) in ids.iter().enumerate() {
            out[[b, i + 1]] = id;
        }
        out[[b, len - 1]] = STOP_TEXT_TOKEN;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punc_norm_empty_string_becomes_placeholder() {
        assert_eq!(punc_norm(""), "You need to add some text for me to talk.");
    }

    #[test]
    fn punc_norm_capitalizes_first_letter() {
        assert_eq!(punc_norm("hello world"), "Hello world.");
    }

    #[test]
    fn punc_norm_collapses_whitespace() {
        assert_eq!(punc_norm("hello   world  "), "Hello world.");
    }

    #[test]
    fn punc_norm_replaces_uncommon_punctuation() {
        // Verified against the live Python reference (`chatterbox.tts.punc_norm`):
        // the double spaces are a real, faithfully-reproduced reference quirk --
        // replacement happens *after* whitespace-collapsing, so `"..."`/`";"`
        // (which leave a space on their replaced-side neighbor) produce a leftover
        // double space, while `" - "` (which consumes both surrounding spaces)
        // does not.
        assert_eq!(punc_norm("wait... really"), "Wait,  really.");
        assert_eq!(punc_norm("a: b"), "A, b.");
        assert_eq!(punc_norm("a - b"), "A, b.");
        assert_eq!(punc_norm("a; b"), "A,  b.");
    }

    #[test]
    fn punc_norm_leaves_existing_ending_punctuation() {
        assert_eq!(punc_norm("already done!"), "Already done!");
        assert_eq!(punc_norm("a question?"), "A question?");
    }

    #[test]
    fn punc_norm_replaces_em_and_en_dash_and_curly_quotes() {
        // Verified against the live Python reference (see the previous test).
        assert_eq!(punc_norm("a\u{2014}b"), "A-b.");
        assert_eq!(punc_norm("a\u{2013}b"), "A-b.");
        assert_eq!(punc_norm("\u{201c}quoted\u{201d}"), "\"quoted\".");
        assert_eq!(punc_norm("\u{2018}single\u{2019}"), "'single'.");
    }

    #[test]
    fn punc_norm_does_not_capitalize_non_letter_first_char() {
        assert_eq!(punc_norm("5 apples"), "5 apples.");
    }

    #[test]
    fn punc_norm_no_extra_period_when_already_comma_terminated() {
        assert_eq!(punc_norm("a,"), "A,");
    }

    #[test]
    fn build_text_tokens_single_batch_when_cfg_double_false() {
        let out = build_text_tokens(&[10, 20, 30], false);
        assert_eq!(out.shape(), &[1, 5]);
        assert_eq!(
            out.row(0).to_vec(),
            vec![START_TEXT_TOKEN, 10, 20, 30, STOP_TEXT_TOKEN]
        );
    }

    #[test]
    fn build_text_tokens_doubles_batch_when_cfg_double_true() {
        let out = build_text_tokens(&[1, 2], true);
        assert_eq!(out.shape(), &[2, 4]);
        for b in 0..2 {
            assert_eq!(
                out.row(b).to_vec(),
                vec![START_TEXT_TOKEN, 1, 2, STOP_TEXT_TOKEN]
            );
        }
    }
}
