"""Tests for export/_common.py's comparison helper and the three Milestone-2 ONNX
exports (HiFiGAN, voice encoder, S3 tokenizer) against their PyTorch references.

The parity tests require the export toolchain (torch, chatterbox-tts, onnx,
onnxruntime — see export/requirements.txt) and download the Chatterbox checkpoint
components (ve.safetensors, s3gen.safetensors) from the HuggingFace Hub on first
run. See docs/dev-setup.md for the one-time venv setup.
"""
from __future__ import annotations

import numpy as np

from _common import allclose_report
import parity_check

ATOL = 1e-4
RTOL = 1e-3


def test_allclose_report_pass_within_tolerance():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 1e-6
    passed, diff = allclose_report(a, b, atol=1e-4, rtol=1e-3)
    assert passed
    assert diff < 1e-4


def test_allclose_report_fails_outside_tolerance():
    a = np.array([1.0, 2.0, 3.0])
    b = a + 1.0
    passed, diff = allclose_report(a, b, atol=1e-4, rtol=1e-3)
    assert not passed
    assert diff == 1.0


def test_allclose_report_fails_on_shape_mismatch():
    passed, diff = allclose_report(np.zeros((1, 2)), np.zeros((1, 3)), atol=1e-4, rtol=1e-3)
    assert not passed
    assert diff == float("inf")


def test_hifigan_export_matches_pytorch_reference():
    result = parity_check.check_hifigan(ATOL, RTOL)
    assert result.passed, f"max_abs_diff={result.max_abs_diff}"


def test_voice_encoder_export_matches_pytorch_reference():
    result = parity_check.check_ve(ATOL, RTOL)
    assert result.passed, f"max_abs_diff={result.max_abs_diff}"


def test_s3tokenizer_export_matches_pytorch_reference():
    result = parity_check.check_s3tokenizer(ATOL, RTOL)
    assert result.passed, f"max_abs_diff={result.max_abs_diff}"
