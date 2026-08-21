from __future__ import annotations

from pathlib import Path

import pytest

import publish_models as pm


def test_require_hf_token_raises_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(pm.PublishError):
        pm.require_hf_token()


def test_require_hf_token_returns_value_when_set(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_fake_token")
    assert pm.require_hf_token() == "hf_fake_token"


def test_publish_rejects_missing_models_dir(tmp_path: Path):
    with pytest.raises(pm.PublishError):
        pm.publish(tmp_path / "nope", "shmmsra/vocal-ai-models", "hf_fake", "msg")


def test_publish_rejects_models_dir_with_no_onnx_files(tmp_path: Path):
    (tmp_path / "stray.txt").write_text("not a model")
    with pytest.raises(pm.PublishError):
        pm.publish(tmp_path, "shmmsra/vocal-ai-models", "hf_fake", "msg")


def test_publish_rejects_missing_third_party_licenses(tmp_path: Path, monkeypatch):
    (tmp_path / "a.onnx").write_bytes(b"not a real onnx file, just needs to exist for this check")
    monkeypatch.setattr(pm, "THIRD_PARTY_LICENSES_SRC", tmp_path / "does-not-exist")
    with pytest.raises(pm.PublishError, match="THIRD_PARTY_LICENSES"):
        pm.publish(tmp_path, "shmmsra/vocal-ai-models", "hf_fake", "msg")
