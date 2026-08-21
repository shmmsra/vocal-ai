from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

import smoke_test_artifact as sta


def _write_valid_onnx(path: Path) -> None:
    node = helper.make_node("Identity", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, producer_name="test")
    model.opset_import[0].version = 17
    onnx.save(model, str(path))


def test_check_models_dir_passes_on_valid_files(tmp_path: Path):
    _write_valid_onnx(tmp_path / "a.onnx")
    np.save(tmp_path / "b.npy", np.zeros(3))
    (tmp_path / "tokenizer.json").write_text(json.dumps({"a": 1}))

    checked = sta.check_models_dir(tmp_path)
    assert len(checked) == 3


def test_check_models_dir_rejects_corrupt_onnx(tmp_path: Path):
    (tmp_path / "broken.onnx").write_bytes(b"not an onnx file")
    with pytest.raises(Exception):
        sta.check_models_dir(tmp_path)


def test_check_models_dir_rejects_no_onnx_files(tmp_path: Path):
    with pytest.raises(sta.SmokeTestError):
        sta.check_models_dir(tmp_path)


def test_check_models_dir_rejects_malformed_tokenizer_json(tmp_path: Path):
    _write_valid_onnx(tmp_path / "a.onnx")
    (tmp_path / "tokenizer.json").write_text("{not valid json")
    with pytest.raises(Exception):
        sta.check_models_dir(tmp_path)


def test_check_models_dir_rejects_missing_dir(tmp_path: Path):
    with pytest.raises(sta.SmokeTestError):
        sta.check_models_dir(tmp_path / "does-not-exist")


def test_check_extra_files_rejects_missing(tmp_path: Path):
    with pytest.raises(sta.SmokeTestError):
        sta.check_extra_files([tmp_path / "missing.txt"])


def test_check_extra_files_passes_when_present(tmp_path: Path):
    f = tmp_path / "LICENSE"
    f.write_text("MIT")
    sta.check_extra_files([f])  # must not raise


def test_check_binary_rejects_missing_binary(tmp_path: Path):
    with pytest.raises(sta.SmokeTestError):
        sta.check_binary(tmp_path / "nope")


def test_main_returns_nonzero_on_empty_models_dir(tmp_path: Path, capsys):
    rc = sta.main(["--models-dir", str(tmp_path)])
    assert rc == 1
    assert "error:" in capsys.readouterr().err


def test_main_returns_zero_on_valid_models_dir(tmp_path: Path):
    _write_valid_onnx(tmp_path / "a.onnx")
    rc = sta.main(["--models-dir", str(tmp_path)])
    assert rc == 0
