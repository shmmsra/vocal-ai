from pathlib import Path

REQUIREMENTS_PATH = Path(__file__).resolve().parent.parent / "requirements.txt"


def _parsed_requirements() -> dict[str, str]:
    packages = {}
    for line in REQUIREMENTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, _, version = line.partition("==")
        packages[name] = version
    return packages


def test_chatterbox_and_onnx_toolchain_pinned():
    packages = _parsed_requirements()
    for name in ("chatterbox-tts", "onnx", "onnxruntime"):
        assert name in packages, f"{name} must be pinned in export/requirements.txt"
        assert packages[name], f"{name} must have an exact version pin"
