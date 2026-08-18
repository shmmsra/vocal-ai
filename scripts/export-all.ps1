#
# Generate every ONNX/npy model artifact vocalai needs, in the required order.
# See docs/dev-setup.md §11.1 for what each script produces.
#
# Usage:
#   scripts\export-all.ps1                        # default-voice path only
#   scripts\export-all.ps1 --with-voice-cloning   # + --voice zero-shot cloning models

$ErrorActionPreference = "Stop"

$RepoRoot = (git rev-parse --show-toplevel 2>$null)
if (-not $RepoRoot) {
    Write-Error "not inside a git repository"
    exit 1
}

$VenvPy = Join-Path $RepoRoot "export\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPy)) {
    Write-Error "$VenvPy not found -- set up the export venv first (docs/dev-setup.md section 2)"
    exit 1
}

$WithVoiceCloning = $false
foreach ($arg in $args) {
    switch ($arg) {
        "--with-voice-cloning" { $WithVoiceCloning = $true }
        default {
            Write-Error "unknown argument '$arg' (expected --with-voice-cloning)"
            exit 1
        }
    }
}

$Scripts = @(
    "fetch_tokenizer.py",
    "export_t3.py",
    "export_s3gen.py",
    "export_s3gen_flow_encoder.py",
    "export_hifigan.py",
    "export_perthnet.py",
    "export_campplus.py",
    "export_default_voice.py"
)

if ($WithVoiceCloning) {
    $Scripts += "export_ve.py"
    $Scripts += "export_s3tokenizer.py"
}

foreach ($script in $Scripts) {
    Write-Host "-> Running export\$script..."
    & $VenvPy (Join-Path $RepoRoot "export\$script")
}

Write-Host "All model artifacts generated in $RepoRoot\models\"
