# Install vocalai: downloads the latest binary release from GitHub + the model
# artifacts from the public HuggingFace Hub repo (no HF token needed -- it's a
# public, anonymously-readable repo), laid out together in .\vocalai\ ready to run.
#
# Usage:
#   irm https://raw.githubusercontent.com/shmmsra/vocal-ai/main/scripts/install.ps1 | iex
#
# Override the install directory with $env:VOCALAI_INSTALL_DIR (default: .\vocalai).

$ErrorActionPreference = "Stop"

$GhRepo = "shmmsra/vocal-ai"
$HfRepo = "shmmsra/vocal-ai-models"
$InstallDir = if ($env:VOCALAI_INSTALL_DIR) { $env:VOCALAI_INSTALL_DIR } else { ".\vocalai" }
$Asset = "vocalai-windows-cpu"

Write-Host "==> Installing vocalai ($Asset) into $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "==> Downloading latest release binary..."
$TmpArchive = New-TemporaryFile
$TmpZip = "$TmpArchive.zip"
Rename-Item -Path $TmpArchive -NewName $TmpZip
try {
    Invoke-WebRequest -Uri "https://github.com/$GhRepo/releases/latest/download/$Asset.zip" -OutFile $TmpZip
    Expand-Archive -Path $TmpZip -DestinationPath $InstallDir -Force
} finally {
    Remove-Item -Force $TmpZip -ErrorAction SilentlyContinue
}

Write-Host "==> Downloading model artifacts from https://huggingface.co/$HfRepo..."
$ModelsDir = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$RepoInfo = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$HfRepo"
foreach ($sibling in $RepoInfo.siblings) {
    $f = $sibling.rfilename
    if ($f -eq "README.md" -or $f -eq ".gitattributes" -or $f -eq "THIRD_PARTY_LICENSES") { continue }  # not needed to run vocalai

    $dest = Join-Path $ModelsDir $f
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Write-Host "    $f"
    Invoke-WebRequest -Uri "https://huggingface.co/$HfRepo/resolve/main/$f" -OutFile $dest
}

Write-Host "==> Done. Run it with:"
Write-Host "    $InstallDir\vocalai.exe --text `"hello world`" --out out.wav --models-dir $ModelsDir"
