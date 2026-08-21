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

function Invoke-DownloadWithRetry {
    param([string]$Uri, [string]$OutFile)
    $maxAttempts = 3
    for ($attempt = 1; $attempt -le $maxAttempts; $attempt++) {
        try {
            Invoke-WebRequest -Uri $Uri -OutFile $OutFile
            return
        } catch {
            if ($attempt -eq $maxAttempts) { throw }
            Write-Host "    retry $attempt/$maxAttempts after transient error: $($_.Exception.Message)"
            Start-Sleep -Seconds 2
        }
    }
}

Write-Host "==> Installing vocalai ($Asset) into $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

Write-Host "==> Downloading latest release binary..."
$TmpArchive = New-TemporaryFile
$TmpZip = "$TmpArchive.zip"
Rename-Item -Path $TmpArchive -NewName $TmpZip
try {
    Invoke-DownloadWithRetry -Uri "https://github.com/$GhRepo/releases/latest/download/$Asset.zip" -OutFile $TmpZip
    Expand-Archive -Path $TmpZip -DestinationPath $InstallDir -Force
} finally {
    Remove-Item -Force $TmpZip -ErrorAction SilentlyContinue
}

Write-Host "==> Downloading model artifacts from https://huggingface.co/$HfRepo..."
$ModelsDir = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

$RepoInfo = Invoke-RestMethod -Uri "https://huggingface.co/api/models/$HfRepo"
$FilesToDownload = @($RepoInfo.siblings | ForEach-Object { $_.rfilename } | Where-Object {
    $_ -ne "README.md" -and $_ -ne ".gitattributes" -and $_ -ne "THIRD_PARTY_LICENSES"  # not needed to run vocalai
})

# Model files are several hundred MB to ~2GB each, and HF Hub's anonymous tier can be slow
# (rate limiting) -- print an [i/N] counter so a slow download doesn't look like a hang.
# Invoke-WebRequest already shows its own per-file progress bar by default.
Write-Host "==> Downloading $($FilesToDownload.Count) model files (this can take a while)"
$i = 0
foreach ($f in $FilesToDownload) {
    $i++
    $dest = Join-Path $ModelsDir $f
    New-Item -ItemType Directory -Force -Path (Split-Path $dest -Parent) | Out-Null
    Write-Host "==> [$i/$($FilesToDownload.Count)] $f"
    Invoke-DownloadWithRetry -Uri "https://huggingface.co/$HfRepo/resolve/main/$f" -OutFile $dest
}

Write-Host "==> Done. Run it with:"
Write-Host "    $InstallDir\vocalai.exe --text `"hello world`" --out out.wav --models-dir $ModelsDir"
