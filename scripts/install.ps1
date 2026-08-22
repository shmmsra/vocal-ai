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
$BinaryName = "vocalai.exe"

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

# Exits 1 with a clear message if the given status/headers indicate this request was
# rate-limited -- a structured, unambiguous signal that falling back to the real (much
# larger) download would not help either. GitHub returns 403 with
# x-ratelimit-remaining: 0 for its rate-limited endpoints (observed live against
# api.github.com during this feature's own testing); both GitHub and HF Hub can also
# return a plain 429. Any OTHER lookup failure (network hiccup, unexpected response
# shape) is handled by the caller, which also exits rather than silently falling back
# to downloading -- a version check that can't be trusted isn't a check at all.
function Test-RateLimited {
    param([string]$What, $StatusCode, $Headers)
    if (-not $StatusCode) { return }
    $code = [int]$StatusCode
    if ($code -eq 429) {
        Write-Error "rate-limited while $What (HTTP 429) -- try again later"
        exit 1
    }
    if ($code -eq 403 -and $Headers -and $Headers["x-ratelimit-remaining"] -eq "0") {
        Write-Error "rate-limited while $What (HTTP 403, rate limit exhausted) -- try again later"
        exit 1
    }
}

# Plain HEAD request that returns status/headers regardless of whether the server
# responded with success or an error status (AllowAutoRedirect = $false so a 3xx is
# surfaced here too, via the WebException's Response, rather than followed).
function Get-HttpHeadInfo {
    param([string]$Uri)
    $statusCode = $null
    $headers = $null
    try {
        $req = [System.Net.HttpWebRequest]::Create($Uri)
        $req.Method = "HEAD"
        $req.AllowAutoRedirect = $false
        $resp = $req.GetResponse()
        $statusCode = $resp.StatusCode
        $headers = $resp.Headers
        $resp.Close()
    } catch [System.Net.WebException] {
        $errResp = $_.Exception.Response
        if ($errResp) {
            $statusCode = $errResp.StatusCode
            $headers = $errResp.Headers
            $errResp.Close()
        }
    }
    return @{ StatusCode = $statusCode; Headers = $headers }
}

# Reads the tag straight off the "latest release" redirect's Location header -- the
# same URL the actual download below hits, just as a HEAD request -- rather than
# api.github.com, which carries its own much stingier unauthenticated rate limit
# (60/hour/IP) shared with unrelated traffic on that IP. Exits 1 (never returns $null)
# if the tag can't be determined, whether due to rate-limiting or any other failure.
function Get-LatestReleaseTag {
    param([string]$Uri)
    $info = Get-HttpHeadInfo -Uri $Uri
    Test-RateLimited -What "checking the latest vocalai release" -StatusCode $info.StatusCode -Headers $info.Headers
    $location = if ($info.Headers) { $info.Headers["Location"] } else { $null }
    if ($location -match "/releases/download/([^/]+)/") {
        return $Matches[1]
    }
    $statusDesc = if ($info.StatusCode) { [int]$info.StatusCode } else { "unknown" }
    Write-Error "could not determine the latest vocalai release tag from $Uri (HTTP $statusDesc) -- check your network connection and that $GhRepo has a published release"
    exit 1
}

Write-Host "==> Installing vocalai ($Asset) into $InstallDir"
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

$ModelsDir = Join-Path $InstallDir "models"
New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null

# Version-tracking files (VAI-012): a plain-text tag/version dropped into the install
# dir after a successful install/update, checked before the next run so an
# already-up-to-date binary/model set isn't re-downloaded. CliVersionFile is our own
# (nothing inside the release archive carries a version marker). ModelsVersionFile
# reuses the name of a file scripts/publish_models.py already copies from this repo's
# root MODELS_VERSION into the HF Hub repo itself -- but it's deliberately written by
# this script only once the whole model set has downloaded successfully (see below),
# not fetched as just another file in the generic per-file loop, so a mid-download
# failure can't leave behind a stamp claiming a genuinely incomplete model set is current.
$CliVersionFile = Join-Path $InstallDir ".vocalai_version"
$ModelsVersionFile = Join-Path $ModelsDir "MODELS_VERSION"

Write-Host "==> Checking latest vocalai release..."
$LatestUrl = "https://github.com/$GhRepo/releases/latest/download/$Asset.zip"
$LatestTag = Get-LatestReleaseTag -Uri $LatestUrl

$binaryPath = Join-Path $InstallDir $BinaryName
$cliUpToDate = (Test-Path $CliVersionFile) -and (Test-Path $binaryPath) `
    -and ((Get-Content $CliVersionFile -Raw).Trim() -eq $LatestTag)

if ($cliUpToDate) {
    Write-Host "==> vocalai binary is up to date ($LatestTag), skipping download"
} else {
    Write-Host "==> Downloading latest release binary ($LatestTag)..."
    $TmpArchive = New-TemporaryFile
    $TmpZip = "$TmpArchive.zip"
    Rename-Item -Path $TmpArchive -NewName $TmpZip
    try {
        Invoke-DownloadWithRetry -Uri $LatestUrl -OutFile $TmpZip
        Expand-Archive -Path $TmpZip -DestinationPath $InstallDir -Force
    } finally {
        Remove-Item -Force $TmpZip -ErrorAction SilentlyContinue
    }
    Set-Content -Path $CliVersionFile -Value $LatestTag -NoNewline
}

Write-Host "==> Checking latest model artifacts version..."
$ModelsVersionUrl = "https://huggingface.co/$HfRepo/resolve/main/MODELS_VERSION"
$modelsVersionHead = Get-HttpHeadInfo -Uri $ModelsVersionUrl
Test-RateLimited -What "checking the latest model artifacts version" -StatusCode $modelsVersionHead.StatusCode -Headers $modelsVersionHead.Headers

try {
    $RemoteModelsVersion = (Invoke-WebRequest -Uri $ModelsVersionUrl -UseBasicParsing).Content.Trim()
} catch {
    Write-Error "could not determine the latest model artifacts version from $ModelsVersionUrl -- $($_.Exception.Message)"
    exit 1
}
if (-not $RemoteModelsVersion) {
    Write-Error "could not determine the latest model artifacts version from $ModelsVersionUrl (empty response)"
    exit 1
}

$modelsUpToDate = (Test-Path $ModelsVersionFile) `
    -and (Get-ChildItem -Path $ModelsDir -Filter '*.onnx' -Recurse -File -ErrorAction SilentlyContinue | Select-Object -First 1) `
    -and ((Get-Content $ModelsVersionFile -Raw).Trim() -eq $RemoteModelsVersion)

if ($modelsUpToDate) {
    Write-Host "==> models are up to date ($RemoteModelsVersion), skipping model download"
} else {
    Write-Host "==> Listing model artifacts from https://huggingface.co/$HfRepo..."
    $ApiUrl = "https://huggingface.co/api/models/$HfRepo"
    $apiHead = Get-HttpHeadInfo -Uri $ApiUrl
    Test-RateLimited -What "listing model artifacts" -StatusCode $apiHead.StatusCode -Headers $apiHead.Headers

    try {
        $RepoInfo = Invoke-RestMethod -Uri $ApiUrl
    } catch {
        Write-Error "could not list files for $HfRepo -- $($_.Exception.Message)"
        exit 1
    }
    # MODELS_VERSION is deliberately excluded from this generic loop too (see below): it
    # must only be written once every other file has downloaded successfully, not
    # incidentally partway through, so a failure mid-loop can't leave behind a version
    # marker for an actually-incomplete model set.
    $FilesToDownload = @($RepoInfo.siblings | ForEach-Object { $_.rfilename } | Where-Object {
        $_ -ne "README.md" -and $_ -ne ".gitattributes" -and $_ -ne "THIRD_PARTY_LICENSES" -and $_ -ne "MODELS_VERSION"
    })
    if ($FilesToDownload.Count -eq 0) {
        Write-Error "could not list files for $HfRepo -- check the repo exists and is public"
        exit 1
    }

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

    # Written last, only once every file above succeeded -- the definitive "this model
    # set is complete" marker, not just a byproduct of MODELS_VERSION happening to be one
    # of the files HF's listing returned.
    Set-Content -Path $ModelsVersionFile -Value $RemoteModelsVersion -NoNewline
}

Write-Host "==> Done. Run it with:"
Write-Host "    $InstallDir\$BinaryName --text `"hello world`" --out out.wav --models-dir $ModelsDir"
