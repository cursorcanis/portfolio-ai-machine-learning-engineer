# Starts a local Qdrant instance for the AI Advisor backend.
#
# Qdrant does not ship a Windows-native binary, so this script picks one of
# two routes (in order of preference):
#   1. Docker (if `docker` is on PATH) — runs the official image with a
#      persistent volume at ./qdrant-storage.
#   2. WSL — runs the Linux release binary inside the default WSL distro,
#      bound to 0.0.0.0:6334 so Windows can reach it on localhost.
#
# Storage on disk is kept next to the repo at ./qdrant-storage so it survives
# container/VM restarts.

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)  # ai-advisor-rs/

$storageDir = Join-Path (Get-Location) "qdrant-storage"
if (-not (Test-Path $storageDir)) {
    New-Item -ItemType Directory -Path $storageDir | Out-Null
    Write-Host "Created $storageDir"
}

# Quick health check — if Qdrant is already up, we're done.
try {
    $probe = Invoke-WebRequest -Uri "http://127.0.0.1:6333/" -TimeoutSec 2 -UseBasicParsing
    if ($probe.StatusCode -eq 200) {
        Write-Host "Qdrant already running on 127.0.0.1:6333."
        exit 0
    }
} catch {
    # not running, continue
}

$dockerCmd = Get-Command docker -ErrorAction SilentlyContinue
if ($null -ne $dockerCmd) {
    Write-Host "Starting Qdrant via Docker..."
    $volumeArg = "${storageDir}:/qdrant/storage"
    & docker run -d --rm --name ai-advisor-qdrant `
        -p 6333:6333 -p 6334:6334 `
        -v $volumeArg `
        qdrant/qdrant:latest | Out-Host
    Write-Host "Qdrant started. HTTP: http://127.0.0.1:6333  gRPC: 127.0.0.1:6334"
    exit 0
}

$wslCmd = Get-Command wsl -ErrorAction SilentlyContinue
if ($null -ne $wslCmd) {
    Write-Host "Docker not found. Falling back to WSL."
    Write-Host "On first run, WSL will download the Qdrant release (~30 MB)."
    $bashScript = @'
set -e
mkdir -p ~/qdrant
cd ~/qdrant
if [ ! -x ./qdrant ]; then
  echo "Downloading Qdrant release..."
  VER=$(curl -s https://api.github.com/repos/qdrant/qdrant/releases/latest | grep tag_name | head -1 | cut -d'"' -f4)
  curl -L -o qdrant.tar.gz "https://github.com/qdrant/qdrant/releases/download/${VER}/qdrant-x86_64-unknown-linux-musl.tar.gz"
  tar xzf qdrant.tar.gz
  rm qdrant.tar.gz
fi
pkill -x qdrant >/dev/null 2>&1 || true
QDRANT__SERVICE__HTTP_PORT=6333 \
QDRANT__SERVICE__GRPC_PORT=6334 \
QDRANT__SERVICE__HOST=0.0.0.0 \
QDRANT__STORAGE__STORAGE_PATH=~/qdrant/storage \
nohup ./qdrant > ~/qdrant/qdrant.log 2>&1 &
sleep 1
echo "Qdrant launched in WSL (logs at ~/qdrant/qdrant.log)"
'@
    & wsl -- bash -lc $bashScript
    Write-Host "Qdrant should now be reachable at http://127.0.0.1:6333"
    exit 0
}

Write-Error "Neither docker nor wsl is available. Install Docker Desktop or WSL to run Qdrant locally."
exit 1
