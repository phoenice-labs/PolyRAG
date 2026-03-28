<#
.SYNOPSIS
    Run PolyRAG load test using Locust.

.DESCRIPTION
    Validates the scalability fixes (PRs #1-#3) under concurrent load.
    Requires the PolyRAG API to be running and data to be ingested.

.PARAMETER Users
    Number of concurrent virtual users. Default: 20.

.PARAMETER SpawnRate
    Users spawned per second until target is reached. Default: 5.

.PARAMETER Duration
    Test duration (e.g. 60s, 2m, 5m). Default: 60s.

.PARAMETER Host
    PolyRAG API base URL. Default: http://localhost:8000.

.PARAMETER UI
    Launch Locust web UI instead of running headless. Default: false.

.PARAMETER Report
    Path for the HTML report. Default: tests/load/report.html.

.EXAMPLE
    # Quick validation (20 users, 60 seconds)
    .\scripts\load-test.ps1

.EXAMPLE
    # Heavier test (50 users, 2 minutes)
    .\scripts\load-test.ps1 -Users 50 -Duration 2m

.EXAMPLE
    # Interactive web UI (http://localhost:8089)
    .\scripts\load-test.ps1 -UI

.NOTES
    Prerequisites:
      1. PolyRAG API running:  uvicorn api.main:app --port 8000
         Or multi-worker:      .\start.ps1 -Workers 4
      2. Data ingested into chromadb + faiss (polyrag_docs_minilm collection)
      3. Locust installed:     pip install locust
#>

param(
    [int]$Users      = 20,
    [int]$SpawnRate  = 5,
    [string]$Duration = "60s",
    [string]$Host    = "http://localhost:8000",
    [switch]$UI,
    [string]$Report  = "tests\load\report.html"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot

# ── Check prerequisites ───────────────────────────────────────────────────────

Write-Host "`n── PolyRAG Load Test ────────────────────────────────" -ForegroundColor Cyan

# Check Locust
$locust = $null
try {
    $locust = & "$root\.venv\Scripts\locust.exe" --version 2>&1
    Write-Host "  Locust:    $locust" -ForegroundColor Green
} catch {
    Write-Host "  ✗ Locust not found. Install with: pip install locust" -ForegroundColor Red
    exit 1
}

# Check API is running
Write-Host "  Checking API at $Host ..." -ForegroundColor Gray
try {
    $health = Invoke-RestMethod -Uri "$Host/api/health" -TimeoutSec 5
    Write-Host "  API:       $($health.service) v$($health.version) — OK" -ForegroundColor Green
} catch {
    Write-Host "  ✗ API not reachable at $Host. Start it first:" -ForegroundColor Red
    Write-Host "      uvicorn api.main:app --port 8000" -ForegroundColor Yellow
    Write-Host "      or: .\start.ps1" -ForegroundColor Yellow
    exit 1
}

# Check cache (shows what's warmed up)
try {
    $cache = Invoke-RestMethod -Uri "$Host/api/system/cache" -TimeoutSec 5
    Write-Host "  Cache:     $($cache.cached)/$($cache.max_pipelines) pipelines cached" -ForegroundColor Green
} catch {
    Write-Host "  Cache:     (system endpoint not available)" -ForegroundColor Yellow
}

Write-Host ""

# ── Build Locust command ──────────────────────────────────────────────────────

$locustFile = Join-Path $root "tests\load\locustfile.py"
$reportPath = Join-Path $root $Report

# Ensure report directory exists
$reportDir = Split-Path $reportPath -Parent
if (-not (Test-Path $reportDir)) {
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
}

if ($UI) {
    Write-Host "Starting Locust web UI..." -ForegroundColor Cyan
    Write-Host "  Open: http://localhost:8089" -ForegroundColor Green
    Write-Host "  Press Ctrl+C to stop`n" -ForegroundColor Gray
    & "$root\.venv\Scripts\locust.exe" `
        -f $locustFile `
        --host $Host
} else {
    Write-Host "Running headless load test:" -ForegroundColor Cyan
    Write-Host "  Users:     $Users (spawn rate: $SpawnRate/s)"
    Write-Host "  Duration:  $Duration"
    Write-Host "  Host:      $Host"
    Write-Host "  Report:    $reportPath"
    Write-Host ""

    & "$root\.venv\Scripts\locust.exe" `
        -f $locustFile `
        --headless `
        --users $Users `
        --spawn-rate $SpawnRate `
        --run-time $Duration `
        --host $Host `
        --html $reportPath `
        --only-summary

    $exitCode = $LASTEXITCODE

    Write-Host ""
    if ($exitCode -eq 0) {
        Write-Host "✅ Load test passed all thresholds" -ForegroundColor Green
        Write-Host "   Report: $reportPath" -ForegroundColor Gray
    } else {
        Write-Host "❌ Load test failed — thresholds breached (see output above)" -ForegroundColor Red
        Write-Host "   Report: $reportPath" -ForegroundColor Gray
    }

    exit $exitCode
}
