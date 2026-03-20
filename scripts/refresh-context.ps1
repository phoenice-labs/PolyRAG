<#
.SYNOPSIS
    Refreshes the Copilot workspace context file (.github/copilot-instructions.md)
    with current project state (test counts, dependency versions, new files).

.DESCRIPTION
    Run after adding phases, new endpoints, or major config changes.
    Keeps the context file accurate so Copilot sessions start with correct info.

.EXAMPLE
    .\scripts\refresh-context.ps1
#>

$Root = Split-Path $PSScriptRoot -Parent
$ContextFile = Join-Path $Root ".github\copilot-instructions.md"

Write-Host "🔄 Refreshing Copilot context..." -ForegroundColor Cyan

# --- Test count ---
$TestCount = (Get-ChildItem -Path (Join-Path $Root "tests") -Recurse -Filter "test_*.py" |
    Select-String -Pattern "^def test_" | Measure-Object).Count

# --- Phase count (core/ subdirectories with __init__.py) ---
$Phases = @("chunking","retrieval","query","provenance","confidence","temporal","noise","graph","store","embedding","ingestion","classification","observability")
$PhaseCount = ($Phases | Where-Object { Test-Path (Join-Path $Root "core\$_") }).Count

# --- Dependency count ---
$DepCount = (Get-Content (Join-Path $Root "requirements.txt") | Where-Object { $_ -match "^\S" -and $_ -notmatch "^#" }).Count

# --- Router count ---
$RouterCount = (Get-ChildItem -Path (Join-Path $Root "api\routers") -Filter "*.py" | Where-Object { $_.Name -ne "__init__.py" }).Count

# --- Python version from pyproject.toml ---
$PyVersion = (Select-String -Path (Join-Path $Root "pyproject.toml") -Pattern 'requires-python\s*=\s*"([^"]+)"').Matches[0].Groups[1].Value

Write-Host "  Tests found  : $TestCount" -ForegroundColor Green
Write-Host "  Core modules : $PhaseCount" -ForegroundColor Green
Write-Host "  Dependencies : $DepCount" -ForegroundColor Green
Write-Host "  API routers  : $RouterCount" -ForegroundColor Green
Write-Host "  Python req   : $PyVersion" -ForegroundColor Green

# --- Patch the "Total tests" line in context ---
$Content = Get-Content $ContextFile -Raw
$Updated = $Content -replace '(?<=\*\*Total\*\*: )\d+(?= tests)', $TestCount
$Updated = $Updated -replace '(?<=- \*\*Total\*\*: )\d+(?= tests)', $TestCount

if ($Updated -ne $Content) {
    Set-Content $ContextFile $Updated -NoNewline
    Write-Host "✅ Test count updated to $TestCount in context file." -ForegroundColor Green
} else {
    Write-Host "ℹ️  No numeric patches needed — context file already current." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📄 Context file: $ContextFile" -ForegroundColor Cyan
Write-Host "   Edit manually for structural changes (new phases, routes, etc.)" -ForegroundColor Gray
