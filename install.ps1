# install.ps1 — Set up Phoenice-PolyRAG local virtual environment (Windows)
# Usage:  .\install.ps1
#         .\install.ps1 -Full   (also installs optional Weaviate/Milvus/PGVector deps)
param(
    [switch]$Full  # Install all optional adapter dependencies
)

$ErrorActionPreference = "Stop"
$base = $PSScriptRoot

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║   Phoenice-PolyRAG  —  Full Installer (Phase 1-13) ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# ── Python version check ──────────────────────────────────────────────────────
$pyVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found on PATH. Install Python 3.10+ first." -ForegroundColor Red
    exit 1
}
Write-Host "  Python : $pyVersion" -ForegroundColor Gray

# ── Virtual environment ───────────────────────────────────────────────────────
$venvPath = Join-Path $base ".venv"
if (-not (Test-Path $venvPath)) {
    Write-Host ""
    Write-Host "==> Creating virtual environment at .venv ..." -ForegroundColor Cyan
    python -m venv $venvPath
} else {
    Write-Host ""
    Write-Host "==> Virtual environment already exists — skipping creation." -ForegroundColor Yellow
}

# Activate
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
. $activateScript
Write-Host "  Venv   : $venvPath" -ForegroundColor Gray

# ── Upgrade pip ───────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Upgrading pip ..." -ForegroundColor Cyan
python -m pip install --upgrade pip --quiet

# ── CPU-only PyTorch (avoids ~2 GB CUDA download) ────────────────────────────
Write-Host ""
Write-Host "==> Installing CPU-only PyTorch (sentence-transformers dependency) ..." -ForegroundColor Cyan
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet

# ── Core requirements ─────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Installing core requirements from requirements.txt ..." -ForegroundColor Cyan
pip install -r (Join-Path $base "requirements.txt") --quiet

# ── Optional full install ─────────────────────────────────────────────────────
if ($Full) {
    Write-Host ""
    Write-Host "==> Installing optional adapter dependencies (--Full flag) ..." -ForegroundColor Cyan
    pip install weaviate-client>=4.5 --quiet
    Write-Host "  weaviate-client : installed" -ForegroundColor Gray

    if ($IsLinux -or $IsMacOS) {
        pip install "pymilvus>=2.4" --quiet
        Write-Host "  pymilvus        : installed (Milvus Lite supported on this OS)" -ForegroundColor Gray
    } else {
        Write-Host "  pymilvus        : skipped (Milvus Lite not supported on Windows; use Docker)" -ForegroundColor Yellow
    }

    pip install psycopg2-binary pgvector --quiet
    Write-Host "  pgvector        : installed (requires PostgreSQL server)" -ForegroundColor Gray
}

# ── Verify key imports ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Verifying key imports ..." -ForegroundColor Cyan
$checks = @(
    "import pydantic; print('  pydantic       :', pydantic.VERSION)",
    "import chromadb; print('  chromadb       :', chromadb.__version__)",
    "import faiss; print('  faiss-cpu      : ok')",
    "import qdrant_client; v=getattr(qdrant_client,'__version__',getattr(qdrant_client.version,'__version__','ok')); print('  qdrant-client  :', v)",
    "import sentence_transformers; print('  sentence-trans :', sentence_transformers.__version__)",
    "import networkx; print('  networkx       :', networkx.__version__)",
    "import kuzu; print('  kuzu           :', kuzu.__version__)",
    "import spacy; print('  spacy          :', spacy.__version__)",
    "import fastapi; print('  fastapi        :', fastapi.__version__)",
    "import uvicorn; print('  uvicorn        : ok')"
)
foreach ($chk in $checks) {
    python -c $chk
}

# ── spaCy English model (Phase 10 NER) ───────────────────────────────────────
Write-Host ""
Write-Host "==> Downloading spaCy English model (en_core_web_sm, ~13 MB) ..." -ForegroundColor Cyan
python -m spacy download en_core_web_sm --quiet

# ── Frontend (Phase 13) ───────────────────────────────────────────────────────
Write-Host ""
Write-Host "==> Installing frontend dependencies (Node.js / npm) ..." -ForegroundColor Cyan
$frontendPath = Join-Path $base "frontend"
if (Test-Path $frontendPath) {
    $nodeCheck = node --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: Node.js not found — skipping frontend install." -ForegroundColor Yellow
        Write-Host "           Install Node.js 20+ from https://nodejs.org then re-run .\install.ps1" -ForegroundColor Yellow
    } else {
        Write-Host "  Node.js : $nodeCheck" -ForegroundColor Gray
        Push-Location $frontendPath
        npm install --silent
        Write-Host "  Frontend dependencies installed ✅" -ForegroundColor Green
        Pop-Location
    }
} else {
    Write-Host "  frontend/ directory not found — skipping." -ForegroundColor Yellow
}

# ── Done ──────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║  ✅  Installation complete! (Phases 1-13)                    ║" -ForegroundColor Green
Write-Host "║                                                              ║" -ForegroundColor Green
Write-Host "║  Next steps:                                                 ║" -ForegroundColor Green
Write-Host "║    .\start.ps1                  — activate env + run API    ║" -ForegroundColor Green
Write-Host "║    pytest tests/ -q             — run all backend tests      ║" -ForegroundColor Green
Write-Host "║    cd frontend && npm test      — run frontend tests         ║" -ForegroundColor Green
Write-Host "║    cd frontend && npm run dev   — start React UI (:3000)    ║" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
