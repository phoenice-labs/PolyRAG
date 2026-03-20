# start.ps1 — Manage Phoenice-PolyRAG services (API, Frontend, Docker vector DBs)
#
# Usage:
#   .\start.ps1                 — start all services (Docker + API + Frontend)
#   .\start.ps1 -Action start   — same as above (explicit)
#   .\start.ps1 -Action stop    — stop all services gracefully
#   .\start.ps1 -Action restart — stop then start all services
#   .\start.ps1 -Action status  — show status of all services
#   .\start.ps1 -Action env     — activate venv only (legacy behaviour)
#
#   Flags (can combine with any Action):
#   -NoDocker    — skip Docker vector DB containers
#   -NoFrontend  — skip React UI (port 3000)
#   -NoApi       — skip FastAPI server (port 8000)

param(
    [ValidateSet("start","stop","restart","status","env")]
    [string]$Action = "start",
    [switch]$NoDocker,
    [switch]$NoFrontend,
    [switch]$NoApi
)

$ErrorActionPreference = "SilentlyContinue"  # avoid crashing on missing processes
$base     = $PSScriptRoot
$pidFile  = Join-Path $base ".service-pids.json"
$compose  = Join-Path $base "docker-compose.polyrag.yml"
$venvPath = Join-Path $base ".venv"
$frontendPath = Join-Path $base "frontend"

# ── Helpers ───────────────────────────────────────────────────────────────────
function Write-Header([string]$text) {
    $line = "=" * 64
    Write-Host ""
    Write-Host $line -ForegroundColor Cyan
    Write-Host "   $text" -ForegroundColor Cyan
    Write-Host $line -ForegroundColor Cyan
    Write-Host ""
}

function Write-Ok([string]$msg)   { Write-Host "  [OK]  $msg" -ForegroundColor Green  }
function Write-Warn([string]$msg) { Write-Host "  [!!]  $msg" -ForegroundColor Yellow }
function Write-Err([string]$msg)  { Write-Host "  [XX]  $msg" -ForegroundColor Red    }
function Write-Info([string]$msg) { Write-Host "  -->   $msg"  -ForegroundColor Gray   }

function Load-Pids {
    if (Test-Path $pidFile) {
        try {
            $obj = Get-Content $pidFile -Raw | ConvertFrom-Json
            $ht = @{}
            foreach ($prop in $obj.PSObject.Properties) { $ht[$prop.Name] = $prop.Value }
            return $ht
        } catch { }
    }
    return @{}
}

function Save-Pids([hashtable]$pids) {
    $pids | ConvertTo-Json | Set-Content $pidFile
}

function Is-ProcessRunning([int]$processId) {
    if ($processId -le 0) { return $false }
    $p = Get-Process -Id $processId -ErrorAction SilentlyContinue
    return ($null -ne $p)
}

function Stop-ServicePid([string]$name, [int]$processId) {
    if ($processId -gt 0 -and (Is-ProcessRunning $processId)) {
        Write-Info "Stopping $name (PID $processId) ..."
        Stop-ProcessTree $processId
        Start-Sleep -Milliseconds 500
        if (-not (Is-ProcessRunning $processId)) {
            Write-Ok "$name stopped."
        } else {
            Write-Warn "$name (PID $processId) may still be running."
        }
    } else {
        Write-Info "$name is not running (PID $processId not found)."
    }
}

function Stop-Port([int]$port) {
    # Kill every process (parent + children) holding the given TCP port.
    # Needed because uvicorn --reload spawns a child worker that outlives the parent PID.
    $pidsOnPort = @()
    netstat -ano 2>$null | Select-String ":$port\s" | ForEach-Object {
        if ($_ -match '\s+(\d+)$') { $pidsOnPort += [int]$Matches[1] }
    }
    foreach ($p in ($pidsOnPort | Sort-Object -Unique)) {
        if ($p -gt 4) {   # skip PID 0/4 (System)
            Stop-Process -Id $p -Force -ErrorAction SilentlyContinue
        }
    }
}

function Stop-ProcessTree([int]$processId) {
    # Recursively collect all descendant PIDs, then kill leaves-first to avoid orphans
    if ($processId -le 0) { return }
    Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $processId } | ForEach-Object {
        Stop-ProcessTree ([int]$_.ProcessId)
    }
    Stop-Process -Id $processId -Force -ErrorAction SilentlyContinue
}

function Wait-PortFree([int]$port, [int]$timeoutSec = 15) {
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Test-Port $port) -and ((Get-Date) -lt $deadline)) {
        Start-Sleep -Milliseconds 500
    }
    if (Test-Port $port) {
        Write-Warn "Port $port still in use after ${timeoutSec}s - proceeding anyway."
    }
}

function Test-Port([int]$port) {
    # Use netstat to check for a process actually LISTENING on this port.
    # TcpClient.Connect() gives false positives on Windows (Hyper-V/WSL2 reserved ranges).
    $listening = netstat -ano 2>$null | Select-String "LISTENING" | Select-String ":$port\s"
    return ($null -ne $listening -and @($listening).Count -gt 0)
}

# ── Activate venv (required for API) ─────────────────────────────────────────
function Activate-Venv {
    if (-not (Test-Path $venvPath)) {
        Write-Err "Virtual environment not found. Run '.\install.ps1' first."
        exit 1
    }
    $activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
    . $activateScript
}

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION: STATUS
# ══════════════════════════════════════════════════════════════════════════════
function Do-Status {
    Write-Header "Phoenice-PolyRAG  —  Service Status"
    $pids = Load-Pids

    # ── Docker vector DBs (per-backend port check) ────────────────────────────
    $dockerServices = @(
        @{ Name = "Qdrant";    Port = 6333;  Label = "Qdrant   (6333 )  " },
        @{ Name = "Weaviate";  Port = 8088;  Label = "Weaviate (8088 )  " },
        @{ Name = "PGVector";  Port = 5433;  Label = "PGVector (5433 )  " },
        @{ Name = "Milvus";    Port = 19530; Label = "Milvus   (19530)  " }
    )
    foreach ($svc in $dockerServices) {
        if (Test-Port $svc.Port) { Write-Ok  "$($svc.Label): ready" }
        else                     { Write-Warn "$($svc.Label): not reachable (Docker stopped?)" }
    }
    Write-Info "FAISS / ChromaDB : in-process (always available when API is running)"

    # ── API (port 8000) ───────────────────────────────────────────────────────
    $apiPid = if ($pids.ContainsKey("api")) { [int]$pids["api"] } else { 0 }
    $apiPort = Test-Port 8000
    if ($apiPort)                      { Write-Ok  "API server         : http://localhost:8000  (PID $apiPid)" }
    elseif (Is-ProcessRunning $apiPid) { Write-Warn "API server         : process alive but port 8000 not yet open" }
    else                               { Write-Warn "API server         : stopped" }

    # ── Frontend (port 3000) ──────────────────────────────────────────────────
    $fePid = if ($pids.ContainsKey("frontend")) { [int]$pids["frontend"] } else { 0 }
    $fePort = Test-Port 3000
    if ($fePort)                      { Write-Ok  "React UI           : http://localhost:3000  (PID $fePid)" }
    elseif (Is-ProcessRunning $fePid) { Write-Warn "React UI           : process alive but port 3000 not yet open" }
    else                              { Write-Warn "React UI           : stopped" }

    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION: STOP
# ══════════════════════════════════════════════════════════════════════════════
function Do-Stop {
    Write-Header "Phoenice-PolyRAG  —  Stopping Services"
    $pids = Load-Pids

    # Stop API
    if (-not $NoApi) {
        $apiPid = if ($pids.ContainsKey("api")) { [int]$pids["api"] } else { 0 }
        Stop-ServicePid "API server" $apiPid
        # Also kill any process still holding port 8000 (uvicorn --reload child worker)
        Stop-Port 8000
        $pids["api"] = 0
    }

    # Stop Frontend — kill the cmd.exe wrapper AND its entire node child tree
    if (-not $NoFrontend) {
        $fePid = if ($pids.ContainsKey("frontend")) { [int]$pids["frontend"] } else { 0 }
        if ($fePid -gt 0 -and (Is-ProcessRunning $fePid)) {
            Write-Info "Stopping React UI (PID $fePid and children) ..."
            Stop-ProcessTree $fePid
            Write-Ok "React UI stopped."
        } else {
            Write-Info "React UI is not running (PID $fePid not found)."
        }
        # Also kill any process still holding port 3000
        Stop-Port 3000
        $pids["frontend"] = 0
    }

    # Stop Docker
    if (-not $NoDocker) {
        if (Test-Path $compose) {
            Write-Info "Stopping Docker vector DB containers ..."
            docker compose -f $compose stop 2>&1 | Out-Null
            if ($LASTEXITCODE -eq 0) { Write-Ok "Docker containers stopped." }
            else                      { Write-Warn "docker compose stop returned non-zero (containers may already be stopped)." }

            # Quit Docker Desktop (daemon)
            Write-Info "Quitting Docker Desktop ..."
            $ddProc = Get-Process "Docker Desktop" -ErrorAction SilentlyContinue
            if ($null -ne $ddProc) {
                # Use the graceful quit CLI first; fall back to process kill
                $ddCli = @(
                    "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
                    "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
                ) | Where-Object { Test-Path $_ } | Select-Object -First 1
                if ($ddCli) {
                    Start-Process -FilePath $ddCli -ArgumentList "quit" -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue
                }
                # If still running, force-close
                $ddProc | Where-Object { -not $_.HasExited } | ForEach-Object {
                    Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue
                }
                Write-Ok "Docker Desktop stopped."
            } else {
                Write-Info "Docker Desktop was not running."
            }
        }
    }

    Save-Pids $pids
    Write-Host ""
    Write-Ok "Done. Run '.\start.ps1' to start all services again."
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION: START
# ══════════════════════════════════════════════════════════════════════════════
function Do-Start {
    Write-Header "Phoenice-PolyRAG  —  Starting Services"
    Activate-Venv

    $pids = Load-Pids

    # ── 1. Docker vector DBs ─────────────────────────────────────────────────
    if (-not $NoDocker) {
        $dockerBin = Get-Command docker -ErrorAction SilentlyContinue
        if ($null -eq $dockerBin) {
            Write-Warn "Docker not found on PATH — skipping vector DB containers."
        } elseif (-not (Test-Path $compose)) {
            Write-Warn "docker-compose.polyrag.yml not found — skipping."
        } else {
            # Check Docker daemon is actually running (not just the CLI)
            docker info 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "Docker daemon is not running — attempting to start Docker Desktop ..."
                # Try common Docker Desktop install locations
                $dockerDesktop = @(
                    "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe",
                    "$env:LOCALAPPDATA\Docker\Docker Desktop.exe"
                ) | Where-Object { Test-Path $_ } | Select-Object -First 1

                if ($null -eq $dockerDesktop) {
                    Write-Err "Docker Desktop not found. Install it from https://www.docker.com/products/docker-desktop"
                } else {
                    Start-Process -FilePath $dockerDesktop -WindowStyle Minimized
                    Write-Info "Waiting for Docker daemon to become ready (up to 60s) ..."
                    $dockerReady = $false
                    $dockerDeadline = (Get-Date).AddSeconds(60)
                    while (-not $dockerReady -and (Get-Date) -lt $dockerDeadline) {
                        Start-Sleep -Seconds 3
                        Write-Host "  ." -NoNewline -ForegroundColor Gray
                        docker info 2>&1 | Out-Null
                        if ($LASTEXITCODE -eq 0) { $dockerReady = $true }
                    }
                    Write-Host ""
                    if (-not $dockerReady) {
                        Write-Err "Docker daemon did not become ready in time — skipping vector DB containers."
                    }
                }
            }
            if ($LASTEXITCODE -eq 0) {
            Write-Info "Starting Docker vector DB containers (Qdrant, Weaviate, Milvus, PGVector) ..."
            $dockerRunning = docker compose -f $compose ps --quiet 2>&1
            if ($LASTEXITCODE -eq 0 -and $dockerRunning) {
                Write-Ok "Docker containers already running — skipping up."
            } else {
                docker compose -f $compose up -d 2>&1 | Out-Null
                if ($LASTEXITCODE -eq 0) {
                    Write-Ok "Docker containers started."
                    # Initialize PGVector extension (idempotent)
                    Start-Sleep -Seconds 3
                    docker exec polyrag_pgvector psql -U postgres -d polyrag -c "CREATE EXTENSION IF NOT EXISTS vector;" 2>&1 | Out-Null

                    # Wait for all 4 Docker backends to be ready
                    $services = @(
                        @{ Name = "Qdrant";    Port = 6333;  Max = 30 },
                        @{ Name = "Weaviate";  Port = 8088;  Max = 30 },
                        @{ Name = "PGVector";  Port = 5433;  Max = 30 },
                        @{ Name = "Milvus";    Port = 19530; Max = 60 }
                    )
                    foreach ($svc in $services) {
                        Write-Info "Waiting for $($svc.Name) to be ready on port $($svc.Port) ..."
                        $ready    = $false
                        $deadline = (Get-Date).AddSeconds($svc.Max)
                        while (-not $ready -and (Get-Date) -lt $deadline) {
                            if (Test-Port $svc.Port) {
                                $ready = $true
                            } else {
                                Start-Sleep -Seconds 3
                                Write-Host "  ." -NoNewline -ForegroundColor Gray
                            }
                        }
                        Write-Host ""
                        if ($ready) { Write-Ok "$($svc.Name) is ready." }
                        else        { Write-Warn "$($svc.Name) port $($svc.Port) not open after $($svc.Max)s — API will retry on first request." }
                    }
                } else {
                    Write-Warn "docker compose up returned non-zero — containers may already be running."
                }
            }
            } # end daemon-running check
        }
    } else {
        Write-Warn "-NoDocker mode: Milvus will use local file data\milvus_lite.db (persists on disk)."
        Write-Warn "  Qdrant/Weaviate will use in-memory mode (data lost on restart)."
    }

    # ── 2. FastAPI API server (port 8000) ────────────────────────────────────
    if (-not $NoApi) {
        # Kill any stale process saved from last run
        $oldApiPid = if ($pids.ContainsKey("api")) { [int]$pids["api"] } else { 0 }
        if ($oldApiPid -gt 0 -and (Is-ProcessRunning $oldApiPid)) {
            Write-Warn "API server already running (PID $oldApiPid) — skipping start."
        } else {
            if (Test-Port 8000) {
                Write-Warn "Port 8000 already in use — clearing stale process ..."
                Stop-Port 8000
                Wait-PortFree 8000 10
            }
            Write-Info "Starting FastAPI server on http://localhost:8000 ..."
            $apiProc = Start-Process -FilePath (Join-Path $venvPath "Scripts\uvicorn.exe") `
                -ArgumentList "api.main:app","--reload","--port","8000" `
                -WorkingDirectory $base `
                -PassThru -WindowStyle Hidden
            if ($null -ne $apiProc) {
                $pids["api"] = $apiProc.Id
                Write-Ok "API server started (PID $($apiProc.Id))."
            } else {
                Write-Err "Failed to start API server."
            }
        }
    }

    # ── 3. React frontend (port 3000) ────────────────────────────────────────
    if (-not $NoFrontend) {
        $nodeCheck = Get-Command node -ErrorAction SilentlyContinue
        if ($null -eq $nodeCheck) {
            Write-Warn "Node.js not found on PATH — skipping React UI."
        } elseif (-not (Test-Path $frontendPath)) {
            Write-Warn "frontend/ directory not found — skipping."
        } else {
            $oldFePid = if ($pids.ContainsKey("frontend")) { [int]$pids["frontend"] } else { 0 }
            if ($oldFePid -gt 0 -and (Is-ProcessRunning $oldFePid)) {
                Write-Warn "React UI already running (PID $oldFePid) — skipping start."
            } else {
                if (Test-Port 3000) {
                    Write-Warn "Port 3000 already in use — clearing stale process ..."
                    Stop-Port 3000
                    Wait-PortFree 3000 10
                }
                Write-Info "Starting React UI on http://localhost:3000 ..."
                $feProc = Start-Process -FilePath "cmd.exe" `
                    -ArgumentList "/c", "npm run dev" `
                    -WorkingDirectory $frontendPath `
                    -PassThru -WindowStyle Hidden
                if ($null -ne $feProc) {
                    # Store the cmd wrapper PID so we can kill it on stop
                    $pids["frontend"] = $feProc.Id
                    Write-Ok "React UI started (PID $($feProc.Id)) — port 3000 initialising..."
                } else {
                    Write-Err "Failed to start React UI."
                }
            }
        }
    }

    Save-Pids $pids

    # ── Wait briefly then show status ────────────────────────────────────────
    Write-Host ""
    Write-Info "Waiting for services to become ready ..."
    Start-Sleep -Seconds 4

    $apiReady = Test-Port 8000
    $feReady  = Test-Port 3000

    Write-Host ""
    Write-Host "  +-----------------------------------------------------+" -ForegroundColor White
    Write-Host "  |  SERVICE            URL                    STATUS   |" -ForegroundColor White
    Write-Host "  +-----------------------------------------------------+" -ForegroundColor White
    $apiStatus = if ($apiReady -or $NoApi) { "[OK]    " } else { "[starting]" }
    $feStatus  = if ($feReady  -or $NoFrontend) { "[OK]    " } else { "[starting]" }
    Write-Host "  |  API Server         http://localhost:8000   $($apiStatus.PadRight(10))|" -ForegroundColor White
    Write-Host "  |  React UI           http://localhost:3000   $($feStatus.PadRight(10))|" -ForegroundColor White
    Write-Host "  |  API Docs           http://localhost:8000/docs       |" -ForegroundColor White
    Write-Host "  +-----------------------------------------------------+" -ForegroundColor White
    Write-Host ""
    Write-Host "  Quick commands:" -ForegroundColor White
    Write-Host "    .\start.ps1 -Action stop      — stop all services" -ForegroundColor Gray
    Write-Host "    .\start.ps1 -Action status    — check service health" -ForegroundColor Gray
    Write-Host "    .\start.ps1 -Action restart   — restart all services" -ForegroundColor Gray
    Write-Host "    .\start.ps1 -NoDocker         — start without Docker DBs" -ForegroundColor Gray
    Write-Host "    pytest tests/ -q              — run backend tests" -ForegroundColor Gray
    Write-Host "    cd frontend; npm test         — run frontend tests" -ForegroundColor Gray
    Write-Host ""

    # Core import sanity check
    $ok = python -c "from core.store.registry import AdapterRegistry; print('ok')" 2>&1
    if ($ok -match "ok") { Write-Ok "Python core import OK" }
    else                  { Write-Err "Python core import FAILED — run .\install.ps1 to fix" }
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  ACTION: ENV  (legacy — activate venv and show quick-reference only)
# ══════════════════════════════════════════════════════════════════════════════
function Do-Env {
    Activate-Venv
    Write-Header "Phoenice-PolyRAG  —  Environment Active"
    Write-Host "  Python   : $(python --version)" -ForegroundColor Gray
    Write-Host "  Venv     : $venvPath"            -ForegroundColor Gray
    Write-Host "  WorkDir  : $base"                -ForegroundColor Gray
    Write-Host ""
    Write-Host "  .\start.ps1                 — start all services"  -ForegroundColor Gray
    Write-Host "  .\start.ps1 -Action stop    — stop all services"   -ForegroundColor Gray
    Write-Host "  .\start.ps1 -Action status  — check service health" -ForegroundColor Gray
    Write-Host ""
}

# ══════════════════════════════════════════════════════════════════════════════
#  DISPATCH
# ══════════════════════════════════════════════════════════════════════════════
switch ($Action) {
    "start"   { Do-Start   }
    "stop"    { Do-Stop    }
    "restart" { Do-Stop; Write-Info "Waiting for ports to be released..."; Wait-PortFree 8000; Wait-PortFree 3000; Do-Start }
    "status"  { Do-Status  }
    "env"     { Do-Env     }
}
