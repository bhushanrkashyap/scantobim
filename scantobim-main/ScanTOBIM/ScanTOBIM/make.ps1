# =============================================================================
# make.ps1 - ScanToBIM Agent (Windows PowerShell)
#
# Drop-in replacement for `make` on Windows. Usage:
#   .\make.ps1 help
#   .\make.ps1 install
#   .\make.ps1 server
#
# Requires: PowerShell 5.1+ (ships with Windows 10+), Python 3.11+ on PATH,
# and either Docker Desktop (for docker targets) or .NET 6+ SDK (for C# tests).
# =============================================================================

param(
    [Parameter(Position = 0)]
    [string]$Target = "help",
    [string]$Python = "py -3.11"       # override with -Python python
)

$ErrorActionPreference = "Stop"

$VENV    = ".venv"
$VenvBin = Join-Path $VENV "Scripts"       # Windows puts scripts under Scripts\, not bin/
$Pip     = Join-Path $VenvBin "pip.exe"
$Pytest  = Join-Path $VenvBin "pytest.exe"
$Ruff    = Join-Path $VenvBin "ruff.exe"
$Uvicorn = Join-Path $VenvBin "uvicorn.exe"
$PyExe   = Join-Path $VenvBin "python.exe"


function Show-Help {
    @"

  ScanToBIM Agent - Windows PowerShell runner
  ---------------------------------------------

  Setup
    .\make.ps1 install        Install Python deps (core + dev)
    .\make.ps1 install-ai     Install OpenAI optional dep
    .\make.ps1 env            Copy .env.example -> .env (first time)

  Run
    .\make.ps1 server         Start FastAPI agent on port 8765
    .\make.ps1 demo           Run standalone demo (no server needed)

  Quality
    .\make.ps1 test           Run pytest + coverage report
    .\make.ps1 test-fast      Run pytest without coverage
    .\make.ps1 lint           Run ruff linter
    .\make.ps1 lint-fix       Run ruff linter with auto-fix

  Docker
    .\make.ps1 docker-up      Start agent in Docker (port 8765)
    .\make.ps1 docker-down    Stop and remove containers
    .\make.ps1 docker-logs    Tail agent logs
    .\make.ps1 docker-build   Rebuild Docker image

  C# Tests
    .\make.ps1 test-csharp    Run xUnit safety gate tests

  Utilities
    .\make.ps1 clean          Remove __pycache__, .db, coverage artifacts
    .\make.ps1 check          Run lint + test in one shot
    .\make.ps1 build-sidecar  Build stb-processor.exe (needs .venv + PyInstaller)

"@
}


function Invoke-Py { & $Python $args }


function Check-Python {
    try {
        $ver = Invoke-Py --version 2>&1
        Write-Host "-> Using Python: $ver"
    } catch {
        Write-Host "ERROR: Python not found. Try one of:" -ForegroundColor Red
        Write-Host "  1. Install Python 3.11 from python.org"
        Write-Host "  2. Override: .\make.ps1 <target> -Python 'C:\Python311\python.exe'"
        exit 1
    }
}


function Do-Install {
    Check-Python
    Write-Host "-> Creating virtual environment in $VENV\..."
    & $Python -m venv $VENV
    & $Pip install --upgrade pip
    & $Pip install `
        fastapi `
        'uvicorn[standard]' `
        pydantic `
        numpy `
        scipy `
        'laspy[lazrs]' `
        python-dotenv `
        structlog `
        httpx `
        aiosqlite `
        python-multipart `
        pytest `
        pytest-asyncio `
        pytest-cov `
        ruff `
        pye57 `
        pyquaternion
    Write-Host "-> Installing Open3D (CPU variant - lighter download)..."
    try { & $Pip install open3d-cpu }
    catch { & $Pip install open3d }
    Write-Host ""
    Write-Host "[OK] Done. Activate with:" -ForegroundColor Green
    Write-Host "    .\$VENV\Scripts\Activate.ps1"
}


function Do-InstallAi {
    Do-Install
    & $Pip install 'openai>=1.50.0'
}


function Do-Env {
    if (Test-Path .env) {
        Write-Host "  .env already exists - skipping."
    } else {
        Copy-Item .env.example .env
        Write-Host "[OK] .env created from .env.example - edit AUDIT_HMAC_SECRET before use."
    }
}


function Do-Server {
    Write-Host "-> Starting ScanToBIM agent on http://localhost:8765"
    Write-Host "  Docs: http://localhost:8765/docs"
    & $Uvicorn agent.main:app --host 0.0.0.0 --port 8765 --reload --log-level info
}


function Do-Demo {
    Write-Host "-> Running standalone demo (no server required)..."
    & $PyExe demo.py
}


function Do-Test {
    Write-Host "-> Running Python tests with coverage..."
    Push-Location agent
    try {
        & "..\$Pytest" tests\ --cov=. --cov-report=term-missing `
            --cov-report=html:..\coverage-html --cov-fail-under=70 -v
    } finally { Pop-Location }
    Write-Host ""
    Write-Host "[OK] Coverage report: coverage-html\index.html"
}


function Do-TestFast {
    Write-Host "-> Running Python tests (no coverage)..."
    Push-Location agent
    try { & "..\$Pytest" tests\ -v } finally { Pop-Location }
}


function Do-TestCsharp {
    Write-Host "-> Running C# xUnit safety gate tests..."
    dotnet test revit-addin\Tests\ScanToBIM.Tests.csproj `
        --logger "console;verbosity=normal" --no-restore
    Write-Host "[OK] C# tests complete."
}


function Do-Lint {
    Write-Host "-> Running ruff linter..."
    & $Ruff check agent\ demo.py
}


function Do-LintFix {
    Write-Host "-> Running ruff linter with auto-fix..."
    & $Ruff check --fix agent\ demo.py
    & $Ruff format agent\ demo.py
}


function Do-DockerUp {
    Write-Host "-> Starting ScanToBIM stack..."
    docker compose up -d
    Write-Host ""
    Write-Host "[OK] Agent running at http://localhost:8765"
    Write-Host "  Docs:    http://localhost:8765/docs"
    Write-Host "  Health:  http://localhost:8765/health"
}


function Do-DockerDown  { docker compose down;               Write-Host "[OK] Stack stopped." }
function Do-DockerLogs  { docker compose logs -f agent }
function Do-DockerBuild { docker compose build --no-cache agent }


function Do-Check { Do-Lint; Do-Test; Write-Host ""; Write-Host "[OK] All checks passed." -ForegroundColor Green }


function Do-BuildSidecar {
    Write-Host "-> Building stb-processor.exe ..."
    if (-not (Test-Path $VENV)) {
        Write-Host "ERROR: .venv not found - run .\make.ps1 install first." -ForegroundColor Red; exit 1
    }
    & $Pip install pyinstaller | Out-Null

    # Use temporary paths outside OneDrive to avoid 'Access is denied' errors from OneDrive file locking.
    $tempDist = Join-Path $env:TEMP "stb-build-dist"
    $tempWork = Join-Path $env:TEMP "stb-build-work"

    # Ensure temporary paths are clean
    if (Test-Path $tempDist) { Remove-Item -Recurse -Force $tempDist }
    if (Test-Path $tempWork) { Remove-Item -Recurse -Force $tempWork }

    # Absolute paths to source elements to prevent relative path resolution relative to --specpath
    $toolsSrc = Join-Path $PWD "agent\tools"
    $modelsSrc = Join-Path $PWD "agent\models.py"
    $rulesSrc = Join-Path $PWD "agent\element_type_rules.json"
    $cliScript = Join-Path $PWD "agent\tools\processor_cli.py"

    & $PyExe -m PyInstaller `
        --onefile `
        --name stb-processor `
        --distpath $tempDist `
        --workpath $tempWork `
        --specpath build `
        --noconfirm --clean `
        --hidden-import=open3d `
        --hidden-import=laspy `
        --hidden-import=scipy.spatial `
        --hidden-import=scipy.spatial._qhull `
        --hidden-import=scipy.spatial.transform `
        --hidden-import=cv2 `
        --hidden-import=numpy `
        --hidden-import=pydantic `
        --hidden-import=structlog `
        --hidden-import=pye57 `
        --hidden-import=pyquaternion `
        --hidden-import=plotly `
        --hidden-import=sklearn `
        --hidden-import=sklearn.cluster `
        --hidden-import=sklearn.neighbors `
        --hidden-import=sklearn.utils._typedefs `
        --hidden-import=sklearn.neighbors._partition_nodes `
        --hidden-import=sklearn.utils._heap `
        --hidden-import=sklearn.utils._sorting `
        --hidden-import=sklearn.utils._vector_sentinel `
        --collect-all=pye57 `
        --collect-all=cv2 `
        --collect-all=sklearn `
        --exclude-module=jedi `
        --exclude-module=IPython `
        --exclude-module=matplotlib `
        --add-data "$toolsSrc;agent\tools" `
        --add-data "$modelsSrc;agent" `
        --add-data "$rulesSrc;agent" `
        $cliScript
        
    $builtExe = Join-Path $tempDist "stb-processor.exe"
    if (Test-Path $builtExe) {
        # Copy back to the repo's expected path
        $targetDir = "build\dist"
        if (-not (Test-Path $targetDir)) { New-Item -ItemType Directory -Path $targetDir -Force | Out-Null }
        $finalExe = Join-Path $targetDir "stb-processor.exe"
        Copy-Item -Path $builtExe -Destination $finalExe -Force
        
        Write-Host "[OK] Built and copied to: $finalExe" -ForegroundColor Green
        & $finalExe --version
    } else {
        Write-Host "ERROR: Build failed." -ForegroundColor Red; exit 1
    }
}


function Do-Clean {
    Write-Host "-> Cleaning build artifacts..."
    Get-ChildItem -Path . -Recurse -Force -Directory `
        | Where-Object { $_.Name -eq "__pycache__" -and $_.FullName -notmatch "\\$VENV\\" } `
        | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Path . -Recurse -Force -File -Filter "*.pyc" `
        | Where-Object { $_.FullName -notmatch "\\$VENV\\" } `
        | Remove-Item -Force -ErrorAction SilentlyContinue
    Remove-Item -Force demo_audit.db, scantobim.db, .coverage -ErrorAction SilentlyContinue
    Remove-Item -Recurse -Force coverage-html, .pytest_cache, agent\.pytest_cache -ErrorAction SilentlyContinue
    Write-Host "[OK] Clean."
}


# -- Dispatcher ----------------------------------------------------------------

switch ($Target.ToLower()) {
    "help"          { Show-Help }
    "install"       { Do-Install }
    "install-ai"    { Do-InstallAi }
    "env"           { Do-Env }
    "server"        { Do-Server }
    "demo"          { Do-Demo }
    "test"          { Do-Test }
    "test-fast"     { Do-TestFast }
    "test-csharp"   { Do-TestCsharp }
    "lint"          { Do-Lint }
    "lint-fix"      { Do-LintFix }
    "docker-up"     { Do-DockerUp }
    "docker-down"   { Do-DockerDown }
    "docker-logs"   { Do-DockerLogs }
    "docker-build"  { Do-DockerBuild }
    "check"         { Do-Check }
    "clean"         { Do-Clean }
    "build-sidecar" { Do-BuildSidecar }
    default {
        Write-Host "Unknown target: $Target" -ForegroundColor Red
        Show-Help
        exit 1
    }
}
