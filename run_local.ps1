<#
  Launch the recommender locally on Windows (PowerShell).

  Starts the FastAPI backend (port 8000) in a new window, then the Streamlit
  app (port 8501) in this window. Run from the project root:

      ./run_local.ps1

  The backend loads ~8 GB of models on its first request, so the first
  recommendation in the UI can take a minute.
#>
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Startup PRELOAD set. Default = none: models load on demand (click "Load" in the
# UI), which keeps memory low. Set RECSYS_MODELS=lite/all to preload a set instead.
if (-not (Test-Path env:RECSYS_MODELS)) { $env:RECSYS_MODELS = "" }
Write-Host "RECSYS_MODELS = '$($env:RECSYS_MODELS)' (empty = on-demand)" -ForegroundColor DarkGray

# Use `python -m ...` so both run under the interpreter that has scikit-surprise
# (the bare uvicorn/streamlit scripts may resolve to a different Python on PATH).
Write-Host "Starting backend  -> http://localhost:8000  (docs at /docs)" -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit", "-Command",
    "Set-Location '$root'; `$env:PYTHONIOENCODING='utf-8'; `$env:RECSYS_MODELS='$($env:RECSYS_MODELS)'; python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000"
)

Start-Sleep -Seconds 2
Write-Host "Starting Streamlit app -> http://localhost:8501" -ForegroundColor Cyan
$env:BACKEND_URL = "http://localhost:8000"
python -m streamlit run app/app.py --server.port 8501
