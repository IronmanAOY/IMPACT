@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
cd /d "%REPO_ROOT%"
if not defined IMPACT_CONDA_ENV set "IMPACT_CONDA_ENV=impact-synergy-clean"

rem Find conda even when it is not on PATH (double-click from Explorer).
set "CONDA_BIN=%IMPACT_CONDA_EXE%"
if not defined CONDA_BIN set "CONDA_BIN=%CONDA_EXE%"
if not defined CONDA_BIN for /f "delims=" %%C in ('where conda 2^>nul') do if not defined CONDA_BIN set "CONDA_BIN=%%C"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniforge3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniforge3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\miniconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\miniconda3\Scripts\conda.exe"
if not defined CONDA_BIN if exist "%USERPROFILE%\anaconda3\Scripts\conda.exe" set "CONDA_BIN=%USERPROFILE%\anaconda3\Scripts\conda.exe"
if not defined CONDA_BIN (
  echo conda was not found. Set IMPACT_CONDA_EXE to the full path of conda.exe and try again.
  pause
  exit /b 1
)

rem On Windows the dashboard can start and stop runs; pause/resume are POSIX-only.
"%CONDA_BIN%" run --no-capture-output -n "%IMPACT_CONDA_ENV%" python scripts\impact_desktop_app.py
if errorlevel 1 pause
endlocal
