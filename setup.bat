@echo off
setlocal enabledelayedexpansion
title YuE2 - One-Click Setup (Windows)

echo =====================================================================
echo                  YuE2 One-Click Installer for Windows
echo               AI Music Generation with Symbolic Planning
echo =====================================================================
echo.

set "PYTHON_EXE="
set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_SYMLINKS=1"

:: 1. Search for existing Python in PATH, User AppData, or uv
where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    for /f "delims=" %%i in ('where python') do (
        if not defined PYTHON_EXE (
            "%%i" -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul
            if !ERRORLEVEL! equ 0 set "PYTHON_EXE=%%i"
        )
    )
)

if not defined PYTHON_EXE (
    if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    ) else if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
        set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    )
)

:: Check if uv is available
where uv >nul 2>nul
if %ERRORLEVEL% equ 0 (
    set "HAS_UV=1"
) else if exist "%USERPROFILE%\.local\bin\uv.exe" (
    set "HAS_UV=1"
    set "PATH=%USERPROFILE%\.local\bin;%PATH%"
) else (
    set "HAS_UV=0"
)

if not defined PYTHON_EXE (
    if "%HAS_UV%"=="1" (
        echo [*] Python not found on PATH, but 'uv' is available.
        echo [*] Installing Python 3.12 via uv...
        uv python install 3.12
        for /f "delims=" %%i in ('uv python find 3.12') do set "PYTHON_EXE=%%i"
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] Python 3.10+ was not found on your system!
    echo Please install Python 3.12 from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

echo [OK] Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -V
echo.

:: 2. Create or verify virtual environment (.venv)
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment in .venv...
    if "%HAS_UV%"=="1" (
        uv venv .venv --python "%PYTHON_EXE%"
    ) else (
        "%PYTHON_EXE%" -m venv .venv
    )
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo [OK] Existing virtual environment found in .venv.
)
echo.

:: 3. Install PyTorch with CUDA 12.6 and dependencies
echo [*] Installing PyTorch with CUDA 12.6 and core dependencies...
if "%HAS_UV%"=="1" (
    uv pip install "torch==2.10.0" "transformers==4.57.6" "huggingface-hub<1.0" "safetensors==0.7.0" "tiktoken==0.12.0" "numpy==2.2.6" "soundfile==0.13.1" "accelerate==1.13.0" "packaging>=24.2" "gradio>=4.0.0" "diffusers" "pillow" --extra-index-url https://download.pytorch.org/whl/cu126 --index-strategy unsafe-best-match
    uv pip install -e . --no-build-isolation
) else (
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\pip.exe" install "torch==2.10.0" --extra-index-url https://download.pytorch.org/whl/cu126
    ".venv\Scripts\pip.exe" install -e ".[ui]" "diffusers" "pillow" "huggingface-hub<1.0"
)

if %ERRORLEVEL% neq 0 (
    echo [WARNING] Some dependencies had warnings or issues. Attempting to verify...
)

echo.
echo =====================================================================
echo                     Verifying Environment
echo =====================================================================
".venv\Scripts\python.exe" -m yue2.cli doctor
echo.
echo =====================================================================
echo  Setup Complete! You can now launch YuE2 anytime using 'start.bat'.
echo =====================================================================
echo.
pause
