@echo off
setlocal enabledelayedexpansion
title YuE2 Studio Launcher

set "PYTHONUTF8=1"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_SYMLINKS=1"

if not exist ".venv\Scripts\python.exe" (
    echo [*] Virtual environment not found. Running setup.bat first...
    call setup.bat
    if not exist ".venv\Scripts\python.exe" (
        echo [ERROR] Setup failed to create .venv. Exiting.
        pause
        exit /b 1
    )
)

echo =====================================================================
echo                     YuE2 Music Studio Launcher
echo =====================================================================
echo.
echo Select an option:
echo   [1] Launch Gradio Web UI (Browser Interface - Recommended)
echo   [2] Generate example song via CLI
echo   [3] Run System Doctor / Hardware Check
echo   [4] Open Interactive Command Shell in .venv
echo.
set /p CHOICE="Enter choice [1-4] (default is 1): "

if "%CHOICE%"=="" set "CHOICE=1"
if "%CHOICE%"=="1" goto webui
if "%CHOICE%"=="2" goto cli_generate
if "%CHOICE%"=="3" goto cli_doctor
if "%CHOICE%"=="4" goto cli_shell

:webui
echo.
echo [*] Launching YuE2 Web UI...
".venv\Scripts\python.exe" webui.py
goto end

:cli_generate
echo.
echo [*] Generating example song via CLI into 'outputs\first-song'...
".venv\Scripts\python.exe" examples\generate.py --output outputs\first-song
goto end

:cli_doctor
echo.
echo [*] Running YuE2 System Doctor...
".venv\Scripts\python.exe" -m yue2.cli doctor
goto end

:cli_shell
echo.
echo [*] Entering virtual environment shell...
cmd /k ".venv\Scripts\activate.bat"
goto end

:end
if %ERRORLEVEL% neq 0 (
    echo.
    echo [NOTE] Process exited with code %ERRORLEVEL%.
)
pause
