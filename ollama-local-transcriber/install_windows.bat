@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo OLLAMA LOCAL TRANSCRIBER - INSTALLER
echo ========================================
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python was not found.
    echo Install Python 3.11 or newer from https://www.python.org/downloads/
    echo Make sure to enable "Add python.exe to PATH".
    pause
    exit /b 1
)

python --version
echo.

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 goto error
) else (
    echo Virtual environment already exists.
)

call venv\Scripts\activate.bat
if errorlevel 1 goto error

echo Upgrading pip...
python -m pip install --upgrade pip
if errorlevel 1 goto error

echo Installing Python dependencies...
pip install -r requirements.txt
if errorlevel 1 goto error

echo Creating runtime folders...
if not exist temp mkdir temp
if not exist outputs mkdir outputs
if not exist logs mkdir logs

echo.
echo Checking Ollama...
where ollama >nul 2>nul
if errorlevel 1 (
    echo WARNING: Ollama was not found on PATH.
    echo Install Ollama from https://ollama.com/download/windows
) else (
    echo Ollama command found.
)

echo.
echo Checking FFmpeg...
where ffmpeg >nul 2>nul
if errorlevel 1 (
    echo WARNING: FFmpeg was not found on PATH.
    echo Install FFmpeg and add its bin folder to PATH.
) else (
    echo FFmpeg command found.
)

echo.
echo Installation complete.
echo.
echo Next steps:
echo 1. Install and start Ollama if needed.
echo 2. Install FFmpeg if the warning appeared.
echo 3. Make sure qwen3:4b is installed in Ollama, or run: ollama pull qwen3:4b
echo 4. Run the app with: run_app.bat
echo.
echo Note: Faster-Whisper downloads the selected Whisper model during first use.
echo Large models are not downloaded automatically by this installer.
pause
exit /b 0

:error
echo.
echo ERROR: Installation failed. Review the messages above.
pause
exit /b 1
