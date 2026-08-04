@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo OLLAMA LOCAL TRANSCRIBER
echo ========================================
echo.

if not exist venv\Scripts\activate.bat (
    echo ERROR: Virtual environment not found.
    echo Run install_windows.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat
if errorlevel 1 goto error

where ollama >nul 2>nul
if errorlevel 1 (
    echo WARNING: Ollama command was not found. AI cleanup and summaries may be unavailable.
) else (
    echo Checking Ollama service...
    ollama list >nul 2>nul
    if errorlevel 1 (
        echo Trying to start Ollama...
        start "" /min ollama serve
        timeout /t 3 /nobreak >nul
    ) else (
        echo Ollama is running.
    )
)

echo Starting Streamlit...
streamlit run app.py
if errorlevel 1 goto error
exit /b 0

:error
echo.
echo ERROR: The application stopped because of an error.
pause
exit /b 1
