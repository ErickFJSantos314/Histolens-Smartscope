@echo off
setlocal EnableDelayedExpansion

REM HistoLens one-click launcher for Windows
REM - Activates conda environment
REM - Validates required runtime variables
REM - Starts microscope digital twin client

call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate histolens
if errorlevel 1 (
  echo [ERROR] Failed to activate conda environment "histolens".
  echo         Confirm Miniconda path and environment name.
  exit /b 1
)

set MISSING_VARS=
if "%GOOGLE_API_KEY%"=="" set MISSING_VARS=!MISSING_VARS! GOOGLE_API_KEY
if "%REMOTE_VISION_API_URL%"=="" if "%COLAB_API_URL%"=="" set MISSING_VARS=!MISSING_VARS! REMOTE_VISION_API_URL
if "%GOOGLE_APPLICATION_CREDENTIALS%"=="" set MISSING_VARS=!MISSING_VARS! GOOGLE_APPLICATION_CREDENTIALS

if not "%MISSING_VARS%"=="" (
  echo [ERROR] Missing required environment variables:%MISSING_VARS%
  echo.
  echo Suggested setup:
  echo   setx GOOGLE_API_KEY "your_google_api_key"
  echo   setx REMOTE_VISION_API_URL "https://your-ngrok-url.ngrok-free.dev/analyze"
  echo   setx GOOGLE_APPLICATION_CREDENTIALS "C:\path\to\service-account.json"
  echo.
  echo Restart terminal after using setx.
  exit /b 1
)

echo [INFO] Launching HistoLens SmartScope AI...
python histolens.py

endlocal
