@echo off
echo ========================================
echo Running Gauss Noise with CUDA
echo ========================================
echo.

REM Add OpenCV to PATH
set PATH=%PATH%;C:\opencv\build\x64\vc16\bin

echo Running: gauss_noise.exe landscape.png landscape_noise.png 0.0 50.0
echo.

gauss_noise.exe landscape.png landscape_noise.png 0.0 50.0

echo.
echo ========================================
echo Exit code: %ERRORLEVEL%
if %ERRORLEVEL% EQU 0 (
    echo Success! Image saved as landscape_noise.png
) else (
    echo Error: Program failed with code %ERRORLEVEL%
)
echo ========================================
pause

