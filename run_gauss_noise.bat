@echo off
echo ========================================
echo Ejecutando Gauss Noise con CUDA
echo ========================================
echo.

REM Agregar OpenCV al PATH
set PATH=%PATH%;C:\opencv\build\x64\vc16\bin

echo Ejecutando: gauss_noise.exe landscape.png landscape_noise.png 0.0 50.0
echo.

gauss_noise.exe landscape.png landscape_noise.png 0.0 50.0

echo.
echo ========================================
echo Codigo de salida: %ERRORLEVEL%
if %ERRORLEVEL% EQU 0 (
    echo Exito! Imagen guardada como landscape_noise.png
) else (
    echo Error: El programa fallo con codigo %ERRORLEVEL%
)
echo ========================================
pause

