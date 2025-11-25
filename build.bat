@echo off
echo ========================================
echo Compiling Gauss Noise with CUDA
echo ========================================
echo.

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d C:\Projects\final_project

echo Compiling gauss_noise.cu...
nvcc .\gauss_noise.cu -o gauss_noise.exe -arch=sm_89 -I"C:\opencv\build\include" -L"C:\opencv\build\x64\vc16\lib" -lopencv_world4120

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo Compilation successful!
    echo ========================================
) else (
    echo.
    echo ========================================
    echo Compilation error
    echo ========================================
)

pause

