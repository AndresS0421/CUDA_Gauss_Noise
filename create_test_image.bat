@echo off
echo Creando imagen de prueba...

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"

cd /d C:\Projects\final_project

REM Compilar el generador
cl /EHsc create_test_image.cpp /I"C:\opencv\build\include" /link /LIBPATH:"C:\opencv\build\x64\vc16\lib" opencv_world4120.lib /OUT:create_test_image.exe

if %ERRORLEVEL% EQU 0 (
    echo Ejecutando generador...
    create_test_image.exe
    del create_test_image.exe
    del create_test_image.obj
    echo.
    echo Imagen de prueba creada: test_image.png
) else (
    echo Error al compilar el generador
)

pause

