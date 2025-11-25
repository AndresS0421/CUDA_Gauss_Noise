# Sobel Edge Detection con CUDA

Este programa implementa la detección de bordes usando el operador Sobel en CUDA.

## Requisitos

1. **CUDA Toolkit** (versión 11.0 o superior recomendada)
   - Descargar desde: https://developer.nvidia.com/cuda-downloads
   - Asegúrate de que `nvcc` esté en tu PATH

2. **OpenCV** (para lectura/escritura de imágenes)
   - Descargar desde: https://opencv.org/releases/
   - O instalar con vcpkg: `vcpkg install opencv`

## Compilación

### Opción 1: Usar el script PowerShell (Windows)

```powershell
.\build.ps1
```

**Nota:** Ajusta la ruta de OpenCV en `build.ps1` según tu instalación.

### Opción 2: Compilar manualmente

#### En Windows (con Visual Studio):

```powershell
nvcc -o sobel_cuda.exe sobel.cu ^
    -I"C:\opencv\include" ^
    -L"C:\opencv\x64\vc15\lib" ^
    -lopencv_world450 ^
    -arch=sm_75 ^
    -std=c++14
```

#### En Linux:

```bash
nvcc -o sobel_cuda sobel.cu \
    -I/usr/include/opencv4 \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -arch=sm_75 \
    -std=c++14
```

**Notas importantes:**
- Ajusta `-arch=sm_75` según tu GPU (sm_75 = Turing, sm_86 = Ampere, etc.)
- Ajusta las rutas de OpenCV según tu instalación
- En Windows, puede que necesites usar `-lopencv_world450` en lugar de las librerías individuales

## Ejecución

```bash
./sobel_cuda.exe <imagen_entrada> <imagen_salida> [umbral]
```

**Parámetros:**
- `imagen_entrada`: Ruta a la imagen de entrada (PNG, JPG, etc.)
- `imagen_salida`: Ruta donde se guardará la imagen procesada
- `umbral`: (Opcional) Umbral para la binarización de bordes (por defecto: 100)

**Ejemplo:**

```bash
./sobel_cuda.exe lena.png lena_sobel.png 100
```

## Verificar instalación de CUDA

```bash
nvcc --version
nvidia-smi
```

## Solución de problemas

1. **Error: "nvcc no se reconoce como comando"**
   - Asegúrate de tener CUDA Toolkit instalado
   - Agrega `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin` a tu PATH

2. **Error: "No se puede encontrar opencv"**
   - Verifica que OpenCV esté instalado
   - Ajusta las rutas `-I` y `-L` en el comando de compilación

3. **Error: "No CUDA-capable device"**
   - Verifica que tengas una GPU NVIDIA compatible
   - Ejecuta `nvidia-smi` para verificar

