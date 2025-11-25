# CUDA_Gauss_Noise

A CUDA-accelerated program for adding Gaussian noise to images, with performance comparison between sequential (CPU) and parallel (GPU) implementations.

## Description

This project implements Gaussian noise generation using the Box-Muller transformation algorithm. It processes images in both sequential (CPU) and parallel (CUDA GPU) modes, allowing for direct performance comparison. The program measures execution time for both implementations and calculates speedup metrics.

### Features

- **Dual Implementation**: Both CPU (sequential) and GPU (CUDA) versions
- **Performance Metrics**: Automatic timing and speedup calculation
- **Multiple Formats**: Supports PNG, JPG, and PPM image formats
- **Configurable Noise**: Adjustable mean and standard deviation parameters
- **CSV Export**: Performance results saved to `tiempos.csv`

## Requirements

### Software

1. **CUDA Toolkit** (version 11.0 or higher recommended)
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Ensure `nvcc` is in your PATH

2. **OpenCV** (for reading/writing PNG and JPG images)
   - Download from: https://opencv.org/releases/
   - Or install with vcpkg: `vcpkg install opencv`

3. **Visual Studio** (Windows) or **GCC** (Linux)
   - Visual Studio 2022 Community or later (Windows)
   - GCC with C++14 support (Linux)

### Hardware

- NVIDIA GPU with CUDA support
- Check your GPU compute capability: https://developer.nvidia.com/cuda-gpus

## Compilation

### Windows (using build.bat)

The project includes a batch script for easy compilation:

```batch
.\build.bat
```

**Note**: Adjust the OpenCV path in `build.bat` according to your installation.

### Manual Compilation

#### Windows (with Visual Studio):

```batch
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
nvcc gauss_noise.cu -o gauss_noise.exe -arch=sm_89 -I"C:\opencv\build\include" -L"C:\opencv\build\x64\vc16\lib" -lopencv_world4120
```

#### Linux:

```bash
nvcc gauss_noise.cu -o gauss_noise \
    -I/usr/include/opencv4 \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs \
    -arch=sm_75 \
    -std=c++14
```

**Important Notes:**
- Adjust `-arch=sm_XX` according to your GPU:
  - `sm_75` = Turing (RTX 20 series)
  - `sm_86` = Ampere (RTX 30 series)
  - `sm_89` = Ada Lovelace (RTX 40 series)
- Adjust OpenCV paths according to your installation
- On Windows, you may need to use `-lopencv_world4120` instead of individual libraries

## Usage

### Basic Usage

```bash
gauss_noise.exe <input_image> <output_base> [mean] [stddev]
```

### Parameters

- `input_image`: Path to input image (PNG, JPG, or PPM)
- `output_base`: Base name for output files (extension will be added automatically)
- `mean`: (Optional) Mean of Gaussian distribution (default: 0.0)
- `stddev`: (Optional) Standard deviation of Gaussian distribution (default: 50.0)

### Examples

```bash
# Using default parameters (mean=0.0, stddev=50.0)
gauss_noise.exe landscape.png landscape_noise

# With custom noise parameters
gauss_noise.exe landscape.png landscape_noise 0.0 30.0
```

### Output Files

The program generates two output files:
- `<output_base>_seq.png`: Image processed with sequential (CPU) version
- `<output_base>_par.png`: Image processed with parallel (CUDA) version
- `tiempos.csv`: Performance metrics (execution times and speedup)

### Windows (using run_gauss_noise.bat)

```batch
.\run_gauss_noise.bat
```

**Note**: Adjust the OpenCV DLL path in `run_gauss_noise.bat` if needed.

## Performance Metrics

The program automatically measures and reports:
- Execution time for sequential (CPU) implementation
- Execution time for parallel (CUDA) implementation
- Speedup ratio (CPU time / GPU time)
- Image dimensions and noise parameters

Results are saved to `tiempos.csv` in CSV format for easy analysis.

## Algorithm

The program uses the **Box-Muller transformation** to generate Gaussian-distributed random numbers:

1. Generate two uniform random numbers (u₁, u₂) in [0, 1)
2. Apply Box-Muller transformation:
   - z₀ = √(-2·ln(u₁)) · cos(2π·u₂)
   - z₁ = √(-2·ln(u₁)) · sin(2π·u₂)
3. Scale by standard deviation and add mean: noise = mean + stddev · z₀
4. Add noise to each pixel channel (clamped to [0, 255])

## Verification

### Check CUDA Installation

```bash
nvcc --version
nvidia-smi
```

### Verify GPU

The program will automatically detect and display:
- GPU name
- Compute capability
- Grid and block dimensions used

## Troubleshooting

### Error: "nvcc is not recognized as a command"
- Ensure CUDA Toolkit is installed
- Add `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vXX.X\bin` to your PATH

### Error: "Cannot find opencv"
- Verify OpenCV is installed
- Adjust `-I` and `-L` paths in the compilation command
- On Windows, ensure OpenCV DLLs are in your PATH or in the same directory as the executable

### Error: "No CUDA-capable device"
- Verify you have an NVIDIA GPU
- Run `nvidia-smi` to check GPU status
- Ensure CUDA drivers are properly installed

### Error: "CUDA error: invalid device function"
- Your GPU compute capability doesn't match the `-arch` flag
- Check your GPU's compute capability and adjust the `-arch` parameter accordingly

## Project Structure

```
final_project/
├── gauss_noise.cu          # Main source code
├── build.bat               # Windows build script
├── run_gauss_noise.bat     # Windows execution script
├── create_test_image.bat   # Test image creation script
├── gauss_noise.exe         # Compiled executable
├── landscape.png           # Sample input image
├── landscape_noise_seq.png # Sequential output
├── landscape_noise_par.png # Parallel output
├── tiempos.csv             # Performance metrics
└── README.md               # This file
```

## License

See LICENSE file for details.

## Author

AndresS0421
