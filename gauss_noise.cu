#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Image structure
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

// Uniform random number generator on GPU
__device__ float random_float(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (*seed) / 2147483648.0f;
}

// Box-Muller transform to generate Gaussian numbers
__device__ void box_muller(float u1, float u2, float* z0, float* z1) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * M_PI * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

// CPU version for uniform random numbers
float random_float_cpu(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (*seed) / 2147483648.0f;
}

// CPU version for Box-Muller
void box_muller_cpu(float u1, float u2, float* z0, float* z1) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * M_PI * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

// Sequential (CPU) version to add Gaussian noise
void add_gaussian_noise_sequential(
    unsigned char* image,
    int width,
    int height,
    int channels,
    float mean,
    float stddev,
    unsigned int seed_offset
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned int seed = (y * width + x) + seed_offset;
            float u1 = random_float_cpu(&seed);
            float u2 = random_float_cpu(&seed);
            
            float z0, z1;
            box_muller_cpu(u1, u2, &z0, &z1);
            float noise = mean + stddev * z0;
            
            for (int c = 0; c < channels; c++) {
                int idx = (y * width + x) * channels + c;
                float pixel_value = (float)image[idx] + noise;
                if (pixel_value < 0.0f) pixel_value = 0.0f;
                if (pixel_value > 255.0f) pixel_value = 255.0f;
                image[idx] = (unsigned char)pixel_value;
            }
        }
    }
}

// Function to get high precision time (milliseconds)
double get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
#endif
}

// Function to copy image
Image copy_image(Image* src) {
    Image dst;
    dst.width = src->width;
    dst.height = src->height;
    dst.channels = src->channels;
    int size = dst.width * dst.height * dst.channels;
    dst.data = (unsigned char*)malloc(size);
    memcpy(dst.data, src->data, size);
    return dst;
}

// CUDA kernel to add Gaussian noise
__global__ void add_gaussian_noise_kernel(
    unsigned char* image,
    int width,
    int height,
    int channels,
    float mean,
    float stddev,
    unsigned int seed_offset
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        unsigned int seed = (y * width + x) + seed_offset;
        float u1 = random_float(&seed);
        float u2 = random_float(&seed);
        
        float z0, z1;
        box_muller(u1, u2, &z0, &z1);
        float noise = mean + stddev * z0;
        
        for (int c = 0; c < channels; c++) {
            int idx = (y * width + x) * channels + c;
            float pixel_value = (float)image[idx] + noise;
            if (pixel_value < 0.0f) pixel_value = 0.0f;
            if (pixel_value > 255.0f) pixel_value = 255.0f;
            image[idx] = (unsigned char)pixel_value;
        }
    }
}

// Read image (PNG, PPM, etc.)
Image read_image(const char* filename) {
    const char* ext = strrchr(filename, '.');
    Image img;
    
    // If PNG, use OpenCV
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0 || 
                strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
                strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0)) {
        printf("Attempting to read with OpenCV...\n");
        fflush(stdout);
        
        cv::Mat cv_img = cv::imread(filename, cv::IMREAD_COLOR);
        if (cv_img.empty()) {
            printf("Error: OpenCV could not read %s\n", filename);
            printf("Verify that OpenCV DLLs are in the PATH\n");
            printf("Or that the file exists and is valid\n");
            fflush(stdout);
            exit(1);
        }
        
        printf("Image read with OpenCV: %dx%d\n", cv_img.cols, cv_img.rows);
        fflush(stdout);
        
        img.width = cv_img.cols;
        img.height = cv_img.rows;
        img.channels = 3;
        int size = img.width * img.height * img.channels;
        img.data = (unsigned char*)malloc(size);
        if (!img.data) {
            printf("Error: Could not allocate memory for image\n");
            fflush(stdout);
            exit(1);
        }
        
        // OpenCV uses BGR, convert to RGB
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                cv::Vec3b pixel = cv_img.at<cv::Vec3b>(y, x);
                int idx = (y * img.width + x) * 3;
                img.data[idx + 0] = pixel[2]; // R
                img.data[idx + 1] = pixel[1]; // G
                img.data[idx + 2] = pixel[0]; // B
            }
        }
    } else {
        // Read PPM
        FILE* file = fopen(filename, "rb");
        if (!file) {
            fprintf(stderr, "Error: Could not open %s\n", filename);
            fflush(stderr);
            exit(1);
        }
        
        char magic[3];
        fscanf(file, "%2s", magic);
        if (magic[0] != 'P' || magic[1] != '6') {
            fprintf(stderr, "Error: PPM P6 format required\n");
            fflush(stderr);
            fclose(file);
            exit(1);
        }
        
        fscanf(file, "%d %d", &img.width, &img.height);
        int max_val;
        fscanf(file, "%d", &max_val);
        fgetc(file);
        
        img.channels = 3;
        int size = img.width * img.height * img.channels;
        img.data = (unsigned char*)malloc(size);
        fread(img.data, 1, size, file);
        fclose(file);
    }
    
    return img;
}

// Write image (PNG, PPM, etc.)
void write_image(const char* filename, Image* img) {
    const char* ext = strrchr(filename, '.');
    
    // If PNG, use OpenCV
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0 ||
                strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
                strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0)) {
        printf("Writing with OpenCV...\n");
        fflush(stdout);
        
        cv::Mat cv_img(img->height, img->width, CV_8UC3);
        
        // Convert RGB to BGR for OpenCV
        for (int y = 0; y < img->height; y++) {
            for (int x = 0; x < img->width; x++) {
                int idx = (y * img->width + x) * 3;
                cv_img.at<cv::Vec3b>(y, x) = cv::Vec3b(
                    img->data[idx + 2], // B
                    img->data[idx + 1], // G
                    img->data[idx + 0]  // R
                );
            }
        }
        
        if (!cv::imwrite(filename, cv_img)) {
            printf("Error: OpenCV could not write %s\n", filename);
            printf("Verify that OpenCV DLLs are in the PATH\n");
            fflush(stdout);
            exit(1);
        }
    } else {
        // Write PPM
        FILE* file = fopen(filename, "wb");
        if (!file) {
            fprintf(stderr, "Error: Could not create %s\n", filename);
            fflush(stderr);
            exit(1);
        }
        
        fprintf(file, "P6\n%d %d\n255\n", img->width, img->height);
        fwrite(img->data, 1, img->width * img->height * img->channels, file);
        fclose(file);
    }
}

// Macro to check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    // Immediate message before any initialization
    fprintf(stdout, "=== STARTING PROGRAM ===\n");
    fprintf(stderr, "=== STARTING PROGRAM (stderr) ===\n");
    fflush(stdout);
    fflush(stderr);
    
    // Force immediate output
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    printf("Starting program...\n");
    fflush(stdout);
    
    if (argc < 3) {
        printf("Usage: %s <input> <output> [mean] [stddev]\n", argv[0]);
        printf("Supported formats: PNG, JPG, PPM\n");
        fflush(stdout);
        return 1;
    }
    
    // Verify that input file exists
    FILE* test = fopen(argv[1], "rb");
    if (!test) {
        printf("Error: Cannot open input file: %s\n", argv[1]);
        fflush(stdout);
        return 1;
    }
    fclose(test);
    
    printf("Verifying CUDA device...\n");
    fflush(stdout);
    
    // Verify CUDA device
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error getting CUDA devices: %s\n", cudaGetErrorString(err));
        printf("Error code: %d\n", err);
        fflush(stdout);
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("Error: No CUDA devices found\n");
        fflush(stdout);
        return 1;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Error getting device properties: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        return 1;
    }
    
    printf("CUDA Device: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    fflush(stdout);
    
    float mean = (argc >= 4) ? atof(argv[3]) : 0.0f;
    float stddev = (argc >= 5) ? atof(argv[4]) : 50.0f;  // More noise by default
    
    printf("Reading image: %s\n", argv[1]);
    fflush(stdout);
    
    Image img = read_image(argv[1]);
    
    printf("Image: %d x %d, Channels: %d\n", img.width, img.height, img.channels);
    printf("Applying Gaussian noise: mean=%.2f, stddev=%.2f\n", mean, stddev);
    fflush(stdout);
    
    size_t image_size = img.width * img.height * img.channels * sizeof(unsigned char);
    unsigned int seed = (unsigned int)time(NULL);
    double time_seq = 0.0, time_par = 0.0;
    double time_seq_start, time_seq_end, time_par_start, time_par_end;
    
    // ========== SEQUENTIAL VERSION (CPU) ==========
    printf("\n=== Executing SEQUENTIAL version (CPU) ===\n");
    fflush(stdout);
    
    Image img_seq = copy_image(&img);
    
    time_seq_start = get_time_ms();
    add_gaussian_noise_sequential(img_seq.data, img_seq.width, img_seq.height, 
                                  img_seq.channels, mean, stddev, seed);
    time_seq_end = get_time_ms();
    time_seq = time_seq_end - time_seq_start;
    
    printf("Sequential time: %.3f ms\n", time_seq);
    fflush(stdout);
    
    // Save sequential image
    char output_seq[256];
    strncpy(output_seq, argv[2], sizeof(output_seq) - 1);
    output_seq[sizeof(output_seq) - 1] = '\0';
    char* dot = strrchr(output_seq, '.');
    if (dot) {
        *dot = '\0';
    }
    strcat(output_seq, "_seq.png");
    write_image(output_seq, &img_seq);
    printf("Sequential image saved: %s\n", output_seq);
    fflush(stdout);
    
    // ========== PARALLEL VERSION (CUDA) ==========
    printf("\n=== Executing PARALLEL version (CUDA) ===\n");
    fflush(stdout);
    
    unsigned char* d_image;
    err = cudaMalloc(&d_image, image_size);
    if (err != cudaSuccess) {
        printf("Error allocating memory on GPU: %s\n", cudaGetErrorString(err));
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    err = cudaMemcpy(d_image, img.data, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error copying data to GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    dim3 blockSize(16, 16);
    dim3 gridSize(
        (img.width + blockSize.x - 1) / blockSize.x,
        (img.height + blockSize.y - 1) / blockSize.y
    );
    
    printf("Grid: %dx%d, Blocks: %dx%d\n", gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    fflush(stdout);
    
    time_par_start = get_time_ms();
    
    add_gaussian_noise_kernel<<<gridSize, blockSize>>>(
        d_image, img.width, img.height, img.channels, mean, stddev, seed
    );
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in kernel: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    err = cudaDeviceSynchronize();
    time_par_end = get_time_ms();
    time_par = time_par_end - time_par_start;
    
    if (err != cudaSuccess) {
        printf("Error in synchronization: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    err = cudaMemcpy(img.data, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error copying data from GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    printf("Parallel time: %.3f ms\n", time_par);
    fflush(stdout);
    
    // Save parallel image
    char output_par[256];
    strncpy(output_par, argv[2], sizeof(output_par) - 1);
    output_par[sizeof(output_par) - 1] = '\0';
    dot = strrchr(output_par, '.');
    if (dot) {
        *dot = '\0';
    }
    strcat(output_par, "_par.png");
    write_image(output_par, &img);
    printf("Parallel image saved: %s\n", output_par);
    fflush(stdout);
    
    // ========== SAVE TIMES TO CSV ==========
    FILE* csv = fopen("tiempos.csv", "w");
    if (csv) {
        fprintf(csv, "Method,Time_ms,Speedup,Width,Height,Channels,Mean,StdDev\n");
        fprintf(csv, "Sequential,%.3f,1.00,%d,%d,%d,%.2f,%.2f\n", 
                time_seq, img.width, img.height, img.channels, mean, stddev);
        double speedup = time_seq / time_par;
        fprintf(csv, "Parallel,%.3f,%.2f,%d,%d,%d,%.2f,%.2f\n", 
                time_par, speedup, img.width, img.height, img.channels, mean, stddev);
        fclose(csv);
        printf("\nTimes saved to: tiempos.csv\n");
        printf("Speedup: %.2fx\n", speedup);
    } else {
        printf("Warning: Could not create tiempos.csv\n");
    }
    fflush(stdout);
    
    // Clean up memory
    cudaFree(d_image);
    free(img.data);
    free(img_seq.data);
    
    printf("\n=== Process completed successfully! ===\n");
    fflush(stdout);
    return 0;
}
