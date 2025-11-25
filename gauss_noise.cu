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

// Estructura para imagen
typedef struct {
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image;

// Generador de números aleatorios uniformes en GPU
__device__ float random_float(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (*seed) / 2147483648.0f;
}

// Transformada Box-Muller para generar números gaussianos
__device__ void box_muller(float u1, float u2, float* z0, float* z1) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * M_PI * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

// Versión CPU para números aleatorios uniformes
float random_float_cpu(unsigned int* seed) {
    *seed = (*seed * 1103515245 + 12345) & 0x7fffffff;
    return (*seed) / 2147483648.0f;
}

// Versión CPU para Box-Muller
void box_muller_cpu(float u1, float u2, float* z0, float* z1) {
    float r = sqrtf(-2.0f * logf(u1 + 1e-10f));
    float theta = 2.0f * M_PI * u2;
    *z0 = r * cosf(theta);
    *z1 = r * sinf(theta);
}

// Versión secuencial (CPU) para agregar ruido gaussiano
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

// Función para obtener tiempo de alta precisión (milisegundos)
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

// Función para copiar imagen
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

// Kernel CUDA para agregar ruido gaussiano
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

// Leer imagen (PNG, PPM, etc.)
Image read_image(const char* filename) {
    const char* ext = strrchr(filename, '.');
    Image img;
    
    // Si es PNG, usar OpenCV
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0 || 
                strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
                strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0)) {
        printf("Intentando leer con OpenCV...\n");
        fflush(stdout);
        
        cv::Mat cv_img = cv::imread(filename, cv::IMREAD_COLOR);
        if (cv_img.empty()) {
            printf("Error: OpenCV no pudo leer %s\n", filename);
            printf("Verifica que las DLLs de OpenCV esten en el PATH\n");
            printf("O que el archivo exista y sea valido\n");
            fflush(stdout);
            exit(1);
        }
        
        printf("Imagen leida con OpenCV: %dx%d\n", cv_img.cols, cv_img.rows);
        fflush(stdout);
        
        img.width = cv_img.cols;
        img.height = cv_img.rows;
        img.channels = 3;
        int size = img.width * img.height * img.channels;
        img.data = (unsigned char*)malloc(size);
        if (!img.data) {
            printf("Error: No se pudo asignar memoria para la imagen\n");
            fflush(stdout);
            exit(1);
        }
        
        // OpenCV usa BGR, convertir a RGB
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
        // Leer PPM
        FILE* file = fopen(filename, "rb");
        if (!file) {
            fprintf(stderr, "Error: No se pudo abrir %s\n", filename);
            fflush(stderr);
            exit(1);
        }
        
        char magic[3];
        fscanf(file, "%2s", magic);
        if (magic[0] != 'P' || magic[1] != '6') {
            fprintf(stderr, "Error: Se requiere formato PPM P6\n");
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

// Escribir imagen (PNG, PPM, etc.)
void write_image(const char* filename, Image* img) {
    const char* ext = strrchr(filename, '.');
    
    // Si es PNG, usar OpenCV
    if (ext && (strcmp(ext, ".png") == 0 || strcmp(ext, ".PNG") == 0 ||
                strcmp(ext, ".jpg") == 0 || strcmp(ext, ".JPG") == 0 ||
                strcmp(ext, ".jpeg") == 0 || strcmp(ext, ".JPEG") == 0)) {
        printf("Escribiendo con OpenCV...\n");
        fflush(stdout);
        
        cv::Mat cv_img(img->height, img->width, CV_8UC3);
        
        // Convertir RGB a BGR para OpenCV
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
            printf("Error: OpenCV no pudo escribir %s\n", filename);
            printf("Verifica que las DLLs de OpenCV esten en el PATH\n");
            fflush(stdout);
            exit(1);
        }
    } else {
        // Escribir PPM
        FILE* file = fopen(filename, "wb");
        if (!file) {
            fprintf(stderr, "Error: No se pudo crear %s\n", filename);
            fflush(stderr);
            exit(1);
        }
        
        fprintf(file, "P6\n%d %d\n255\n", img->width, img->height);
        fwrite(img->data, 1, img->width * img->height * img->channels, file);
        fclose(file);
    }
}

// Macro para verificar errores de CUDA
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "Error CUDA en %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            fflush(stderr); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char* argv[]) {
    // Mensaje inmediato antes de cualquier inicialización
    fprintf(stdout, "=== INICIANDO PROGRAMA ===\n");
    fprintf(stderr, "=== INICIANDO PROGRAMA (stderr) ===\n");
    fflush(stdout);
    fflush(stderr);
    
    // Forzar salida inmediata
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    
    printf("Iniciando programa...\n");
    fflush(stdout);
    
    if (argc < 3) {
        printf("Uso: %s <entrada> <salida> [media] [desviacion]\n", argv[0]);
        printf("Formatos soportados: PNG, JPG, PPM\n");
        fflush(stdout);
        return 1;
    }
    
    // Verificar que el archivo de entrada existe
    FILE* test = fopen(argv[1], "rb");
    if (!test) {
        printf("Error: No se puede abrir el archivo de entrada: %s\n", argv[1]);
        fflush(stdout);
        return 1;
    }
    fclose(test);
    
    printf("Verificando dispositivo CUDA...\n");
    fflush(stdout);
    
    // Verificar dispositivo CUDA
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        printf("Error al obtener dispositivos CUDA: %s\n", cudaGetErrorString(err));
        printf("Codigo de error: %d\n", err);
        fflush(stdout);
        return 1;
    }
    
    if (deviceCount == 0) {
        printf("Error: No se encontraron dispositivos CUDA\n");
        fflush(stdout);
        return 1;
    }
    
    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        printf("Error al obtener propiedades del dispositivo: %s\n", cudaGetErrorString(err));
        fflush(stdout);
        return 1;
    }
    
    printf("Dispositivo CUDA: %s\n", prop.name);
    printf("Capacidad de computo: %d.%d\n", prop.major, prop.minor);
    fflush(stdout);
    
    float mean = (argc >= 4) ? atof(argv[3]) : 0.0f;
    float stddev = (argc >= 5) ? atof(argv[4]) : 50.0f;  // Más ruido por defecto
    
    printf("Leyendo imagen: %s\n", argv[1]);
    fflush(stdout);
    
    Image img = read_image(argv[1]);
    
    printf("Imagen: %d x %d, Canales: %d\n", img.width, img.height, img.channels);
    printf("Aplicando ruido gaussiano: media=%.2f, desviacion=%.2f\n", mean, stddev);
    fflush(stdout);
    
    size_t image_size = img.width * img.height * img.channels * sizeof(unsigned char);
    unsigned int seed = (unsigned int)time(NULL);
    double time_seq = 0.0, time_par = 0.0;
    double time_seq_start, time_seq_end, time_par_start, time_par_end;
    
    // ========== VERSIÓN SECUENCIAL (CPU) ==========
    printf("\n=== Ejecutando version SECUENCIAL (CPU) ===\n");
    fflush(stdout);
    
    Image img_seq = copy_image(&img);
    
    time_seq_start = get_time_ms();
    add_gaussian_noise_sequential(img_seq.data, img_seq.width, img_seq.height, 
                                  img_seq.channels, mean, stddev, seed);
    time_seq_end = get_time_ms();
    time_seq = time_seq_end - time_seq_start;
    
    printf("Tiempo secuencial: %.3f ms\n", time_seq);
    fflush(stdout);
    
    // Guardar imagen secuencial
    char output_seq[256];
    strncpy(output_seq, argv[2], sizeof(output_seq) - 1);
    output_seq[sizeof(output_seq) - 1] = '\0';
    char* dot = strrchr(output_seq, '.');
    if (dot) {
        *dot = '\0';
    }
    strcat(output_seq, "_seq.png");
    write_image(output_seq, &img_seq);
    printf("Imagen secuencial guardada: %s\n", output_seq);
    fflush(stdout);
    
    // ========== VERSIÓN PARALELA (CUDA) ==========
    printf("\n=== Ejecutando version PARALELA (CUDA) ===\n");
    fflush(stdout);
    
    unsigned char* d_image;
    err = cudaMalloc(&d_image, image_size);
    if (err != cudaSuccess) {
        printf("Error al asignar memoria en GPU: %s\n", cudaGetErrorString(err));
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    err = cudaMemcpy(d_image, img.data, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        printf("Error al copiar datos a GPU: %s\n", cudaGetErrorString(err));
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
        printf("Error en kernel: %s\n", cudaGetErrorString(err));
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
        printf("Error en sincronizacion: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    err = cudaMemcpy(img.data, d_image, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        printf("Error al copiar datos de GPU: %s\n", cudaGetErrorString(err));
        cudaFree(d_image);
        free(img.data);
        free(img_seq.data);
        fflush(stdout);
        return 1;
    }
    
    printf("Tiempo paralelo: %.3f ms\n", time_par);
    fflush(stdout);
    
    // Guardar imagen paralela
    char output_par[256];
    strncpy(output_par, argv[2], sizeof(output_par) - 1);
    output_par[sizeof(output_par) - 1] = '\0';
    dot = strrchr(output_par, '.');
    if (dot) {
        *dot = '\0';
    }
    strcat(output_par, "_par.png");
    write_image(output_par, &img);
    printf("Imagen paralela guardada: %s\n", output_par);
    fflush(stdout);
    
    // ========== GUARDAR TIEMPOS EN CSV ==========
    FILE* csv = fopen("tiempos.csv", "w");
    if (csv) {
        fprintf(csv, "Metodo,Tiempo_ms,Speedup,Ancho,Alto,Canales,Media,Desviacion\n");
        fprintf(csv, "Secuencial,%.3f,1.00,%d,%d,%d,%.2f,%.2f\n", 
                time_seq, img.width, img.height, img.channels, mean, stddev);
        double speedup = time_seq / time_par;
        fprintf(csv, "Paralelo,%.3f,%.2f,%d,%d,%d,%.2f,%.2f\n", 
                time_par, speedup, img.width, img.height, img.channels, mean, stddev);
        fclose(csv);
        printf("\nTiempos guardados en: tiempos.csv\n");
        printf("Speedup: %.2fx\n", speedup);
    } else {
        printf("Advertencia: No se pudo crear tiempos.csv\n");
    }
    fflush(stdout);
    
    // Limpiar memoria
    cudaFree(d_image);
    free(img.data);
    free(img_seq.data);
    
    printf("\n=== Proceso completado exitosamente! ===\n");
    fflush(stdout);
    return 0;
}
