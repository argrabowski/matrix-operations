#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>

// Define matrix dimensions
const int N = 1024;
const int M = 1024;
const int P = 1024;

// Kernel to perform matrix multiplication on GPU
__global__ void matrixMultiply(float* A, float* B, float* C, int n, int m, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += A[row * m + k] * B[k * p + col];
        }
        C[row * p + col] = sum;
    }
}

// CPU-based matrix multiplication
void cpuMatrixMultiply(float* A, float* B, float* C, int n, int m, int p) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[i * m + k] * B[k * p + j];
            }
            C[i * p + j] = sum;
        }
    }
}

int main() {
    // Initialize random matrices A and B
    float* h_A = new float[N * M];
    float* h_B = new float[M * P];
    float* h_C = new float[N * P];
    float* h_C_cpu = new float[N * P];
    float* d_A, * d_B, * d_C;

    for (int i = 0; i < N * M; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * P; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate memory on GPU
    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, M * P * sizeof(float));
    cudaMalloc((void**)&d_C, N * P * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * P * sizeof(float), cudaMemcpyHostToDevice);

    // Set thread block size
    dim3 threadsPerBlock(32, 32);

    // Calculate number of thread blocks
    dim3 numBlocks((N + 31) / 32, (P + 31) / 32);

    // Measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel
    matrixMultiply<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, M, P);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU execution time: " << milliseconds << " ms" << std::endl;

    // Copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure CPU execution time
    clock_t start_cpu, end_cpu;
    start_cpu = clock();

    // Perform CPU-based matrix multiplication
    cpuMatrixMultiply(h_A, h_B, h_C_cpu, N, M, P);

    end_cpu = clock();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0;

    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;

    // Compare GPU and CPU results
    for (int i = 0; i < N * P; i++) {
        if (h_C[i] != h_C_cpu[i]) {
            std::cerr << "Error: GPU and CPU results do not match!" << std::endl;
            break;
        }
    }

    std::cout << "Matrix multiplication completed successfully." << std::endl;

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
