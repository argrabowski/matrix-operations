#include "gpu_functions.h"
#include "cpu_functions.h"

#include <iostream>
#include <cstdlib>
#include <ctime>

void procMatMult(int N, int M, int P) {
    // Allocate memory for input and output matrices
    float* h_A = new float[N * M];
    float* h_B = new float[M * P];
    float* h_C = new float[N * P];
    float* h_C_cpu = new float[N * P];
    float* d_A, * d_B, * d_C;

    // Initialize input matrices with random values
    for (int i = 0; i < N * M; i++) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < M * P; i++) {
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate GPU memory for input and output matrices
    cudaMalloc((void**)&d_A, N * M * sizeof(float));
    cudaMalloc((void**)&d_B, M * P * sizeof(float));
    cudaMalloc((void**)&d_C, N * P * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_A, h_A, N * M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, M * P * sizeof(float), cudaMemcpyHostToDevice);

    // Set thread block size and calculate number of thread blocks
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((N + 31) / 32, (P + 31) / 32);

    // Measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch CUDA kernel for matrix multiplication
    gpuMatMult<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N, M, P);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print GPU execution time
    std::cout << "GPU execution time: " << milliseconds << " ms" << std::endl;

    // Copy result from GPU to CPU
    cudaMemcpy(h_C, d_C, N * P * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure CPU execution time
    clock_t start_cpu, end_cpu;
    start_cpu = clock();

    // Perform CPU-based matrix multiplication
    cpuMatMult(h_A, h_B, h_C_cpu, N, M, P);

    end_cpu = clock();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0;

    // Print CPU execution time
    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;

    // Compare GPU and CPU results
    bool match = true;
    for (int i = 0; i < N * P; i++) {
        if (h_C[i] != h_C_cpu[i]) {
            std::cerr << "Error: GPU and CPU results do not match!" << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Matrix multiplication completed successfully." << std::endl;
    }

    // Clean up memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void procConv(int imgW, int imgH, int kernel) {
    // Allocate memory for input image, output image, and kernel
    float* h_inImg = new float[imgW * imgH];
    float* h_outImg = new float[imgW * imgH];
    float* h_outImg_cpu = new float[imgW * imgH];
    float* h_kernel = new float[kernel * kernel];
    float* d_inImg, * d_outImg, * d_kernel;

    // Initialize input image and kernel with random values
    for (int i = 0; i < imgW * imgH; i++) {
        h_inImg[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < kernel * kernel; i++) {
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate GPU memory for input image, output image, and kernel
    cudaMalloc((void**)&d_inImg, imgW * imgH * sizeof(float));
    cudaMalloc((void**)&d_outImg, imgW * imgH * sizeof(float));
    cudaMalloc((void**)&d_kernel, kernel * kernel * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(d_inImg, h_inImg, imgW * imgH * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernel * kernel * sizeof(float), cudaMemcpyHostToDevice);

    // Set thread block size and calculate number of thread blocks
    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks((imgW + 31) / 32, (imgH + 31) / 32);

    // Measure GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Launch CUDA kernel for convolution
    gpuConv<<<numBlocks, threadsPerBlock>>>(d_inImg, d_outImg, d_kernel, imgW, imgH, kernel);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print GPU execution time
    std::cout << "GPU execution time: " << milliseconds << " ms" << std::endl;

    // Copy result from GPU to CPU
    cudaMemcpy(h_outImg, d_outImg, imgW * imgH * sizeof(float), cudaMemcpyDeviceToHost);

    // Measure CPU execution time for convolution
    clock_t start_cpu, end_cpu;
    start_cpu = clock();

    // Perform CPU-based convolution
    cpuConv(h_inImg, h_outImg_cpu, h_kernel, imgW, imgH, kernel);

    end_cpu = clock();
    double cpu_time = static_cast<double>(end_cpu - start_cpu) / CLOCKS_PER_SEC * 1000.0;

    // Print CPU execution time
    std::cout << "CPU execution time: " << cpu_time << " ms" << std::endl;

    // Compare GPU and CPU results
    bool match = true;
    for (int i = 0; i < imgW * imgH; i++) {
        if (std::abs(h_outImg[i] - h_outImg_cpu[i]) > 1e-5) {
            std::cerr << "Error: GPU and CPU results do not match!" << std::endl;
            match = false;
            break;
        }
    }

    if (match) {
        std::cout << "Convolution completed successfully." << std::endl;
    }

    // Clean up memory
    delete[] h_inImg;
    delete[] h_outImg;
    delete[] h_outImg_cpu;
    delete[] h_kernel;
    cudaFree(d_inImg);
    cudaFree(d_outImg);
    cudaFree(d_kernel);
}

int main() {
    // Define matrix dimensions
    const int N = 1024;
    const int M = 1024;
    const int P = 1024;

    // Process matrix multiplication test
    procMatMult(N, M, P);

    // Define image dimensions and kernel size
    const int imgW = 1024;
    const int imgH = 1024;
    const int kernel = 5;

    // Process convolution test
    procConv(imgW, imgH, kernel);

    return 0;
}
