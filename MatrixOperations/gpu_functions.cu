#include "gpu_functions.h"

__global__ void gpuMatMult(float* A, float* B, float* C, int n, int m, int p) {
    // Calculate row and column indices for current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread within valid range of result matrix
    if (row < n && col < p) {
        float sum = 0.0f;

        // Perform matrix multiplication for row and column
        for (int k = 0; k < m; k++) {
            sum += A[row * m + k] * B[k * p + col];
        }

        // Store result in output matrix
        C[row * p + col] = sum;
    }
}

__global__ void gpuConv(float* inImg, float* outImg, float* kernel, int imgW, int imgH, int kSize) {
    // Calculate row and column indices for current thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if thread within valid range of output image
    if (row < imgH && col < imgW) {
        float sum = 0.0f;
        int kHalf = kSize / 2;

        // Perform convolution for pixel
        for (int i = -kHalf; i <= kHalf; i++) {
            for (int j = -kHalf; j <= kHalf; j++) {
                int imageX = col + j;
                int imageY = row + i;

                // Check if current position within valid range of input image
                if (imageX >= 0 && imageX < imgW && imageY >= 0 && imageY < imgH) {
                    int kernelX = j + kHalf;
                    int kernelY = i + kHalf;

                    // Perform convolution operation and accumulate result
                    sum += inImg[imageY * imgW + imageX] * kernel[kernelY * kSize + kernelX];
                }
            }
        }

        // Store convolution result in output image
        outImg[row * imgW + col] = sum;
    }
}
