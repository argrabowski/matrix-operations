#include "cpu_functions.h"

void cpuMatMult(float* A, float* B, float* C, int n, int m, int p) {
    // Iterate over rows and columns of result matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < p; j++) {
            float sum = 0.0f;

            // Perform matrix multiplication for current result element
            for (int k = 0; k < m; k++) {
                sum += A[i * m + k] * B[k * p + j];
            }

            // Store result in output matrix
            C[i * p + j] = sum;
        }
    }
}

void cpuConv(float* inputImage, float* outputImage, float* kernel, int imageWidth, int imageHeight, int kernelSize) {
    int halfKernel = kernelSize / 2;

    // Iterate over rows and columns of output image
    for (int row = 0; row < imageHeight; row++) {
        for (int col = 0; col < imageWidth; col++) {
            float sum = 0.0f;

            // Iterate over kernel elements for convolution
            for (int i = -halfKernel; i <= halfKernel; i++) {
                for (int j = -halfKernel; j <= halfKernel; j++) {
                    int imageX = col + j;
                    int imageY = row + i;

                    // Check if current position within valid range of input image
                    if (imageX >= 0 && imageX < imageWidth && imageY >= 0 && imageY < imageHeight) {
                        int kernelX = j + halfKernel;
                        int kernelY = i + halfKernel;

                        // Perform convolution operation and accumulate result
                        sum += inputImage[imageY * imageWidth + imageX] * kernel[kernelY * kernelSize + kernelX];
                    }
                }
            }

            // Store convolution result in output image
            outputImage[row * imageWidth + col] = sum;
        }
    }
}
