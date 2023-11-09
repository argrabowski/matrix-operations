#ifndef GPU_FUNCTIONS_H
#define GPU_FUNCTIONS_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void gpuMatMult(float* A, float* B, float* C, int n, int m, int p);
__global__ void gpuConv(float* inputImage, float* outputImage, float* kernel, int imageWidth, int imageHeight, int kernelSize);

#endif
