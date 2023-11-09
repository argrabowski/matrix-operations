#ifndef CPU_FUNCTIONS_H
#define CPU_FUNCTIONS_H

void cpuMatMult(float* A, float* B, float* C, int n, int m, int p);
void cpuConv(float* inputImage, float* outputImage, float* kernel, int imageWidth, int imageHeight, int kernelSize);

#endif
