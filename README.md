# CUDA Matrix Multiplication and Convolution

This project demonstrates how to perform matrix multiplication and image convolution using CUDA (Compute Unified Device Architecture). The project includes both GPU and CPU implementations for these operations, allowing you to compare the execution times and results of the two approaches.

## Features

- Matrix multiplication on GPU and CPU
- Image convolution on GPU and CPU
- Timing measurements to compare execution times
- Error checking to verify the consistency of results between CPU and GPU computations
- Modular code structure with separate files for GPU and CPU functions

## Prerequisites

- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- C++ development environment (e.g. Visual Studio with CUDA support)
- CMake (for building the project)

## Getting Started

1. Clone the repository to your local machine.
2. Open the project in your preferred C++ development environment with CUDA support (e.g. Visual Studio with the CUDA extension).
3. Build the project. Make sure you have configured the project to use the CUDA Toolkit.
4. Run the project to perform matrix multiplication and image convolution on both GPU and CPU. The program will output execution times and check the consistency of results.

## Usage

The main functionality of the project is split into two functions: `procMatMult` and `procConv`. These functions are called from the main function, which is responsible for executing matrix multiplication and image convolution on both GPU and CPU. To customize the matrix dimensions and kernel size, you can modify the parameters passed to `procMatMult` and `procConv` in the main function.
