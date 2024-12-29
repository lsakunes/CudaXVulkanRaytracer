#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <cuda_runtime.h>
#include <iostream>
#include "camera.hpp"


#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

__global__ void plainUV(cudaSurfaceObject_t* surface, int nWidth, int nHeight);

__global__ void copySurfaceToBuffer(cudaSurfaceObject_t* surface, unsigned char* buffer, int width, int height);

void launchPlainUV(uint32_t height, uint32_t width, cudaStream_t stream, cudaSurfaceObject_t* surface); // Function to launch the kernel

#endif