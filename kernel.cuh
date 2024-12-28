#include <cuda_runtime.h>

#ifndef KERNEL_CUH
#define KERNEL_CUH

// Declaration of a simple kernel function
__global__ void plainUV(cudaSurfaceObject_t* surface, int nWidth, int nHeight);

__global__ void copySurfaceToBuffer(cudaSurfaceObject_t* surface, unsigned char* buffer, int width, int height);

void launchPlainUV(uint32_t height, uint32_t width, cudaStream_t stream, cudaSurfaceObject_t* surface); // Function to launch the kernel

#endif