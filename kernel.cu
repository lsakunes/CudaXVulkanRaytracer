#include "kernel.cuh"
#include <iostream>
#include <device_launch_parameters.h>
//#include "camera.hpp"

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	{
		if (result != cudaSuccess) {
			std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << " " << "' \n";
			cudaDeviceReset();
			exit(EXIT_FAILURE);
		}
	}

}


union RGBA32 {
	uint32_t d;
	uchar4 v;
	struct {
		uint8_t r, g, b, a;
	} c;
};

__device__ int rgbaFloatToInt(float4 rgba) {
	int r = static_cast<int>(rgba.x * 255.0f);
	int g = static_cast<int>(rgba.y * 255.0f);
	int b = static_cast<int>(rgba.z * 255.0f);
	int a = static_cast<int>(rgba.w * 255.0f);

	//return (r << 24) | (g << 16) | (b << 8) | a;
	return (a << 24) | (b << 16) | (g << 8) | r; //TODO: investigate
}


template<class Rgb>
__global__ void copySurfaceToBuffer(cudaSurfaceObject_t* surface, unsigned char* buffer, int width, int height) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x < width && y < height) {
		Rgb pixel;
		surf2Dread(&pixel.d, surface, x * sizeof(Rgb), y);

		int idx = (y * width + x) * 4;
		buffer[idx + 0] = pixel.c.r;  
		buffer[idx + 1] = pixel.c.g;  
		buffer[idx + 2] = pixel.c.b;  
		buffer[idx + 3] = pixel.c.a;  
	}
}

template<class Rgb>
__global__ void plainUV(cudaSurfaceObject_t* surface, int nWidth, int nHeight) {
	int x = (threadIdx.x + blockIdx.x * blockDim.x);
	int y = (threadIdx.y + blockIdx.y * blockDim.y);
	if (x + 1 >= nWidth || y + 1 >= nHeight) {
		return;
	}
	float4 rgba{};
	rgba.x = (x & 0xFF) / 255.0f;
	rgba.y = (y & 0xFF) / 255.0f;
	rgba.z = 0.0f;
	rgba.w = 1.0f;
	int color = rgbaFloatToInt(rgba);
	surf2Dwrite(color, surface[0], x * sizeof(Rgb), y);

}

void launchPlainUV(uint32_t height, uint32_t width, cudaStream_t stream, cudaSurfaceObject_t* surface) {
	uint32_t idealSquareSize = 50; // ???
	int tx = ceil(width / idealSquareSize);
	int ty = ceil(height / idealSquareSize);

	dim3 blocks(width / tx + 1, height / ty + 1);
	dim3 threads(tx, ty);


	//camera** d_cam;
	//checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));

	plainUV<RGBA32> << <blocks, threads, 0, stream >> > (surface, width, height);

	//cudaDeviceSynchronize();

//	unsigned char* h_buffer = new unsigned char[width * height * 4]; // For RGBA
//	unsigned char* d_buffer;
//	cudaMalloc(&d_buffer, width * height * 4);
//	copySurfaceToBuffer<RGBA32> <<<blocks, threads, 0, stream>>>(surface, d_buffer, width, height);
//	cudaMemcpy(h_buffer, d_buffer, width * height * 4, cudaMemcpyDeviceToHost);
//
//	// Print contents to console
//	for (int y = 0; y < height; y+=10) {
//		for (int x = 0; x < width; x+=10) {
//			int idx = (y * width + x) * 4;
//			printf("Pixel(%d, %d): R=%d G=%d B=%d A=%d\n", x, y, h_buffer[idx], h_buffer[idx + 1], h_buffer[idx + 2], h_buffer[idx + 3]);
//		}
//	}
//
//	cudaDeviceSynchronize();
//	cudaFree(d_buffer);
//	delete[] h_buffer;
}