#ifndef DENOISERHPP
#define DENOISERHPP
#include "vec3.hpp"
#include "stats.hpp"

__device__ void getPixel(color* pix, int index, color* image, int pixel_index) {
	pix[index] = image[pixel_index];
}

__device__ void getPixel(color& pix, color* image, int pixel_index) {
	pix = image[pixel_index];
}

__device__ color clamp(vec3 col) {
	color ret(col[0], col[1], col[2]);
	if (col[0] > 1) ret[0] = 1;
	if (col[0] < 0) ret[0] = 0;
	if (col[1] > 1) ret[1] = 1;
	if (col[1] < 0) ret[1] = 0;
	if (col[2] > 1) ret[2] = 1;
	if (col[2] < 0) ret[2] = 0;
	return ret;
}

__global__ void denoise(color* raw, color* denoised, int max_x, int max_y, int windowSize, const uint3 threadIdx, const uint3 blockIdx, const uint3 blockDim) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i <= windowSize) || (j <= windowSize)) return;
	if ((i >= max_x - windowSize) || (j >= max_y - windowSize)) return;
	int pixel_index = j * max_x + i;
	color* pool = new color[windowSize * windowSize];
	for (int x = 0; x < windowSize; x++) {
		for (int y = 0; y < windowSize; y++) {
			getPixel(pool, x * windowSize + y, raw, (pixel_index + (max_x * (x - windowSize) + (y - windowSize / 2))));
		}
	}
	sortVecs(pool, windowSize * windowSize);
	denoised[pixel_index] = pool[(windowSize * windowSize) / 2];
	printf("wrote to denoised\n");
}


#endif