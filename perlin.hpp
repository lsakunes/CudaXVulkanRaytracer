#ifndef PERLINHPP
#define PERLINHPP

#include <random>
#include "vec3.hpp"
#include <random>

__device__ inline float trilinear_interp(float c[2][2][2], float u, float v, float w) {
	u = u * u * (3 - 2 * u);
	v = v * v * (3 - 2 * v);
	w = w * w * (3 - 2 * w);
	float accum = 0; 
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++)
				accum +=	(i * u + (1 - i) * (1 - u)) *
							(j * v + (1 - j) * (1 - v)) *
							(k * w + (1 - k) * (1 - w)) * c[i][j][k];
	return accum;
}
__device__ inline float perlin_interp(vec3 c[2][2][2], float u, float v, float w) {
	u = u * u * (3 - 2 * u);
	v = v * v * (3 - 2 * v);
	w = w * w * (3 - 2 * w);
	float accum = 0;
	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++)
			for (int k = 0; k < 2; k++) {
				vec3 weight_v(u - i, v - j, w - k);
				accum += (i * u + (1 - i) * (1 - u)) *
					(j * v + (1 - j) * (1 - v)) *
					(k * w + (1 - k) * (1 - w)) * dot(c[i][j][k], weight_v);
			}
	return accum;
}

class perlin {
public:
	__device__ float noise(const vec3& p, float* ranfloat, int* perm_x, int* perm_y, int* perm_z) const {
		int i = int(4 * p.x()) & 255;
		int j = int(4 * p.y()) & 255;
		int k = int(4 * p.z()) & 255;
		float retval = ranfloat[perm_x[i] ^ perm_y[j] ^ perm_z[k]];
		return retval;
	}
	__device__ float smooth_noise(const vec3& p, vec3* ranvec, int* perm_x, int* perm_y, int* perm_z) const {
		float u = p.x() - floor(p.x());
		float v = p.y() - floor(p.y());
		float w = p.z() - floor(p.z());
		int i = floor(p.x());
		int j = floor(p.y());
		int k = floor(p.z());
		vec3 c[2][2][2];
		for (int di = 0; di < 2; di++)
			for (int dj = 0; dj < 2; dj++)
				for (int dk = 0; dk < 2; dk++)
					c[di][dj][dk] = ranvec[perm_x[(i + di) & 255] ^ perm_y[(j + dj) & 255] ^ perm_z[(k + dk) & 255]];
		return perlin_interp(c, u, v, w);
	}
};

__device__ static vec3* perlin_generate(curandState localState) {
	vec3* p = new vec3[256];
	for (int i = 0; i < 256; i++)
		p[i] = unit_vector(vec3(2*curand_uniform(&localState)-1, 2 * curand_uniform(&localState) - 1, 2 * curand_uniform(&localState) - 1));
	return p;
}

__device__ void permute(int* p, int n, curandState localState) {

	for (int i = n - 1; i > 0; i--) {
		int target = int(curand_uniform(&localState) * (i + 1));
		int tmp = p[i];
		p[i] = p[target];
		p[target] = tmp;
	}
	return;
}

__device__ static int* perlin_generate_perm(curandState localState) {
	int* p = new int[256];
	for (int i = 0; i < 256; i++)
		p[i] = i;
	permute(p, 256, localState);
	return p;
}

#endif