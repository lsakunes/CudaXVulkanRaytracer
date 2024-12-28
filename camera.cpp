#include "camera.hpp"

__device__ vec3 random_in_unit_disk(curandState* localState) {
	vec3 p;
	do {
		p = 2.0 * vec3(curand_uniform(localState), curand_uniform(localState), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}