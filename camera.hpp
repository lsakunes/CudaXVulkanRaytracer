#ifndef CAMERAH
#define CAMERAH

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>

#include "ray.hpp"
#include <curand_kernel.h>
#define PI 3.14159

__device__ vec3 random_in_unit_disk(curandState* localState) {
	vec3 p;
	do {
		p = 2.0 * vec3(curand_uniform(localState), curand_uniform(localState), 0) - vec3(1, 1, 0);
	} while (dot(p, p) >= 1.0);
	return p;
}

class camera {
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float asp, float aper, float focus, float t0, float t1) : origin(lookfrom), lookAt(lookat), vUp(vup), vFov(vfov), aspectRatio(asp), aperture(aper), focusDist(focus) {
		time0 = t0;
		time1 = t1;
		originalLookAt = lookAt;
		origin = lookfrom;
		lens_radius = aperture / 2;
		float theta = vFov * PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspectRatio * half_height;
		w = unit_vector(origin - lookAt);
		u = unit_vector(cross(vUp, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
		vertical = 2.0 * half_height * focusDist * v;
		horizontal = 2.0 * half_width * focusDist * u;
	}

	__device__ void moveCamera(glm::vec3 position, glm::vec3 vec, glm::mat3 matrix, vec3& out) {
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		float sum = 0;
		if (tid < 3) {
			sum += vec[0] * matrix[0][tid];
			sum += vec[1] * matrix[1][tid];
			sum += vec[2] * matrix[2][tid];
			out[tid] = sum;
		}
	}

	__device__ void moveCamPoints(glm::vec3 position, glm::mat3 matrix) {
		moveCamera(position, glm::vec3(originalLookAt.x(), originalLookAt.y(), originalLookAt.z()), matrix, lookAt);
		lookAt += origin;
	}


	__device__ void updateCam(glm::vec3 position, glm::mat3 matrix) {
		origin = vec3(position.x, position.y, position.z);
		moveCamPoints(position, matrix);
		lens_radius = aperture / 2;
		float theta = vFov * PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspectRatio * half_height;
		w = unit_vector(origin - lookAt);
		u = unit_vector(cross(vUp, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focusDist * u - half_height * focusDist * v - focusDist * w;
		vertical = 2.0 * half_height * focusDist * v;
		horizontal = 2.0 * half_width * focusDist * u;
	}

	__device__ ray get_ray(float s, float t, curandState* localState) {
		vec3 rd = lens_radius * random_in_unit_disk(localState);
		vec3 offset = u * rd.x() + v * rd.y();
		float time = time0 + curand_uniform(localState) * (time1 - time0);
		return ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset, time);
	}

	vec3 origin;
	vec3 horizontal;
	vec3 vertical;
	vec3 lower_left_corner;
	vec3 u, v, w;
	float time0, time1;
	float lens_radius;
	vec3 originalLookAt;
	vec3 lookAt, vUp;
	float vFov, aspectRatio, aperture, focusDist;
};
#endif