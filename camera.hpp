#ifndef CAMERAH
#define CAMERAH

#include "ray.hpp"
#include <curand_kernel.h>
#define PI 3.14159

__device__ vec3 random_in_unit_disk(curandState* localState);

class camera {
public:
	__device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist, float t0, float t1) {
		time0 = t0;
		time1 = t1;
		lens_radius = aperture / 2;
		float theta = vfov * PI / 180;
		float half_height = tan(theta / 2);
		float half_width = aspect * half_height;
		origin = lookfrom;
		w = unit_vector(lookfrom - lookat);
		u = unit_vector(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width * focus_dist * u - half_height * focus_dist * v - focus_dist * w;
		vertical = 2.0 * half_height * focus_dist * v;
		horizontal = 2.0*half_width * focus_dist * u;
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
};
#endif