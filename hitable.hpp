#ifndef HITABLEH
#define HITABLEH

#include "ray.hpp"
#include "aabb.hpp"

class material;

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	material* mat_ptr;
	float u;
	float v;
};


class hitable {
public:
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
	__device__ virtual bool bounding_box(float t0, float t1, aabb& box) const = 0;
	material* mat_ptr;
};

__device__ aabb surrounding_box(aabb box0, aabb box1) {
	vec3 small(fmin(box0.min().x(), box1.min().x()),
		fmin(box0.min().y(), box1.min().y()),
		fmin(box0.min().z(), box1.min().z()));
	vec3 big(fmax(box0.max().x(), box1.max().x()),
		fmax(box0.max().y(), box1.max().y()),
		fmax(box0.max().z(), box1.max().z()));
	return aabb(small, big);

}
#endif