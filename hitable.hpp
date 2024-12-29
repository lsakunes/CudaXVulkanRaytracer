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
	vec3 smallBox(fmin(box0.minPoint().x(), box1.minPoint().x()),
		fmin(box0.minPoint().y(), box1.minPoint().y()),
		fmin(box0.minPoint().z(), box1.minPoint().z()));
	vec3 big(fmax(box0.maxPoint().x(), box1.maxPoint().x()),
		fmax(box0.maxPoint().y(), box1.maxPoint().y()),
		fmax(box0.maxPoint().z(), box1.maxPoint().z()));
	return aabb(smallBox, big);
}
#endif