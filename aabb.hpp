//axis aligned bounding boxes
#ifndef AABBHPP
#define AABBHPP
#include "ray.hpp"
inline float ffmin(float a, float b) { return a < b ? a : b; }
inline float ffmax(float a, float b) { return a > b ? a : b; }
class aabb {
public:
	__device__ aabb() {}
	__device__ aabb(const vec3& a, const vec3& b) : _min(a), _max(b) {}
	__device__ vec3 min() const { return _min; }
	__device__ vec3 max()  const { return _max; }
	__device__ bool hit(const ray& r, float tmin, float tmax) const {
		for (int a = 0; a < 3; a++) {
			float invD = 1.0f / r.direction()[a];
			float t0 = (_min[a] - r.origin()[a]) * invD;
			float t1 = (_min[a] - r.origin()[a]) * invD;
			if (invD < 0.0f)
				std::swap(t0, t1);
			tmin = ffmax(t0, tmin);
			tmax = ffmin(t1, tmax);
			if (tmax <= tmin)
				return false;
		}
		return true;
	}
	vec3 _min;
	vec3 _max;
};
#endif