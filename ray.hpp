#ifndef RAY_HPP
#define RAY_HPP

#include "vec3.hpp"

class ray {
public:
	__device__ ray() {}
	__device__ ray(const point3& origin, const vec3& direction, float ti = 0.0)
		: orig(origin), dir(direction), _time(ti)
	{}

	__device__ point3 origin() const { return orig; }
	__device__ vec3 direction() const { return dir; }
	__device__ float time() const { return _time; }
	__device__ point3 at(double t) const {
		return orig + t * dir;
	}


public:
	point3 orig;
	vec3 dir;
	float _time;
};


#endif