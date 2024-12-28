#ifndef TRIANGLEHPP
#define TRIANGLEHPP

#include "hitable.hpp"
#ifndef EPSILON
#define EPSILON 0.000001
#endif

//Cross product bitch
//idk which way it's facing
__device__ vec3 get_triangle_normal(vec3 p1, vec3 p2, vec3 p3) {
	vec3 A = p2 - p1;
	vec3 B = p3 - p1;
	return cross(A, B);
}

class triangle : public hitable {
public:
	__device__ triangle() {};
	__device__ triangle(vec3 p1, vec3 p2, vec3 p3, material* m) : point1(p1), point2(p2), point3(p3), mat_ptr(m) {
		center = vec3(0, 0, 0);
		normal = get_triangle_normal(p1, p2, p3);
	};
	__device__ virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	__device__ bool bounding_box(float t0, float t1, aabb& box) const {
		bounding_box(box);
	}
	__device__ virtual bool bounding_box(aabb& box) const;
	vec3 point1, point2, point3;
	vec3 normal;
	vec3 center;
	material* mat_ptr;
};


// P = ray.orig + rec.t*ray.dir = p1 + u(p2 - p1) + v(p3-p1);
// ray.orig - p1 = -rec.t*ray.dir + u(p2 - p1) + v(p3-p1);
// investigate further later
__device__ bool get_triangle_uv(ray r, vec3 p1, vec3 p2, vec3 p3, float t_min, float t_max, float& u, float& v, float& t) {
	vec3 A = p2 - p1;
	vec3 B = p3 - p1;

	//yeah
	vec3 ray_cross_B = cross(r.dir, B);
	float det = dot(A, ray_cross_B);
	if (det > -EPSILON && det < EPSILON) 
		return false;

	//hm
	float inv_det = 1.0 / det;
	vec3 s = r.orig - p1;
	u = inv_det * dot(s, ray_cross_B);
	if (u < 0 || u > 1) 
		return false;

	//?
	vec3 s_cross_A = cross(s, A);
	v = inv_det * dot(r.dir, s_cross_A);

	if (v < 0 || u + v > 1)
		return false;

	t = inv_det * dot(B, s_cross_A);
	if (t > t_min && t < t_max)
		return true;
	else
		return false;
}

__device__ bool triangle::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	if (!get_triangle_uv(r, point1, point2, point3, t_min, t_max, rec.u, rec.v, rec.t)) return false;
	rec.p = r.at(rec.t);
	rec.normal = normal;
	rec.mat_ptr = mat_ptr;
	return true;
}

__device__ vec3 minVec(vec3 p1, vec3 p2, vec3 p3) {
	float x = p1.x();
	x = p2.x() < x ? x : p2.x();
	x = p3.x() < x ? x : p3.x();
	float y = p1.y();
	y = p2.y() < y ? y : p2.y();
	y = p3.y() < y ? y : p3.y();
	float z = p1.z();
	z = p2.z() < z ? z : p2.z();
	z = p3.z() < z ? z : p3.z();
}
__device__ vec3 maxVec(vec3 p1, vec3 p2, vec3 p3) {
	float x = p1.x();
	x = p2.x() > x ? x : p2.x();
	x = p3.x() > x ? x : p3.x();
	float y = p1.y();
	y = p2.y() > y ? y : p2.y();
	y = p3.y() > y ? y : p3.y();
	float z = p1.z();
	z = p2.z() > z ? z : p2.z();
	z = p3.z() > z ? z : p3.z();
}

__device__ bool triangle::bounding_box(aabb& box) const {
	box = aabb(minVec(point1, point2, point3), maxVec(point1, point2, point3));
	return true;
}
#endif