#ifndef BVHHPP
#define BVHHPP
#include <cmath>
#include <curand_kernel.h>

#include "aabb.hpp"
#include "hitable.hpp"
#include "stats.hpp"

class bvh_node : public hitable {
public:
	bvh_node() {}
	bvh_node(hitable** l, int n, float time0, float time1, curandState* localState);
	virtual bool hit(const ray& r, float tmin, float tmax, hit_record& rec) const;
	virtual bool bounding_box(float t0, float t1, aabb& box) const;
	hitable* left;
	hitable* right;
	aabb box;
};

bool bvh_node::bounding_box(float t0, float t1, aabb& b) const {
	b = box;
	return true;
}

bool bvh_node::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	if (box.hit(r, t_min, t_max)) {
		hit_record left_rec, right_rec;
		bool hit_left = left->hit(r, t_min, t_max, left_rec);
		bool hit_right = right->hit(r, t_min, t_max, right_rec);
		if (hit_left && hit_right) {
			if (left_rec.t < right_rec.t)
				rec = left_rec;
			else
				rec = right_rec;
			return true;
		}
		else if (hit_right) {
			rec = right_rec;
			return true;
		}
		else if (hit_left) {
			rec = left_rec;
			return true;
		}
		else return false;
	}
	else return false;
}

bvh_node::bvh_node(hitable** l, int n, float time0, float time1, curandState* localState) {
	int axis = int(3 * curand_uniform(localState));
	if (axis == 0)
		quickSortHitables(l, n, sizeof(hitable*), box_x_compare);
	else if (axis == 1)
		quickSortHitables(l, n, sizeof(hitable*), box_y_compare);
	else
		quickSortHitables(l, n, sizeof(hitable*), box_z_compare);
	if (n == 1) {
		left = right = l[0];
	}
	else if (n == 2) {
		left = l[0];
		right = l[1];
	}
	else {
		left = new bvh_node(l, n / 2, time0, time1, localState);
		right = new bvh_node(l + n / 2, n - n / 2, time0, time1, localState);
	}
	aabb box_left, box_right;
	if (!left->bounding_box(time0, time1, box_left) || !right->bounding_box(time0, time1, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	box = surrounding_box(box_left, box_right);
}

int box_x_compare(const void* a, const void* b) {
	aabb box_left, box_right;
	hitable* ah = *(hitable**)a;
	hitable* bh = *(hitable**)b;
	if (!ah->bounding_box(0, 0, box_left) || !bh->bounding_box(0, 0, box_right))
		std::cerr << "no bounding box in bvh_node constructor\n";
	if (box_left.min().x() - box_right.min().x() < 0.0)
		return -1;
	else
		return 1;
}
int box_y_compare(const void* a, const void* b) {

}
int box_z_compare(const void* a, const void* b) {

}
#endif