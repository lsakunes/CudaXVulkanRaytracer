#ifndef COLOR_HPP
#define COLOR_HPP

#include "vec3.hpp"

#include <iostream>

float clamp(float x) {
	if (x > 1) return 1;
	if (x < 0) return 0;
	return x;
}

void write_color(std::ostream& out, color pixel_color) {
	out << static_cast<int>(255.999 * sqrt(clamp(pixel_color.x()))) << ' '
		<< static_cast<int>(255.999 * sqrt(clamp(pixel_color.y()))) << ' '
		<< static_cast<int>(255.999 * sqrt(clamp(pixel_color.z()))) << '\n';
}


#endif