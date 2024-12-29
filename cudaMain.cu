#include <iostream>                                                   
#include <fstream>    
#include <time.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "stdio.h"

#include "vec3.hpp"
#include "color.hpp"
#include "ray.hpp"
#include "hitable_list.hpp"
#include "sphere.hpp"
#include "triangle.hpp"
#include "sphere.hpp"
#include "camera.hpp"
#include "material.hpp"
#include "stats.hpp"

#define checkCudaErrors(val) checkCuda((val), #val, __FILE__, __LINE__)

__device__ float MAXFLOAT = 999;
__device__ int MAX_BOUNCES = 5;

__device__ int rgbaFloatToInt(float4 rgba) {
	int r = static_cast<int>(rgba.x * 255.0f);
	int g = static_cast<int>(rgba.y * 255.0f);
	int b = static_cast<int>(rgba.z * 255.0f);
	int a = static_cast<int>(rgba.w * 255.0f);

	//return (r << 24) | (g << 16) | (b << 8) | a;
	return (a << 24) | (b << 16) | (g << 8) | r; //TODO: investigate
}

int numSpheres = 5;

void checkCuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result != cudaSuccess) {
		std::cerr << "CUDA error = " << cudaGetErrorString(result) << " at " << file << ":" << line << " '" << func << " " << "' \n";
		cudaDeviceReset();
		exit(EXIT_FAILURE);
	}
}

//DO NOT TOUCH
void bar(int j,int ny){;;;;;;;;;;
;;;;std::cout<<"\r";;;;;;;;;;;;;;
;;;;std::cout<<"[";;;;;;;;;;;;;;;
;;;;float ratio=(float)j/(ny-1);;
;;;;int space=10*ratio;;;;;;;;;;;
;;;;int equals=10-space;;;;;;;;;;
;;;;for(int i=equals;i>0;i--){;;;
;;;;;;;;std::cout<<"=";;;;;;;;;;;
;;;;};;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;for(int i=space;i>0;i--){;;;;
;;;;;;;;std::cout<<" ";;;;;;;;;;;
;;;;};;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;std::cout<<"]";;;;;;;;;;;;;;;
};;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

__device__ color ray_color(const ray& r, hitable **world, curandState* local_rand_state) {
	vec3 bgColor(0.05, 0.05, 0.2);
	ray cur_ray = r;
	vec3 cur_attenuation = vec3(1.0, 1.0, 1.0);
	vec3 emitted = vec3(0, 0, 0);
	for (int i = 0; i < MAX_BOUNCES; i++) {
		hit_record rec;
		if ((*world)->hit(cur_ray, 0.001f, MAXFLOAT, rec)) {
			ray scattered;
			vec3 attenuation;
			vec3 curemitted = rec.mat_ptr->emitted(rec.u, rec.v, rec.p) * cur_attenuation;
			emitted += curemitted;
			if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
				cur_attenuation *= attenuation;
				cur_ray = scattered;
			}
			else
				return emitted;
		}
		else {
			vec3 unit_direction = unit_vector(cur_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			vec3 c = (1.0f - t) * vec3(0.1, 0.1, 0.1) + t * bgColor;
			return cur_attenuation * c + emitted;
		}
	}
	return vec3(0,0,0);
}

__global__ void render_init(int max_x, int max_y, curandState* rand_state, int seed) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curand_init(1984 * seed, pixel_index, 0, &rand_state[pixel_index]);
}

template<class Rgb>
__global__ void render(cudaSurfaceObject_t* surface, int max_x, int max_y, int samples, camera **d_cam, hitable** world, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	curandState local_rand_state = rand_state[pixel_index];
	vec3 col(0, 0, 0);
	for (int s = 0; s < samples; s++) {
		// if (pixel_index == 50000) printf("sample: %d\n", s);
		auto u = double(i + curand_uniform(&local_rand_state)) / (max_x - 1);
		auto v = double(j + curand_uniform(&local_rand_state)) / (max_y - 1);
		ray r = (*d_cam)->get_ray(u, v, &local_rand_state);
		col += ray_color(r, world, &local_rand_state);
	}
	rand_state[pixel_index] = local_rand_state;
	col = col / float(samples);
	int color = rgbaFloatToInt(float4{ col.x(), col.y(), col.z(), 1 });
	// surf2Dwrite(color, surface[0], i * sizeof(Rgb), j);
	surf2Dwrite(color, surface[0], i * sizeof(Rgb), j);
}

__global__ void create_world(hitable** d_list, hitable** d_world, camera** d_cam, int numSpheres, int nx, int ny, vec3* ranvec, int* perm_x, int* perm_y, int* perm_z, curandState* localState, unsigned char* tex_data, int texnx, int texny) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		ranvec = perlin_generate(localState[0]);
		perm_x = perlin_generate_perm(localState[1]);
		perm_y = perlin_generate_perm(localState[2]);
		perm_z = perlin_generate_perm(localState[3]);

		//BASE
		d_list[0] = new sphere(vec3(0, -100, -1), 100,
			new lambertian(new marble_texture(ranvec, perm_x, perm_y, perm_z, 30, vec3(1,1,1))));
		//d_list[1] = new sphere(vec3(0, 0.5, -1), 0.5, 
		//	new lambertian(new checker_texture(vec3(1, 0.2, 0.2), vec3(0.2, 0.2, 1))));

		d_list[1] = new sphere(vec3(0, 0.5, -1), 0.5,
			new metal(new image_texture(tex_data, texnx, texny), 0.9));
		d_list[2] = new sphere(vec3(1, 0.8, 0.3), 0.8,
			new Emit(new constant_texture(vec3(2, 2, 2))));
		d_list[3] = new triangle(vec3(0, 0, -3), vec3(2, 0, -3), vec3(1, 2, -3), 
			new metal(new constant_texture(vec3(1, 1,1)), 0));
		d_list[4] = new sphere(vec3(-0.25, 0.3, 0), 0.3,
			new dielectric(1.5));
		*d_world = new hitable_list(d_list, numSpheres);
		float R = cos(PI / 4);

		vec3 lookfrom(0, 0, 0);
		vec3 lookat(0, 0, 1);
		float dist_to_focus = 5;
		float aperture = 0.05;
		*d_cam = new camera(lookfrom, lookat, vec3(0,-1,0), 20, float(nx)/float(ny), aperture, dist_to_focus, 0, 1);
	}
}

__global__ void free_world(hitable** d_list, hitable** d_world, camera** d_cam, int numSpheres, vec3* ranvec, int* perm_x, int* perm_y, int* perm_z, unsigned char* tex_data) {
	for (int i = 0; i < numSpheres; i++) {
		delete (d_list[i])->mat_ptr;
		delete d_list[i];
	}
	delete ranvec;
	delete perm_x;
	delete perm_y;
	delete perm_z;
	delete* d_world;
	delete* d_cam;
	delete tex_data;
}

//int cmain() {
//	// hmmmm
//	// hmmmmmmmm
//
//
//	int num_pixels = nx * ny;
//	size_t fb_size = num_pixels * sizeof(color);
//
//	
//	
//	vec3* ranvec;
//	checkCudaErrors(cudaMalloc((void**)&ranvec, sizeof(vec3*)));
//	int* perm_x;
//	checkCudaErrors(cudaMalloc((void**)&perm_x, sizeof(int*)));
//	int* perm_y;
//	checkCudaErrors(cudaMalloc((void**)&perm_y, sizeof(int*)));
//	int* perm_z;
//	checkCudaErrors(cudaMalloc((void**)&perm_z, sizeof(int*)));
//
//	camera** d_cam;
//	checkCudaErrors(cudaMalloc((void**)&d_cam, sizeof(camera*)));
//	hitable** d_list;
//	checkCudaErrors(cudaMalloc((void**)& d_list, numSpheres * sizeof(hitable*)));
//	hitable** d_world;
//	checkCudaErrors(cudaMalloc((void**)& d_world, sizeof(hitable*)));
//	create_world<<<1,1>>>(d_list, d_world, d_cam, numSpheres, nx, ny, ranvec, perm_x, perm_y, perm_z, d_rand_state, d_tex_data, texnx, texny);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	color* fb;
//	checkCudaErrors(cudaMallocManaged((void**)&fb, fb_size));
//
//	color* finalFb;
//	checkCudaErrors(cudaMallocManaged((void**)&finalFb, fb_size));
//
//	render << <blocks, threads >> > (fb, nx, ny, samples, d_cam, d_world, d_rand_state);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaDeviceSynchronize());
//
//	stop = clock();
//	float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
//	std::cerr << "took " << timer_seconds << " seconds.\n";
//
//	std::ofstream file_streamRaw;
//	file_streamRaw.open("fileRaw.ppm");
//
//	file_streamRaw << "P3\n" << nx << ' ' << ny << "\n255\n";
//
//	for (int j = ny - 1; j >= 0; j--) {
//		for (int i = 0; i < nx; i++) {
//			size_t pixel_index = j * nx + i;
//			write_color(file_streamRaw, fb[pixel_index]);
//		}
//		if (j % 100 == 0)
//			bar(j, ny);
//	}
//
//	checkCudaErrors(cudaDeviceSynchronize());
//	free_world << <1, 1>> > (d_list, d_world, d_cam, numSpheres, ranvec, perm_x, perm_y, perm_z, d_tex_data);
//	checkCudaErrors(cudaGetLastError());
//	checkCudaErrors(cudaFree(d_tex_data));
//	checkCudaErrors(cudaFree(d_cam));
//	checkCudaErrors(cudaFree(d_world));
//	checkCudaErrors(cudaFree(d_list));
//	checkCudaErrors(cudaFree(d_rand_state));
//	checkCudaErrors(cudaFree(fb));
//
//	cudaDeviceReset();
//
//	return 0;
//}



