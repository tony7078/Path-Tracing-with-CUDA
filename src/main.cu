#include <iostream>
#include <time.h>
#include <float.h>
#include <curand_kernel.h>
#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "hittable_list.h"
#include "camera.h"
#include "material.h"
#include "interval.h"
#include <device_launch_parameters.h>
#include <time.h>

#define RND (curand_uniform(&local_rand_state))                                 // Range [0,1)
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )      // In order to check errors from CUDA API call results

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
        
        // CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

// Initialize cuRAND state for a single thread
__global__ void rand_init(curandState* rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);            // Custom seed : 1984
    }
}

// Initialize cuRAND state for each pixel, ensuring a unique random sequence for each pixel
__global__ void render_init(int max_x, int max_y, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // range [0, 1200]
    int j = threadIdx.y + blockIdx.y * blockDim.y;      // range [0, 679]

    if ((i >= max_x) || (j >= max_y)) return;

    int pixel_index = j * max_x + i;
    curand_init(1984 + pixel_index, 0, 0, &rand_state[pixel_index]);    // Each thread gets different seed number (Each ramdom number generation pattern must be independent per each thread)
}


// Calculate multi-sampled pixel-based color and storage in the frame buffer
__global__ void render(vec3* fb, int max_x, int max_y, int samples_per_pixel, camera** d_camera, hittable** d_world, curandState* rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // Range [0, 1200]
    int j = threadIdx.y + blockIdx.y * blockDim.y;      // Range [0, 679]

    if ((i >= max_x) || (j >= max_y)) return;           // Maximun range check according to the screen resolution

    // 2D image-to-1D image transform
    int pixel_index = j * max_x + i;
    curandState local_rand_state = rand_state[pixel_index];

    color col(0, 0, 0);
    for (int s = 0; s < samples_per_pixel; s++) {
        // Sampling 
        // float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
        // float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

        ray r = (*d_camera)->get_ray(i, j, &local_rand_state);
        col += ray_color(r, d_world, &local_rand_state);
    }

    rand_state[pixel_index] = local_rand_state;

    // Calculate avg value of pixel
    col /= float(samples_per_pixel);         

    // Gamma transform
    col[0] = linear_to_gamma(col[0]);       
    col[1] = linear_to_gamma(col[1]);     
    col[2] = linear_to_gamma(col[2]);  

    fb[pixel_index] = col;
}

__global__ void create_world(hittable** objects, hittable** d_world, camera** d_camera, int image_width, int image_height, curandState* rand_state, int num_hittables) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int i = 0;

        // Sphere Setup
        curandState local_rand_state = *rand_state;
        objects[0] = new sphere(point3(0, -1000.0, -1), 1000, new lambertian(color(0.5, 0.5, 0.5)));
  
        for (int a = -11; a < 11; a++) {
            for (int b = -11; b < 11; b++) {
                float choose_mat = RND;
                point3 center(a + 0.9 * RND, 0.2, b + 0.9 * RND);

                if (choose_mat < 0.8) {
                    // diffuse
                    objects[++i] = new sphere(center, 0.2, new lambertian(color(RND * RND, RND * RND, RND * RND)));
                }
                else if (choose_mat < 0.95) {
                    // metal
                    objects[++i] = new sphere(center, 0.2, new metal(color(0.5f * (1.0f + RND), 0.5f * (1.0f + RND), 0.5f * (1.0f + RND)), 0.5f * RND));
                }
                else {
                    // glass
                    objects[++i] = new sphere(center, 0.2, new dielectric(1.5));
                }
            }
        }

        objects[++i] = new sphere(point3(0, 1, 0), 1.0, new dielectric(1.5));
        objects[++i] = new sphere(point3(-4, 1, 0), 1.0, new lambertian(color(0.4, 0.2, 0.1)));
        objects[++i] = new sphere(point3(4, 1, 0), 1.0, new metal(color(0.7, 0.6, 0.5), 0.0));

        *rand_state = local_rand_state;
        *d_world = new hittable_list(objects, num_hittables);

        // Camera Setup
        vec3 lookfrom(13, 2, 3);
        vec3 lookat(0, 0, 0);
        vec3 vup(0, 1, 0),
        float dist_to_focus = 10.0;
        float aperture = 0.1;
        float vfov = 20.0f;
        float aspect = float(image_width) / float(image_height);

        *d_camera = new camera(lookfrom, lookat, vup, vfov, aspect, dist_to_focus, image_width, image_height);
    }
}

__global__ void free_world(hittable** d_object_list, hittable** d_world, camera** d_camera, int num_hittables) {
    for (int i = 0; i < num_hittables; i++) {
        delete ((sphere*)d_object_list[i])->mat_ptr;
        delete d_object_list[i];
    }
    delete* d_world;
    delete* d_camera;
}


int main() {
    // Resolution Setup
    int image_width = 1200; 
    int image_height = 675;
    int sample_per_pixel = 500;                  // Sample number of each pixel.
    int tx = 8;                                  // Thread x dimension
    int ty = 8;                                  // Thread y dimension

    int num_pixels = image_width * image_height;
    size_t fb_size = num_pixels * sizeof(vec3);


    // Allocate Frame Buffer 1 (in GPU memory)
    vec3* frame_buffer_1;
    checkCudaErrors(cudaMalloc((void**)&frame_buffer_1, fb_size));


    // Allocate random states
    curandState* d_rand_state_1;                // For rendering        
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_1, num_pixels * sizeof(curandState)));
    curandState* d_rand_state_2;                // For world creation
    checkCudaErrors(cudaMalloc((void**)&d_rand_state_2, 1 * sizeof(curandState)));

    rand_init << <1, 1 >> > (d_rand_state_2);   // 2nd random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Scene Setup
    hittable** d_object_list;
    int num_hittables = 22 * 22 + 1 + 3;         // Total object number
    checkCudaErrors(cudaMalloc((void**)&d_object_list, num_hittables * sizeof(hittable*)));

    hittable** d_world;
    checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable*)));

    camera** d_camera;
    checkCudaErrors(cudaMalloc((void**)&d_camera, sizeof(camera*)));

    create_world << <1, 1 >> > (d_object_list, d_world, d_camera, image_width, image_height, d_rand_state_2, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());


    // Performance measurement Setup
    clock_t start, stop;
    start = clock();


    // Render our buffer
    dim3 blocks(image_width / tx + 1, image_height / ty + 1);
    dim3 threads(tx, ty);

    render_init << <blocks, threads >> > (image_width, image_height, d_rand_state_1);   // 1st random state initialization
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    render << <blocks, threads >> > (frame_buffer_1, image_width, image_height, sample_per_pixel, d_camera, d_world, d_rand_state_1);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Allocate Frame Buffer 2 (in CPU memory)
    vec3* frame_buffer_2 = (vec3*)malloc(fb_size);
    checkCudaErrors(cudaMemcpy(frame_buffer_2, frame_buffer_1, fb_size, cudaMemcpyDeviceToHost));

    // Check time
    stop = clock();
    float timer_seconds = ((float)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "Took " << timer_seconds << " seconds.\n";


    // Directly making a PPM image
    FILE* f = fopen("image.ppm", "w");
    std::fprintf(f, "P3\n%d %d\n%d\n", image_width, image_height, 255);
    static const interval intensity(0.000, 0.999);
    for (int j = 0; j < image_height; j++) {
        for (int i = 0; i < image_width; i++) {
            size_t pixel_index = j * image_width + i;
            auto pixel_color = frame_buffer_2[pixel_index];
            auto r = pixel_color.x();
            auto g = pixel_color.y();
            auto b = pixel_color.z();

            int rbyte = int(256 * intensity.clamp(r));
            int gbyte = int(256 * intensity.clamp(g));
            int bbyte = int(256 * intensity.clamp(b));

            std::fprintf(f, "%d %d %d ", int(rbyte), int(gbyte), int(bbyte));
        }
    }
    std::clog << "\rDone.                 \n";

    // Freedom for all resources
    checkCudaErrors(cudaDeviceSynchronize());
    free_world << <1, 1 >> > (d_object_list, d_world, d_camera, num_hittables);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaFree(d_camera));
    checkCudaErrors(cudaFree(d_world));
    checkCudaErrors(cudaFree(d_object_list));
    checkCudaErrors(cudaFree(d_rand_state_1));
    checkCudaErrors(cudaFree(d_rand_state_2));
    checkCudaErrors(cudaFree(frame_buffer_1));
    free(frame_buffer_2);

    cudaDeviceReset();
}