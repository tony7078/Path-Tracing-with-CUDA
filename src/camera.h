#pragma once
#include "material.h"

class camera {
public:
    __device__ camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height) { 
        initialize(lookfrom, lookat, vup, vfov, aspect, focus_dist, image_width, image_height);
    }
    __device__ void initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height);
    __device__ ray get_ray(float i, float j, curandState* local_rand_state);
    __device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state);

    int     samples_per_pixel = 10;      // Count of random samples for each pixel
    float   defocus_angle = 0;           // Variation angle of rays through each pixel
    float   focus_dist = 10;             // Distance from camera lookfrom point to plane of perfect foc
    point3  center;                      // Camera center
    point3  pixel00_loc;                 // Location of pixel 0, 0
    vec3    pixel_delta_u;               // Offset to pixel to the right
    vec3    pixel_delta_v;               // Offset to pixel below
    vec3    u, v, w;                     // Camera frame basis vectors
    vec3    defocus_disk_u;              // Defocus disk horizontal radius
    vec3    defocus_disk_v;              // Defocus disk vertical radius

};

__device__ void camera::initialize(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect, float focus_dist, int image_width, int image_height){
    // Initialize
    defocus_angle = 0.6;
    center = lookfrom;

    // Determine viewport dimensions.
    float theta = degrees_to_radians(vfov);
    float h = tan(theta / 2.0f);
    float viewport_height = 2 * h * focus_dist;
    float viewport_width = viewport_height * aspect;

    // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
    w = unit_vector(lookfrom - lookat);
    u = unit_vector(cross(vup, w));
    v = cross(w, u);

    // Calculate the vectors across the horizontal and down the vertical viewport edges.
    vec3 viewport_u = viewport_width * u;       // Vector across viewport horizontal edge
    vec3 viewport_v = viewport_height * -v;     // Vector down viewport vertical edge

    // Calculate the horizontal and vertical delta vectors from pixel to pixel.
    pixel_delta_u = viewport_u / image_width;
    pixel_delta_v = viewport_v / image_height;

    // Calculate the location of the upper left pixel.
    vec3 viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
    pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

    // Calculate the camera defocus disk basis vectors.
    float defocus_radius = focus_dist * tan(degrees_to_radians(defocus_angle / 2));
    defocus_disk_u = u * defocus_radius;
    defocus_disk_v = v * defocus_radius;
}

__device__ ray camera::get_ray(float i, float j, curandState* local_rand_state) {
    // Range transform
    //i *= 1200.f;
    //j *= 675.0f;

    // Sampling 
    vec3 offset = sample_square(local_rand_state);
    vec3 pixel_sample = pixel00_loc
        + ((i + offset.x()) * pixel_delta_u)
        + ((j + offset.y()) * pixel_delta_v);

    point3 ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(center, defocus_disk_u, defocus_disk_v, local_rand_state);
    vec3 ray_direction = pixel_sample - ray_origin;

    return ray(ray_origin, ray_direction);
}

__device__ color ray_color(const ray& r, hittable** world, curandState* local_rand_state) {
    int maxDepth = 50;      // Ray tree maximum depth

    ray cur_ray = r;
    color cur_attenuation = color(1.0, 1.0, 1.0);

    for (int i = 0; i < maxDepth; i++) {
        hit_record rec;
        if ((*world)->hit(cur_ray, interval(0.001f, FLT_MAX), rec)) {
            ray scattered;
            color attenuation;
            if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, local_rand_state)) {
                cur_attenuation *= attenuation;
                cur_ray = scattered;
            }
            else {
                return vec3(0.0, 0.0, 0.0);
            }
        }
        else {
            vec3 unit_direction = unit_vector(cur_ray.direction());
            float t = 0.5f * (unit_direction.y() + 1.0f);
            vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
            return cur_attenuation * c;
        }
    }

    vec3 unit_direction = unit_vector(cur_ray.direction());
    float t = 0.5f * (unit_direction.y() + 1.0f);
    vec3 c = (1.0f - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    return cur_attenuation * c;
}

