#pragma once
#include <curand_kernel.h>
#include "ray.h"

class camera {
public:
    float  aspect_ratio = 1.0;          // Ratio of image width over height
    int    image_width = 100;           // Rendered image width in pixel count
    int    samples_per_pixel = 10;      // Count of random samples for each pixel
    int    max_depth = 10;              // Maximum number of ray bounces into scene

    float vfov = 90;                    // Vertical view angle (field of view)
    point3 lookfrom = point3(0, 0, 0);  // Point camera is looking from
    point3 lookat = point3(0, 0, -1);   // Point camera is looking at
    vec3   vup = vec3(0, 1, 0);         // Camera-relative "up" direction

    float defocus_angle = 0;            // Variation angle of rays through each pixel
    float focus_dist = 10;              // Distance from camera lookfrom point to plane of perfect foc

    __device__ camera() {
        initialize();
    }

    __device__  ray get_ray(float i, float j, curandState* rand_state) {
        // Construct a camera ray originating from the defocus disk and directed at a randomly
        // sampled point around the pixel location i, j.

        auto offset = sample_square(rand_state);
        auto pixel_sample = pixel00_loc
            + ((i + offset.x()) * pixel_delta_u)
            + ((j + offset.y()) * pixel_delta_v);

        auto ray_origin = (defocus_angle <= 0) ? center : defocus_disk_sample(rand_state);
        auto ray_direction = pixel_sample - ray_origin;

        return ray(ray_origin, ray_direction);
    }

private:
    int     image_height;            // Rendered image height
    float   pixel_samples_scale;     // Color scale factor for a sum of pixel samples
    point3  center;                  // Camera center
    point3  pixel00_loc;             // Location of pixel 0, 0
    vec3    pixel_delta_u;           // Offset to pixel to the right
    vec3    pixel_delta_v;           // Offset to pixel below
    vec3    u, v, w;                 // Camera frame basis vectors
    vec3    defocus_disk_u;          // Defocus disk horizontal radius
    vec3    defocus_disk_v;          // Defocus disk vertical radius

    __device__ void initialize() {
        // Camera Setup
        aspect_ratio = 16.0 / 9.0;
        image_width = 1200;
        samples_per_pixel = 10;
        max_depth = 50;

        vfov = 20;
        lookfrom = point3(13, 2, 3);
        lookat = point3(0, 0, 0);
        vup = vec3(0, 1, 0);

        defocus_angle = 0.6;
        focus_dist = 10.0;

        // Image Setup
        //image_height = int(image_width / aspect_ratio);
        //image_height = (image_height < 1) ? 1 : image_height;
        image_height = 800;

        pixel_samples_scale = 1.0 / samples_per_pixel;

        center = lookfrom;

        // Determine viewport dimensions.
        auto theta = degrees_to_radians(vfov);
        auto h = std::tan(theta / 2);
        auto viewport_height = 2 * h * focus_dist;
        auto viewport_width = viewport_height * (float(image_width) / image_height);

        // Calculate the u,v,w unit basis vectors for the camera coordinate frame.
        w = unit_vector(lookfrom - lookat);
        u = unit_vector(cross(vup, w));
        v = cross(w, u);

        // Calculate the vectors across the horizontal and down the vertical viewport edges.
        vec3 viewport_u = viewport_width * u;    // Vector across viewport horizontal edge
        vec3 viewport_v = viewport_height * -v;  // Vector down viewport vertical edge

        // Calculate the horizontal and vertical delta vectors from pixel to pixel.
        pixel_delta_u = viewport_u / image_width;
        pixel_delta_v = viewport_v / image_height;

        // Calculate the location of the upper left pixel.
        auto viewport_upper_left = center - (focus_dist * w) - viewport_u / 2 - viewport_v / 2;
        pixel00_loc = viewport_upper_left + 0.5 * (pixel_delta_u + pixel_delta_v);

        // Calculate the camera defocus disk basis vectors.
        auto defocus_radius = focus_dist * std::tan(degrees_to_radians(defocus_angle / 2));
        defocus_disk_u = u * defocus_radius;
        defocus_disk_v = v * defocus_radius;

        std::cerr << "Rendering a " << image_width << "x" << image_height << " image with " << samples_per_pixel << " samples per pixel ";
    }

    __device__ vec3 sample_square(curandState* rand_state) const {
        // Returns the vector to a random point in the [-.5,-.5]-[+.5,+.5] unit square.
        return vec3(random_float(rand_state) - 0.5, random_float(rand_state) - 0.5, 0);
    }

    __device__ point3 defocus_disk_sample(curandState* rand_state) const {
        // Returns a random point in the camera defocus disk.
        auto p = random_in_unit_disk(rand_state);
        return center + (p[0] * defocus_disk_u) + (p[1] * defocus_disk_v);
    }
};