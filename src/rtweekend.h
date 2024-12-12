#pragma once
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <random>

#include <curand_kernel.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


// C++ Std Usings

using std::make_shared;
using std::shared_ptr;
using std::sqrt;

// Constants

const float infinity = std::numeric_limits<float>::infinity();
const float pi = 3.1415926535897932385;

// Utility Functions

__host__ __device__ inline float degrees_to_radians(float degrees) {
    return degrees * pi / 180.0;
}

__device__ inline float random_float(curandState* state) {
    return curand_uniform(state);
}

__device__ inline float random_float(float min, float max, curandState* state) {
    // Returns a random real in [min,max).
    return min + (max - min) * curand_uniform(state);
}

// Common Headers

#include "ray.h"
#include "vec3.h"