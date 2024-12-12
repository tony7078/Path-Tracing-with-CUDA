﻿#pragma once
#include <math.h>
#include <stdlib.h>
#include <iostream>

using std::sqrt;

class vec3 {
public:
    __host__ __device__ vec3() : e{ 0,0,0 } {}
    __host__ __device__ vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

    __host__ __device__ inline float x() const { return e[0]; }
    __host__ __device__ inline float y() const { return e[1]; }
    __host__ __device__ inline float z() const { return e[2]; }
    __host__ __device__ inline float r() const { return e[0]; }
    __host__ __device__ inline float g() const { return e[1]; }
    __host__ __device__ inline float b() const { return e[2]; }

    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline float operator[](int i) const { return e[i]; }
    __host__ __device__ inline float& operator[](int i) { return e[i]; }

    __host__ __device__ inline vec3& operator+=(const vec3& v) {
        e[0] += v.e[0];
        e[1] += v.e[1];
        e[2] += v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(float t) {
        e[0] *= t;
        e[1] *= t;
        e[2] *= t;
        return *this;
    }

    __host__ __device__ inline vec3& operator*=(const vec3& v) {
        e[0] *= v.e[0];
        e[1] *= v.e[1];
        e[2] *= v.e[2];
        return *this;
    }

    __host__ __device__ inline vec3& operator/=(float t) {
        return *this *= 1 / t;
    }

    __host__ __device__ inline float length() const { return std::sqrt(length_squared()); }

    __host__ __device__ inline float length_squared() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

    __host__ __device__ inline bool near_zero() const {
        // Return true if the vector is close to zero in all dimensions.
        auto s = 1e-8;
        return (std::fabs(e[0]) < s) && (std::fabs(e[1]) < s) && (std::fabs(e[2]) < s);
    }

    __device__ static vec3 random(curandState* rand_state) { return vec3(random_float(rand_state), random_float(rand_state), random_float(rand_state)); }

    __device__ static vec3 random(float min, float max, curandState* rand_state) { return vec3(random_float(min, max, rand_state), random_float(min, max, rand_state), random_float(min, max, rand_state)); }

    float e[3];
};

// point3 is just an alias for vec3, but useful for geometric clarity in the code.
using point3 = vec3;
using color = vec3;


// Vector Utility Functions
inline std::ostream& operator<<(std::ostream& out, const vec3& v) {
    return out << v.e[0] << ' ' << v.e[1] << ' ' << v.e[2];
}

__host__ __device__ inline vec3 operator+(const vec3& u, const vec3& v) {
    return vec3(u.e[0] + v.e[0], u.e[1] + v.e[1], u.e[2] + v.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3& u, const vec3& v) {
    return vec3(u.e[0] - v.e[0], u.e[1] - v.e[1], u.e[2] - v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& u, const vec3& v) {
    return vec3(u.e[0] * v.e[0], u.e[1] * v.e[1], u.e[2] * v.e[2]);
}

__host__ __device__ inline vec3 operator*(float t, const vec3& v) {
    return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3& v, float t) {
    return t * v;
}

__host__ __device__ inline vec3 operator/(const vec3& v, float t) {
    return (1 / t) * v;
}

__host__ __device__ inline float dot(const vec3& u, const vec3& v) {
    return u.e[0] * v.e[0]
        + u.e[1] * v.e[1]
        + u.e[2] * v.e[2];
}

__host__ __device__ inline vec3 cross(const vec3& u, const vec3& v) {
    return vec3(u.e[1] * v.e[2] - u.e[2] * v.e[1],
        u.e[2] * v.e[0] - u.e[0] * v.e[2],
        u.e[0] * v.e[1] - u.e[1] * v.e[0]);
}

__host__ __device__ inline vec3 unit_vector(const vec3& v) {
    return v / v.length();
}

__device__ inline vec3 random_in_unit_disk(curandState* rand_state) {
    while (true) {
        vec3 p = vec3(random_float(-1.0f, 1.0f, rand_state), random_float(-1.0f, 1.0f, rand_state), 0);
        if (p.length_squared() < 1) return p;
    }
}

__device__ inline vec3 random_unit_vector(curandState* rand_state) {
    while (true) {
        auto p = vec3::random(-1.0f, 1.0f, rand_state);
        auto lensq = p.length_squared();
        if (1e-160 < lensq && lensq <= 1) return p / sqrt(lensq);  // Checking whether point P is inside the sphere
    }
}

__device__ inline vec3 random_on_hemisphere(const vec3& normal, curandState* rand_state) {
    vec3 on_unit_sphere = random_unit_vector(rand_state);
    if (dot(on_unit_sphere, normal) > 0.0)  // In the same hemisphere as the normal
        return on_unit_sphere;
    else
        return -on_unit_sphere;
}

__device__ inline vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2 * dot(v, n) * n;
}

__device__ inline vec3 refract(const vec3& uv, const vec3& n, float etai_over_etat) {
    auto cos_theta = std::fmin(dot(-uv, n), 1.0);
    vec3 r_out_perp = etai_over_etat * (uv + cos_theta * n);
    vec3 r_out_parallel = -std::sqrt(std::fabs(1.0 - r_out_perp.length_squared())) * n;
    return r_out_perp + r_out_parallel;
}