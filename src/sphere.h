#pragma once
#include "hittable.h"

class sphere : public hittable {
public:
    __device__ sphere(const point3& center, float radius, material* m) : center(center), radius(std::fmax(0.0f, radius)), mat_ptr(m) {}

    __device__ bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
        // Intersection point calculation
        vec3 oc = center - r.origin();
        auto a = r.direction().length_squared();
        auto h = dot(r.direction(), oc);
        auto c = oc.length_squared() - radius * radius;

        auto discriminant = h * h - a * c;
        if (discriminant < 0)
            return false;

        auto sqrtd = std::sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = (h - sqrtd) / a;
        bool surrounds = t_min < root && root < t_max;
        if (!surrounds) {
            root = (h + sqrtd) / a;
            surrounds = t_min < root && root < t_max;
            if (!surrounds)
                return false;
        }

        rec.t = root;                                       // t
        rec.p = r.at(rec.t);                                // intersection point
        vec3 outward_normal = (rec.p - center) / radius;    // normalize with radius
        rec.set_face_normal(r, outward_normal);             // inside/outside determination
        rec.mat_ptr = mat_ptr;

        return true;
    }

    point3 center;
    float radius;
    material* mat_ptr;
};