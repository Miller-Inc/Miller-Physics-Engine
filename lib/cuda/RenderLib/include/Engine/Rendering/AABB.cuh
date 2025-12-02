//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include "Engine/CudaMacros.cuh"
#include "Engine/Vector.cuh" // use your existing Vector type
#include <limits>

/// Axis-Aligned Bounding Box: Used for spatial partitioning and collision detection (specifically in BVH)
typedef struct AABB
{
    /// Minimum corner
    Vector min;

    /// Maximum corner
    Vector max;

    AABB() noexcept {
        const float inf = std::numeric_limits<float>::infinity();
        min = { inf,  inf,  inf};
        max = {-inf, -inf, -inf};
    }

    /// Expand to include point p
    CUDA_CALLABLE_MEMBER void expand(const Vector& p) noexcept {
        min.x = fminf(min.x, p.x); min.y = fminf(min.y, p.y); min.z = fminf(min.z, p.z);
        max.x = fmaxf(max.x, p.x); max.y = fmaxf(max.y, p.y); max.z = fmaxf(max.z, p.z);
    }

    /// Expand to include another AABB
    CUDA_CALLABLE_MEMBER void expand(const AABB& o) noexcept {
        expand(o.min); expand(o.max);
    }

    /// slab method: returns true if ray (orig + t*dir) intersects in [tmin,tmax]
    CUDA_CALLABLE_MEMBER bool intersect(const Vector& orig, const Vector& invDir, const int dirIsNeg[3], float tmin = 0.0f, float tmax = 1e30f) const noexcept
    {
        // X
        float t0 = ((dirIsNeg[0] ? max.x : min.x) - orig.x) * invDir.x;
        float t1 = ((dirIsNeg[0] ? min.x : max.x) - orig.x) * invDir.x;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
        // Y
        t0 = ((dirIsNeg[1] ? max.y : min.y) - orig.y) * invDir.y;
        t1 = ((dirIsNeg[1] ? min.y : max.y) - orig.y) * invDir.y;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));
        // Z
        t0 = ((dirIsNeg[2] ? max.z : min.z) - orig.z) * invDir.z;
        t1 = ((dirIsNeg[2] ? min.z : max.z) - orig.z) * invDir.z;
        tmin = fmaxf(tmin, fminf(t0, t1));
        tmax = fminf(tmax, fmaxf(t0, t1));

        return tmax >= tmin;
    }
} AABB;