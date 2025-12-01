//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include "defns.h"
#include "CudaMacros.cuh"

typedef struct Triangle
{
    unsigned long int i0 = 0, i1 = 0, i2 = 0;
    Triangle() = default;
    Triangle(const size_t size, const size_t idx2, const size_t idx3) : i0(size), i1(idx2), i2(idx3) {}
    CUDA_CALLABLE_MEMBER Vector centroid(const Vector* vertices) const noexcept
    {
        return {(vertices[i0].x+vertices[i1].x+vertices[i2].x)/3.0f,
            (vertices[i0].y+vertices[i1].y+vertices[i2].y)/3.0f,
            (vertices[i0].z+vertices[i1].z+vertices[i2].z)/3.0f};
    };
} Triangle;
