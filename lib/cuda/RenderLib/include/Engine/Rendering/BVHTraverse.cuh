//
// Created by James Miller on 11/29/2025.
//

#pragma once

#include <cmath>
#include "../../Engine/CudaMacros.cuh"
#include "../../Engine/Vector.cuh"
#include "../../Engine/Triangle.cuh"
#include "../Rendering/BVHNode.cuh"

// Fallback for tools that don't define CUDA_CALLABLE_MEMBER
#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER
#include <cuda_runtime.h>
#endif

// ray-triangle intersection (Möller–Trumbore). Returns t if hit.
CUDA_CALLABLE_MEMBER inline bool intersectTriangle(const Vector& orig, const Vector& dir, const Triangle& tri, const Vector* points, float& outT) noexcept
{
    const float EPS = 1e-6f;
    Vector edge1 = points[tri.i1] - points[tri.i0];
    Vector edge2 = points[tri.i2] - points[tri.i0];
    Vector pvec = dir.cross(edge2);
    float det = edge1.dot(pvec);
    if (fabsf(det) < EPS) return false;
    float invDet = 1.0f / det;
    Vector tvec = orig - points[tri.i0];
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    Vector qvec = tvec.cross(edge1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = edge2.dot(qvec) * invDet;
    if (t <= EPS) return false;
    outT = t;
    return true;
}

// Traversal: nodes and triangles are in device memory arrays
CUDA_CALLABLE_MEMBER inline int traverseBVH(const BVHNode* nodes, int nodeCount, const Triangle* tris, const Vector* points, const Vector& orig, const Vector& dir) noexcept
{
    // Precompute invDir and sign
    Vector invDir = { 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z };
    int dirSign[3] = { invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f };

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root index
    int hitTri = -1;
    float closestT = 1e30f;

    while (sp > 0)
    {
        int nodeIdx = stack[--sp];
        const BVHNode& node = nodes[nodeIdx];
        if (!node.bounds.intersect(orig, invDir, dirSign, 0.0f, closestT)) continue;

        if (node.triCount > 0) // leaf
        {
            const int firstTri = node.right;
            for (int i = 0; i < node.triCount; ++i)
            {
                const Triangle& t = tris[firstTri + i];
                float tHit;
                if (intersectTriangle(orig, dir, t, points, tHit))
                {
                    if (tHit < closestT)
                    {
                        closestT = tHit;
                        hitTri = firstTri + i;
                    }
                }
            }
        }
        else // internal
        {
            // push children; preserve order so nearer child could be pushed last if desired
            if (node.right >= 0) stack[sp++] = node.right;
            if (node.left >= 0)  stack[sp++] = node.left;
        }
    }
    return hitTri;
}

// Hit info returned by traversal
struct HitInfo
{
    int triIndex;   // index into device triangle array (or -1)
    float t;        // hit distance
    float u;        // barycentric u
    float v;        // barycentric v
};

// Möller–Trumbore that returns barycentrics (u,v) and t
CUDA_CALLABLE_MEMBER inline bool intersectTriangleBary(const Vector& orig, const Vector& dir, const Triangle& tri, const Vector* points, float& outT, float& outU, float& outV) noexcept
{
    const float EPS = 1e-6f;
    const Vector& v0 = points[tri.i0];
    const Vector& v1 = points[tri.i1];
    const Vector& v2 = points[tri.i2];

    Vector edge1 = v1 - v0;
    Vector edge2 = v2 - v0;
    Vector pvec = dir.cross(edge2);
    float det = edge1.dot(pvec);
    if (fabsf(det) < EPS) return false;
    float invDet = 1.0f / det;
    Vector tvec = orig - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;
    Vector qvec = tvec.cross(edge1);
    float v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = edge2.dot(qvec) * invDet;
    if (t <= EPS) return false;
    outT = t;
    outU = u;
    outV = v;
    return true;
}

// Traversal: returns index of closest hit triangle, fills out HitInfo
CUDA_CALLABLE_MEMBER inline int traverseBVHHit(const BVHNode* nodes, int nodeCount, const Triangle* tris, const Vector* points, const Vector& orig, const Vector& dir, HitInfo& outHit) noexcept
{
    // Precompute invDir and sign
    Vector invDir = { 1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z };
    int dirSign[3] = { invDir.x < 0.0f, invDir.y < 0.0f, invDir.z < 0.0f };

    int stack[64];
    int sp = 0;
    stack[sp++] = 0; // root index
    int hitTri = -1;
    float closestT = 1e30f;
    outHit.triIndex = -1; outHit.t = 0.0f; outHit.u = 0.0f; outHit.v = 0.0f;

    while (sp > 0)
    {
        int nodeIdx = stack[--sp];
        const BVHNode& node = nodes[nodeIdx];
        if (!node.bounds.intersect(orig, invDir, dirSign, 0.0f, closestT)) continue;

        if (node.triCount > 0) // leaf
        {
            const int firstTri = node.right;
            for (int i = 0; i < node.triCount; ++i)
            {
                const Triangle& t = tris[firstTri + i];
                float tHit, u, v;
                if (intersectTriangleBary(orig, dir, t, points, tHit, u, v))
                {
                    if (tHit < closestT)
                    {
                        closestT = tHit;
                        hitTri = firstTri + i;
                        outHit.triIndex = hitTri;
                        outHit.t = tHit;
                        outHit.u = u;
                        outHit.v = v;
                    }
                }
            }
        }
        else // internal
        {
            if (node.right >= 0) stack[sp++] = node.right;
            if (node.left >= 0)  stack[sp++] = node.left;
        }
    }
    return hitTri;
}

