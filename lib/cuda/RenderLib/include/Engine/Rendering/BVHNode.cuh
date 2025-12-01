//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include "Engine/CudaMacros.cuh"
#include "AABB.cuh"

struct BVHNode
{
    AABB bounds;
    // If leaf: left == -1 and right == firstTriangle (index), triCount > 0
    // If internal: triCount == 0 and left/right are child indices
    int left;    // index of left child or -1 for leaf
    int right;   // index of right child or first triangle index when leaf
    int triCount;// 0 for internal node, >0 for leaf
};