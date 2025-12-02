//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include <vector>
#include <algorithm>
#include "Engine/Triangle.cuh"
#include "Engine/Rendering/BVHNode.cuh"
#include "Engine/Vector.cuh"

class BVHBuilder
{
public:
    // Build returns nodes and reordered triangles. Simple median-split recursion.
    // Now requires a pointer to the vertex array so triangle centroids can be computed.
    static void Build(const std::vector<Triangle>& triangles, const Vector* points, std::vector<BVHNode>& outNodes, std::vector<Triangle>& outTriangles)
    {
        outNodes.clear();
        outTriangles.clear();
        if (triangles.empty()) return;
        std::vector<int> indices(triangles.size());
        for (size_t i=0;i<indices.size();++i) indices[i] = (int)i;
        buildRecursive(triangles, indices, 0, (int)indices.size(), outNodes, outTriangles, points);
    }

private:

    static int buildRecursive(const std::vector<Triangle>& tris, std::vector<int>& indices, const int start, const int end,
                                  std::vector<BVHNode>& nodes, std::vector<Triangle>& outTris, const Vector* points)
    {
        const int nodeIndex = (int)nodes.size();
        nodes.push_back(BVHNode{}); // placeholder
        AABB bounds;
        AABB centroidBounds;
        for (int i = start; i < end; ++i) {
            const Triangle& t = tris[indices[i]];
            bounds.expand(points[t.i0]); bounds.expand(points[t.i1]); bounds.expand(points[t.i2]);
            centroidBounds.expand(tris[indices[i]].centroid(points));
        }

        if (const int count = end - start; count <= 4) // make leaf
        {
            // append triangles to outTris and fill leaf node
            int first = (int)outTris.size();
            for (int i = start; i < end; ++i) outTris.push_back(tris[indices[i]]);
            BVHNode leaf; leaf.bounds = bounds; leaf.left = -1; leaf.right = first; leaf.triCount = count;
            nodes[nodeIndex] = leaf;
            return nodeIndex;
        }

        // choose split axis by largest extent
        const Vector extent = { centroidBounds.max.x - centroidBounds.min.x, centroidBounds.max.y - centroidBounds.min.y, centroidBounds.max.z - centroidBounds.min.z };
        int axis = 0;
        if (extent.y > extent.x) axis = 1;
        if (extent.z > (axis==1 ? extent.y : extent.x)) axis = 2;

        // Precompute centroids for the indices in [start,end)
        std::vector<Vector> centroids(tris.size());
        for (int i = start; i < end; ++i) {
            centroids[indices[i]] = tris[indices[i]].centroid(points);
        }

        // median split: single sort with axis-aware comparator (use const refs)
        std::sort(indices.begin() + start, indices.begin() + end, [&](const int& a, const int& b){
            const Vector& ca = centroids[a];
            const Vector& cb = centroids[b];
            if (axis == 0) return ca.x < cb.x;
            if (axis == 1) return ca.y < cb.y;
            return ca.z < cb.z;
        });

        const int mid = (start + end) / 2;
        const int left = buildRecursive(tris, indices, start, mid, nodes, outTris, points);
        const int right = buildRecursive(tris, indices, mid, end, nodes, outTris, points);

        BVHNode internal; internal.bounds = bounds; internal.left = left; internal.right = right; internal.triCount = 0;
        nodes[nodeIndex] = internal;
        return nodeIndex;
    }
};