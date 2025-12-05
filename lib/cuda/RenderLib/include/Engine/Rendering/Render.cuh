//
// Created by James Miller on 12/5/2025.
//

#pragma once
#include <cmath>
#include "Engine/CudaMacros.cuh"
#include "Engine/Vector.cuh"
#include "Engine/Triangle.cuh"
#include "BVHNode.cuh"
#include "BVHTraverse.cuh"

// Fallback for tools that don't define CUDA_CALLABLE_MEMBER
#ifndef CUDA_CALLABLE_MEMBER
#define CUDA_CALLABLE_MEMBER
#include <cuda_runtime.h>
#endif


// pack to 0xAARRGGBB
__device__ __host__ inline uint32_t packRGBA8(const uint8_t b, const uint8_t g, const uint8_t r, const uint8_t a = 255)
{
    return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
}

// device helper: sample equirectangular panorama (nearest)
// expects 'dir' normalized
__device__ inline Vector SampleEquirectangular(const uint8_t* pixels, const unsigned int w,
    const unsigned int h,const unsigned int c, const Vector dir)
{
    // u: [0,1] = 0.5 + atan2(z,x) / (2*pi)
    // v: [0,1] = 0.5 - asin(y) / pi
    float u = 0.5f + atan2f(dir.z, dir.x) * (1.0f / (2.0f * PI));
    float v = 0.5f - asinf(fmaxf(-1.0f, fminf(1.0f, dir.y))) * (1.0f / PI);

    // wrap u, clamp v
    u = u - floorf(u);
    v = fminf(1.0f, fmaxf(0.0f, v));

    unsigned int ix = int(u * float(w));
    unsigned int iy = int(v * float(h));
    if (ix < 0) ix = 0; if (ix >= w) ix = w - 1;
    if (iy < 0) iy = 0; if (iy >= h) iy = h - 1;

    const unsigned int idx = (iy * w + ix) * c;
    Vector out; out.x = out.y = out.z = 0.0f;
    if (c >= 3) {
        out.x = float(pixels[idx + 0]) / 255.0f;
        out.y = float(pixels[idx + 1]) / 255.0f;
        out.z = float(pixels[idx + 2]) / 255.0f;
    } else if (c == 1) {
        float v0 = float(pixels[idx]) / 255.0f;
        out.x = out.y = out.z = v0;
    }
    return out;
}

__device__ inline Vector SampleTextureNearest(const uint8_t* pixels, unsigned int w, unsigned int h, unsigned int c, float u, float v)
{
    if (!pixels || w == 0 || h == 0) return Vector(1.0f, 1.0f, 1.0f);
    // wrap/clamp
    u = u - floorf(u);
    v = fminf(1.0f, fmaxf(0.0f, v));
    int ix = int(u * float(w));
    int iy = int(v * float(h));
    if (ix < 0) ix = 0; if (ix >= (int)w) ix = (int)w - 1;
    if (iy < 0) iy = 0; if (iy >= (int)h) iy = (int)h - 1;
    const unsigned int idx = (iy * w + ix) * c;
    Vector out{0.0f,0.0f,0.0f};
    if (c >= 3) {
        out.x = float(pixels[idx + 0]) / 255.0f;
        out.y = float(pixels[idx + 1]) / 255.0f;
        out.z = float(pixels[idx + 2]) / 255.0f;
    } else if (c == 1) {
        float v0 = float(pixels[idx]) / 255.0f;
        out.x = out.y = out.z = v0;
    }
    return out;
}

__device__ inline float hash2D(float x, float y) {
    // sin/dot hash — inexpensive and consistent
    float h = sinf(x * 127.1f + y * 311.7f) * 43758.5453f;
    return h - floorf(h);
}

// smooth noise (single octave, bilinear-ish by smoothing sample points)
__device__ inline float noise2D(float x, float y) {
    // fractional positions
    float xf = floorf(x);
    float yf = floorf(y);
    float fracx = x - xf;
    float fracy = y - yf;

    float v00 = hash2D(xf + 0.0f, yf + 0.0f);
    float v10 = hash2D(xf + 1.0f, yf + 0.0f);
    float v01 = hash2D(xf + 0.0f, yf + 1.0f);
    float v11 = hash2D(xf + 1.0f, yf + 1.0f);

    // smoothstep for smoother interpolation
    float sx = fracx * fracx * (3.0f - 2.0f * fracx);
    float sy = fracy * fracy * (3.0f - 2.0f * fracy);

    float ix0 = v00 + sx * (v10 - v00);
    float ix1 = v01 + sx * (v11 - v01);
    return ix0 + sy * (ix1 - ix0);
}

// fractal brownian motion (few octaves) for smooth variation
__device__ inline float fbm2D(float x, float y, int octaves = 4) {
    float sum = 0.0f;
    float amp = 0.5f;
    for (int i = 0; i < octaves; ++i) {
        sum += amp * noise2D(x, y);
        x *= 2.0f;
        y *= 2.0f;
        amp *= 0.5f;
    }
    // normalize approx to [0,1]
    return fminf(1.0f, fmaxf(0.0f, sum));
}

__device__ inline Vector SampleTextureBilinear(const uint8_t* pixels, unsigned int w, unsigned int h, unsigned int c, float u, float v)
{
    if (!pixels || w == 0 || h == 0) return Vector(1.0f, 1.0f, 1.0f);
    // wrap/clamp
    u = u - floorf(u);
    v = fminf(1.0f, fmaxf(0.0f, v));

    // Map to texel space with pixel center offset
    float fx = u * float(w) - 0.5f;
    float fy = v * float(h) - 0.5f;

    int ix0 = int(floorf(fx));
    int iy0 = int(floorf(fy));
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float tx = fx - float(ix0);
    float ty = fy - float(iy0);

    // clamp indices
    if (ix0 < 0) ix0 = 0; if (ix0 >= (int)w) ix0 = (int)w - 1;
    if (ix1 < 0) ix1 = 0; if (ix1 >= (int)w) ix1 = (int)w - 1;
    if (iy0 < 0) iy0 = 0; if (iy0 >= (int)h) iy0 = (int)h - 1;
    if (iy1 < 0) iy1 = 0; if (iy1 >= (int)h) iy1 = (int)h - 1;

    const unsigned int idx00 = (iy0 * w + ix0) * c;
    const unsigned int idx10 = (iy0 * w + ix1) * c;
    const unsigned int idx01 = (iy1 * w + ix0) * c;
    const unsigned int idx11 = (iy1 * w + ix1) * c;

    auto fetch = [&](unsigned int idx)->Vector {
        Vector out{0.0f,0.0f,0.0f};
        if (c >= 3) {
            out.x = float(pixels[idx + 0]) / 255.0f;
            out.y = float(pixels[idx + 1]) / 255.0f;
            out.z = float(pixels[idx + 2]) / 255.0f;
        } else if (c == 1) {
            float v0 = float(pixels[idx]) / 255.0f;
            out.x = out.y = out.z = v0;
        }
        return out;
    };

    Vector v00 = fetch(idx00);
    Vector v10 = fetch(idx10);
    Vector v01 = fetch(idx01);
    Vector v11 = fetch(idx11);

    // bilinear interp
    Vector ix0v = v00 + (v10 - v00) * tx;
    Vector ix1v = v01 + (v11 - v01) * tx;
    return ix0v + (ix1v - ix0v) * ty;
}

__global__ void inline renderKernel(const BVHNode* d_nodes, const int nodeCount,
                             const Triangle* d_tris, const Vector* d_points,
                             uint32_t* d_pixels, const int width, const int height,
                             const Vector camOrigin, const Quaternion camOrient, const float fov,
                             const uint8_t* skyboxPtr, const unsigned int skyboxWidth = 0,
                             const unsigned int skyboxHeight = 0, const unsigned int skyboxChannels = 0,
                             // per-texture arrays (device pointers)
                             const uint8_t** albedoPtrs = nullptr, const unsigned int* albedoW = nullptr,
                             const unsigned int* albedoH = nullptr, const unsigned int* albedoC = nullptr,
                             const unsigned int albedoCount = 0,
                             // per-triangle material indices
                             const unsigned int* triMaterialIndex = nullptr, const unsigned int triCount = 0)
{
    const unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Normalized Device Coordinates -> View Space
    const float ndcX = ( ((float)px + 0.5f) / float(width) ) * 2.0f - 1.0f;
    const float ndcY = ( ((float)py + 0.5f) / float(height) ) * 2.0f - 1.0f;
    const float aspect = float(width) / float(height);
    const float tanFov = tanf(fov * 0.5f);

    Vector dir;
    dir.x = ndcX * aspect * tanFov;
    dir.y = -ndcY * tanFov;
    dir.z = -1.0f;

    dir = Quaternion::RotateVectorByQuaternion(dir.normalize(), camOrient);

    // Use new traversal that returns barycentrics
    HitInfo hitInfo{};
    const int hit = traverseBVHHit(d_nodes, nodeCount, d_tris, d_points, camOrigin, dir, hitInfo);

    uint32_t color;
    if (hit >= 0)
    {
        const float t = hitInfo.t;
        const Vector hitPos = Vector(camOrigin.x + dir.x * t, camOrigin.y + dir.y * t, camOrigin.z + dir.z * t);

        // Determine material for this triangle (default -> none)
        unsigned int matIndex = 0xFFFFFFFFu;
        if (triMaterialIndex != nullptr && (unsigned int)hitInfo.triIndex < triCount) {
            matIndex = triMaterialIndex[hitInfo.triIndex];
        }

        // Determine triangle & normal early so we can choose a projection for UVs
        const Triangle tri = d_tris[hitInfo.triIndex];
        const Vector p0 = d_points[tri.i0];
        const Vector p1 = d_points[tri.i1];
        const Vector p2 = d_points[tri.i2];
        Vector N = (p1 - p0).cross(p2 - p0).normalize();

        Vector albedo;
        bool usedTexture = false;
        if (matIndex != 0xFFFFFFFFu && albedoPtrs != nullptr && matIndex < albedoCount) {
            const uint8_t* texPtr = albedoPtrs[matIndex];
            const unsigned int w = albedoW ? albedoW[matIndex] : 0u;
            const unsigned int h = albedoH ? albedoH[matIndex] : 0u;
            const unsigned int c = albedoC ? albedoC[matIndex] : 0u;
            if (texPtr && w > 0 && h > 0 && c > 0) {
                // Choose projection axes based on dominant normal component (planar mapping)
                float ax = fabsf(N.x), ay = fabsf(N.y), az = fabsf(N.z);
                float coordA = 0.0f, coordB = 0.0f;
                if (ay >= ax && ay >= az) {
                    // normal mostly up/down -> project X,Z
                    coordA = hitPos.x;
                    coordB = hitPos.z;
                } else if (ax >= ay && ax >= az) {
                    // normal mostly along X -> project Y,Z
                    coordA = hitPos.y;
                    coordB = hitPos.z;
                } else {
                    // normal mostly along Z -> project X,Y
                    coordA = hitPos.x;
                    coordB = hitPos.y;
                }

                // tileScale controls repeats per world unit; tweak to taste
                const float tileScale = 0.2f; // 0.2 -> 5 repeats per unit
                float u = coordA * tileScale;
                float v = coordB * tileScale;

                // Sample with bilinear filtering to avoid visible texel lines
                albedo = SampleTextureBilinear(texPtr, w, h, c, u, v);
                usedTexture = true;
            }
        }

        if (!usedTexture) {
            // single pink fallback for untextured geometry
            albedo = Vector(0.961f, 0.149f, 0.898f);
        }

        Vector L = Vector(0.57735f, -0.57735f, -0.57735f);
        float ndotl = fmaxf(0.0f, N.dot(L));
        Vector shaded = albedo * ndotl;

        const auto r = uint8_t(fminf(255.0f, fmaxf(0.0f, shaded.x * 255.0f)));
        const auto g = uint8_t(fminf(255.0f, fmaxf(0.0f, shaded.y * 255.0f)));
        const auto b = uint8_t(fminf(255.0f, fmaxf(0.0f, shaded.z * 255.0f)));
        color = packRGBA8(b, g, r);
    }
    else
    {
        // skybox or gradient (unchanged)
        if (skyboxPtr && skyboxWidth > 0 && skyboxHeight > 0 && skyboxChannels > 0) {
            // sample panorama
            Vector col = SampleEquirectangular(skyboxPtr, skyboxWidth, skyboxHeight, skyboxChannels, dir);
            const auto r = uint8_t(fminf(255.0f, fmaxf(0.0f, col.x * 255.0f)));
            const auto g = uint8_t(fminf(255.0f, fmaxf(0.0f, col.y * 255.0f)));
            const auto b = uint8_t(fminf(255.0f, fmaxf(0.0f, col.z * 255.0f)));
            color = packRGBA8(b, g, r);
        } else {
            // simple gradient sky
            float t = 0.5f * (dir.y + 1.0f);
            Vector top = Vector(0.55f, 0.6f, 0.65f);
            Vector bot = Vector(0.9f, 0.92f, 0.95f);
            Vector col = bot * (1.0f - t) + top * t;
            const auto r = uint8_t(fminf(255.0f, fmaxf(0.0f, col.x * 255.0f)));
            const auto g = uint8_t(fminf(255.0f, fmaxf(0.0f, col.y * 255.0f)));
            const auto b = uint8_t(fminf(255.0f, fmaxf(0.0f, col.z * 255.0f)));
            color = packRGBA8(b, g, r);
        }
    }

    d_pixels[py * width + px] = color;
}