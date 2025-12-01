//
// Created by James Miller on 11/14/2025.
//

#include "Engine/Environment.h"
#include "Engine/Rendering/Camera.cuh"
#include "Engine/Physics/PhysicsObject.cuh"
#include "Engine/Rendering/BVHBuilder.h"
#include "Engine/Rendering/BVHTraverse.cuh"
#include <cuda_runtime.h>

// pack to 0xAARRGGBB
__device__ __host__ inline uint32_t packRGBA8(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255)
{
    return (uint32_t(a) << 24) | (uint32_t(r) << 16) | (uint32_t(g) << 8) | uint32_t(b);
}

__global__ void renderKernel(const BVHNode* d_nodes, const int nodeCount,
                             const Triangle* d_tris, const Vector* d_points,
                             uint32_t* d_pixels, const int width, const int height,
                             const Vector camOrigin, const Quaternion camOrient, const float fov)
{
    const unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // NDC -> screen
    const float ndcX = ( (px + 0.5f) / float(width) ) * 2.0f - 1.0f;
    const float ndcY = ( (py + 0.5f) / float(height) ) * 2.0f - 1.0f;
    const float aspect = float(width) / float(height);
    const float tanFov = tanf(fov * 0.5f);

    Vector dir;
    dir.x = ndcX * aspect * tanFov;
    dir.y = -ndcY * tanFov;
    dir.z = -1.0f;

    // normalize dir (manual)
    const float len = sqrtf(dir.x*dir.x + dir.y*dir.y + dir.z*dir.z);
    dir.x /= len; dir.y /= len; dir.z /= len;

    Vector unrot = dir; // dir computed and normalized
    dir = quat_rotate_vector(camOrient, unrot);

    int hit = traverseBVH(d_nodes, nodeCount, d_tris, d_points, camOrigin, dir);

    uint32_t color;
    if (hit >= 0) {
        // simple hit color
        color = packRGBA8(200, 40, 40);
    } else {
        // sky gradient
        const float t = 0.5f * (dir.y + 1.0f);
        const auto r = uint8_t((1.0f - t) * 135 + t * 25);
        const auto g = uint8_t((1.0f - t) * 206 + t * 25);
        const auto b = uint8_t((1.0f - t) * 235 + t * 112);
        color = packRGBA8(r, g, b);
    }

    d_pixels[py * width + px] = color;
}

void Environment::Init()
{
    static bool init = false;
    if (init)
        return;
    init = true;

    MainCamera.position = Vector(0.0f, 0.0f, 7.5f);
    MainCamera.orientation = Quaternion(1.0f, 0.0f, 0.0f, 0.0f);

    EngineClock.Init(); // Initialize the delta timer
    EngineClock.Tick();
}

void Environment::TickAll(const float deltaTime)
{
    // Physics Update
    for (PhysicsObject* obj : PhysicsObjects)
    {
        if (obj)
        {
            obj->Tick(deltaTime, PhysicsObjects.data(), (int)PhysicsObjects.size());
        }
    }
    cudaDeviceSynchronize();


}

RawImage Environment::RenderScene(const int width, const int height) const
{
    return RenderScene(MainCamera, width, height);
}

RawImage Environment::RenderScene(const Camera& camera, int width, int height) const
{
    if (width <= 0 && height > 0)
    {
        width = static_cast<int>(camera.aspectRatio * float(height));
    }
    else if (height <= 0 && width > 0)
    {
        height = static_cast<int>(float(width) / camera.aspectRatio);
    }
    else if (width <= 0 && height <= 0)
    {
        // Default resolution
        height = 740;
        width = static_cast<int>(camera.aspectRatio * float(height));
    }
    const int H = height;
    const int W = width;

    std::vector<Vector> points;
    std::vector<Triangle> tris;

    for (PhysicsObject* obj : PhysicsObjects)
    {
        if (!obj) continue;
        const std::vector<Vector>& objPoints = obj->GetPoints();
        const std::vector<Triangle>& objTris   = obj->GetTriangles();

        const int baseIndex = (int)points.size();
        points.insert(points.end(), objPoints.begin(), objPoints.end());

        for (const Triangle& t : objTris)
        {
            Triangle remap = t;
            remap.i0 += baseIndex;
            remap.i1 += baseIndex;
            remap.i2 += baseIndex;
            tris.push_back(remap);
        }
    }

    if (points.empty() || tris.empty())
    {
        points.emplace_back(-1.0f, -1.0f, -3.0f);
        points.emplace_back(1.0f, -1.0f, -3.0f);
        points.emplace_back(0.0f, 1.0f, -3.0f);
        Triangle t; t.i0 = 0; t.i1 = 1; t.i2 = 2;
        tris.push_back(t);
    }

    // Build BVH (host)
    std::vector<BVHNode> nodes;
    std::vector<Triangle> outTris;
    BVHBuilder::Build(tris, points.data(), nodes, outTris);

    // Device buffers
    BVHNode* d_nodes = nullptr;
    Triangle* d_tris = nullptr;
    Vector* d_points = nullptr;
    uint32_t* d_pixels = nullptr;

    const int nodeCount = (int)nodes.size();
    const size_t nodeBytes  = nodes.size() * sizeof(BVHNode);
    const size_t triBytes   = outTris.size() * sizeof(Triangle);
    const size_t pointBytes = points.size() * sizeof(Vector);
    const size_t imgBytes   = size_t(W) * size_t(H) * sizeof(uint32_t);

    cudaMalloc(&d_nodes, nodeBytes);
    cudaMalloc(&d_tris, triBytes);
    cudaMalloc(&d_points, pointBytes);
    cudaMalloc(&d_pixels, imgBytes);

    cudaMemcpy(d_nodes, nodes.data(), nodeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, outTris.data(), triBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, points.data(), pointBytes, cudaMemcpyHostToDevice);
    cudaMemset(d_pixels, 0, imgBytes);

    // Launch kernel using camera
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    const auto camOrigin = camera.position;
    const auto camOrient = camera.orientation;
    const float fovRad = camera.fov * 3.14159265f / 180.0f;

    renderKernel<<<grid, block>>>(d_nodes, nodeCount, d_tris, d_points, d_pixels,
        W, H, camOrigin, camOrient, fovRad);
    cudaDeviceSynchronize();

    // DO NOT copy back. Return a GPU-backed RawImage.
    RawImage img(W, H);
    img.Pixels = nullptr;
    img.DevicePixels = reinterpret_cast<void*>(d_pixels);
    img.IsDevice = true;

    // Provide a release callback that frees all GPU allocations created for this image.
    // Caller must call img.ReleaseDevice() when done (or you can invoke immediately in destructor if you add that behavior).
    img.ReleaseCallback = [d_nodes, d_tris, d_points, d_pixels](RawImage& /*unused*/) {
        if (d_nodes)   cudaFree(d_nodes);
        if (d_tris)    cudaFree(d_tris);
        if (d_points)  cudaFree(d_points);
        if (d_pixels)  cudaFree(d_pixels);
    };

    return img;
}
