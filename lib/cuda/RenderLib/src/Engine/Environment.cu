//
// Created by James Miller on 11/14/2025.
//

#include "Engine/Environment.h"
#include "Engine/Rendering/Camera.cuh"
#include "Engine/Physics/PhysicsObject.cuh"
#include "Engine/Rendering/BVHBuilder.h"
#include "Engine/Rendering/BVHTraverse.cuh"
#include "Engine/Rendering/Render.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <fstream>
#include <ranges>
#include <unordered_map>
#include "nlohmann/json.hpp"
using json = nlohmann::json;




void Environment::Init()
{
    static bool init = false;
    if (init)
        return;
    init = true;

    MainCamera.position = Vector(0.0f, 0.0f, 7.5f);
    MainCamera.orientation = Quaternion(1.0f, 0.0f, 0.0f, 0.0f);

    if (ScanIndexedResources(DEFAULT_ENGINE_RESOURCES_PATH))
    {
        for (const auto& [name, data] : mResourceMap)
        {
            if (name.find("skybox") != std::string::npos || name.find("Skybox") != std::string::npos)
            {
                SetSkybox(data);
                break;
            }
        }
    }


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

void Environment::SetSkybox(const ImageFileData& SkyboxImageData)
{
    this->SkyboxImageData = SkyboxImageData;
    this->SkyboxImageData.LoadImageFromFileGPU();
}

void Environment::SetSkybox(const std::string& SkyboxImagePath)
{
    std::string name = "Skybox";
    SkyboxImageData = ImageFileData(name, SkyboxImagePath);
    this->SkyboxImageData.LoadImageFromFileGPU();
}

void Environment::FreeSkybox()
{
    cudaDeviceSynchronize();
    SkyboxImageData.FreeImageData();
}

bool Environment::ScanIndexedResources(const std::string& ResourceFilePath)
{
    mResourceMap.clear();

    ResourcesPath = ResourceFilePath;

    std::ifstream file(ResourceFilePath);
    if (!file.is_open())
    {
        return false;
    }

    json json_object = json::parse(file);
    auto images = json_object["images"];

    if (images == json::value_t::discarded || !images.is_array())
    {
        return false;
    }

    // auto img_arr = images->array();
    static unsigned int un_labeled_resource_id = 0;

    for (const auto& resource : images)
    {
        ImageFileData data;
        data.path = resource.value("path", "");
        data.name = resource.value("name", "Object_" + std::to_string(un_labeled_resource_id++));

        mResourceMap.emplace(data.name, data);
    }

    return true;
}

void Environment::AddResource(const std::string& ResourceName, const std::string& ResourceFilePath)
{
    ImageFileData data;
    data.path = ResourceFilePath;
    data.name = ResourceName;

    mResourceMap.emplace(data.name, data);
}

RawImage Environment::RenderScene(const Camera& cam, const int _width, const int _height) const
{
    // --- Configuration / resolve members (adjust these names to match your Environment class) ---
    const unsigned int width  = (_width  > 0) ? (unsigned int)_width  : 1280u;
    const unsigned int height = (_height > 0) ? (unsigned int)_height : 720u;

    // Get list of physics objects in the scene
    // TODO: replace GetPhysicsObjects() with your container access (e.g. mObjects)
    std::vector<PhysicsObject*> sceneObjects = PhysicsObjects;

    // --- Gather points & triangles into contiguous host arrays ---
    std::vector<Vector> hostPoints;
    std::vector<Triangle> hostTris;
    std::vector<unsigned int> triOwners; // owner object index per tri (object index into sceneObjects)

    hostPoints.reserve(1024);
    hostTris.reserve(1024);
    triOwners.reserve(1024);

    for (size_t objIdx = 0; objIdx < sceneObjects.size(); ++objIdx)
    {
        PhysicsObject* obj = sceneObjects[objIdx];
        if (!obj) continue;

        // get host copies (PhysicsObject provides GetPoints/GetTriangles)
        const std::vector<Vector> objPoints = obj->GetPoints();
        const std::vector<Triangle> objTris  = obj->GetTriangles();

        if (objPoints.empty() || objTris.empty()) continue;

        // index offset for this object's points inside hostPoints
        const unsigned int baseIndex = (unsigned int)hostPoints.size();
        // append points
        hostPoints.insert(hostPoints.end(), objPoints.begin(), objPoints.end());

        // append triangles, remapping indices and preserving materialIndex as object index (owner)
        for (const Triangle &t : objTris)
        {
            Triangle t2 = t;
            t2.i0 = t.i0 + baseIndex;
            t2.i1 = t.i1 + baseIndex;
            t2.i2 = t.i2 + baseIndex;
            // store owner index in materialIndex so Environment can later map to albedos / materials
            t2.materialIndex = static_cast<unsigned int>(objIdx);
            hostTris.push_back(t2);
            triOwners.push_back(static_cast<unsigned int>(objIdx));
        }
    }

    // If nothing to render, return an empty gradient image to avoid crashes
    RawImage outImg;
    outImg.Width = width;
    outImg.Height = height;
    if (hostTris.empty() || hostPoints.empty())
    {
        // Create a small device buffer with gradient and return
        const size_t pxCount = size_t(width) * size_t(height);
        uint32_t* d_pixels = nullptr;
        cudaMalloc(&d_pixels, pxCount * sizeof(uint32_t));
        // launch a tiny kernel or memset to clear; here clear to sky gradient color (simple)
        std::vector<uint32_t> tmp(pxCount, 0xFFB3C2D1u); // packed ARGB-like placeholder
        cudaMemcpy(d_pixels, tmp.data(), pxCount * sizeof(uint32_t), cudaMemcpyHostToDevice);

        outImg.DevicePixels = d_pixels;
        outImg.IsDevice = true;
        outImg.ReleaseCallback = DefaultGPURawImageRelease;
        return outImg;
    }

    // --- Build BVH (host-side) ---
    std::vector<BVHNode> nodes;
    std::vector<Triangle> outTris;
    outTris.reserve(hostTris.size());
    // BVHBuilder::Build expects hostTris and hostPoints
    BVHBuilder::Build(hostTris, hostPoints.data(), nodes, outTris);

    // NOTE: do NOT overwrite outTris[].materialIndex here (preserve owner info set above)

    // --- Upload BVH, triangles, points to device ---
    BVHNode* d_nodes = nullptr;
    Triangle* d_tris = nullptr;
    Vector* d_points = nullptr;

    const size_t nodeBytes  = nodes.size() * sizeof(BVHNode);
    const size_t trisBytes  = outTris.size() * sizeof(Triangle);
    const size_t pointsBytes= hostPoints.size() * sizeof(Vector);

    cudaMalloc(&d_nodes, nodeBytes);
    cudaMalloc(&d_tris, trisBytes);
    cudaMalloc(&d_points, pointsBytes);

    cudaMemcpy(d_nodes, nodes.data(), nodeBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_tris, outTris.data(), trisBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_points, hostPoints.data(), pointsBytes, cudaMemcpyHostToDevice);

    // --- Prepare albedo textures: allocate device buffers for each object's albedo, and build pointer arrays ---
    // Map objects that actually have textures -> compact albedo slot indices
    std::vector<unsigned int> objToAlbedoIndex(sceneObjects.size(), 0xFFFFFFFFu);
    std::vector<uint8_t*> albedoDevicePtrs;
    std::vector<unsigned int> albedoW;
    std::vector<unsigned int> albedoH;
    std::vector<unsigned int> albedoC;

    for (size_t i = 0; i < sceneObjects.size(); ++i)
    {
        PhysicsObject* obj = sceneObjects[i];
        if (!obj) continue;
        if (obj->AlbedoW == 0 || obj->AlbedoH == 0 || obj->AlbedoC == 0 || obj->AlbedoPixels.empty()) {
            continue; // no texture for this object
        }
        const size_t bytes = size_t(obj->AlbedoW) * size_t(obj->AlbedoH) * size_t(obj->AlbedoC);
        uint8_t* d_tex = nullptr;
        cudaMalloc(&d_tex, bytes);
        cudaMemcpy(d_tex, obj->AlbedoPixels.data(), bytes, cudaMemcpyHostToDevice);

        objToAlbedoIndex[i] = (unsigned int)albedoDevicePtrs.size();
        albedoDevicePtrs.push_back(d_tex);
        albedoW.push_back(obj->AlbedoW);
        albedoH.push_back(obj->AlbedoH);
        albedoC.push_back(obj->AlbedoC);
    }

    // Copy albedo pointers/metadata to device arrays expected by kernel
    uint8_t** d_albedoPtrs = nullptr;
    unsigned int* d_albedoW = nullptr;
    unsigned int* d_albedoH = nullptr;
    unsigned int* d_albedoC = nullptr;

    const unsigned int albedoCount = (unsigned int)albedoDevicePtrs.size();
    if (albedoCount > 0)
    {
        cudaMalloc(&d_albedoPtrs, sizeof(uint8_t*) * albedoCount);
        cudaMalloc(&d_albedoW,   sizeof(unsigned int) * albedoCount);
        cudaMalloc(&d_albedoH,   sizeof(unsigned int) * albedoCount);
        cudaMalloc(&d_albedoC,   sizeof(unsigned int) * albedoCount);

        cudaMemcpy(d_albedoPtrs, albedoDevicePtrs.data(), sizeof(uint8_t*) * albedoCount, cudaMemcpyHostToDevice);
        cudaMemcpy(d_albedoW,   albedoW.data(), sizeof(unsigned int) * albedoCount, cudaMemcpyHostToDevice);
        cudaMemcpy(d_albedoH,   albedoH.data(), sizeof(unsigned int) * albedoCount, cudaMemcpyHostToDevice);
        cudaMemcpy(d_albedoC,   albedoC.data(), sizeof(unsigned int) * albedoCount, cudaMemcpyHostToDevice);
    }

    // --- Build per-triangle materialIndex array that references albedo slots (or sentinel) ---
    std::vector<unsigned int> h_triMaterialIndex(outTris.size(), 0xFFFFFFFFu);
    for (size_t i = 0; i < outTris.size(); ++i)
    {
        const unsigned int owner = outTris[i].materialIndex; // owner was stored above
        if (owner < objToAlbedoIndex.size() && objToAlbedoIndex[owner] != 0xFFFFFFFFu) {
            h_triMaterialIndex[i] = objToAlbedoIndex[owner];
        } else {
            h_triMaterialIndex[i] = 0xFFFFFFFFu;
        }
    }
    unsigned int* d_triMaterialIndex = nullptr;
    if (!h_triMaterialIndex.empty())
    {
        cudaMalloc(&d_triMaterialIndex, sizeof(unsigned int) * h_triMaterialIndex.size());
        cudaMemcpy(d_triMaterialIndex, h_triMaterialIndex.data(), sizeof(unsigned int) * h_triMaterialIndex.size(), cudaMemcpyHostToDevice);
    }

    // --- Allocate device pixel buffer ---
    const size_t pixelCount = size_t(width) * size_t(height);
    uint32_t* d_pixels = nullptr;
    cudaMalloc(&d_pixels, pixelCount * sizeof(uint32_t));
    cudaMemset(d_pixels, 0, pixelCount * sizeof(uint32_t));

    // --- Launch kernel ---
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    // cam parameters: convert field-of-view degrees to radians for kernel (kernel expects fov in radians)
    const float fovRad = cam.fov * (PI / 180.0f);
    // kernel arguments: skybox pointer/size (TODO: adjust if you store skybox differently)
    const uint8_t* skyboxPtr = (uint8_t*)SkyboxImageData.GPUImage->DevicePixels;
    const unsigned int skyW = SkyboxImageData.metadata.width;
    const unsigned int skyH = SkyboxImageData.metadata.height;
    const unsigned int skyC = SkyboxImageData.metadata.numChannels;

    renderKernel<<<gridDim, blockDim>>>(d_nodes, (int)nodes.size(),
                                        d_tris, d_points,
                                        d_pixels, (int)width, (int)height,
                                        cam.position, cam.orientation, fovRad,
                                        skyboxPtr, skyW, skyH, skyC,
                                        (const uint8_t**)d_albedoPtrs, d_albedoW, d_albedoH, d_albedoC, albedoCount,
                                        (const unsigned int*)d_triMaterialIndex, (unsigned int)outTris.size());

    // Wait for kernel to finish and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::fprintf(stderr, "Render kernel error: %s\n", cudaGetErrorString(err));
    }

    // --- Free GPU objects that are no longer needed (we keep d_pixels alive for the returned RawImage) ---
    if (d_nodes) cudaFree(d_nodes);
    if (d_tris)  cudaFree(d_tris);
    if (d_points) cudaFree(d_points);

    if (d_albedoPtrs) cudaFree(d_albedoPtrs);
    if (d_albedoW) cudaFree(d_albedoW);
    if (d_albedoH) cudaFree(d_albedoH);
    if (d_albedoC) cudaFree(d_albedoC);

    // free each albedo texture device allocation (safe now since kernel finished)
    for (uint8_t* p : albedoDevicePtrs) if (p) cudaFree(p);

    if (d_triMaterialIndex) cudaFree(d_triMaterialIndex);

    // --- Package result as GPU-backed RawImage ---
    outImg.Width = width;
    outImg.Height = height;
    outImg.DevicePixels = d_pixels;
    outImg.IsDevice = true;
    // Use DefaultGPURawImageRelease from Camera.cuh which will cudaFree(DevicePixels)
    outImg.ReleaseCallback = DefaultGPURawImageRelease;

    return outImg;
}
