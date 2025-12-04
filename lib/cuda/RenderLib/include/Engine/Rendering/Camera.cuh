//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include "../CudaMacros.cuh"
#include "defns.h"
#include "Engine/Vector.cuh"
#include "Engine/Quaternion.cuh"
#include "Color.cuh"
#include <functional>
#include <utility>

typedef struct ImageMetadata
{
    unsigned int width;
    unsigned int height;
    unsigned int numChannels; // e.g., 3 for RGB, 4 for RGBA
    unsigned int bitsPerChannel; // e.g., 8, 16, 32
} ImageMetadata;

typedef struct RawImage
{
    RawImage() noexcept = default;
    RawImage(const int width, const int height) noexcept : Width(width), Height(height) {}
    Color* Pixels = nullptr;                 // host pointer (if populated)
    void* DevicePixels = nullptr;            // device pointer (if populated)
    bool IsDevice = false;                   // true when DevicePixels is valid
    unsigned int Width = 0;
    unsigned int Height = 0;

    std::function<void(RawImage&)> ReleaseCallback; // call to free device resources

    NO_DISCARD unsigned int NumPixels() const { return Width * Height; }

    // Call to release GPU allocations (safe to call multiple times)
    void ReleaseDevice()
    {
        if (ReleaseCallback && IsDevice)
        {
            ReleaseCallback(*this);
            ReleaseCallback = nullptr;
            DevicePixels = nullptr;
            IsDevice = false;
        }
    }

    void Release()
    {
        if (ReleaseCallback && !IsDevice)
        {
            ReleaseCallback(*this);
            ReleaseCallback = nullptr;
            DevicePixels = nullptr;
            IsDevice = false;
        }
    }
} RawImage;

static void DefaultGPURawImageRelease(RawImage& img)
{
    if (img.DevicePixels)
    {
        cudaFree(img.DevicePixels);
        img.DevicePixels = nullptr;
    }
    img.IsDevice = false;
}

static void DefaultHostRawImageRelease(RawImage& img)
{
    if (img.Pixels)
    {
        delete[] img.Pixels;
        img.Pixels = nullptr;
    }

    img.IsDevice = false;
}

typedef struct ImageFileData
{
    ImageFileData() noexcept = default;
    ImageFileData(std::string  name, std::string  path) : name(std::move(name)), path(std::move(path)) {}

    /// Image name or identifier
    std::string name;
    /// File path on disk
    std::string path;
    /// Metadata about the image
    ImageMetadata metadata{};

    /// Flag indicating if the image was successfully loaded
    bool bLoaded = false;

    /// Flag indicating if the image was successfully loaded to GPU
    bool bLoadedGPU = false;

    /// Pointer to the raw image data (only valid if loaded is true)
    RawImage* image = nullptr;

    /// Pointer to the GPU image data (only valid if bLoadedGPU is true)
    RawImage* GPUImage = nullptr; // GPU version of the image

    bool LoadImageFromFile();
    bool LoadImageFromFileGPU();
    void FreeImageData();
    ~ImageFileData() {
        FreeImageData();
    }
} ImageFileData;

typedef struct Camera
{
    Vector position;
    Quaternion orientation;
    float fov = 90.0f; // Field of View in degrees
    float aspectRatio = 16.0f / 9.0f; // Width / Height
    float nearClip = 0.1f;
    float farClip = 1000.0f;

    Camera() noexcept = default;
    CUDA_CALLABLE_MEMBER Camera(const Vector& pos, const Quaternion& orient,
        const float fov, const float aspect, const float nearC, const float farC) noexcept
        : position(pos), orientation(orient), fov(fov), aspectRatio(aspect), nearClip(nearC), farClip(farC) {}
} Camera;

// Camera helpers (add as inline functions / methods you can call from host/device)
static __host__ __device__ void Camera_RotateYawPitchRoll(Camera &cam, const float yawRad,
        const float pitchRad, const float rollRad = 0.0f) {
    // Yaw around world up (0,1,0), pitch around camera's local right axis, roll around forward
    // Create yaw quaternion (global Y)
    const Quaternion yawQ = Quaternion::fromAxisAngle({0.0f, 1.0f, 0.0f}, yawRad);

    // compute camera's local right axis by rotating world right (1,0,0) by current orientation
    const Vector right = Quaternion::RotateVectorByQuaternion({1.0f, 0.0f, 0.0f}, cam.orientation);
    const Quaternion pitchQ = Quaternion::fromAxisAngle(right, pitchRad);

    // forward axis for roll
    const Vector forward = Quaternion::RotateVectorByQuaternion(Vector{0.0f, 0.0f, -1.0f}, cam.orientation);
    const Quaternion rollQ = Quaternion::fromAxisAngle(forward, rollRad);

    // apply: newOrientation = yaw * pitch * roll * orientation
    cam.orientation = (yawQ * (pitchQ * (rollQ * cam.orientation))).normalize();
}

static __host__ __device__ Vector Camera_GetForward(const Camera &cam) {
    return Quaternion::RotateVectorByQuaternion({0.0f, 0.0f, -1.0f}, cam.orientation);
}
static __host__ __device__ Vector Camera_GetRight(const Camera &cam) {
    return Quaternion::RotateVectorByQuaternion({1.0f, 0.0f, 0.0f}, cam.orientation);
}
static __host__ __device__ Vector Camera_GetUp(const Camera &cam) {
    return Quaternion::RotateVectorByQuaternion({0.0f, 1.0f, 0.0f}, cam.orientation);
}
