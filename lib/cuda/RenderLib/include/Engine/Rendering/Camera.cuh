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

typedef struct RawImage
{
    RawImage(): Width(0), Height(0) {};
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
        if (ReleaseCallback)
        {
            ReleaseCallback(*this);
            ReleaseCallback = nullptr;
            DevicePixels = nullptr;
            IsDevice = false;
        }
    }
} RawImage;

typedef struct Camera
{
    Vector position;
    Quaternion orientation;
    float fov = 90.0f; // Field of View in degrees
    float aspectRatio = 16.0f / 9.0f; // Width / Height
    float nearClip = 0.1f;
    float farClip = 1000.0f;

    CUDA_CALLABLE_MEMBER Camera() noexcept = default;
    CUDA_CALLABLE_MEMBER Camera(const Vector& pos, const Quaternion& orient,
        const float fov, const float aspect, const float nearC, const float farC) noexcept
        : position(pos), orientation(orient), fov(fov), aspectRatio(aspect), nearClip(nearC), farClip(farC) {}
} Camera;

static inline __host__ __device__ Quaternion quat_normalize(const Quaternion &q) {
    float mag = sqrtf(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
    if (mag == 0.0f) return Quaternion{1.0f,0.0f,0.0f,0.0f};
    return Quaternion{ q.w/mag, q.x/mag, q.y/mag, q.z/mag };
}

static inline __host__ __device__ Quaternion quat_mul(const Quaternion &a, const Quaternion &b) {
    return Quaternion{
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w
    };
}

static inline __host__ __device__ Quaternion quat_from_axis_angle(const Vector &axis, float angleRad) {
    const float half = angleRad * 0.5f;
    const float s = sinf(half);
    return quat_normalize(Quaternion{ cosf(half), axis.x * s, axis.y * s, axis.z * s });
}

static inline __host__ __device__ Vector quat_rotate_vector(const Quaternion &q, const Vector &v) {
    // v' = q * (0, v) * q^-1
    Quaternion qn = quat_normalize(q);
    Quaternion vq{ 0.0f, v.x, v.y, v.z };
    // compute q * vq
    Quaternion tmp = quat_mul(qn, vq);
    // compute (q * vq) * q_conjugate
    Quaternion qconj{ qn.w, -qn.x, -qn.y, -qn.z };
    Quaternion res = quat_mul(tmp, qconj);
    return Vector{ res.x, res.y, res.z };
}

// Camera helpers (add as inline functions / methods you can call from host/device)
static inline __host__ __device__ void Camera_RotateYawPitchRoll(Camera &cam, const float yawRad,
        const float pitchRad, const float rollRad = 0.0f) {
    // Yaw around world up (0,1,0), pitch around camera's local right axis, roll around forward
    // Create yaw quaternion (global Y)
    const Quaternion yawQ = quat_from_axis_angle(Vector{0.0f, 1.0f, 0.0f}, yawRad);

    // compute camera's local right axis by rotating world right (1,0,0) by current orientation
    Vector right = quat_rotate_vector(cam.orientation, Vector{1.0f, 0.0f, 0.0f});
    const Quaternion pitchQ = quat_from_axis_angle(right, pitchRad);

    // forward axis for roll
    Vector forward = quat_rotate_vector(cam.orientation, Vector{0.0f, 0.0f, -1.0f});
    const Quaternion rollQ = quat_from_axis_angle(forward, rollRad);

    // apply: newOrientation = yaw * pitch * roll * orientation
    Quaternion newQ = quat_mul(yawQ, quat_mul(pitchQ, quat_mul(rollQ, cam.orientation)));
    cam.orientation = quat_normalize(newQ);
}

static inline __host__ __device__ Vector Camera_GetForward(const Camera &cam) {
    return quat_rotate_vector(cam.orientation, Vector{0.0f, 0.0f, -1.0f});
}
static inline __host__ __device__ Vector Camera_GetRight(const Camera &cam) {
    return quat_rotate_vector(cam.orientation, Vector{1.0f, 0.0f, 0.0f});
}
static inline __host__ __device__ Vector Camera_GetUp(const Camera &cam) {
    return quat_rotate_vector(cam.orientation, Vector{0.0f, 1.0f, 0.0f});
}
