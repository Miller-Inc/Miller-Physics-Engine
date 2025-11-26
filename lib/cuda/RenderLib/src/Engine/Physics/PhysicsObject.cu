//
// Created by James Miller on 11/13/2025.
//

#include "Engine/Physics/PhysicsObject.cuh"
#include <iostream>
#include <ostream>

FUNC_DEF __global__ void Translate_Kernel(Vector* Points, const Vector translation, const size_t points_count)
{
    if (const size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < points_count)
    {
        Points[i] = Points[i] + translation;
        printf("Point %llu: (%f, %f, %f)\n", (unsigned long long)i, Points[i].x, Points[i].y, Points[i].z);
    }
}

FUNC_DEF __global__ void Rotate_Kernel(Vector* Points, const Quaternion Rotation, const Vector Center, const size_t points_count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points_count) return;

    // Translate to origin
    Vector v = Points[i] - Center;

    // Normalize quaternion on device
    Quaternion q = Rotation;
    float mag = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag > 0.0f) {
        q.w /= mag; q.x /= mag; q.y /= mag; q.z /= mag;
    }

    // u = q.xyz
    Vector u; u.x = q.x; u.y = q.y; u.z = q.z;

    // t = 2 * cross(u, v)
    Vector t;
    t.x = 2.0f * (u.y * v.z - u.z * v.y);
    t.y = 2.0f * (u.z * v.x - u.x * v.z);
    t.z = 2.0f * (u.x * v.y - u.y * v.x);

    // v' = v + q.w * t + cross(u, t)
    Vector cross_ut;
    cross_ut.x = u.y * t.z - u.z * t.y;
    cross_ut.y = u.z * t.x - u.x * t.z;
    cross_ut.z = u.x * t.y - u.y * t.x;

    Vector out;
    out.x = v.x + q.w * t.x + cross_ut.x;
    out.y = v.y + q.w * t.y + cross_ut.y;
    out.z = v.z + q.w * t.z + cross_ut.z;

    // Translate back
    Points[i] = out + Center;

    printf("Point %llu: (%f, %f, %f)\n", (unsigned long long)i, Points[i].x, Points[i].y, Points[i].z);
}

FUNC_DEF __global__ void TransRot_Kernel(Vector* Points, const Vector translation, const Quaternion Rotation, const Vector Center, const size_t points_count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points_count) return;

    // Translate to origin
    Vector v = Points[i] - Center;

    // Normalize quaternion on device
    Quaternion q = Rotation;
    float mag = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag > 0.0f) {
        q.w /= mag; q.x /= mag; q.y /= mag; q.z /= mag;
    }

    // u = q.xyz
    Vector u; u.x = q.x; u.y = q.y; u.z = q.z;

    // t = 2 * cross(u, v)
    Vector t;
    t.x = 2.0f * (u.y * v.z - u.z * v.y);
    t.y = 2.0f * (u.z * v.x - u.x * v.z);
    t.z = 2.0f * (u.x * v.y - u.y * v.x);

    // v' = v + q.w * t + cross(u, t)
    Vector cross_ut;
    cross_ut.x = u.y * t.z - u.z * t.y;
    cross_ut.y = u.z * t.x - u.x * t.z;
    cross_ut.z = u.x * t.y - u.y * t.x;

    Vector out;
    out.x = v.x + q.w * t.x + cross_ut.x;
    out.y = v.y + q.w * t.y + cross_ut.y;
    out.z = v.z + q.w * t.z + cross_ut.z;

    // Translate back
    Points[i] = out + Center;

    // Apply translation
    Points[i] = Points[i] + translation;

    printf("Point %llu: (%f, %f, %f)\n", (unsigned long long)i, Points[i].x, Points[i].y, Points[i].z);
}

PhysicsObject::PhysicsObject()
{
    mIdentifier = GetNextIdentifier();
    Points = nullptr;
    PointsCount = 0;
}

PhysicsObject::PhysicsObject(const Vector* points, const size_t points_count)
{
    mIdentifier = GetNextIdentifier();
    Points = nullptr;
    PointsCount = 0;
    SetPoints(points, points_count);
}

PhysicsObject::~PhysicsObject() = default;

void PhysicsObject::BeginPlay()
{

}

void PhysicsObject::EndPlay()
{
}

int64_t PhysicsObject::GetIdentifier() const
{
    return mIdentifier;
}

Constructor<PhysicsObject>* PhysicsObject::StaticClass()
{
    static Constructor<PhysicsObject> instance {&PhysicsObject::BeginPlay, &PhysicsObject::Tick};
    return &instance;
}

void* PhysicsObject::GetWorld() const
{
    return WorldContext;
}

void PhysicsObject::Translate(const Vector& translation)
{
    const size_t threadsPerBlock = (PointsCount > 128) ? 128 : PointsCount;
    const size_t blocks = (PointsCount + threadsPerBlock - 1) / threadsPerBlock;

    // Pass translation by value to avoid a device reference to host memory.
    Translate_Kernel<<<blocks, threadsPerBlock>>>(Points, translation, PointsCount);

    Position += translation;

#if DEBUG

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cerr << "Kernel execution/sync error: " << cudaGetErrorString(err) << std::endl;

#endif
}

void PhysicsObject::Rotate(const Vector& rotation)
{
    const size_t threadsPerBlock = (PointsCount > 128) ? 128 : PointsCount;
    const size_t blocks = (PointsCount + threadsPerBlock - 1) / threadsPerBlock;

    const Quaternion rot = Quaternion::fromEuler(rotation);

    Rotate_Kernel<<<blocks, threadsPerBlock>>>(Points, rot, Position, PointsCount);

    Rotation += rot;
    Rotation.normalize();

#if DEBUG

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cerr << "Kernel execution/sync error: " << cudaGetErrorString(err) << std::endl;

#endif
}

void PhysicsObject::TranslateRotate(const Vector& translation, const Vector& rotation)
{
    const size_t threadsPerBlock = (PointsCount > 128) ? 128 : PointsCount;
    const size_t blocks = (PointsCount + threadsPerBlock - 1) / threadsPerBlock;

    const Quaternion rot = Quaternion::fromEuler(rotation);

    TransRot_Kernel<<<blocks, threadsPerBlock>>>(Points, translation, rot, Position, PointsCount);

    // Update rotation
    Rotation += rot;
    Rotation.normalize();

    // Update position
    Position += translation;

    std::cout << "Rotation: " << Rotation.toString() << std::endl;
    std::cout << "Position: " << Position.toString() << std::endl;
    std::cout << "Rot amount: " << rot.toString() << std::endl;

#if DEBUG

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) std::cerr << "Kernel execution/sync error: " << cudaGetErrorString(err) << std::endl;

#endif
}


void PhysicsObject::SetPoints(const Vector* points, const size_t points_count)
{
    PointsCount = points_count;
    if (Points != nullptr)
    {
        cudaFree(Points);
        Points = nullptr;
    }

    cudaError_t err = cudaMalloc(&Points, points_count * sizeof(Vector));
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << std::endl;
        Points = nullptr; PointsCount = 0;
        return;
    }

    err = cudaMemcpy(Points, points, points_count * sizeof(Vector), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "cudaMemcpy H2D failed: " << cudaGetErrorString(err) << std::endl;
        cudaFree(Points); Points = nullptr; PointsCount = 0;
        return;
    }
}

void PhysicsObject::Tick(const float deltaTime)
{
    static Vector translation_amount {0.1f, -1.0f, 0.5f};
    static Vector rotation_amount { PI/4.0f, 0.0f, 0.0f }; // Rotate 45 degrees around X axis per second

    std::cout << "Ticking Physics Object " << GetIdentifier() << " " << deltaTime << " seconds." << std::endl;
    if (PointsCount == 0 || Points == nullptr) return;

    TranslateRotate(translation_amount * deltaTime, rotation_amount * deltaTime);

}
std::vector<CollisionResult> PhysicsObject::CollisionCheck(PhysicsObject* environment)
{
    return {}; // Placeholder implementation
}
