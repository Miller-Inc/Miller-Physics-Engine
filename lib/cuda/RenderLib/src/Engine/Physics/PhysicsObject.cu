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
        // printf("Point %llu: (%f, %f, %f)\n", (unsigned long long)i, Points[i].x, Points[i].y, Points[i].z);
    }
}

FUNC_DEF __global__ void Rotate_Kernel(Vector* Points, const Quaternion Rotation, const Vector Center, const size_t points_count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points_count) return;

    // Translate to center
    Vector v = Points[i] - Center;

    // Normalize quaternion on device
    Quaternion q = Rotation;
    float mag = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag > 0.0f) {
        q.w /= mag; q.x /= mag; q.y /= mag; q.z /= mag;
    } else {
        q.w = 1.0f; q.x = q.y = q.z = 0.0f;
    }

    // q * (0, v)
    Quaternion vq{ 0.0f, v.x, v.y, v.z };
    Quaternion tmp;
    tmp.w = q.w * vq.w - q.x * vq.x - q.y * vq.y - q.z * vq.z;
    tmp.x = q.w * vq.x + q.x * vq.w + q.y * vq.z - q.z * vq.y;
    tmp.y = q.w * vq.y - q.x * vq.z + q.y * vq.w + q.z * vq.x;
    tmp.z = q.w * vq.z + q.x * vq.y - q.y * vq.x + q.z * vq.w;

    // (q * vq) * q_conjugate
    Quaternion qconj{ q.w, -q.x, -q.y, -q.z };
    Quaternion res;
    res.w = tmp.w * qconj.w - tmp.x * qconj.x - tmp.y * qconj.y - tmp.z * qconj.z;
    res.x = tmp.w * qconj.x + tmp.x * qconj.w + tmp.y * qconj.z - tmp.z * qconj.y;
    res.y = tmp.w * qconj.y - tmp.x * qconj.z + tmp.y * qconj.w + tmp.z * qconj.x;
    res.z = tmp.w * qconj.z + tmp.x * qconj.y - tmp.y * qconj.x + tmp.z * qconj.w;

    // rotated vector + add center back
    Points[i].x = res.x + Center.x;
    Points[i].y = res.y + Center.y;
    Points[i].z = res.z + Center.z;
}

FUNC_DEF __global__ void TransRot_Kernel(Vector* Points, const Vector translation, const Quaternion Rotation, const Vector Center, const size_t points_count)
{
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= points_count) return;

    // Translate to center
    const Vector v = Points[i] - Center;

    // Normalize quaternion on device
    Quaternion q = Rotation;
    float mag = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (mag > 0.0f) {
        q.w /= mag; q.x /= mag; q.y /= mag; q.z /= mag;
    } else {
        q.w = 1.0f; q.x = q.y = q.z = 0.0f;
    }

    // q * (0, v)
    Quaternion vq{ 0.0f, v.x, v.y, v.z };
    Quaternion tmp;
    tmp.w = q.w * vq.w - q.x * vq.x - q.y * vq.y - q.z * vq.z;
    tmp.x = q.w * vq.x + q.x * vq.w + q.y * vq.z - q.z * vq.y;
    tmp.y = q.w * vq.y - q.x * vq.z + q.y * vq.w + q.z * vq.x;
    tmp.z = q.w * vq.z + q.x * vq.y - q.y * vq.x + q.z * vq.w;

    // (q * vq) * q_conjugate
    Quaternion qconj{ q.w, -q.x, -q.y, -q.z };
    Quaternion res;
    res.w = tmp.w * qconj.w - tmp.x * qconj.x - tmp.y * qconj.y - tmp.z * qconj.z;
    res.x = tmp.w * qconj.x + tmp.x * qconj.w + tmp.y * qconj.z - tmp.z * qconj.y;
    res.y = tmp.w * qconj.y - tmp.x * qconj.z + tmp.y * qconj.w + tmp.z * qconj.x;
    res.z = tmp.w * qconj.z + tmp.x * qconj.y - tmp.y * qconj.x + tmp.z * qconj.w;

    // rotated vector + add center back
    Points[i].x = res.x + Center.x;
    Points[i].y = res.y + Center.y;
    Points[i].z = res.z + Center.z;

    // Apply translation
    Points[i] = Points[i] + translation;

    // printf("Point %llu: (%f, %f, %f)\n", (unsigned long long)i, Points[i].x, Points[i].y, Points[i].z);
}

PhysicsObject::PhysicsObject()
{
    mIdentifier = (long long)GetNextIdentifier();
    Points = nullptr;
    PointsCount = 0;
}

PhysicsObject::PhysicsObject(const Vector* points, const size_t points_count)
{
    mIdentifier = (long long)GetNextIdentifier();
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
    cudaDeviceSynchronize();
    if (Points != nullptr)
    {
        cudaFree(Points);
        Points = nullptr;
        PointsCount = 0;
    }
    if (Triangles != nullptr)
    {
        cudaFree(Triangles);
        Triangles = nullptr;
        TrianglesCount = 0;
    }
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
    const size_t count = PointsCount; // assume this member exists
    if (count == 0) return;

    // Compute centroid on host from stored host copy or from GetPoints()
    // Use GetPoints() which returns host-side std::vector<Vector>
    std::vector<Vector> hostPoints = GetPoints();
    Vector centroid{0.0f, 0.0f, 0.0f};
    for (const auto &p : hostPoints) {
        centroid.x += p.x; centroid.y += p.y; centroid.z += p.z;
    }
    centroid.x /= float(hostPoints.size());
    centroid.y /= float(hostPoints.size());
    centroid.z /= float(hostPoints.size());

    // Create rotation quaternion from Euler angles
    const Quaternion rotationQuat = Quaternion::fromEuler(rotation);

    // Copy centroid to device kernel parameter (passed by value here)
    // Launch kernel using the device-side points buffer (assume d_Points is device pointer)
    // Choose thread/block sizing
    const int threads = 256;
    const int blocks = int((count + threads - 1) / threads);

    // If using a device-stored quaternion, pass rotationQuat directly.
    Rotate_Kernel<<<blocks, threads>>>(Points, rotationQuat, centroid, count);

    // If PhysicsObject caches a host copy, update it (optional): copy device -> host
    // cudaMemcpy(hostPoints.data(), d_Points, count * sizeof(Vector), cudaMemcpyDeviceToHost);
    // Update any stored host-side structures as needed.
}

void PhysicsObject::TranslateRotate(const Vector& translation, const Vector& rotation) const
{
    const size_t count = PointsCount; // assume this member exists
    if (count == 0) return;

    // Compute centroid on host from stored host copy or from GetPoints()
    // Use GetPoints() which returns host-side std::vector<Vector>
    std::vector<Vector> hostPoints = GetPoints();
    Vector centroid{0.0f, 0.0f, 0.0f};
    for (const auto &p : hostPoints) {
        centroid.x += p.x; centroid.y += p.y; centroid.z += p.z;
    }
    centroid.x /= float(hostPoints.size());
    centroid.y /= float(hostPoints.size());
    centroid.z /= float(hostPoints.size());

    // Create rotation quaternion from Euler angles
    const Quaternion rotationQuat = Quaternion::fromEuler(rotation);

    // Copy centroid to device kernel parameter (passed by value here)
    // Launch kernel using the device-side points buffer (assume d_Points is device pointer)
    // Choose thread/block sizing
    constexpr int threads = 256;
    const int blocks = int((count + threads - 1) / threads);

    // If using a device-stored quaternion, pass rotationQuat directly.
    TransRot_Kernel<<<blocks, threads>>>(Points, translation, rotationQuat, centroid, count);
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

void PhysicsObject::SetTriangles(const Triangle* triangles, size_t triangle_count)
{
    TrianglesCount = triangle_count;
    if (Triangles != nullptr)
    {
        cudaFree(Triangles);
        Triangles = nullptr;
    }

    cudaMalloc(&Triangles, triangle_count * sizeof(Triangle));
    cudaMemcpy(Triangles, triangles, triangle_count * sizeof(Triangle), cudaMemcpyHostToDevice);
}

void PhysicsObject::CreateTriangles(const Vector* points, const size_t points_count, const size_t* triangle_indices,
                                    const size_t triangle_count)
{
    SetPoints(points, points_count);
    CreateTrianglesFromPoints(triangle_indices, triangle_count);
}

void PhysicsObject::CreateTrianglesFromPoints(const size_t* triangle_indices, size_t triangle_count)
{
    if (triangle_count > PointsCount)
    {
        printf("Error: triangle_count (%llu) exceeds PointsCount (%llu)\n",
            (unsigned long long)triangle_count, (unsigned long long)PointsCount);
        return; // Not enough points to create triangles
    }

    std::vector<Triangle> triangles_host(triangle_count / 3);

    for (size_t t = 0; t < triangle_count; t += 3)
    {
        const size_t idx1 = triangle_indices[t];
        const size_t idx2 = triangle_indices[t + 1];
        const size_t idx3 = triangle_indices[t + 2];
        if (idx1 >= PointsCount || idx2 >= PointsCount || idx3 >= PointsCount)
        {
            printf("Error: Triangle index out of bounds (idx1: %llu, idx2: %llu, idx3: %llu, PointsCount: %llu)\n",
                (unsigned long long)idx1, (unsigned long long)idx2, (unsigned long long)idx3, (unsigned long long)PointsCount);
            continue;
        }

        Triangle tri;
        tri.i0 = static_cast<unsigned int>(idx1);
        tri.i1 = static_cast<unsigned int>(idx2);
        tri.i2 = static_cast<unsigned int>(idx3);

        // Do NOT encode per-triangle owner here. Leave as invalid sentinel so
        // the Environment can set the owner/material id when assembling the scene.
        tri.materialIndex = 0xFFFFFFFFu;

        triangles_host[t / 3] = tri;
    }

    const auto tris = triangles_host.data();
    const size_t trisCount = triangles_host.size();

    if (Triangles != nullptr)
    {
        cudaFree(Triangles);
        Triangles = nullptr;
        TrianglesCount = 0;
    }

    cudaMalloc(&Triangles, trisCount * sizeof(Triangle));
    cudaMemcpy(Triangles, tris, trisCount * sizeof(Triangle), cudaMemcpyHostToDevice);
    TrianglesCount = trisCount;
}

std::vector<Vector> PhysicsObject::GetPoints() const
{
    cudaDeviceSynchronize();
    auto points_host = std::vector<Vector>(PointsCount);
    cudaMemcpy(points_host.data(), Points, PointsCount * sizeof(Vector), cudaMemcpyDeviceToHost);
    return points_host;
}

size_t PhysicsObject::GetPointsCount() const
{
    return PointsCount;
}

std::vector<Triangle> PhysicsObject::GetTriangles() const
{
    cudaDeviceSynchronize();
    auto triangles = std::vector<Triangle>(TrianglesCount);
    cudaMemcpy(triangles.data(), Triangles, TrianglesCount * sizeof(Triangle), cudaMemcpyDeviceToHost);
    return triangles;
}

size_t PhysicsObject::GetTrianglesCount() const
{
    return TrianglesCount;
}

void PhysicsObject::SetPosition(const Vector& position)
{
    Translate(position - Position);
    cudaDeviceSynchronize();
}

EPhysicsObjectType PhysicsObject::GetType() const
{
    if (mType & EPhysicsObjectType_Static)
    {
        return EPhysicsObjectType_Static;
    }
    if (mType & EPhysicsObjectType_Kinematic)
    {
        return EPhysicsObjectType_Kinematic;
    }
    if (mType & EPhysicsObjectType_All)
    {
        return EPhysicsObjectType_All;
    }
    if (mType & EPhysicsObjectType_Dynamic)
    {
        return EPhysicsObjectType_Dynamic;
    }
    return EPhysicsObjectType_Generic;
}

ECollisionChannel PhysicsObject::GetCollisionChannel() const
{
    if (mCollisionChannel & ECollisionChannel_World)
    {
        return ECollisionChannel_World;
    }
    if (mCollisionChannel & ECollisionChannel_Dynamic)
    {
        return ECollisionChannel_Dynamic;
    }
    if (mCollisionChannel & ECollisionChannel_All)
    {
        return ECollisionChannel_All;
    }
    if (mCollisionChannel & ECollisionChannel_PhysicsBody)
    {
        return ECollisionChannel_PhysicsBody;
    }
    return ECollisionChannel_None;
}

void PhysicsObject::SetType(const EPhysicsObjectType type)
{
    mType = type;
}

void PhysicsObject::SetCollisionChannel(const ECollisionChannel channel)
{
    mCollisionChannel = channel;
}

void PhysicsObject::Tick(const float deltaTime, PhysicsObject** allObjects, const int objectCount)
{
    if (PointsCount == 0 || Points == nullptr) return;

    if (bUsePhysics && PhysicsCallback)
    {
        PhysicsCallback(deltaTime, this, allObjects, objectCount);
    }

}

void PhysicsObject::SetAlbedo(const uint8_t* pixels, unsigned int w, unsigned int h, unsigned int c)
{
    if (w == 0 || h == 0 || c == 0) { AlbedoPixels.clear(); AlbedoW = AlbedoH = AlbedoC = 0; return; }
    AlbedoW = w; AlbedoH = h; AlbedoC = c;
    AlbedoPixels.assign(pixels, pixels + size_t(w) * size_t(h) * size_t(c));
}

std::vector<CollisionResult> PhysicsObject::CollisionCheck(PhysicsObject** allObjects, const int objectCount) const
{
    switch (mCollisionChannel)
    {
        case ECollisionChannel_World:
        case ECollisionChannel_PhysicsBody:
        case ECollisionChannel_Dynamic:
        case ECollisionChannel_All:
            {
                std::vector<CollisionResult> results;
                // Placeholder for collision detection logic
                return results;
            }
        case ECollisionChannel_None:
        default:
            return {};
    }
}
