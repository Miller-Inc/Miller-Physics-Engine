//
// Created by James Miller on 11/13/2025.
//

#pragma once
#include <vector>
#include <Engine/EngineCommon.h>
#include <functional>

#include "PhysicsObject.cuh"

enum EPhysicsObjectType : uint8_t
{
    EPhysicsObjectType_Generic = 0,
    EPhysicsObjectType_Static = 1 << 0,
    EPhysicsObjectType_Dynamic = 1 << 1,
    EPhysicsObjectType_Kinematic = 1 << 2,
    EPhysicsObjectType_All = 0xFF
};

enum ECollisionChannel : uint8_t
{
    ECollisionChannel_None        = 0,
    ECollisionChannel_World       = 1 << 0,
    ECollisionChannel_PhysicsBody = 1 << 1,
    ECollisionChannel_Dynamic    = 1 << 2,
    ECollisionChannel_All         = 0xFF
};

class PhysicsObject; // Forward declaration

/// Callback type for physics events (returns void, takes delta time,
///     physics object, other physics objects, and number of objects)
typedef std::function<void(float, PhysicsObject*, PhysicsObject**, int)> PhysicsCallback;

typedef struct CollisionResult {
    PhysicsObject** CollidedObjects;
    size_t CollidedObjectCount;
} CollisionResult;

class PhysicsObject
{
public:
    PhysicsObject();
    PhysicsObject(const Vector* points, size_t points_count);
    virtual ~PhysicsObject();

    virtual void BeginPlay();
    virtual void Tick(float deltaTime, PhysicsObject** allObjects, int objectCount);
    virtual void EndPlay();

    NO_DISCARD int64_t GetIdentifier() const;

    static Constructor<PhysicsObject>* StaticClass(); // Placeholder for static class retrieval

    void SetPoints(const Vector* points, size_t points_count);

    void SetTriangles(const Triangle* triangles, size_t triangle_count);

    void CreateTriangles(const Vector* points, size_t points_count, const size_t* triangle_indices, size_t triangle_count);

    void CreateTrianglesFromPoints(const size_t* triangle_indices, size_t triangle_count);

    Vector translation_amount {0.0f, 0.0f, 0.0f};
    Vector rotation_amount { PI/0.50f, (PI/0.50f) , 0.0f };

    NO_DISCARD std::vector<Vector> GetPoints() const;
    NO_DISCARD size_t GetPointsCount() const;
    NO_DISCARD std::vector<Triangle> GetTriangles() const;
    NO_DISCARD size_t GetTrianglesCount() const;

    void SetPosition(const Vector& position);

    bool bUsePhysics = true;

    PhysicsCallback PhysicsCallback;

    NO_DISCARD EPhysicsObjectType GetType() const;

    NO_DISCARD ECollisionChannel GetCollisionChannel() const;

    void SetType(EPhysicsObjectType type);

    void SetCollisionChannel(ECollisionChannel channel);

    void SetWorld(void* world) { WorldContext = world; }

protected:
    Vector Position = {0.0f, 0.0f, 0.0f};
    Quaternion Rotation = {1.0f, 0.0f, 0.0f, 0.0f};
    Vector Scale = {1.0f, 1.0f, 1.0f};

public:
    /// Collision check for this physics object within the given environment (type not yet defined)
    std::vector<CollisionResult> CollisionCheck(PhysicsObject** allObjects, int objectCount) const;

    NO_DISCARD void* GetWorld() const;
    // Placeholder for getting the world context


    /// Translate the object by the given translation vector
    void Translate(const Vector& translation);

    /// Rotate the object around its center by the given rotation euler angles (in radians)
    void Rotate(const Vector& rotation);

    void TranslateRotate(const Vector& translation, const Vector& rotation) const;
protected:
    Vector* Points; // Points defining the physics object
    size_t PointsCount;
    Triangle* Triangles{}; // Triangles defining the physics object
    size_t TrianglesCount{};

private:
    void* WorldContext = nullptr; // Placeholder for world context
    int64_t mIdentifier = -1; // Unique identifier for this physics object

    ECollisionChannel mCollisionChannel = ECollisionChannel_None;
    EPhysicsObjectType mType = EPhysicsObjectType_Dynamic;
};
