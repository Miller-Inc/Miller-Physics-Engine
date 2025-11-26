//
// Created by James Miller on 11/13/2025.
//

#pragma once
#include <vector>
#include <Engine/EngineCommon.h>

class PhysicsObject; // Forward declaration

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
    virtual void Tick(float deltaTime);
    virtual void EndPlay();

    NO_DISCARD int64_t GetIdentifier() const;

    static Constructor<PhysicsObject>* StaticClass(); // Placeholder for static class retrieval

    void SetPoints(const Vector* points, size_t points_count);

protected:
    Vector Position = {0.0f, 0.0f, 0.0f};
    Quaternion Rotation = {1.0f, 0.0f, 0.0f, 0.0f};
    Vector Scale = {1.0f, 1.0f, 1.0f};

    /// Collision check for this physics object within the given environment (type not yet defined)
    std::vector<CollisionResult> CollisionCheck(PhysicsObject* environment);

    NO_DISCARD void* GetWorld() const; // Placeholder for getting the world context

    /// Translate the object by the given translation vector
    void Translate(const Vector& translation);

    /// Rotate the object around its center by the given rotation euler angles (in radians)
    void Rotate(const Vector& rotation);

    void TranslateRotate(const Vector& translation, const Vector& rotation);

    Vector* Points; // Points defining the physics object
    size_t PointsCount;

private:
    void* WorldContext = nullptr; // Placeholder for world context
    int64_t mIdentifier = -1; // Unique identifier for this physics object
};
