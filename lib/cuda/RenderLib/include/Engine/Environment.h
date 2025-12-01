//
// Created by James Miller on 11/14/2025.
//

#pragma once
#include <vector>
#include "EngineClock.h"
#include "Engine/EngineCommon.h"
#include "Physics/PhysicsObject.cuh"
#include "Rendering/Camera.cuh"


class Environment
{
public:
    Environment();
    virtual ~Environment();

    virtual void Init();
    virtual void Destroy();
    virtual void TickAll(float deltaTime);
    NO_DISCARD RawImage RenderScene(int width = -1, int height = -1) const;

    double GetDeltaTime() { return EngineClock.Tick(); }

    NO_DISCARD bool IsTicking() const { return bIsTicking; }
    void SetTicking(const bool bTick) { bIsTicking = bTick; }

    template<typename T>
    PhysicsObject* SpawnPhysicsObject(EPhysicsObjectType Type = EPhysicsObjectType_Dynamic,
        ECollisionChannel CollisionChannel = ECollisionChannel_All)
    {
        T* TypedObject = new T();

        TypedObject->SetWorld(this);
        TypedObject->SetType(Type);
        TypedObject->SetCollisionChannel(CollisionChannel);

        TypedObject->BeginPlay();

        PhysicsObjects.emplace_back(TypedObject);

        switch (Type)
        {
        case EPhysicsObjectType_Generic:
            GenericPhysicsObjects.emplace_back(TypedObject);
            break;
        case EPhysicsObjectType_Static:
            StaticPhysicsObjects.emplace_back(TypedObject);
            break;
        case EPhysicsObjectType_Dynamic:
            DynamicPhysicsObjects.emplace_back(TypedObject);
            break;
        case EPhysicsObjectType_Kinematic:
            KinematicPhysicsObjects.emplace_back(TypedObject);
            break;
        default:
            break;
        }

        return TypedObject;
    }

protected:
    std::vector<PhysicsObject*> PhysicsObjects{};

public:
    Camera MainCamera{};

protected:
    NO_DISCARD RawImage RenderScene(const Camera& camera, int width = -1, int height = -1) const;

    std::vector<PhysicsObject*> GenericPhysicsObjects{};
    std::vector<PhysicsObject*> StaticPhysicsObjects{};
    std::vector<PhysicsObject*> DynamicPhysicsObjects{};
    std::vector<PhysicsObject*> KinematicPhysicsObjects{};
    std::vector<PhysicsObject*> AllTypePhysicsObjects{};

private:
    bool bIsTicking = false;
    int64_t mIdentifier = 0;

    DeltaTimer EngineClock{}; // Delta timer for tracking time between ticks
};

