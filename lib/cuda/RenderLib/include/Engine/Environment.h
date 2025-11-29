//
// Created by James Miller on 11/14/2025.
//

#pragma once
#include <vector>
#include "EngineClock.h"
#include "Engine/EngineCommon.h"
#include "Physics/PhysicsObject.cuh"

class Environment
{
public:
    Environment();
    virtual ~Environment();

    virtual void Init();
    virtual void Destroy();
    virtual void TickAll(float deltaTime);

    double GetDeltaTime() { return EngineClock.Tick(); }

    NO_DISCARD bool IsTicking() const { return bIsTicking; }
    void SetTicking(const bool bTick) { bIsTicking = bTick; }

    template<typename T>
    PhysicsObject* SpawnPhysicsObject()
    {
        T* TypedObject = new T();

        TypedObject->BeginPlay();

        PhysicsObjects.emplace_back(TypedObject);
        return TypedObject;
    }

protected:
    std::vector<PhysicsObject*> PhysicsObjects{};

private:
    bool bIsTicking = false;
    int64_t mIdentifier = 0;

    DeltaTimer EngineClock{}; // Delta timer for tracking time between ticks
};

