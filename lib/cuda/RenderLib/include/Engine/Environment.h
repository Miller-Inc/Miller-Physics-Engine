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
    NO_DISCARD T* SpawnPhysicsObject(Constructor<T>* ObjectStaticClass)
    {
        static_assert(std::is_base_of<PhysicsObject, T>::value, "T must derive from PhysicsObject");

        // Construct the object (requires T to be default-constructible)
        T* obj = new T();

        // If a Constructor<T> is provided and has a BeginPlay function, invoke it via pointer-to-member.
        if (ObjectStaticClass && ObjectStaticClass->BeginPlayFunc) {
            (obj->*ObjectStaticClass->BeginPlayFunc)();
        } else {
            // Fallback: call the virtual BeginPlay directly.
            obj->BeginPlay();
        }

        PhysicsObjects.push_back(static_cast<PhysicsObject*>(obj));
        return obj;
    }

protected:
    std::vector<PhysicsObject*> PhysicsObjects;

private:
    bool bIsTicking = true;
    int64_t mIdentifier = 0;

    DeltaTimer EngineClock; // Delta timer for tracking time between ticks
};

