//
// Created by James Miller on 11/14/2025.
//

#pragma once
#include <map>
#include <vector>
#include "EngineClock.h"
#include "Engine/EngineCommon.h"
#include "Physics/PhysicsObject.cuh"
#include "Rendering/Camera.cuh"
#define DEFAULT_ENGINE_RESOURCES_PATH "Engine_Resources/Resources.json"


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

    void SetSkybox(const ImageFileData& SkyboxImageData);

    void SetSkybox(const std::string& SkyboxImagePath);

    void FreeSkybox();

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

    bool ScanIndexedResources(const std::string& ResourceFilePath);

    void AddResource(const std::string& ResourceName, const std::string& ResourceFilePath);

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

    /// Path to resource json index
    std::string ResourcesPath{};

    ImageFileData SkyboxImageData{};

private:
    bool bIsTicking = false;
    int64_t mIdentifier = 0;

    std::map<std::string, ImageFileData> mResourceMap{};

    DeltaTimer EngineClock{}; // Delta timer for tracking time between ticks
};

