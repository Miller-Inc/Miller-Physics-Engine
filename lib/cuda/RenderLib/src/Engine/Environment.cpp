//
// Created by James Miller on 11/14/2025.
//

#include "Engine/Environment.h"
#include "Engine/EngineClock.h"

Environment::Environment()
{
    mIdentifier = GetNextIdentifier(); // Initialize identifier counter
    PhysicsObjects.clear();
}

Environment::~Environment()
{
    Environment::Destroy(); // Clean up resources
};

void Environment::Init()
{
    static bool init = false;
    if (init)
        return;
    init = true;
    EngineClock.Init(); // Initialize the delta timer
    EngineClock.Tick();
}

void Environment::Destroy()
{
}

