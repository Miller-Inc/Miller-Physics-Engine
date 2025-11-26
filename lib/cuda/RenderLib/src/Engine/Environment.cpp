//
// Created by James Miller on 11/14/2025.
//

#include "Engine/Environment.h"
#include "Engine/EngineClock.h"

Environment::Environment()
{
    mIdentifier = GetNextIdentifier(); // Initialize identifier counter
}

Environment::~Environment()
{
    Environment::Destroy(); // Clean up resources
};

void Environment::Init()
{
    EngineClock.Init(); // Initialize the delta timer
    EngineClock.Tick();
}

void Environment::Destroy()
{
}

