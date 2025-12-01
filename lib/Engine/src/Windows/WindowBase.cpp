//
// Created by James Miller & Joshua Miller on 9/10/2025.
//

#include "Windows/WindowBase.h"

void WindowBase::Init()
{
    isOpen = true;

    if (mInstance != nullptr)
    {
        mTextures = mInstance->LoadResources(Name);
    } else {
        M_LOGGER(Logger::LogCore, Logger::Warning, "GameInstance is null. Cannot load resources for window: " + Name);
    }

    Open(); // Open the window by default after initialization
}

void WindowBase::Init(const std::string& WindowName, GInstance* Instance)
{
    Name = WindowName;
    this->mInstance = Instance;
    WindowBase::Init();
}

void WindowBase::Init(GInstance* GameInstance)
{
    mInstance = GameInstance;
    WindowBase::Init();
}

void WindowBase::Open()
{
    isOpen = true;
}

void WindowBase::Draw(const float deltaTime)
{
    // Base draw does nothing
}

void WindowBase::Close()
{
    isOpen = false;
}

void WindowBase::Tick(float deltaTime)
{
    // Base tick does nothing
}