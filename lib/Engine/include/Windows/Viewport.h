//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include <MillerCudaLibrary.h>
#include "WindowBase.h"

class Viewport final : public WindowBase
{
public:
    Viewport();
    Viewport(const Viewport& other);
    ~Viewport() override = default;
    void Init() override;
    void Init(const std::string& WindowName, GInstance* Instance) override;
    void Init(GInstance* GameInstance) override;
    void Open() override;
    void Draw(float deltaTime) override;
    void Close() override;

    /// Tick function for updating physics and rendering
    void Tick(float deltaTime) override;

    std::string Name = "Viewport";

private:
    Environment mEnvironment{};

    VkImage   mImg = VK_NULL_HANDLE;
    VkDeviceMemory mImgMem = VK_NULL_HANDLE;
    VkImageView mImgView = VK_NULL_HANDLE;
    ImTextureID mTex = 0;

    // Mouse capture state for viewport input handling
    bool mMouseCaptured = false;
    float mMouseSensitivity = 0.005f; // radians per pixel (tweak as needed)
    bool mPrevEscDown = false;
    float mMoveSpeed = 5.0f; // units per second
};
