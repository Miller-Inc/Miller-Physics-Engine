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
};
