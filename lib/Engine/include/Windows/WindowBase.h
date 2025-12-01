//
// Created by James Miller on 9/10/2025.
//

/// WindowBase.h
/// This file contains the declaration of the WindowBase class, which serves as a base class for
///     all GUI windows in the application. It provides common functionality such as initialization,
///     opening, closing, and drawing the window using ImGui. The class also manages a collection
///     of textures that can be used within the window. Derived classes can override the virtual
///     methods to implement specific behaviors for different types of windows. This class is essential
///     for creating a consistent and reusable window management system in the GUI framework. This allows
///     a developer to create new windows by inheriting from this base class and implementing
///     the necessary functionality and adding the window to the render loop in the GameInstance class.

#pragma once
#include "GInstance.h" // Forward declaration of Instance
#include "Core.h" // Include core engine types
#include <map>
#include <string>
#include "imgui.h" // Include ImGui for GUI rendering for this class and all derived classes

class WindowBase
{
public:
    virtual void Init();
    virtual void Init(const std::string& WindowName, GInstance* Instance);
    virtual void Init(GInstance* GameInstance);
    WindowBase(const WindowBase& other)
    {
        Name = other.Name;
        isOpen = other.isOpen;
    }
    virtual ~WindowBase() = default;

    /// Open the window (should only call once)
    virtual void Open();

    /// Draw the window (called every frame)
    virtual void Draw(float deltaTime);

    /// Close the window (should only call once)
    virtual void Close();

    /// Tick function for updating window state and other logic
    virtual void Tick(float deltaTime);

    /// Indicates if the window is currently open
    bool isOpen = false;

    /// Name of the window, used to identify it to the resource manager
    std::string Name = "WindowBase";
protected:
    /// Pointer to the engine instance
    GInstance* mInstance = nullptr;

    /// Map to hold images/textures associated with this window
    std::map<std::string, MImage*> mTextures{}; // Map to hold textures with string keys
public:
    WindowBase() = default;

    /// Get the name of the window
    [[nodiscard]] std::string GetName() const { return Name; }
};