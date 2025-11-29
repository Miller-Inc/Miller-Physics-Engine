//
// Created by James Miller on 9/4/2025.
//

/// Instance.h
/// This file contains the declaration of the Instance class, which manages the core game loop,
///     as well as window and texture management. This covers up most of the backend code, allowing
///     developers to focus on game logic and content. The main code has been taken and adapted from the
///     imgui Vulkan SDL3 example. Over time, the goal is to fully move this code to use the engine's
///     class structures and types, but for now, it is a hybrid of imgui example code and engine code.

#pragma once

#include <functional> // For std::function
#include "Core.h" // Core engine types and utilities
#include "ResourceLoader.h" // Resource loading utilities

/// Define the path to the resources JSON file
#define RESOURCE_RELATIVE_PATH "Resources/resources.json"

/// Forward declarations and type definitions

/// Forward declaration of Instance since it is used in callback typedefs
class GInstance;

/// Callback typedefs for window drawing, initialization, ticking, and client connections
typedef std::function<void()> DrawWindowCallback;
typedef std::function<void(GInstance*)> InitWindowCallback;
typedef std::function<void(float)> TickCallback;

typedef struct m_window
{
    std::string Name;
    DrawWindowCallback DrawCallback;
    InitWindowCallback InitCallback;
    TickCallback Ticker;
    bool* OpenWindow = nullptr;

} MWindow;

class GInstance
{
public:
    GInstance() = default;
    ~GInstance() = default;

    /// Prepare and run the game instance
    void Init(); // Initialize the game instance

    /// Update the game instance logic, called every frame
    void Tick(float deltaTime);

    /// Render the game instance, called every frame
    void Render(); // Render the game instance

    /// Texture Management

    /// Open an image and add it to the texture map
    bool OpenImage(const std::string& PathToImage, const std::string& ImageName, const MVector2& Position, const MVector2& Scale = {1.0f, 1.0f}); // Add an image to the texture map

    /// Get an image from the texture map
    MImage* GetImage(const std::string& ImageName); // Retrieve an image from the texture map

    /// Open an image if it doesn't exist, otherwise return the existing image
    MImage* OpenGetImage(const std::string& PathToImage, const std::string& ImageName, const MVector2& Position, const MVector2& Scale = {1.0f, 1.0f});

    /// Remove an image from the texture map
    bool RemoveImage(const std::string& ImageName);

        /// Delete an image from memory and the texture map
    bool DeleteImage(const std::string& ImageName);

    /// Window Management

    /// Add a new window to the game instance
    bool AddWindow(MWindow& newWindow); // Add a new window to the game instance

    /// Load resources for a specific window
    std::map<std::string, MImage*> LoadResources(const std::string& WindowName);

    /// Load all resources
    bool LoadResources();

    /// Exits the main loop and shuts down the game instance
    void StopMainLoop() { RunLoop = false; }

protected:
    /// Pre-window initialization (called before any windows are created) (Called once)
    void PreWindowInit();

    /// Cleanup resources (called on shutdown) (Called once)
    void Cleanup();

private:
    MVector2 mWindowSize = {800, 600}; // Default window size
    std::map<std::string, MImage> mTextures{};
    std::map<std::string, MWindow> mWindows{};

    GPU::VulkanSetup *mSetup = nullptr; // Pointer to Vulkan setup
    int Program(); // Main program loop
    bool IsRunning = false;
    bool RunLoop = true;

    /// Resource loader instance, handles loading resources from disk and labeling them for use
    ResourceLoader mResourceLoader{};

    void ParseResources();


};