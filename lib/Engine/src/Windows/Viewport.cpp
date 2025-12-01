//
// Created by James Miller on 11/29/2025.
//

#include "Windows/Viewport.h"
#include "Vulkan/VulkanHelpers.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

#include "imgui_impl_vulkan.h"
#include <SDL3/SDL.h>

static VkImage   s_img = VK_NULL_HANDLE;
static VkDeviceMemory s_imgMem = VK_NULL_HANDLE;
static VkImageView s_imgView = VK_NULL_HANDLE;
static ImTextureID s_tex = 0;

// Mouse capture state for viewport input handling
static bool s_mouseCaptured = false;
static float s_mouseSensitivity = 0.005f; // radians per pixel (tweak as needed)
static bool s_prevEscDown = false;
static float s_moveSpeed = 5.0f; // units per second

Viewport::Viewport()
{
    Name = "Viewport";
    isOpen = false;
}

Viewport::Viewport(const Viewport& other)
 : WindowBase(other) {
    Name = other.Name;
    isOpen = false; // Always start closed
}

void Viewport::Init()
{
    Viewport::Init("Viewport", nullptr);
}

void Viewport::Init(const std::string& WindowName, GInstance* Instance)
{
    WindowBase::Init(WindowName, Instance);
    mEnvironment.Init();

    // Spawn a physics object and validate the returned pointer
    PhysicsObject* obj1 = mEnvironment.SpawnPhysicsObject<PhysicsObject>();
    if (!obj1) {
        std::fprintf(stderr, "ERROR: SpawnPhysicsObject returned null\n");
        std::abort();
    }

    PhysicsObject* obj2 = mEnvironment.SpawnPhysicsObject<PhysicsObject>();

    // Create a more complex mesh: a unit cube centered at (0,0,-3)
    // 8 vertices, 12 triangles (two per face)
    const float zOffset = -3.0f;
    const float half = 0.5f;
    const std::vector<Vector> cubePoints = {
        Vector(-half, -half, -half + zOffset), // 0
        Vector( half, -half, -half + zOffset), // 1
        Vector( half,  half, -half + zOffset), // 2
        Vector(-half,  half, -half + zOffset), // 3
        Vector(-half, -half,  half + zOffset), // 4
        Vector( half, -half,  half + zOffset), // 5
        Vector( half,  half,  half + zOffset), // 6
        Vector(-half,  half,  half + zOffset)  // 7
    };

    // Helper lambda to build a Triangle
    auto tri = [](int a, int b, int c) {
        Triangle t; t.i0 = a; t.i1 = b; t.i2 = c; return t;
    };

    const std::vector<Triangle> cubeTris = {
        // -Z face
        tri(0, 1, 2), tri(0, 2, 3),
        // +Z face
        tri(5, 4, 7), tri(5, 7, 6),
        // -Y face
        tri(4, 0, 3), tri(4, 3, 7),
        // +Y face
        tri(1, 5, 6), tri(1, 6, 2),
        // -X face
        tri(4, 5, 1), tri(4, 1, 0),
        // +X face
        tri(3, 2, 6), tri(3, 6, 7)
    };

    // Upload points & triangles to the PhysicsObject.
    // Existing code used SetPoints(points, count) so assume SetTriangles(tris, count) exists.
    obj1->SetPoints(cubePoints.data(), (int)cubePoints.size());
    obj1->SetTriangles(cubeTris.data(), (int)cubeTris.size());
    obj1->SetPosition({-2.0f, 0.0f, 0.0f}); // Move object 2 to the left
    obj1->PhysicsCallback = [](const float deltaTime, PhysicsObject* obj, PhysicsObject**, int) {
        // Simple rotation over time
        static float time = 0.0f; time += deltaTime;
        obj->TranslateRotate({0.0f, deltaTime * sin(time), 0.0f},{0.0f, deltaTime * PI, 0.0f});
    };

    obj2->SetPoints(cubePoints.data(), (int)cubePoints.size());
    obj2->SetTriangles(cubeTris.data(), (int)cubeTris.size());
    obj2->SetPosition(Vector(2.0f, 0.0f, 0.0f)); // Move object 2 to the right
    obj2->rotation_amount = { PI/1.5f, -(PI/1.5f) , 0.0f };
    obj2->PhysicsCallback = [](const float deltaTime, PhysicsObject* obj, PhysicsObject**, int) {
        // Simple rotation over time
        obj->Rotate({deltaTime * PI, 0.0f,0.0f});
    };
}

void Viewport::Init(GInstance* GameInstance)
{
    Init("Viewport", GameInstance);
}

void Viewport::Open()
{
    WindowBase::Open();
    mEnvironment.GetDeltaTime();
}

void Viewport::Draw(float deltaTime)
{
    WindowBase::Draw(deltaTime);

    M_LOGGER(Logger::LogCore, Logger::Info, "Drawing Viewport Window");

    static bool window = true;
    ImGui::SetNextWindowSize(ImVec2(1280, 760));

    if (ImGui::Begin("Viewport", &window, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar))
    {
        // --- Input / capture handling before rendering so camera update affects this frame ---
        ImGuiIO& io = ImGui::GetIO();

        const bool windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_ChildWindows);
        const bool leftClick = ImGui::IsMouseClicked(ImGuiMouseButton_Left);

        // Start capture on left click inside the viewport
        if (!s_mouseCaptured && windowHovered && leftClick) {
            s_mouseCaptured = true;
            // SDL_SetRelativeMouseMode(true);
            // SDL_SetRelativeMouseMode(SDL_TRUE); // capture & hide cursor
            SDL_HideCursor();
            // reset previous escape state
            s_prevEscDown = false;
        }

        // If captured, apply rotation from mouse delta
        if (s_mouseCaptured) {
            // Apply small rotation based on mouse movement
            const ImVec2 md = io.MouseDelta; // delta in pixels
            const float yaw = md.x * s_mouseSensitivity;
            const float pitch = md.y * s_mouseSensitivity;

            // Rotate the environment's main camera.
            // Assumes Environment exposes MainCamera publicly; adjust if accessor exists.
            Camera_RotateYawPitchRoll(mEnvironment.MainCamera, yaw, pitch);

            // Release capture on Escape key press
            const bool escDown = ImGui::IsKeyPressed(ImGuiKey_Escape);

            if (escDown && !s_prevEscDown) {
                // Escape pressed -> release capture
                s_mouseCaptured = false;
                // SDL_SetRelativeMouseMode(false);
                SDL_ShowCursor();
            }
            s_prevEscDown = escDown;

            // Also release capture if viewport loses focus/hover
            if (!windowHovered) {
                s_mouseCaptured = false;
                // SDL_SetRelativeMouseMode(false);
                SDL_ShowCursor();
            }

            Camera_RotateYawPitchRoll(mEnvironment.MainCamera, yaw, pitch);

            // movement: use environment delta time
            const float dt = deltaTime;
            const float step = s_moveSpeed * dt;

            // accumulate movement in world (camera-relative) axes
            Vector mv{ 0.0f, 0.0f, 0.0f };
            if (ImGui::IsKeyDown(ImGuiKey_W)) {
                Vector f = Camera_GetForward(mEnvironment.MainCamera);
                mv.x += f.x * step; mv.y += f.y * step; mv.z += f.z * step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_S)) {
                Vector f = Camera_GetForward(mEnvironment.MainCamera);
                mv.x -= f.x * step; mv.y -= f.y * step; mv.z -= f.z * step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_A)) {
                Vector r = Camera_GetRight(mEnvironment.MainCamera);
                mv.x -= r.x * step; mv.y -= r.y * step; mv.z -= r.z * step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_D)) {
                Vector r = Camera_GetRight(mEnvironment.MainCamera);
                mv.x += r.x * step; mv.y += r.y * step; mv.z += r.z * step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_Space)) {
                Vector u = Camera_GetUp(mEnvironment.MainCamera);
                mv.x += u.x * step; mv.y += u.y * step; mv.z += u.z * step;
            }
            if (ImGui::IsKeyDown(ImGuiKey_LeftCtrl) || ImGui::IsKeyDown(ImGuiKey_LeftAlt)) {
                Vector u = Camera_GetUp(mEnvironment.MainCamera);
                mv.x -= u.x * step; mv.y -= u.y * step; mv.z -= u.z * step;
            }

            // apply movement
            mEnvironment.MainCamera.position.x += mv.x;
            mEnvironment.MainCamera.position.y += mv.y;
            mEnvironment.MainCamera.position.z += mv.z;
        }

        // --- Now render the scene using the (possibly updated) camera ---
        RawImage img = mEnvironment.RenderScene();

        if (mInstance)
        {
            if (const GPU::VulkanSetup* setup = mInstance->GetVulkanSetup())
            {
                VkDevice device = setup->device;
                VkPhysicalDevice physicalDevice = setup->physicalDevice;
                VkQueue graphicsQueue = setup->queue;
                VkCommandPool commandPool = setup->commandPool;

                // Use file-scope statics (persist across frames) to avoid re-creating/destroying each frame
                const VkDescriptorSet tex = UploadRawImageToVulkan(device, physicalDevice, graphicsQueue, commandPool,
                                                                 img, &s_img, &s_imgMem, &s_imgView, &s_tex);
                if (tex)
                {
                    ImGui::Image((ImTextureID)tex, ImVec2((float)img.Width, (float)img.Height));
                    if (img.ReleaseCallback) img.ReleaseCallback(const_cast<RawImage&>(img));
                }
                else
                {
                    ImGui::Text("Texture upload failed");
                }
            }
            else
            {
                ImGui::Text("Vulkan setup not available");
            }
        }
        else
        {
            ImGui::Text("No GInstance available");
        }
    }
    ImGui::End();
}

void Viewport::Close()
{
    // Ensure mouse capture is released if still active
    if (s_mouseCaptured) {
        s_mouseCaptured = false;
        // SDL_SetRelativeMouseMode(false);
        SDL_ShowCursor();
    }

    // Existing cleanup that destroys Vulkan resources and ImGui texture
    if (mInstance)
    {
        if (const GPU::VulkanSetup* setup = mInstance->GetVulkanSetup())
        {
            VkDevice device = setup->device;
            // Ensure GPU finished using descriptors/images
            vkDeviceWaitIdle(device);

            // Remove ImGui descriptor/registration first
            if (s_tex) {
                // ImGui_ImplVulkan_RemoveTexture(s_tex);
                s_tex = 0;
            }

            // Destroy image view, image, and free memory (in correct order)
            if (s_imgView != VK_NULL_HANDLE) {
                vkDestroyImageView(device, s_imgView, nullptr);
                s_imgView = VK_NULL_HANDLE;
            }
            if (s_img != VK_NULL_HANDLE) {
                vkDestroyImage(device, s_img, nullptr);
                s_img = VK_NULL_HANDLE;
            }
            if (s_imgMem != VK_NULL_HANDLE) {
                vkFreeMemory(device, s_imgMem, nullptr);
                s_imgMem = VK_NULL_HANDLE;
            }

            // Destroy helper-owned resources such as the static sampler
            DestroyVulkanHelpersResources(device);
        }
    }

    WindowBase::Close();
    mEnvironment.Destroy();
}

void Viewport::Tick(const float deltaTime)
{
    mEnvironment.TickAll(deltaTime); // Update physics environment
}



