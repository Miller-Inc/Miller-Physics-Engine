//
// Created by James Miller on 11/29/2025.
//

#include "Windows/Viewport.h"
#include <array>
#include "Vulkan/VulkanHelpers.h"
#include <vector>
#include <cstdint>
#include "imgui_impl_vulkan.h"
#include <SDL3/SDL.h>

static std::vector<uint8_t> MakeCheckerboard(unsigned int w, unsigned int h, unsigned int c,
                                             const std::array<uint8_t,3>& a,
                                             const std::array<uint8_t,3>& b,
                                             unsigned int tileSize = 8)
{
    if (c < 3) c = 3; // ensure at least RGB
    std::vector<uint8_t> pixels;
    pixels.reserve(size_t(w) * h * c);
    for (unsigned int y = 0; y < h; ++y) {
        for (unsigned int x = 0; x < w; ++x) {
            const bool useA = ((x / tileSize) + (y / tileSize)) % 2 == 0;
            const auto &col = useA ? a : b;
            pixels.push_back(col[0]); // R
            pixels.push_back(col[1]); // G
            pixels.push_back(col[2]); // B
            // optional alpha channel: pixels.push_back(255);
        }
    }
    return pixels;
}

static inline float lerp(float a, float b, float t) { return a + t * (b - a); }
static inline float smoothstep(float t) { return t * t * (3.0f - 2.0f * t); }

// cheap integer hash -> [0,1)
static inline float ihash(int x, int y, unsigned int seed)
{
    uint32_t h = uint32_t(x) * 374761393u + uint32_t(y) * 668265263u + seed * 974761741u;
    h = (h ^ (h >> 13)) * 1274126177u;
    return float(h & 0xFFFFFFu) / float(0x1000000u);
}

// value noise (bilinear)
static inline float valueNoise2D(float x, float y, unsigned int seed)
{
    const int x0 = int(floorf(x));
    const int y0 = int(floorf(y));
    const float fx = x - float(x0);
    const float fy = y - float(y0);
    const float sx = smoothstep(fx);
    const float sy = smoothstep(fy);

    const float n00 = ihash(x0,     y0,     seed);
    const float n10 = ihash(x0 + 1, y0,     seed);
    const float n01 = ihash(x0,     y0 + 1, seed);
    const float n11 = ihash(x0 + 1, y0 + 1, seed);

    const float ix0 = lerp(n00, n10, sx);
    const float ix1 = lerp(n01, n11, sx);
    return lerp(ix0, ix1, sy);
}

// simple FBM
static inline float fbm2D(float x, float y, unsigned int seed, int octaves = 5, float lacunarity = 2.0f, float gain = 0.5f)
{
    float amp = 1.0f;
    float freq = 1.0f;
    float sum = 0.0f;
    float norm = 0.0f;
    for (int i = 0; i < octaves; ++i)
    {
        sum += valueNoise2D(x * freq, y * freq, seed + i * 101) * amp;
        norm += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return sum / norm;
}

// Generate a natural ground texture: blended grass/dirt with rock speckles.
// - w/h: texture size
// - c: channels (will use at least 3)
// - seed: random seed
// - scale: base frequency scale (smaller -> larger features)
static std::vector<uint8_t> MakeNaturalGround(unsigned int w, unsigned int h, unsigned int c,
                                              unsigned int seed = 1337u, float scale = 0.0045f)
{
    if (c < 3) c = 3;
    std::vector<uint8_t> pixels;
    pixels.reserve(size_t(w) * h * c);

    // base colors (RGB)
    const std::array<uint8_t,3> grassA = {100, 150, 70};  // darker grass
    const std::array<uint8_t,3> grassB = {140, 190, 90};  // lighter grass
    const std::array<uint8_t,3> dirt   = {115, 85, 55};   // dirt
    const std::array<uint8_t,3> rock   = {90, 90, 95};    // rock speckle

    // per-pixel
    for (unsigned int y = 0; y < h; ++y)
    {
        for (unsigned int x = 0; x < w; ++x)
        {
            // normalized coords
            const float nx = float(x) / float(w);
            const float ny = float(y) / float(h);

            // fbm for large-scale terrain variation (controls grass vs dirt)
            float base = fbm2D(nx / scale, ny / scale, seed + 11, 5, 2.0f, 0.5f);

            // small detail noise for blades/variation
            float detail = fbm2D(nx / (scale * 4.0f), ny / (scale * 4.0f), seed + 77, 4, 2.0f, 0.45f);

            // rocky speckles (high frequency)
            float speck = valueNoise2D(nx / (scale * 12.0f), ny / (scale * 12.0f), seed + 333);

            // combine to a single factor in [0,1]
            float mixVal = std::clamp(base * 0.7f + detail * 0.25f, 0.0f, 1.0f);

            // bias so most plains are grassy, lower values -> dirt
            const float grassThreshold = 0.45f;
            float grassFactor = (mixVal - grassThreshold) / (1.0f - grassThreshold);
            grassFactor = std::clamp(grassFactor, 0.0f, 1.0f);

            // interpolate between dirt and two-tone grass
            const float grassBlend = (ihash(x, y, seed + 9) * 0.6f + 0.4f); // subtle per-pixel grass tint factor
            std::array<float,3> col{};
            for (int k = 0; k < 3; ++k)
            {
                float gcol = lerp(float(grassA[k]), float(grassB[k]), grassBlend);
                float baseCol = lerp(float(dirt[k]), gcol, grassFactor);

                // add fine detail
                baseCol += (detail - 0.5f) * 18.0f; // small bright/dark variation
                // rock speckles darken or tint toward rock color
                if (speck > 0.78f && ihash(x + 7, y + 13, seed + 5) > 0.6f) {
                    // stronger rock spot
                    baseCol = lerp(baseCol, float(rock[k]), 0.65f);
                } else if (speck > 0.72f) {
                    baseCol = lerp(baseCol, float(rock[k]), 0.28f);
                }

                // clamp to byte range
                col[k] = std::clamp(baseCol, 0.0f, 255.0f);
            }

            // push bytes (RGB)
            pixels.push_back(uint8_t(col[0]));
            pixels.push_back(uint8_t(col[1]));
            pixels.push_back(uint8_t(col[2]));
            // optional alpha funnel
            if (c >= 4) pixels.push_back(255);
        }
    }

    return pixels;
}

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

    const unsigned int texW = 64;
    const unsigned int texH = 64;
    const unsigned int texC = 3; // RGB
    constexpr std::array<uint8_t,3> lightCol = {200, 50, 200};
    constexpr std::array<uint8_t,3> darkCol  = {30, 30, 50};
    auto checker = MakeCheckerboard(texW, texH, texC, lightCol, darkCol, 8);

    // Upload points & triangles to the PhysicsObject.
    // Existing code used SetPoints(points, count) so assume SetTriangles(tris, count) exists.
    obj1->SetPoints(cubePoints.data(), (int)cubePoints.size());
    obj1->SetTriangles(cubeTris.data(), (int)cubeTris.size());
    obj1->SetPosition({0.0f, 5.0f, 0.0f}); // Move object 2 to the left
    obj1->PhysicsCallback = [](const float deltaTime, PhysicsObject* obj, PhysicsObject**, int) {
        // Simple rotation over time
        static float time = 0.0f; time += deltaTime;
        obj->TranslateRotate({deltaTime * 5 * cos(time), deltaTime * 5 * sin(time), 0.0f},{0.0f, deltaTime * PI, 0.0f});
    };

    obj2->SetPoints(cubePoints.data(), (int)cubePoints.size());
    obj2->SetTriangles(cubeTris.data(), (int)cubeTris.size());
    obj2->SetPosition(Vector(0.0f, 10.0f, 0.0f)); // Move object 2 to the right
    obj2->rotation_amount = { PI/1.5f, -(PI/1.5f) , 0.0f };
    obj2->PhysicsCallback = [](const float deltaTime, PhysicsObject* obj, PhysicsObject**, int) {
        // Simple rotation over time
        obj->Rotate({deltaTime * PI, 0.0f,0.0f});
    };

    obj1->SetAlbedo(checker.data(), texW, texH, texC);
    obj2->SetAlbedo(checker.data(), texW, texH, texC);

    PhysicsObject* floor = mEnvironment.SpawnPhysicsObject<PhysicsObject>();
    const std::vector<Vector> floorPoints = {
        Vector(-100.0f, 0.0f, -100.0f),
        Vector( 100.0f, 0.0f, -100.0f),
        Vector( 100.0f, 0.0f,  100.0f),
        Vector(-100.0f, 0.0f,  100.0f)
    };
    const std::vector<Triangle> floorTris = {
        tri(0, 1, 2), tri(0, 2, 3)
    };
    floor->SetPoints(floorPoints.data(), (int)floorPoints.size());
    floor->SetTriangles(floorTris.data(), (int)floorTris.size());
    floor->SetPosition(Vector(0.0f, -1.0f, 0.0f));
    floor->SetType(EPhysicsObjectType_Static);

    const unsigned int groundW = 1024;
    const unsigned int groundH = 1024;
    const unsigned int groundC = 3;
    auto groundTex = MakeNaturalGround(groundW, groundH, groundC, 123456u, 0.0045f);
    floor->SetAlbedo(groundTex.data(), groundW, groundH, groundC);

    floor->PhysicsCallback = [](const float deltaTime, PhysicsObject* obj, PhysicsObject**, int)
    {
        // Static object; no update needed as yet (collisions to come later)
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
    ImGui::SetNextWindowSize(ImVec2(1320, 750));

    if (ImGui::Begin("Viewport", &window, ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize))
    {
        // --- Input / capture handling before rendering so camera update affects this frame ---
        ImGuiIO& io = ImGui::GetIO();
        // float H = ImGui::GetWindowHeight(), W = ImGui::GetWindowWidth();

        const bool windowHovered = ImGui::IsWindowHovered(ImGuiHoveredFlags_AllowWhenBlockedByPopup | ImGuiHoveredFlags_ChildWindows);
        const bool leftClick = ImGui::IsMouseClicked(ImGuiMouseButton_Left);

        // Start capture on left click inside the viewport
        if (!mMouseCaptured && windowHovered && leftClick) {
            mMouseCaptured = true;
            // SDL_SetRelativeMouseMode(true);
            // SDL_SetRelativeMouseMode(SDL_TRUE); // capture & hide cursor
            SDL_HideCursor();
            // reset previous escape state
            mPrevEscDown = false;
        }

        // If captured, apply rotation from mouse delta
        if (mMouseCaptured) {
            // Apply small rotation based on mouse movement
            const ImVec2 md = io.MouseDelta; // delta in pixels
            const float yaw = md.x * mMouseSensitivity;
            const float pitch = md.y * mMouseSensitivity;

            // Rotate the environment's main camera.
            // Assumes Environment exposes MainCamera publicly; adjust if accessor exists.
            Camera_RotateYawPitchRoll(mEnvironment.MainCamera, yaw, pitch);

            // Release capture on Escape key press
            const bool escDown = ImGui::IsKeyPressed(ImGuiKey_Escape);

            if (escDown && !mPrevEscDown) {
                // Escape pressed -> release capture
                mMouseCaptured = false;
                // SDL_SetRelativeMouseMode(false);
                SDL_ShowCursor();
            }
            mPrevEscDown = escDown;

            // Also release capture if viewport loses focus/hover
            if (!windowHovered) {
                mMouseCaptured = false;
                // SDL_SetRelativeMouseMode(false);
                SDL_ShowCursor();
            }

            Camera_RotateYawPitchRoll(mEnvironment.MainCamera, yaw, pitch);

            // movement: use environment delta time
            const float dt = deltaTime;
            const float step = mMoveSpeed * dt;

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
                                                                 img, &mImg, &mImgMem, &mImgView, &mTex);
                if (tex)
                {
                    ImGui::SetCursorPos(ImVec2(0, 0));
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
    if (mMouseCaptured) {
        mMouseCaptured = false;
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
            if (mTex) {
                // ImGui_ImplVulkan_RemoveTexture(s_tex);
                mTex = 0;
            }

            // Destroy image view, image, and free memory (in correct order)
            if (mImgView != VK_NULL_HANDLE) {
                vkDestroyImageView(device, mImgView, nullptr);
                mImgView = VK_NULL_HANDLE;
            }
            if (mImg != VK_NULL_HANDLE) {
                vkDestroyImage(device, mImg, nullptr);
                mImg = VK_NULL_HANDLE;
            }
            if (mImgMem != VK_NULL_HANDLE) {
                vkFreeMemory(device, mImgMem, nullptr);
                mImgMem = VK_NULL_HANDLE;
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



