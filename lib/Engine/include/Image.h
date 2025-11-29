//
// Created by James Miller on 9/3/2025.
//

#pragma once

#include "Vector2.h"
#include "VulkanSetup.hpp"
#include <cstdint>
#include <string>

typedef struct mImage
{
    GPU::VulkanSetup::TextureImage TextureHandle;
    std::string Name;
    MVector2 Size{}; // Size of the image
    MVector2 Position{}; // Position of the image
    MVector2 Scale{ 1.0f, 1.0f}; // Scale of the image
    int32_t ref_count = 0; // Reference count for shared usage
} MImage;
