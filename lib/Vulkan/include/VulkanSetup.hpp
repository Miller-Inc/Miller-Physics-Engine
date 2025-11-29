//
// Created by James Miller on 9/3/2025.
//

/// VulkanSetup.hpp
/// This file defines the VulkanSetup class, which provides utility functions for setting up Vulkan resources such
///     as buffers, images, and textures. It is designed to work with an existing Vulkan instance and device.
///     The main purpose of this class is to simplify common Vulkan operations and resource management. Most of the
///     code has been taken from Vulkan official examples and adapted for this engine.

#pragma once
#include <string>
#include "MacroDefs.h"
#include "Vector2.h"
#if PLATFORM_MACOSX || PLATFORM_UNIX
#include <vulkan/vulkan.h> // For vulkan functions
#elif PLATFORM_WINDOWS
#include <Vulkan/vulkan.h> // For vulkan functions
#endif

namespace GPU
{


    class VulkanSetup
    {
    public:

        typedef struct textureImage
        {
            VkImage image{};
            VkDeviceMemory memory{};
            VkDescriptorSet textureId{};
            MVector2 size;
            VkImageView ImageView{ VK_NULL_HANDLE };
            VkSampler Sampler{ VK_NULL_HANDLE };
        } TextureImage;

        VulkanSetup(VkInstance instance, VkDevice device, VkPhysicalDevice physical, VkQueue graphics_queue, VkCommandPool cmd_pool); // Constructor to initialize with a Vulkan device

        void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) const;
        [[nodiscard]] uint32_t FindMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
        void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags properties, VkImage& image, VkDeviceMemory& imageMemory) const;
        [[nodiscard]] TextureImage CreateTextureImage(const std::string& filename) const;

        [[nodiscard]] VkCommandBuffer BeginSingleTimeCommands() const;
        void EndSingleTimeCommands(VkCommandBuffer commandBuffer) const;

        VkInstance instance{ VK_NULL_HANDLE };
        VkPhysicalDevice physicalDevice{ VK_NULL_HANDLE };
        VkDevice device{ VK_NULL_HANDLE };
        VkQueue queue{ VK_NULL_HANDLE };
        VkCommandPool commandPool{ VK_NULL_HANDLE };
    };
}