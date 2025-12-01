//
// Created by James Miller on 11/30/2025.
//

#pragma once
#include <vulkan/vulkan.h>

#include "imgui.h"
#include "Engine/Rendering/Camera.cuh"

void DestroyVulkanHelpersResources(VkDevice device);
VkBuffer createStagingBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkDeviceMemory* outMem);
void createOrResizeImage(VkDevice device, VkPhysicalDevice phys, int w, int h, VkFormat fmt, VkImage* outImage, VkDeviceMemory* outMemory, VkImageView* outView = nullptr);
VkCommandBuffer beginSingleUseCommands(VkDevice device, VkCommandPool pool);
void endSingleUseCommands(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd);
void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkFormat fmt, VkImageLayout oldLayout, VkImageLayout newLayout);
void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t w, uint32_t h);
VkImageView createImageView(VkDevice device, VkImage image, VkFormat fmt);
VkSampler getOrCreateSampler(VkDevice device);
VkDescriptorSet UploadRawImageToVulkan(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkQueue queue,
    VkCommandPool cmdPool,
    const RawImage& img,
    VkImage* outImage,
    VkDeviceMemory* outImageMemory,
    VkImageView* outImageView,
    ImTextureID* outTex /* = nullptr */);