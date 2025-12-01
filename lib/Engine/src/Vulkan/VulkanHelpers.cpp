//
// Created by James Miller on 11/30/2025.
//

#include "Vulkan/VulkanHelpers.h"
#include <cstdio>
#include <vector>
#include <cassert>
#include <cstring>

#include "imgui.h"
#include "imgui_impl_vulkan.h"
#include "Engine/Rendering/Camera.cuh"

static uint32_t findMemoryType(VkPhysicalDevice phys, uint32_t typeFilter, VkMemoryPropertyFlags properties)
{
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);
    for (uint32_t i = 0; i < memProps.memoryTypeCount; ++i) {
        if ((typeFilter & (1u << i)) && (memProps.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    std::fprintf(stderr, "findMemoryType: suitable memory type not found\n");
    return UINT32_MAX;
}

VkBuffer createStagingBuffer(VkDevice device, VkPhysicalDevice physicalDevice, VkDeviceSize size, VkDeviceMemory* outMem)
{
    VkBufferCreateInfo bufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufInfo.size = size;
    bufInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    bufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf = VK_NULL_HANDLE;
    if (vkCreateBuffer(device, &bufInfo, nullptr, &buf) != VK_SUCCESS) {
        std::fprintf(stderr, "createStagingBuffer: vkCreateBuffer failed\n");
        return VK_NULL_HANDLE;
    }

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buf, &req);

    uint32_t memType = findMemoryType(physicalDevice, req.memoryTypeBits,
                                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (memType == UINT32_MAX) {
        vkDestroyBuffer(device, buf, nullptr);
        return VK_NULL_HANDLE;
    }

    VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memType;

    VkDeviceMemory mem = VK_NULL_HANDLE;
    if (vkAllocateMemory(device, &alloc, nullptr, &mem) != VK_SUCCESS) {
        std::fprintf(stderr, "createStagingBuffer: vkAllocateMemory failed\n");
        vkDestroyBuffer(device, buf, nullptr);
        return VK_NULL_HANDLE;
    }

    vkBindBufferMemory(device, buf, mem, 0);
    if (outMem) *outMem = mem;
    return buf;
}

void createOrResizeImage(VkDevice device, VkPhysicalDevice phys, int w, int h, VkFormat fmt, VkImage* outImage, VkDeviceMemory* outMemory, VkImageView* outView /*= nullptr*/)
{
    if (!outImage || !outMemory) return;

    // Ensure all GPU work that might reference the old image/view has completed.
    // This is coarse (blocks until idle) but prevents "image in use by VkImageView" and
    // "bound memory was freed" validation errors. Replace with proper lifetime
    // management for better performance later.
    vkDeviceWaitIdle(device);

    // Destroy existing image view first (views reference the image).
    if (outView && *outView != VK_NULL_HANDLE) {
        vkDestroyImageView(device, *outView, nullptr);
        *outView = VK_NULL_HANDLE;
    }

    // Now safe to destroy the image and free memory.
    if (*outImage != VK_NULL_HANDLE) {
        vkDestroyImage(device, *outImage, nullptr);
        *outImage = VK_NULL_HANDLE;
    }
    if (*outMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, *outMemory, nullptr);
        *outMemory = VK_NULL_HANDLE;
    }

    // Create new image
    VkImageCreateInfo imgInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.extent.width = uint32_t(w);
    imgInfo.extent.height = uint32_t(h);
    imgInfo.extent.depth = 1;
    imgInfo.mipLevels = 1;
    imgInfo.arrayLayers = 1;
    imgInfo.format = fmt;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imgInfo.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateImage(device, &imgInfo, nullptr, outImage) != VK_SUCCESS) {
        std::fprintf(stderr, "createOrResizeImage: vkCreateImage failed\n");
        *outImage = VK_NULL_HANDLE;
        return;
    }

    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(device, *outImage, &req);

    uint32_t memType = findMemoryType(phys, req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    assert(memType != UINT32_MAX);

    VkMemoryAllocateInfo alloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    alloc.allocationSize = req.size;
    alloc.memoryTypeIndex = memType;

    if (vkAllocateMemory(device, &alloc, nullptr, outMemory) != VK_SUCCESS) {
        std::fprintf(stderr, "createOrResizeImage: vkAllocateMemory failed\n");
        vkDestroyImage(device, *outImage, nullptr);
        *outImage = VK_NULL_HANDLE;
        *outMemory = VK_NULL_HANDLE;
        return;
    }

    vkBindImageMemory(device, *outImage, *outMemory, 0);
}

VkCommandBuffer beginSingleUseCommands(VkDevice device, VkCommandPool pool)
{
    VkCommandBufferAllocateInfo alloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc.commandPool = pool;
    alloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &alloc, &cmd);

    VkCommandBufferBeginInfo begin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &begin);
    return cmd;
}

void endSingleUseCommands(VkDevice device, VkQueue queue, VkCommandPool pool, VkCommandBuffer cmd)
{
    vkEndCommandBuffer(cmd);

    VkSubmitInfo submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    vkFreeCommandBuffers(device, pool, 1, &cmd);
}

void transitionImageLayout(VkCommandBuffer cmd, VkImage image, VkFormat fmt, VkImageLayout oldLayout, VkImageLayout newLayout)
{
    VkImageMemoryBarrier barrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags srcStage = 0;
    VkPipelineStageFlags dstStage = 0;

    if (oldLayout == VK_IMAGE_LAYOUT_UNDEFINED && newLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (oldLayout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && newLayout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        // Generic fallback
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        srcStage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        dstStage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    }

    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void copyBufferToImage(VkCommandBuffer cmd, VkBuffer buffer, VkImage image, uint32_t w, uint32_t h)
{
    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0; // tightly packed
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = { 0, 0, 0 };
    region.imageExtent = { w, h, 1 };

    vkCmdCopyBufferToImage(cmd, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
}

VkImageView createImageView(VkDevice device, VkImage image, VkFormat fmt)
{
    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = fmt;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    VkImageView view = VK_NULL_HANDLE;
    if (vkCreateImageView(device, &viewInfo, nullptr, &view) != VK_SUCCESS) {
        std::fprintf(stderr, "createImageView: vkCreateImageView failed\n");
        return VK_NULL_HANDLE;
    }
    return view;
}

static VkSampler s_sampler = VK_NULL_HANDLE;

VkSampler getOrCreateSampler(VkDevice device)
{
    if (s_sampler != VK_NULL_HANDLE) return s_sampler;

    VkSamplerCreateInfo samp{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    samp.magFilter = VK_FILTER_LINEAR;
    samp.minFilter = VK_FILTER_LINEAR;
    samp.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samp.anisotropyEnable = VK_FALSE;
    samp.maxAnisotropy = 1.0f;
    samp.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    samp.unnormalizedCoordinates = VK_FALSE;
    samp.compareEnable = VK_FALSE;
    samp.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samp.mipLodBias = 0.0f;
    samp.minLod = 0.0f;
    samp.maxLod = 0.0f;

    if (vkCreateSampler(device, &samp, nullptr, &s_sampler) != VK_SUCCESS) {
        std::fprintf(stderr, "getOrCreateSampler: vkCreateSampler failed\n");
        return VK_NULL_HANDLE;
    }
    return s_sampler;
}

void DestroyVulkanHelpersResources(VkDevice device)
{
    // Ensure device isn't executing commands that reference these objects.
    vkDeviceWaitIdle(device);

    if (s_sampler != VK_NULL_HANDLE) {
        vkDestroySampler(device, s_sampler, nullptr);
        s_sampler = VK_NULL_HANDLE;
    }
}


VkDescriptorSet UploadRawImageToVulkan(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VkQueue queue,
    VkCommandPool cmdPool,
    const RawImage& img,
    VkImage* outImage,
    VkDeviceMemory* outImageMemory,
    VkImageView* outImageView,
    ImTextureID* outTex /* = nullptr */
)
{
    if (img.Width <= 0 || img.Height <= 0) {
        std::fprintf(stderr, "UploadRawImageToVulkan: invalid image size\n");
        return VK_NULL_HANDLE;
    }

    const int W = img.Width;
    const int H = img.Height;
    const auto imgBytes = VkDeviceSize(size_t(W) * size_t(H) * sizeof(uint32_t));
    const VkFormat fmt = VK_FORMAT_R8G8B8A8_UNORM;

    std::vector<uint8_t> hostBuf((size_t)imgBytes);
    if (img.IsDevice && img.DevicePixels) {
        cudaError_t cerr = cudaMemcpy(hostBuf.data(), img.DevicePixels, imgBytes, cudaMemcpyDeviceToHost);
        if (cerr != cudaSuccess) {
            std::fprintf(stderr, "UploadRawImageToVulkan: cudaMemcpyDeviceToHost failed: %d\n", (int)cerr);
            return VK_NULL_HANDLE;
        }
    } else if (img.Pixels) {
        std::memcpy(hostBuf.data(), reinterpret_cast<const void*>(img.Pixels), (size_t)imgBytes);
    } else {
        std::fprintf(stderr, "UploadRawImageToVulkan: no source pixels\n");
        return VK_NULL_HANDLE;
    }

    VkDeviceMemory stagingMem = VK_NULL_HANDLE;
    VkBuffer stagingBuf = createStagingBuffer(device, physicalDevice, imgBytes, &stagingMem);
    if (stagingBuf == VK_NULL_HANDLE || stagingMem == VK_NULL_HANDLE) {
        std::fprintf(stderr, "UploadRawImageToVulkan: createStagingBuffer failed\n");
        return VK_NULL_HANDLE;
    }

    void* mapped = nullptr;
    VkResult res = vkMapMemory(device, stagingMem, 0, imgBytes, 0, &mapped);
    if (res != VK_SUCCESS || !mapped) {
        std::fprintf(stderr, "UploadRawImageToVulkan: vkMapMemory failed (%d)\n", (int)res);
        vkDestroyBuffer(device, stagingBuf, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
        return VK_NULL_HANDLE;
    }
    std::memcpy(mapped, hostBuf.data(), (size_t)imgBytes);
    vkUnmapMemory(device, stagingMem);

    // Only create image/view if not already created.
    if (!outImage || *outImage == VK_NULL_HANDLE) {
        createOrResizeImage(device, physicalDevice, W, H, fmt, outImage, outImageMemory, outImageView);
        if (!outImage || *outImage == VK_NULL_HANDLE) {
            std::fprintf(stderr, "UploadRawImageToVulkan: createOrResizeImage failed\n");
            vkDestroyBuffer(device, stagingBuf, nullptr);
            vkFreeMemory(device, stagingMem, nullptr);
            return VK_NULL_HANDLE;
        }
    }

    VkCommandBuffer cmd = beginSingleUseCommands(device, cmdPool);
    if (cmd == VK_NULL_HANDLE) {
        std::fprintf(stderr, "UploadRawImageToVulkan: beginSingleUseCommands failed\n");
        vkDestroyBuffer(device, stagingBuf, nullptr);
        vkFreeMemory(device, stagingMem, nullptr);
        return VK_NULL_HANDLE;
    }

    // Transition/Copy
    transitionImageLayout(cmd, *outImage, fmt, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    copyBufferToImage(cmd, stagingBuf, *outImage, uint32_t(W), uint32_t(H));
    transitionImageLayout(cmd, *outImage, fmt, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    endSingleUseCommands(device, queue, cmdPool, cmd);

    vkDestroyBuffer(device, stagingBuf, nullptr);
    vkFreeMemory(device, stagingMem, nullptr);

    bool createdView = false;
    if (outImageView) {
        if (*outImageView == VK_NULL_HANDLE) {
            *outImageView = createImageView(device, *outImage, fmt);
            if (*outImageView == VK_NULL_HANDLE) {
                std::fprintf(stderr, "UploadRawImageToVulkan: createImageView failed\n");
                return VK_NULL_HANDLE;
            }
            createdView = true;
        }
    }

    VkSampler sampler = getOrCreateSampler(device);
    if (sampler == VK_NULL_HANDLE) {
        std::fprintf(stderr, "UploadRawImageToVulkan: getOrCreateSampler failed\n");
        return VK_NULL_HANDLE;
    }

    if (!outImageView || *outImageView == VK_NULL_HANDLE) {
        std::fprintf(stderr, "UploadRawImageToVulkan: image view missing\n");
        return VK_NULL_HANDLE;
    }

    // Only allocate/register a new ImGui texture if caller provided a storage pointer
    // and it is still null. This prevents leaking descriptor sets and exhausting the pool.
    if (outTex) {
        if (*outTex == 0) {
            VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(sampler, *outImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            if (!ds) {
                std::fprintf(stderr, "UploadRawImageToVulkan: ImGui_ImplVulkan_AddTexture failed\n");
                return VK_NULL_HANDLE;
            }
            *outTex = (ImTextureID)ds;
            return ds;
        } else {
            return (VkDescriptorSet)(*outTex);
        }
    }

    // Fallback: caller didn't request caching; still return a descriptor set but this will allocate each call.
    VkDescriptorSet ds = ImGui_ImplVulkan_AddTexture(sampler, *outImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    return ds;
}