//
// Created by James Miller on 11/29/2025.
//

#include "Engine/Rendering/Camera.cuh"
#if defined (STB_IMAGE_IMPLEMENTATION)
#include "stb_image.h"
#else
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#endif

bool ImageFileData::LoadImageFromFile()
{
    if (bLoaded && image)
        image->ReleaseCallback(*image);

    if (image)
    {
        delete image;
        image = nullptr;
    }

    int width = 0, height = 0, originalChannels = 0;
    constexpr int desiredChannels = STBI_rgb_alpha; // force RGBA (4)
    unsigned char* data = stbi_load(path.c_str(), &width, &height, &originalChannels, desiredChannels);
    if (!data)
    {
        printf("Could not open file: %s\n", path.c_str());
        fflush(stdout);
        bLoaded = false;
        return false;
    }

    metadata.width = width;
    metadata.height = height;
    metadata.numChannels = desiredChannels; // returned buffer has 'desiredChannels'
    metadata.bitsPerChannel = 8; // stbi_load returns 8-bit channels here

    const size_t bytes = size_t(metadata.width) * size_t(metadata.height) * size_t(metadata.numChannels) * (metadata.bitsPerChannel / 8);

    auto cpuImage = RawImage(width, height);
    cpuImage.IsDevice = false;
    cpuImage.Pixels = (Color*)malloc(bytes);
    cpuImage.ReleaseCallback = DefaultHostRawImageRelease;
    if (!cpuImage.Pixels)
    {
        stbi_image_free(data);
        bLoaded = false;
        return false;
    }

    memcpy(cpuImage.Pixels, data, bytes);
    stbi_image_free(data);

    image = new RawImage(cpuImage);
    bLoaded = true;
    return true;
}

bool ImageFileData::LoadImageFromFileGPU()
{
    if (bLoadedGPU && GPUImage)
    {
        GPUImage->ReleaseCallback(*GPUImage);
        delete GPUImage;
        GPUImage = nullptr;
    }

    // Ensure host image is loaded
    if (!bLoaded && !LoadImageFromFile())
    {
        bLoadedGPU = false;
        return false;
    }

    const size_t bytes = size_t(metadata.width) * size_t(metadata.height) * size_t(metadata.numChannels) * (metadata.bitsPerChannel / 8);

    auto gpuImage = RawImage((int)metadata.width, (int)metadata.height);
    gpuImage.IsDevice = true;
    gpuImage.ReleaseCallback = DefaultGPURawImageRelease;

    cudaError_t err = cudaMalloc(&(gpuImage.DevicePixels), bytes);
    if (err != cudaSuccess)
    {
        bLoadedGPU = false;
        return false;
    }

    err = cudaMemcpy(gpuImage.DevicePixels, image->Pixels, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cudaFree(gpuImage.DevicePixels);
        bLoadedGPU = false;
        return false;
    }


    delete GPUImage;

    GPUImage = new RawImage(gpuImage);
    bLoadedGPU = true;
    return true;
}

void ImageFileData::FreeImageData()
{
    if (bLoaded && image)
    {
        image->Release();
        // image->ReleaseCallback(*image);
        delete image;
        image = nullptr;
    }

    if (bLoadedGPU && GPUImage)
    {
        GPUImage->ReleaseDevice();
        // GPUImage->ReleaseCallback(*GPUImage);
        delete GPUImage;
        GPUImage = nullptr;
    }

    bLoaded = false;
    bLoadedGPU = false;
}
