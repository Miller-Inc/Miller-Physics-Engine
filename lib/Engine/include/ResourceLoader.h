//
// Created by James Miller on 9/10/2025.
//

#pragma once
#include "Image.h"
#include <string>
#include <map>
#include <vector>

#include "Vector2.h"

typedef struct mImageLoadData
{
    std::string Path;
    std::string Name;
    MVector2 Position;
    MVector2 Scale;
    MVector2 Size;
    std::vector<std::string> WindowTags; // Tags for categorizing or grouping images
} MImageLoadData;

class ResourceLoader
{
public:
    ResourceLoader() = default;
    ~ResourceLoader() = default;

    std::map<std::string, MImageLoadData> ImageData{}; // Map to hold image load data with string keys
    bool LoadResources(const std::string& FilePath);
};
