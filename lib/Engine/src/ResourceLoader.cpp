//
// Created by James Miller on 9/10/2025.
//

#include "ResourceLoader.h"
#include "Logger.h"
#include <fstream>
#include <ranges>
#include "nlohmann/json.hpp"
using json = nlohmann::json;

bool ResourceLoader::LoadResources(const std::string& FilePath)
{
    ImageData.clear();
    std::ifstream file(FilePath);
    if (!file.is_open())
    {
        M_LOGGER(Logger::LogCore, Logger::Error, "Failed to open resource file: " + FilePath);
        return false;
    }

    json json_object = json::parse(file);
    auto images = json_object["images"];

    if (images == json::value_t::discarded || !images.is_array())
    {
        M_LOGGER(Logger::LogCore, Logger::Error, "Invalid or missing 'images' array in resource file: " + FilePath);
        return false;
    }

    // auto img_arr = images->array();

    for (const auto& image : images)
    {
        MImageLoadData data;
        data.Path = image.value("path", "");
        data.Name = image.value("name", "");
        if (image.contains("position") && image["position"].is_object())
        {
            data.Position.x = image["position"].value("x", 0.0f);
            data.Position.y = image["position"].value("y", 0.0f);
        } else
        {
            data.Position = {0.0f, 0.0f};
        }

        if (image.contains("scale") && image["scale"].is_object())
        {
            data.Scale.x = image["scale"].value("x", 1.0f);
            data.Scale.y = image["scale"].value("y", 1.0f);
        } else
        {
            data.Scale = {1.0f, 1.0f};
        }

        data.Size = {-1.0f, -1.0f};

        if (image.contains("pages") && image["pages"].is_array())
        {
            for (const auto& tag : image["pages"])
            {
                data.WindowTags.emplace_back(tag.get<std::string>());
            }
        }

        ImageData.emplace(data.Name, data);
    }

    M_LOGGER(Logger::LogCore, Logger::Info, "Successfully loaded resources from: " + FilePath);
    return true;
}

