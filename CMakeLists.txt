cmake_minimum_required(VERSION 4.0)
project(MillerRenderLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find modern CMake CUDA package (gives CUDA::cudart_static)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIGURATION>")

find_package(Vulkan REQUIRED)

# Icon Handling
if (WIN32)
    set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_SOURCE_DIR}/Resources/ico/Windows/app.rc")
    set(APP_ICON_RESOURCE ${APP_ICON_RESOURCE_WINDOWS})
    add_definitions(-DAPP_ICON_RESOURCE="${APP_ICON_RESOURCE}")
    message(STATUS "App icon resource set for Windows: ${APP_ICON_RESOURCE}")
elseif(UNIX)
    set(APP_ICON_RESOURCE "")  # No icon resource for Linux in this example
else()
    set(APP_ICON_RESOURCE "")  # No icon resource for other platforms
endif()


# This assumes the SDL source is available in the third_party/sdl3 directory.
add_subdirectory(third_party/sdl3 EXCLUDE_FROM_ALL)
add_subdirectory(third_party/sdl3_net)
add_subdirectory(third_party/json)


# build CUDA static library
add_subdirectory(lib/cuda/RenderLib)


# ****************** Executable Configuration ****************** #
# Determine executable flags based on platform and build type
if (WIN32 AND CMAKE_BUILD_TYPE STREQUAL "Release")
    # For Windows Release builds, we use the WIN32 flag to suppress the console window.
    set(EXE_FLAGS WIN32)
    message(STATUS "Building for Windows Release with flags: ${EXE_FLAGS}")
else()
    set(EXE_FLAGS "")
endif()

# List of source files for the MillerRenderLib executable
set(MILLER_RENDER_LIB_SOURCES
        main.cpp
        lib/Engine/src/InputValidation.cpp
        lib/Engine/src/ResourceLoader.cpp
        lib/Engine/src/Vector2.cpp
        lib/Vulkan/src/VulkanSetup.cpp
        lib/Engine/src/GInstance.cpp
        lib/Engine/src/Windows/WindowBase.cpp
        lib/Engine/src/Windows/Viewport.cpp
        lib/Engine/src/Vulkan/VulkanHelpers.cpp
)

# List of header files for the MillerRenderLib executable
set(MILLER_RENDER_LIB_HEADERS
    lib/Engine/include/Core.h
    lib/Engine/include/Image.h
    lib/Engine/include/InputValidation.h
    lib/Engine/include/Logger.h
    lib/Engine/include/MacroDefs.h
    lib/Engine/include/ResourceLoader.h
    lib/Engine/include/Vector2.h
    lib/Vulkan/include/VulkanSetup.hpp
    lib/Engine/include/GInstance.h
    lib/Engine/include/Windows/WindowBase.h
        lib/Engine/include/Windows/Viewport.h
        lib/Engine/include/Vulkan/VulkanHelpers.h
)

# create executable
add_executable(MillerRenderLib ${EXE_FLAGS}
    ${MILLER_RENDER_LIB_SOURCES}
    ${MILLER_RENDER_LIB_HEADERS}
)
# Add the application icon resource if it exists
target_sources(MillerRenderLib PRIVATE ${APP_ICON_RESOURCE})

# include library headers and statically link the CUDA runtime into the exe
target_include_directories(MillerRenderLib PRIVATE ${CMAKE_SOURCE_DIR}/lib/cuda/RenderLib/include)
target_link_libraries(MillerRenderLib PRIVATE RenderLib CUDA::cudart_static)

# Include directories for the library target
target_include_directories(MillerRenderLib PRIVATE imgui third_party/imgui)
target_include_directories(MillerRenderLib PRIVATE imgui third_party/imgui/backends)
target_include_directories(MillerRenderLib PRIVATE stb third_party/stb)
target_include_directories(MillerRenderLib PRIVATE Miller lib/Engine/include)
target_include_directories(MillerRenderLib PRIVATE Miller lib/Vulkan/include)
target_include_directories(MillerRenderLib PRIVATE third_party/json/include)

# Define preprocessor definitions for static linking of SDL3 and SDL3_net
target_compile_definitions(MillerRenderLib PUBLIC SDL_STATIC SDL3_STATIC SDL3_NET_STATIC)

# ImGui setup
set(IMGUI_DIR third_party/imgui)
set(IMGUI_BACKEND_DIR ${IMGUI_DIR}/backends)

# Add ImGui source files to the library target
target_sources(MillerRenderLib PRIVATE
        ${IMGUI_DIR}/imgui.cpp
        ${IMGUI_DIR}/imgui_draw.cpp
        ${IMGUI_DIR}/imgui_tables.cpp
        ${IMGUI_DIR}/imgui_widgets.cpp
        ${IMGUI_BACKEND_DIR}/imgui_impl_vulkan.cpp
        ${IMGUI_BACKEND_DIR}/imgui_impl_sdl3.cpp
)

# Link to the actual SDL3 and Vulkan libraries.
target_link_libraries(MillerRenderLib PUBLIC SDL3::SDL3 SDL3_net::SDL3_net Vulkan::Vulkan)

# Link to the nlohmann_json library
target_link_libraries(MillerRenderLib PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(MillerRenderLib PUBLIC ${NLOHMANN_JSON_TARGET_NAME})

# Copy Resources folder to the output directory after building
add_custom_command(TARGET MillerRenderLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/Resources
        $<TARGET_FILE_DIR:MillerRenderLib>/Resources
)

add_custom_command(TARGET MillerRenderLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/lib/cuda/RenderLib/Engine_Resources
        $<TARGET_FILE_DIR:MillerRenderLib>/Engine_Resources
)

# tell the compiler we link the static CUDA runtime
target_compile_definitions(MillerRenderLib PRIVATE CUDART_STATIC)

add_custom_command(TARGET MillerRenderLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:RenderLib>
        $<TARGET_FILE:SDL3::SDL3>
        $<TARGET_FILE:SDL3_net::SDL3_net>
        $<TARGET_FILE_DIR:MillerRenderLib>
)