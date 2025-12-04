cmake_minimum_required(VERSION 4.0)
project(RenderLib LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/json EXCLUDE_FROM_ALL)

add_library(RenderLib SHARED
        src/Test.cu
        include/MillerCudaLibrary.h
        include/Test.h
        include/defns.h
        include/Engine/Physics/PhysicsObject.cuh
        src/Engine/Physics/PhysicsObject.cu
        include/Engine/Vector.cuh
        src/Engine/Vector.cu
        include/Engine/Environment.h
        src/Engine/Environment.cu
        src/Engine/Environment.cpp
        include/Engine/Constructor.h
        include/Engine/EngineClock.h
        include/Engine/Quaternion.cuh
        src/Engine/Quaternion.cu
        include/Engine/Triangle.cuh
        include/Engine/CudaMacros.cuh
        include/Engine/Rendering/Camera.cuh
        src/Engine/Rendering/Camera.cu
        include/Engine/Rendering/Color.cuh
        include/Engine/Rendering/AABB.cuh
        include/Engine/Rendering/BVHNode.cuh
        include/Engine/Rendering/BVHBuilder.h
        include/Engine/Rendering/BVHTraverse.cuh)

target_include_directories(RenderLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(RenderLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)
target_include_directories(RenderLib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/json/include)

target_link_libraries(RenderLib PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(RenderLib PUBLIC ${NLOHMANN_JSON_TARGET_NAME})

set_target_properties(RenderLib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_STANDARD 20
        WINDOWS_EXPORT_ALL_SYMBOLS ON
)

# optional: define export macro for MSVC builds
if (MSVC)
    target_compile_definitions(RenderLib PRIVATE RENDERLIB_EXPORTS)
endif()