cmake_minimum_required(VERSION 4.0)
project(MillerRenderLib LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

# Find modern CMake CUDA package (gives CUDA::cudart_static)
find_package(CUDAToolkit REQUIRED)

# build CUDA static library
add_subdirectory(lib/cuda/RenderLib)

# create executable
add_executable(MillerRenderLib main.cpp)

# include library headers and statically link the CUDA runtime into the exe
target_include_directories(MillerRenderLib PRIVATE ${CMAKE_SOURCE_DIR}/lib/cuda/RenderLib/include)
target_link_libraries(MillerRenderLib PRIVATE RenderLib CUDA::cudart_static)

# tell the compiler we link the static CUDA runtime
target_compile_definitions(MillerRenderLib PRIVATE CUDART_STATIC)

add_custom_command(TARGET MillerRenderLib POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:RenderLib>
        $<TARGET_FILE_DIR:MillerRenderLib>
)