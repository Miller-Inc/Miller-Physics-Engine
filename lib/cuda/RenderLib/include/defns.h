//
// Created by James Miller on 11/13/2025.
//

#pragma once
#if defined(_WIN32) || defined(_WIN64)
    #define DLL_EXPORT __declspec(dllexport)
#else
    #define DLL_EXPORT __attribute__((visibility("default")))
#endif

#define CHECK_CUDA(e) \
{ \
if (e != cudaSuccess) {\
std::cerr << "CUDA error at " << __FILE__ << " line " << __LINE__ << ": " << cudaGetErrorString(e) << std::endl;\
}\
}

#if defined(__cplusplus)
#define FUNC_DEF extern "C" DLL_EXPORT
#define CLASS_DEF extern "C" DLL_EXPORT
#else
#define LIB_HEADER_START
#define LIB_HEADER_END
#define FUNC_DEF DLL_EXPORT
#endif

#define FUNCTION_SIGNATURE(rettype, name, ...) FUNC_DEF rettype name(__VA_ARGS__)
#define NO_DISCARD [[nodiscard]]

#define DEBUG false

#define PI 3.14159265358979323846f

static unsigned long long GetNextIdentifier()
{
    static unsigned long long currentId = 0;
    return currentId++;
}