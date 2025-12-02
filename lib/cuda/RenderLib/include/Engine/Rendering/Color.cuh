//
// Created by James Miller on 11/29/2025.
//

#pragma once
#include "Engine/CudaMacros.cuh"
#include <cstdint>
#include <algorithm>
#include <cmath>

typedef struct Color
{
    float r{}, g{}, b{}, a{};

    CUDA_CALLABLE_MEMBER Color() noexcept : r(0), g(0), b(0), a(1) {}
    CUDA_CALLABLE_MEMBER Color(float rr, float gg, float bb, float aa = 1.0f) noexcept : r(rr), g(gg), b(bb), a(aa) {}

    CUDA_CALLABLE_MEMBER Color operator+(const Color& o) const noexcept { return {r+o.r, g+o.g, b+o.b, a+o.a}; }
    CUDA_CALLABLE_MEMBER Color operator*(float s) const noexcept { return {r*s, g*s, b*s, a*s}; }
    CUDA_CALLABLE_MEMBER Color& operator+=(const Color& o) noexcept { r+=o.r; g+=o.g; b+=o.b; a+=o.a; return *this; }

    CUDA_CALLABLE_MEMBER void clamp01() noexcept {
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        b = fminf(fmaxf(b, 0.0f), 1.0f);
        a = fminf(fmaxf(a, 0.0f), 1.0f);
    }

    // simple gamma correction (to sRGB)
    CUDA_CALLABLE_MEMBER void applyGamma(float gamma = 1.0f/2.2f) noexcept {
        r = powf(fmaxf(r, 0.0f), gamma);
        g = powf(fmaxf(g, 0.0f), gamma);
        b = powf(fmaxf(b, 0.0f), gamma);
    }

    // pack to 0xAARRGGBB (8-bit per channel)
    CUDA_CALLABLE_MEMBER uint32_t toRGBA8() const noexcept {
        Color c = *this;
        c.clamp01();
        const uint32_t R = static_cast<uint32_t>(c.r * 255.0f + 0.5f) & 0xFF;
        const uint32_t G = static_cast<uint32_t>(c.g * 255.0f + 0.5f) & 0xFF;
        const uint32_t B = static_cast<uint32_t>(c.b * 255.0f + 0.5f) & 0xFF;
        const uint32_t A = static_cast<uint32_t>(c.a * 255.0f + 0.5f) & 0xFF;
        return (A << 24) | (R << 16) | (G << 8) | (B);
    }
} Color;