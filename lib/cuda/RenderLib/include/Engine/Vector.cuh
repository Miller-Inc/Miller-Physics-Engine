//
// Created by James Miller on 11/13/2025.
//

#pragma once
#include "defns.h"
#include "CudaMacros.cuh"
#include <string>

typedef struct Vector
{
    float x = 0.0f, y = 0.0f, z = 0.0f;

    CUDA_CALLABLE_MEMBER Vector() noexcept;
    CUDA_CALLABLE_MEMBER Vector(float x, float y, float z) noexcept;


    CUDA_CALLABLE_MEMBER NO_DISCARD float magnitude() const;
    CUDA_CALLABLE_MEMBER NO_DISCARD float magnitudeSquared() const;
    CUDA_CALLABLE_MEMBER NO_DISCARD Vector normalize() const;
    CUDA_CALLABLE_MEMBER NO_DISCARD Vector add(const Vector& other) const;
    CUDA_CALLABLE_MEMBER NO_DISCARD Vector subtract(const Vector& other) const;
    CUDA_CALLABLE_MEMBER NO_DISCARD Vector multiply(float scalar) const;
    CUDA_CALLABLE_MEMBER NO_DISCARD float dot(const Vector& other) const;
    CUDA_CALLABLE_MEMBER NO_DISCARD Vector cross(const Vector& other) const;

    CUDA_CALLABLE_MEMBER Vector operator+(const Vector& other) const;
    CUDA_CALLABLE_MEMBER Vector operator-(const Vector& other) const;
    CUDA_CALLABLE_MEMBER Vector operator*(float scalar) const;
    CUDA_CALLABLE_MEMBER Vector operator/(float scalar) const;
    CUDA_CALLABLE_MEMBER Vector& operator+=(const Vector& other);
    CUDA_CALLABLE_MEMBER Vector& operator-=(const Vector& other);
    CUDA_CALLABLE_MEMBER Vector& operator*=(float scalar);
    CUDA_CALLABLE_MEMBER Vector& operator/=(float scalar);

    /// Cross product
    CUDA_CALLABLE_MEMBER Vector operator*(const Vector& other) const;
    CUDA_CALLABLE_MEMBER Vector operator*=(const Vector& other);

    /// Dot product
    CUDA_CALLABLE_MEMBER float operator|(const Vector& other) const;

    CUDA_CALLABLE_MEMBER std::string toString() const;

} Vector;
