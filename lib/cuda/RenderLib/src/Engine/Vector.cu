//
// Created by James Miller on 11/13/2025.
//
#include "../../include/Engine/Vector.cuh"
#include <cuda_runtime.h>

static constexpr float VECTOR_EPS = 1e-8f;

CUDA_CALLABLE_MEMBER Vector::Vector() noexcept
{
    x = y = z = 0.0f;
}

CUDA_CALLABLE_MEMBER Vector::Vector(const float x, const float y, const float z) noexcept
    : x(x), y(y), z(z)
{}

CUDA_CALLABLE_MEMBER float Vector::magnitude() const
{
    return sqrtf(magnitudeSquared());
}

CUDA_CALLABLE_MEMBER float Vector::magnitudeSquared() const
{
    return x*x + y*y + z*z;
}

CUDA_CALLABLE_MEMBER Vector Vector::normalize() const
{
    float mag = magnitude();
    if (mag <= VECTOR_EPS) return Vector{0.0f, 0.0f, 0.0f};
    float inv = 1.0f / mag;
    return Vector{x * inv, y * inv, z * inv};
}

CUDA_CALLABLE_MEMBER Vector Vector::add(const Vector& other) const
{
    return Vector{x + other.x, y + other.y, z + other.z};
}

CUDA_CALLABLE_MEMBER Vector Vector::subtract(const Vector& other) const
{
    return Vector{x - other.x, y - other.y, z - other.z};
}

CUDA_CALLABLE_MEMBER Vector Vector::multiply(float scalar) const
{
    return Vector{x * scalar, y * scalar, z * scalar};
}

CUDA_CALLABLE_MEMBER float Vector::dot(const Vector& other) const
{
    return x * other.x + y * other.y + z * other.z;
}

CUDA_CALLABLE_MEMBER Vector Vector::cross(const Vector& other) const
{
    return Vector{
        y * other.z - z * other.y,
        z * other.x - x * other.z,
        x * other.y - y * other.x
    };
}

/* Operators */

CUDA_CALLABLE_MEMBER Vector Vector::operator+(const Vector& other) const { return add(other); }
CUDA_CALLABLE_MEMBER Vector Vector::operator-(const Vector& other) const { return subtract(other); }
CUDA_CALLABLE_MEMBER Vector Vector::operator*(float scalar) const { return multiply(scalar); }

CUDA_CALLABLE_MEMBER Vector Vector::operator/(float scalar) const
{
    if (fabsf(scalar) <= VECTOR_EPS) return Vector{0.0f, 0.0f, 0.0f};
    float inv = 1.0f / scalar;
    return Vector{x * inv, y * inv, z * inv};
}

CUDA_CALLABLE_MEMBER Vector& Vector::operator+=(const Vector& other)
{
    x += other.x; y += other.y; z += other.z;
    return *this;
}

CUDA_CALLABLE_MEMBER Vector& Vector::operator-=(const Vector& other)
{
    x -= other.x; y -= other.y; z -= other.z;
    return *this;
}

CUDA_CALLABLE_MEMBER Vector& Vector::operator*=(float scalar)
{
    x *= scalar; y *= scalar; z *= scalar;
    return *this;
}

CUDA_CALLABLE_MEMBER Vector& Vector::operator/=(float scalar)
{
    if (fabsf(scalar) <= VECTOR_EPS) { x = y = z = 0.0f; return *this; }
    float inv = 1.0f / scalar;
    x *= inv; y *= inv; z *= inv;
    return *this;
}

/* Cross product operator signatures from header */
CUDA_CALLABLE_MEMBER Vector Vector::operator*(const Vector& other) const
{
    return cross(other);
}

/* Header declared return type as Vector for operator*= with Vector */
CUDA_CALLABLE_MEMBER Vector Vector::operator*=(const Vector& other)
{
    x = y * other.z - z * other.y;
    y = z * other.x - x * other.z; // careful: this uses updated x above -> compute temp to be safe
    z = x * other.y - y * other.x;
    // To avoid using updated components incorrectly, do the proper component-wise cross:
    const float cx = y * other.z - z * other.y;
    const float cy = z * other.x - x * other.z;
    const float cz = x * other.y - y * other.x;
    x = cx; y = cy; z = cz;
    return Vector{x, y, z};
}

/* Dot-product operator */
CUDA_CALLABLE_MEMBER float Vector::operator|(const Vector& other) const
{
    return dot(other);
}

CUDA_CALLABLE_MEMBER std::string Vector::toString() const
{
    return "(" + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

