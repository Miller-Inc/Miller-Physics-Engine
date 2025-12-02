//
// Created by James Miller on 11/25/2025.
//

#pragma once
#include "../defns.h"
#include "CudaMacros.cuh"

#include "Vector.cuh"

typedef struct Quaternion
{
    float w = 1.0f, x = 0.0f, y = 0.0f, z = 0.0f;

    /// Default constructor
    CUDA_CALLABLE_MEMBER Quaternion() noexcept;

    /// Component-wise constructor
    CUDA_CALLABLE_MEMBER Quaternion(float w, float x, float y, float z) noexcept;

    /// Create a vector with w=0 and (x,y,z) from the given vector
    CUDA_CALLABLE_MEMBER explicit Quaternion(const Vector& vec) noexcept;

    /// Normalize
    CUDA_CALLABLE_MEMBER Quaternion& normalize();

    /// Conjugate
    CUDA_CALLABLE_MEMBER NO_DISCARD Quaternion conjugate() const;

    /// Magnitude
    CUDA_CALLABLE_MEMBER float magnitude() const;

    /// Squared magnitude
    CUDA_CALLABLE_MEMBER float magnitudeSquared() const;

    /// Dot product
    CUDA_CALLABLE_MEMBER float dot(const Quaternion& other) const;

    /// Cross product
    CUDA_CALLABLE_MEMBER Quaternion hamilton_product(const Quaternion& other) const;

    /// Dot product with a vector treated as a quaternion with w=0
    CUDA_CALLABLE_MEMBER float dot(const Vector& other) const;

    /// Cross product with a vector treated as a quaternion with w=0
    CUDA_CALLABLE_MEMBER Quaternion hamilton_product(const Vector& other) const;

    /// Addition
    CUDA_CALLABLE_MEMBER Quaternion operator+(const Quaternion& other) const;

    /// Subtraction
    CUDA_CALLABLE_MEMBER Quaternion operator-(const Quaternion& other) const;

    /// Scalar multiplication
    CUDA_CALLABLE_MEMBER Quaternion operator*(float scalar) const;

    /// Scalar division
    CUDA_CALLABLE_MEMBER Quaternion operator/(float scalar) const;

    /// Addition assignment
    CUDA_CALLABLE_MEMBER Quaternion& operator+=(const Quaternion& other);

    /// Subtraction assignment
    CUDA_CALLABLE_MEMBER Quaternion& operator-=(const Quaternion& other);

    /// Scalar multiplication assignment
    CUDA_CALLABLE_MEMBER Quaternion& operator*=(float scalar);

    /// Scalar division assignment
    CUDA_CALLABLE_MEMBER Quaternion& operator/=(float scalar);

    /// Cross product
    CUDA_CALLABLE_MEMBER Quaternion operator*(const Quaternion& other) const;

    /// Cross product assignment
    CUDA_CALLABLE_MEMBER Quaternion& operator*=(const Quaternion& other);

    /// Expects a unit axis vector and an angle in radians, returns a normalized quaternion
    static CUDA_CALLABLE_MEMBER Quaternion fromAxisAngle(const Vector& axis, float angleRad);

    /// Expects Euler angles in radians, returns a normalized quaternion
    static CUDA_CALLABLE_MEMBER Quaternion fromEuler(const Vector& euler) noexcept;

    /// Expects a 3x3 rotation matrix in column-major order, returns a normalized quaternion
    static CUDA_CALLABLE_MEMBER Quaternion fromRotationMatrix(const float m[3][3]);

    /// Returns Euler angles in radians
    static CUDA_CALLABLE_MEMBER Vector toEuler(const Quaternion& q);

    /// Returns the rotation angle in radians
    static CUDA_CALLABLE_MEMBER float rotationAngle(const Quaternion& q);

    /// Returns the rotation axis (unit vector)
    static CUDA_CALLABLE_MEMBER Vector rotationAxis(const Quaternion& q);

    /// Returns a pointer to a 3x3 rotation matrix (column-major order)
    static CUDA_CALLABLE_MEMBER float* toRotationMatrix(const Quaternion& q);

    /// Spherical linear interpolation, returns a normalized quaternion
    static CUDA_CALLABLE_MEMBER Quaternion slerp(const Quaternion& q1, const Quaternion& q2, float t);

    /// Linear interpolation, not normalized, returns a non-unit quaternion
    static CUDA_CALLABLE_MEMBER Quaternion lerp(const Quaternion& q1, const Quaternion& q2, float t);

    /// Normalized linear interpolation
    static CUDA_CALLABLE_MEMBER Quaternion nlerp(const Quaternion& q1, const Quaternion& q2, float t);

    /// Creates a pure quaternion from a vector (w=0)
    static CUDA_CALLABLE_MEMBER Quaternion PureFromVector(const Vector& v);

    /// Rotates a vector by a quaternion
    static CUDA_CALLABLE_MEMBER Vector RotateVectorByQuaternion(const Vector& vec, Quaternion quat);


    /// Instance methods

    /// Spherical linear interpolation, returns a normalized quaternion
    CUDA_CALLABLE_MEMBER Quaternion slerp(const Quaternion& other, float t) const;

    /// Linear interpolation, not normalized, returns a non-unit quaternion
    CUDA_CALLABLE_MEMBER Quaternion lerp(const Quaternion& other, float t) const;

    /// Normalized linear interpolation, returns a normalized quaternion
    CUDA_CALLABLE_MEMBER Quaternion nlerp(const Quaternion& other, float t) const;

    CUDA_CALLABLE_MEMBER Vector PureAsVector() const;

    CUDA_CALLABLE_MEMBER Quaternion PureAsQuaternion() const;

    CUDA_CALLABLE_MEMBER float re() const;

    CUDA_CALLABLE_MEMBER Vector im() const;

    NO_DISCARD std::string toString() const;

    CUDA_CALLABLE_MEMBER Quaternion to_normalized() const;


} Quaternion;
