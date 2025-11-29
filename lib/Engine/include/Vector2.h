//
// Created by James Miller on 11/29/2025.
//

#pragma once
typedef struct mVector2
{
    float x;
    float y;

    mVector2();
    mVector2(float x, float y);
    mVector2 operator+(const mVector2& other) const;
    mVector2 operator-(const mVector2& other) const;
    mVector2 operator*(float scalar) const;
    mVector2 operator/(float scalar) const;
    mVector2& operator+=(const mVector2& other);
    mVector2& operator-=(const mVector2& other);
    mVector2& operator*=(float scalar);
    mVector2& operator/=(float scalar);
    [[nodiscard]] float Length() const;
    [[nodiscard]] mVector2 Normalize() const;
} MVector2;