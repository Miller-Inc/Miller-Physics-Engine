//
// Created by James Miller on 9/3/2025.
//

#include "Vector2.h"
#include <cmath>

mVector2::mVector2()
{
    this->x = 0.0f;
    this->y = 0.0f;
}

mVector2::mVector2(const float x, const float y)
{
    this->x = x;
    this->y = y;
}

mVector2 mVector2::operator+(const mVector2& other) const
{
    return {x+other.x, y+other.y};
}

mVector2 mVector2::operator-(const mVector2& other) const
{
    return {x-other.x, y-other.y};
}

mVector2 mVector2::operator*(float scalar) const
{
    return {x*scalar, y*scalar};
}

mVector2 mVector2::operator/(float scalar) const
{
    return {x/scalar, y/scalar};
}

mVector2& mVector2::operator+=(const mVector2& other)
{
    return *this = *this + other;
}

mVector2& mVector2::operator-=(const mVector2& other)
{
    return *this = *this - other;
}

mVector2& mVector2::operator*=(const float scalar)
{
    return *this = *this * scalar;
}

mVector2& mVector2::operator/=(const float scalar)
{
    return *this = *this / scalar;
}

float mVector2::Length() const
{
    return sqrt(x*x + y*y);
}

mVector2 mVector2::Normalize() const
{
    return *this / Length();
}