//
// Created by James Miller on 11/25/2025.
//

#include "Engine/Quaternion.cuh"

CUDA_CALLABLE_MEMBER Quaternion::Quaternion() noexcept
{
    w = 1.0f;
    x = y = z = 0.0f;
}

CUDA_CALLABLE_MEMBER Quaternion::Quaternion(const float w, const float x, const float y, const float z) noexcept
    : w(w), x(x), y(y), z(z)
{}

Quaternion::Quaternion(const Vector& vec) noexcept
{
    w = 0.0f;
    x = vec.x;
    y = vec.y;
    z = vec.z;
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::normalize()
{
    const float mag = magnitude();
    w /= mag;
    x /= mag;
    y /= mag;
    z /= mag;
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::conjugate() const
{
    return Quaternion{w, -x, -y, -z};
}

CUDA_CALLABLE_MEMBER float Quaternion::magnitude() const
{
    return sqrtf(magnitudeSquared());
}

CUDA_CALLABLE_MEMBER float Quaternion::magnitudeSquared() const
{
    return w*w + x*x + y*y + z*z;
}

CUDA_CALLABLE_MEMBER float Quaternion::dot(const Quaternion& other) const
{
    return w * other.w + x * other.x + y * other.y + z * other.z;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::hamilton_product(const Quaternion& other) const
{
    return Quaternion{
        w * other.w - x * other.x - y * other.y - z * other.z,
        w * other.x + x * other.w + y * other.z - z * other.y,
        w * other.y - x * other.z + y * other.w + z * other.x,
        w * other.z + x * other.y - y * other.x + z * other.w
    };
}

CUDA_CALLABLE_MEMBER float Quaternion::dot(const Vector& other) const
{
    return w * 0.0f + x * other.x + y * other.y + z * other.z;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::hamilton_product(const Vector& other) const
{
    return Quaternion{
        - (x * other.x + y * other.y + z * other.z),
        w * other.x + y * other.z - z * other.y,
        w * other.y - x * other.z + z * other.x,
        w * other.z + x * other.y - y * other.x
    };
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator+(const Quaternion& other) const
{
    return Quaternion{
        w + other.w,
        x + other.x,
        y + other.y,
        z + other.z
    };
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator-(const Quaternion& other) const
{
    return Quaternion{
        w - other.w,
        x - other.x,
        y - other.y,
        z - other.z
    };
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator*(float scalar) const
{
    return Quaternion{
        w * scalar,
        x * scalar,
        y * scalar,
        z * scalar
    };
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator/(float scalar) const
{
    return Quaternion{
        w / scalar,
        x / scalar,
        y / scalar,
        z / scalar
    };
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::operator+=(const Quaternion& other)
{
    w += other.w;
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::operator-=(const Quaternion& other)
{
    w -= other.w;
    x -= other.x;
    y -= other.y;
    z -= other.z;
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::operator*=(float scalar)
{
    w *= scalar;
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::operator/=(float scalar)
{
    w /= scalar;
    x /= scalar;
    y /= scalar;
    z /= scalar;
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::operator*(const Quaternion& other) const
{
    return hamilton_product(other);
}

CUDA_CALLABLE_MEMBER Quaternion& Quaternion::operator*=(const Quaternion& other)
{
    *this = hamilton_product(other);
    return *this;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::fromAxisAngle(const Vector& axis, const float angleRad)
{
    const float half = 0.5f * angleRad;
    const float s = sinf(half);
    // normalize axis
    float lx = axis.x, ly = axis.y, lz = axis.z;
    if (const float len = sqrtf(lx*lx + ly*ly + lz*lz); len > 0.0f) {
        lx /= len; ly /= len; lz /= len;
    } else {
        // zero axis -> identity
        return {};
    }
    Quaternion q{ cosf(half), lx * s, ly * s, lz * s };
    return q.normalize();
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::fromEuler(const Vector& euler) noexcept
{
    // roll  = rotation around X
    // pitch = rotation around Y
    // yaw   = rotation around Z
    const float roll  = euler.x;
    const float pitch = euler.y;
    const float yaw   = euler.z;
    const float hr = 0.5f * roll;
    const float hp = 0.5f * pitch;
    const float hy = 0.5f * yaw;

    const float cr = cosf(hr), sr = sinf(hr);
    const float cp = cosf(hp), sp = sinf(hp);
    const float cy = cosf(hy), sy = sinf(hy);

    // quaternion for ZYX (yaw, pitch, roll) composition
    Quaternion q{
        cr * cp * cy + sr * sp * sy, // w
        sr * cp * cy - cr * sp * sy, // x
        cr * sp * cy + sr * cp * sy, // y
        cr * cp * sy - sr * sp * cy  // z
    };
    return q.normalize();
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::fromRotationMatrix(const float m[3][3])
{
    // Uses standard matrix->quaternion conversion
    const float trace = m[0][0] + m[1][1] + m[2][2];
    Quaternion q;
    if (trace > 0.0f) {
        const float s = sqrtf(trace + 1.0f) * 2.0f; // s = 4*w
        q.w = 0.25f * s;
        q.x = (m[2][1] - m[1][2]) / s;
        q.y = (m[0][2] - m[2][0]) / s;
        q.z = (m[1][0] - m[0][1]) / s;
    } else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) {
        const float s = sqrtf(1.0f + m[0][0] - m[1][1] - m[2][2]) * 2.0f; // s = 4*x
        q.w = (m[2][1] - m[1][2]) / s;
        q.x = 0.25f * s;
        q.y = (m[0][1] + m[1][0]) / s;
        q.z = (m[0][2] + m[2][0]) / s;
    } else if (m[1][1] > m[2][2]) {
        const float s = sqrtf(1.0f + m[1][1] - m[0][0] - m[2][2]) * 2.0f; // s = 4*y
        q.w = (m[0][2] - m[2][0]) / s;
        q.x = (m[0][1] + m[1][0]) / s;
        q.y = 0.25f * s;
        q.z = (m[1][2] + m[2][1]) / s;
    } else {
        const float s = sqrtf(1.0f + m[2][2] - m[0][0] - m[1][1]) * 2.0f; // s = 4*z
        q.w = (m[1][0] - m[0][1]) / s;
        q.x = (m[0][2] + m[2][0]) / s;
        q.y = (m[1][2] + m[2][1]) / s;
        q.z = 0.25f * s;
    }
    return q.normalize();
}

CUDA_CALLABLE_MEMBER Vector Quaternion::toEuler(const Quaternion& q)
{
    // Assumes q is normalized. Returns (roll, pitch, yaw) = (X, Y, Z) in radians (Tait-Bryan ZYX)
    const Quaternion nq = q.magnitudeSquared() > 0.0f ? Quaternion{q.w, q.x, q.y, q.z}.normalize() : q;

    const float sinr_cosp = 2.0f * (nq.w * nq.x + nq.y * nq.z);
    const float cosr_cosp = 1.0f - 2.0f * (nq.x * nq.x + nq.y * nq.y);
    const float roll = atan2f(sinr_cosp, cosr_cosp);

    const float sinp = 2.0f * (nq.w * nq.y - nq.z * nq.x);
    float pitch;
    if (fabsf(sinp) >= 1.0f) {
        pitch = copysignf(PI / 2.0f, sinp); // use 90 degrees if out of range
    } else {
        pitch = asinf(sinp);
    }

    const float siny_cosp = 2.0f * (nq.w * nq.z + nq.x * nq.y);
    const float cosy_cosp = 1.0f - 2.0f * (nq.y * nq.y + nq.z * nq.z);
    const float yaw = atan2f(siny_cosp, cosy_cosp);

    return Vector{ roll, pitch, yaw };
}

CUDA_CALLABLE_MEMBER float Quaternion::rotationAngle(const Quaternion& q)
{
    // angle in radians: 2 * acos(w). Clamp w to [-1,1].
    float w_clamped = fmaxf(-1.0f, fminf(1.0f, q.w));
    return 2.0f * acosf(w_clamped);
}

CUDA_CALLABLE_MEMBER Vector Quaternion::rotationAxis(const Quaternion& q)
{
    // axis = (x,y,z) / sin(theta/2) where sin(theta/2) = sqrt(1 - w*w)
    const float s = sqrtf(fmaxf(0.0f, 1.0f - q.w * q.w));
    if (s < 1e-6f) {
        // If angle is near zero, return default axis
        return Vector{ 1.0f, 0.0f, 0.0f };
    }
    return Vector{ q.x / s, q.y / s, q.z / s };
}

float* Quaternion::toRotationMatrix(const Quaternion& q)
{
    // Returns pointer to a static 3x3 row-major matrix { m00, m01, m02, m10, ... }
    static float m[9];
    // Ensure normalized for stable results
    Quaternion nq = q;
    if (nq.magnitudeSquared() <= 0.0f) nq = Quaternion{};
    nq.normalize();

    const float xx = nq.x * nq.x;
    const float yy = nq.y * nq.y;
    const float zz = nq.z * nq.z;

    m[0] = 1.0f - 2.0f * (yy + zz);            // m00
    m[1] = 2.0f * (nq.x * nq.y - nq.z * nq.w); // m01
    m[2] = 2.0f * (nq.x * nq.z + nq.y * nq.w); // m02

    m[3] = 2.0f * (nq.x * nq.y + nq.z * nq.w); // m10
    m[4] = 1.0f - 2.0f * (xx + zz);            // m11
    m[5] = 2.0f * (nq.y * nq.z - nq.x * nq.w); // m12

    m[6] = 2.0f * (nq.x * nq.z - nq.y * nq.w); // m20
    m[7] = 2.0f * (nq.y * nq.z + nq.x * nq.w); // m21
    m[8] = 1.0f - 2.0f * (xx + yy);            // m22

    return m;
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::slerp(const Quaternion& q1, const Quaternion& q2, const float t)
{
    // Spherical linear interpolation, ensures shortest path
    Quaternion qa = q1;
    Quaternion qb = q2;
    // normalize inputs
    qa.normalize();
    qb.normalize();

    float dot = qa.dot(qb);

    // If dot < 0, negate qb to take the shortest path
    if (dot < 0.0f) {
        qb = qb * -1.0f;
        dot = -dot;
    }

    if (constexpr float DOT_THRESHOLD = 0.9995f; dot > DOT_THRESHOLD) {
        // Very close - use nlerp for stability
        return nlerp(qa, qb, t);
    }

    // Clamp dot to be safe for acos
    dot = fmaxf(-1.0f, fminf(1.0f, dot));
    const float theta = acosf(dot);
    const float sinTheta = sinf(theta);

    const float invSin = 1.0f / sinTheta;
    const float scale0 = sinf((1.0f - t) * theta) * invSin;
    const float scale1 = sinf(t * theta) * invSin;

    Quaternion res = qa * scale0 + qb * scale1;
    return res.normalize();
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::lerp(const Quaternion& q1, const Quaternion& q2, const float t)
{
    Quaternion qb = q2;
    // Ensure the shortest path by flipping sign if dot < 0
    if (q1.dot(qb) < 0.0f) {
        qb = qb * -1.0f;
    }
    return q1 * (1.0f - t) + qb * t;
}

/// Normalized linear interpolation with shortest-path correction
CUDA_CALLABLE_MEMBER Quaternion Quaternion::nlerp(const Quaternion& q1, const Quaternion& q2, float t)
{
    // Normalized linear interpolation with shortest-path correction
    Quaternion qa = q1;
    Quaternion qb = q2;
    qa.normalize();
    qb.normalize();

    if (qa.dot(qb) < 0.0f) {
        qb = qb * -1.0f;
    }

    Quaternion res = qa * (1.0f - t) + qb * t;
    return res.normalize();
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::PureFromVector(const Vector& v)
{
    return Quaternion{0.0f, v.x, v.y, v.z};
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::slerp(const Quaternion& other, float t) const
{
    return slerp(*this, other, t);
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::lerp(const Quaternion& other, float t) const
{
    return lerp(*this, other, t);
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::nlerp(const Quaternion& other, float t) const
{
    return nlerp(*this, other, t);
}

CUDA_CALLABLE_MEMBER Vector Quaternion::PureAsVector() const
{
    return Vector{x, y, z};
}

CUDA_CALLABLE_MEMBER Quaternion Quaternion::PureAsQuaternion() const
{
    return Quaternion{0.0f, x, y, z};
}

CUDA_CALLABLE_MEMBER float Quaternion::re() const
{
    return w;
}

CUDA_CALLABLE_MEMBER Vector Quaternion::im() const
{
    return Vector{x, y, z};
}

std::string Quaternion::toString() const
{
    return "(" + std::to_string(w) + ", " + std::to_string(x) + ", " + std::to_string(y) + ", " + std::to_string(z) + ")";
}

Quaternion Quaternion::to_normalized() const
{
    return Quaternion{w, x, y, z}.normalize();
}

CUDA_CALLABLE_MEMBER Vector Quaternion::RotateVectorByQuaternion(const Vector& vec, Quaternion quat)
{
    // Ensure the rotation quaternion is normalized
    quat.normalize(); // modifies local copy

    // Rotate: v' = q * (0, v) * q^{-1}
    const Quaternion res  = quat * PureFromVector(vec) * quat.conjugate();

    // Don't normalize 'res' — return its vector (pure) part directly
    return res.PureAsVector();
}
