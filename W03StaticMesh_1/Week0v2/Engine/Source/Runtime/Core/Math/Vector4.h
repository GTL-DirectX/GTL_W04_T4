#pragma once

// 4D Vector
struct FVector4 {
    //struct
    //{
    float x, y, z, a;
    //};
    //__m128 V;
    FVector4(float _x = 0, float _y = 0, float _z = 0, float _a = 0) : x(_x), y(_y), z(_z), a(_a) {}
    //    : V(_mm_setr_ps(_x, _y, _z, _a)) {}
    //FVector4(__m128 _v) : V(_v) {}

    FVector4 operator-(const FVector4& other) const {
        //return FVector4(_mm_sub_ps(V, other.V));
        return FVector4(x - other.x, y - other.y, z - other.z, a - other.a);
    }
    FVector4 operator+(const FVector4& other) const {
        //return FVector4(_mm_add_ps(V, other.V));
        return FVector4(x + other.x, y + other.y, z + other.z, a + other.a);
    }
    FVector4 operator/(float scalar) const
    {
        //return FVector4(_mm_div_ps(V, _mm_set1_ps(scalar)));
        return FVector4{ x / scalar, y / scalar, z / scalar, a / scalar };
    }
};
