#pragma once

#include "Runtime/Launch/Define.h"
#include "Math/Vector.h"
#include "Math/Quat.h"

class FEditorViewportClient;
class UStaticMeshComponent;

struct FPlane
{
    float A, B, C, D;
    
    // Normal x, y, z, Distance.
    FPlane() : A(0), B(0), C(0), D(0) {}
    FPlane(float InA, float InB, float InC, float InD) : A(InA), B(InB), C(InC), D(InD) {}

    float DistanceTo(const FVector& Point) const
    {
        return A * Point.x + B * Point.y + C * Point.z + D;
    }

};

class FCameraFrustum
{
public:
    void BuildFromView(FEditorViewportClient* ViewportCamera);
    bool IntersectMesh(const FBoundingBox& Box) const;
    bool IntersectMesh(const FBoundingSphere& Sphere) const;
    //bool IntersectSphere(const FVector& Center, float Radius) const;

private:
    // Near, Far, Left, Right, Top, Bottom
    FPlane Planes[6];

}; 