#include "CameraFrustum.h"

#include <cmath>
#include "Math/MathUtility.h"
#include "Math/JungleMath.h"
#include "Math/Matrix.h"
#include "Editor/UnrealEd/EditorViewportClient.h"
#include "Components/StaticMeshComponent.h"

void FCameraFrustum::BuildFromView(FEditorViewportClient* ViewportCamera)
{
    FMatrix View = ViewportCamera->GetViewMatrix();
    FMatrix Proj = ViewportCamera->GetProjectionMatrix();
    FMatrix ViewProj = View * Proj;

    // Near4
    Planes[0] = FPlane(
        ViewProj.M[0][3] + ViewProj.M[0][2],
        ViewProj.M[1][3] + ViewProj.M[1][2],
        ViewProj.M[2][3] + ViewProj.M[2][2],
        ViewProj.M[3][3] + ViewProj.M[3][2]
    );
    // Far
    Planes[1] = FPlane(
        ViewProj.M[0][3] - ViewProj.M[0][2],
        ViewProj.M[1][3] - ViewProj.M[1][2],
        ViewProj.M[2][3] - ViewProj.M[2][2],
        ViewProj.M[3][3] - ViewProj.M[3][2]
    );
    // Left
    Planes[2] = FPlane(
        ViewProj.M[0][3] + ViewProj.M[0][0],
        ViewProj.M[1][3] + ViewProj.M[1][0],
        ViewProj.M[2][3] + ViewProj.M[2][0],
        ViewProj.M[3][3] + ViewProj.M[3][0]
    );
    // Right
    Planes[3] = FPlane(
        ViewProj.M[0][3] - ViewProj.M[0][0],
        ViewProj.M[1][3] - ViewProj.M[1][0],
        ViewProj.M[2][3] - ViewProj.M[2][0],
        ViewProj.M[3][3] - ViewProj.M[3][0]
    );
    // Top
    Planes[4] = FPlane(
        ViewProj.M[0][3] - ViewProj.M[0][1],
        ViewProj.M[1][3] - ViewProj.M[1][1],
        ViewProj.M[2][3] - ViewProj.M[2][1],
        ViewProj.M[3][3] - ViewProj.M[3][1]
    );
    // Bottom
    Planes[5] = FPlane(
        ViewProj.M[0][3] + ViewProj.M[0][1],
        ViewProj.M[1][3] + ViewProj.M[1][1],
        ViewProj.M[2][3] + ViewProj.M[2][1],
        ViewProj.M[3][3] + ViewProj.M[3][1]
    );


    for (int i = 0; i < 6; i++)
    {
        float Len = std::sqrt(Planes[i].A * Planes[i].A + Planes[i].B * Planes[i].B + Planes[i].C * Planes[i].C);
        if (Len != 0.0f)
        {
            Planes[i].A /= Len;
            Planes[i].B /= Len;
            Planes[i].C /= Len;
            Planes[i].D /= Len;
        }
    }   
}

bool FCameraFrustum::IntersectMesh(const FBoundingBox& Box) const
{
    for (const FPlane& Plane : Planes)
    {
        // P-Vertex와 N-Vertex 계산
        FVector PVertex(
        Plane.A >= 0 ? Box.max.x : Box.min.x,
        Plane.B >= 0 ? Box.max.y : Box.min.y,
        Plane.C >= 0 ? Box.max.z : Box.min.z
        );
        if (Plane.DistanceTo(PVertex) < 0)
        {
            return false;
        }
    }
    return true;
}

bool FCameraFrustum::IntersectMesh(const FBoundingSphere& Sphere) const
{
    for (int i = 0; i < 6; ++i)
    {
        if (Planes[i].DistanceTo(Sphere.Center) < -Sphere.Radius)
            return false;
    }
    return true;
}
