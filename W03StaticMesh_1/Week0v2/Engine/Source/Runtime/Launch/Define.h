#pragma once
#include <cmath>
#include <algorithm>
#include "Core/Container/String.h"
#include "Core/Container/Array.h"
#include "UObject/NameTypes.h"

// 수학 관련
#include "Math/Vector.h"
#include "Math/Vector4.h"
#include "Math/Matrix.h"


#define UE_LOG Console::GetInstance().AddLog

#define _TCHAR_DEFINED
#include <d3d11.h>

#include "UserInterface/Console.h"

class UActorComponent;

struct FVertexSimple
{
    float x, y, z;    // Position
    //float r, g, b, a; // Color
    //float nx, ny, nz;
    float u=0, v=0;
    //uint32 MaterialIndex;
};

// Material Subset
struct FMaterialSubset
{
    uint32 IndexStart; // Index Buffer Start pos
    uint32 IndexCount; // Index Count
    uint32 MaterialIndex; // Material Index
    FString MaterialName; // Material Name
};

struct FStaticMaterial
{
    class UMaterial* Material;
    FName MaterialSlotName;
    //FMeshUVChannelInfo UVChannelData;
};

// OBJ File Raw Data
struct FObjInfo
{
    FWString ObjectName; // OBJ File Name
    FWString PathName; // OBJ File Paths
    FString DisplayName; // Display Name
    FString MatName; // OBJ MTL File Name
    
    // Group
    uint32 NumOfGroup = 0; // token 'g' or 'o'
    TArray<FString> GroupName;
    
    // Vertex, UV, Normal List
    TArray<FVector> Vertices;
    TArray<FVector> Normals;
    TArray<FVector2D> UVs;
    
    // Faces
    TArray<int32> Faces;

    // Index
    TArray<uint32> VertexIndices;
    TArray<uint32> NormalIndices;
    TArray<uint32> TextureIndices;
    
    // Material
    TArray<FMaterialSubset> MaterialSubsets;
};

// Face
struct FObjFace
{
    TArray<uint32> VertexIndices;
    TArray<uint32> TextureIndices;
    TArray<uint32> NormalIndices;
};

struct FObjMaterialInfo
{
    FString MTLName;  // newmtl : Material Name.

    bool bHasTexture = false;  // Has Texture?
    bool bTransparent = false; // Has alpha channel?

    FVector Diffuse;  // Kd : Diffuse (Vector4)
    FVector Specular;  // Ks : Specular (Vector) 
    FVector Ambient;   // Ka : Ambient (Vector)
    FVector Emissive;  // Ke : Emissive (Vector)

    float SpecularScalar; // Ns : Specular Power (Float)
    float DensityScalar;  // Ni : Optical Density (Float)
    float TransparencyScalar; // d or Tr  : Transparency of surface (Float)

    uint32 IlluminanceModel; // illum: illumination Model between 0 and 10. (UINT)

    /* Texture */
    FString DiffuseTextureName;  // map_Kd : Diffuse texture
    FWString DiffuseTexturePath;
    
    FString AmbientTextureName;  // map_Ka : Ambient texture
    FWString AmbientTexturePath;
    
    FString SpecularTextureName; // map_Ks : Specular texture
    FWString SpecularTexturePath;
    
    FString BumpTextureName;     // map_Bump : Bump texture
    FWString BumpTexturePath;
    
    FString AlphaTextureName;    // map_d : Alpha texture
    FWString AlphaTexturePath;
};

// Cooked Data
namespace OBJ
{
    struct FStaticMeshRenderData
    {
        FWString ObjectName;
        FWString PathName;
        FString DisplayName;
        
        TArray<FVertexSimple> Vertices;
        TArray<UINT> Indices;   // 삼각형 단위로 정렬된 인덱스

        ID3D11Buffer* VertexBuffer;
        ID3D11Buffer* IndexBuffer;
        
        TArray<FObjMaterialInfo> Materials;
        TArray<FMaterialSubset> MaterialSubsets;

        FVector BoundingBoxMin;
        FVector BoundingBoxMax;

        FVector BoundingSphereCenter;
        float BoundingSphereRadius;
    };
}

struct FVertexTexture
{
	float x, y, z;    // Position
	float u, v; // Texture
};
struct FGridParameters
{
	float gridSpacing;
	int   numGridLines;
	FVector gridOrigin;
	float pad;
};
struct FSimpleVertex
{
	float dummy; // 내용은 사용되지 않음
    float padding[11];
};
struct FOBB {
    FVector corners[8];
};
struct FRect
{
    FRect() : leftTopX(0), leftTopY(0), width(0), height(0) {}
    FRect(float x, float y, float w, float h) : leftTopX(x), leftTopY(y), width(w), height(h) {}
    float leftTopX, leftTopY, width, height;
};
struct FPoint
{
    FPoint() : x(0), y(0) {}
    FPoint(float _x, float _y) : x(_x), y(_y) {}
    FPoint(long _x, long _y) : x(_x), y(_y) {}
    FPoint(int _x, int _y) : x(_x), y(_y) {}

    float x, y;
};
struct FRay
{
    FVector Origin;
    FVector Direction;
};
struct HitResult {
    const UActorComponent* Component = nullptr;
    float Distance = FLT_MAX;
    int IntersectCount = 0;
};

struct FBoundingBox
{
    FBoundingBox() : min(0, 0, 0), pad(), max(0, 0, 0), pad1(0) {}
    FBoundingBox(FVector _min, FVector _max) : min(_min), pad(), max(_max), pad1(0) {}
	FVector min; // Minimum extents
	float pad;
	FVector max; // Maximum extents
	float pad1;

    float Size() const
    {
        FVector diff = max - min;
        return std::max({ diff.x, diff.y, diff.z });
    }

    // 포함
    bool Contains(const FBoundingBox Other) const
    {
        return (Other.min.x >= min.x) &&
           (Other.min.y >= min.y) &&
           (Other.min.z >= min.z) &&
           (Other.max.x <= max.x) &&
           (Other.max.y <= max.y) &&
           (Other.max.z <= max.z);
    }

    // 겹침
    bool Intersect(const FBoundingBox& Other) const
    {
        return (min.x <= Other.max.x && max.x >= Other.min.x) && // X축 겹침
               (min.y <= Other.max.y && max.y >= Other.min.y) && // Y축 겹침
               (min.z <= Other.max.z && max.z >= Other.min.z);   // Z축 겹침
    }
    
    bool IntersectToRay(const FVector& rayOrigin, const FVector& rayDir, float& outDistance) const
    {
        float tmin = -FLT_MAX;
        float tmax = FLT_MAX;
        constexpr float epsilon = 1e-6f;

        // X축 처리
        if (fabs(rayDir.x) < epsilon)
        {
            // 레이가 X축 방향으로 거의 평행한 경우,
            // 원점의 x가 박스 [min.x, max.x] 범위 밖이면 교차 없음
            if (rayOrigin.x < min.x || rayOrigin.x > max.x)
                return false;
        }
        else
        {
            float t1 = (min.x - rayOrigin.x) / rayDir.x;
            float t2 = (max.x - rayOrigin.x) / rayDir.x;
            if (t1 > t2)  std::swap(t1, t2);

            // tmin은 "현재까지의 교차 구간 중 가장 큰 min"
            tmin = (t1 > tmin) ? t1 : tmin;
            // tmax는 "현재까지의 교차 구간 중 가장 작은 max"
            tmax = (t2 < tmax) ? t2 : tmax;
            if (tmin > tmax)
                return false;
        }

        // Y축 처리
        if (fabs(rayDir.y) < epsilon)
        {
            if (rayOrigin.y < min.y || rayOrigin.y > max.y)
                return false;
        }
        else
        {
            float t1 = (min.y - rayOrigin.y) / rayDir.y;
            float t2 = (max.y - rayOrigin.y) / rayDir.y;
            if (t1 > t2)  std::swap(t1, t2);

            tmin = (t1 > tmin) ? t1 : tmin;
            tmax = (t2 < tmax) ? t2 : tmax;
            if (tmin > tmax)
                return false;
        }

        // Z축 처리
        if (fabs(rayDir.z) < epsilon)
        {
            if (rayOrigin.z < min.z || rayOrigin.z > max.z)
                return false;
        }
        else
        {
            float t1 = (min.z - rayOrigin.z) / rayDir.z;
            float t2 = (max.z - rayOrigin.z) / rayDir.z;
            if (t1 > t2)  std::swap(t1, t2);

            tmin = (t1 > tmin) ? t1 : tmin;
            tmax = (t2 < tmax) ? t2 : tmax;
            if (tmin > tmax)
                return false;
        }

        // 여기까지 왔으면 교차 구간 [tmin, tmax]가 유효하다.
        // tmax < 0 이면, 레이가 박스 뒤쪽에서 교차하므로 화면상 보기엔 교차 안 한다고 볼 수 있음
        if (tmax < 0.0f)
            return false;

        // outDistance = tmin이 0보다 크면 그게 레이가 처음으로 박스를 만나는 지점
        // 만약 tmin < 0 이면, 레이의 시작점이 박스 내부에 있다는 의미이므로, 거리를 0으로 처리해도 됨.
        outDistance = (tmin >= 0.0f) ? tmin : 0.0f;

        return true;
    }

};

struct FBoundingSphere
{
    FBoundingSphere() : Center(0, 0, 0), Radius(0) {}
    FBoundingSphere(FVector InCenter, float InRadius) : Center(InCenter), Radius(InRadius) {}

    FVector Center;
    float Radius;

    bool Intersect(const FVector& rayOrigin, const FVector& rayDir, float& outDistance)
    {
        // 레이의 원점에서 구의 중심으로 향하는 벡터
        FVector L = Center - rayOrigin;

        // 레이의 방향과 구의 중심 사이의 거리
        float tca = L.Dot(rayDir);
        // tca가 음수이면 구의 중심이 레이의 반대 방향에 있다는 의미
        if (tca < 0)
            return false;

        // L과 구의 중심 사이의 거리의 제곱
        float d2 = L.Dot(L) - tca * tca;
        // d2가 구의 반지름의 제곱보다 크면 레이가 구와 교차하지 않음
        if (d2 > Radius * Radius)
            return false;

        // 구의 중심에서 교차 지점까지의 거리
        float thc = sqrt(Radius * Radius - d2);

        float t0 = tca - thc; // 레이가 구와 처음 교차하는 지점
        float t1 = tca + thc; // 레이가 구와 두 번째 교차하는 지점

        // t0이 t1보다 크면 두 교차 지점이 바뀐 것이므로 교체
        if (t0 > t1)
            std::swap(t0, t1);

        // t0이 음수이면 레이의 시작점이 구 내부에 있음
        if (t0 < 0)
            return false;
        // outDistance에 t0 대입
        outDistance = t0;
        return true;
    }

};

struct FCone
{
    FVector ConeApex; // 원뿔의 꼭짓점
    float ConeRadius; // 원뿔 밑면 반지름

    FVector ConeBaseCenter; // 원뿔 밑면 중심
    float ConeHeight; // 원뿔 높이 (Apex와 BaseCenter 간 차이)
    FVector4 Color;

    int ConeSegmentCount; // 원뿔 밑면 분할 수
    float pad[3];

};
struct FPrimitiveCounts 
{
	int BoundingBoxCount;
	int pad;
	int ConeCount; 
	int pad1;
};
//struct FLighting
//{
//	float lightDirX, lightDirY, lightDirZ; // 조명 방향
//	float pad1;                      // 16바이트 정렬용 패딩
//	float lightColorX, lightColorY, lightColorZ;    // 조명 색상
//	float pad2;                      // 16바이트 정렬용 패딩
//	float AmbientFactor;             // ambient 계수
//	float pad3; // 16바이트 정렬 맞춤 추가 패딩
//	float pad4; // 16바이트 정렬 맞춤 추가 패딩
//	float pad5; // 16바이트 정렬 맞춤 추가 패딩
//};

//struct FMaterialConstants {
//    FVector DiffuseColor;
//    float TransparencyScalar;
//    FVector AmbientColor;
//    float DensityScalar;
//    FVector SpecularColor;
//    float SpecularScalar;
//    FVector EmmisiveColor;
//    float MaterialPad0;
//};

struct FConstants {
    FMatrix MVP;      // 모델
    //FMatrix ModelMatrixInverseTranspose; // normal 변환을 위한 행렬
    //FVector4 UUIDColor;
    int IsSelected;
    FVector pad;
};
struct FUUIDConstants {
    FMatrix MVP;
    FVector4 UUIDColor;
};
//struct FLitUnlitConstants {
//    int isLit; // 1 = Lit, 0 = Unlit 
//    FVector pad;
//};

struct FSubMeshConstants {
    float isSelectedSubMesh;
    FVector pad;
};

struct FTextureConstants {
    float UOffset;
    float VOffset;
    float pad0;
    float pad1;
};