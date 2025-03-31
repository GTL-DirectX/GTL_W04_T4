#include "Octree.h"

#include "Camera/CameraFrustum.h"
#include "Components/StaticMeshComponent.h"
#include "Math/JungleMath.h"

bool Octree::bReadyTree = false;
bool Octree::bBuildTree = false;
uint32 Octree::Capacity = 127;

Octree::Octree() : Parent(nullptr), ActiveNodeMask(0)
{
}

Octree::Octree(const FBoundingBox& InRegion) : Region(InRegion), Parent(nullptr), ActiveNodeMask(0)
{
}

Octree::Octree(const FBoundingBox& InRegion, const TArray<UStaticMeshComponent*>& InActors) : Region(InRegion), ActorComps(InActors), Parent(nullptr), ActiveNodeMask(0)
{
}

int Octree::GetOctant(const FVector& Center, const FVector& HalfSize) const
{
    int Octant = 0;
    if (Center.x >= Region.min.x + HalfSize.x) Octant |= 1;
    if (Center.y >= Region.min.y + HalfSize.y) Octant |= 2;
    if (Center.z >= Region.min.z + HalfSize.z) Octant |= 4;
    return Octant;
}

FBoundingBox Octree::GetLooseRegion()
{
    if (LooseRegion.Size() != 0)
    {
        return LooseRegion;
    }
    
    FVector Center = (Region.min + Region.max) * 0.5f;
    FVector HalfSize = (Region.max - Region.min) * 0.5f;
    FVector LooseHalfSize = HalfSize * LooseFactor;
    LooseRegion = FBoundingBox(Center - LooseHalfSize, Center + LooseHalfSize);

    return LooseRegion;
}

void Octree::Insert(UStaticMeshComponent* ActorComp)
{
    FBoundingBox ActorBoundingBox = ActorComp->GetWorldSpaceBoundingBox();
    
    // 현재 Actor의 바운딩 박스가 영역에 포함되는지 확인
    if (!GetLooseRegion().IntersectToRay(ActorBoundingBox.min, ActorBoundingBox.max, *(new float)))
    {
        return;
    }

    // 현재 노드가 Leaf-node (자식이 없을 때)
    if (!Children[0])
    {
        ActorComps.Add(ActorComp);
        if (ActorComps.Num() > Capacity && Region.Size() > MinSize)
        {
            BuildTree();
        }
    }
    else // 자식 노드가 있다면, 적절한 위치에 삽입
    {
        FVector ActorCenter = (ActorBoundingBox.min + ActorBoundingBox.max) * 0.5f;
        FVector BoxSize = Region.max - Region.min;
        FVector HalfSize = BoxSize * 0.5f;

        int Octant = GetOctant(ActorCenter, HalfSize);
        Children[Octant]->Insert(ActorComp);
    }
}


void Octree::UpdateTree()
{
    if (!bBuildTree)
    {
        BuildTree();
        bBuildTree = true;
    }
    bReadyTree = true;
}

void Octree::BuildTree()
{
    UE_LOG(LogLevel::Display, "BuildTree: Actors=%d, Size=%f", ActorComps.Num(), Region.Size());
    // 현재 노드가 이미 분할되어 있다면 리턴
    if (Children[0])
    {
        return;
    }

    FVector HalfSize = (Region.max - Region.min) * 0.5f;
    
    for (uint32 i = 0; i < 8; ++i)
    {
        FVector ChildMin = Region.min + FVector(
            (i & 1) ? HalfSize.x : 0,
            (i & 2) ? HalfSize.y : 0,
            (i & 4) ? HalfSize.z : 0
        );
        FBoundingBox ChildRegion(ChildMin, ChildMin + HalfSize);
        Children[i] = std::make_unique<Octree>(ChildRegion);
        Children[i]->Parent = this;
    }

    // 기존 액터들을 자식 노드로 분배
    TArray<UStaticMeshComponent*> ActorsToDistribute = ActorComps;
    ActorComps.Empty(); // 현재 노드의 Actors를 비움

    for (UStaticMeshComponent* ActorComp : ActorsToDistribute)
    {
        FBoundingBox ActorBoundingBox = ActorComp->GetWorldSpaceBoundingBox();
        FVector actorCenter = (ActorBoundingBox.min + ActorBoundingBox.max) * 0.5f;
        int octant = GetOctant(actorCenter, HalfSize);

        // 자식 노드에 삽입 (재귀적으로 분할 가능)
        Children[octant]->Insert(ActorComp);
        ActiveNodeMask |= (1 << octant);
    }
}

void Octree::QueryTree(const FVector& RayOrigin, const FVector& RayDirection, TArray<UStaticMeshComponent*>& OutActors)
{
    float OutDistance = 0.0f;
    
    if (!GetLooseRegion().IntersectToRay(RayOrigin, RayDirection, OutDistance))
    {
        return;
    }
    
    // 2. Leaf 노드라면 액터 추가
    if (!Children[0]) // 자식이 없으면 Leaf 노드
    {
        for (auto Actor : ActorComps)
        {
            OutActors.Add(Actor);
        }
        return;
    }

    // 3. 중간 노드: 자식 노드로 내려가기 전에 추가 필터링
    for (const auto& Child : Children)
    {
        // 자식 노드의 느슨한 영역과 Ray 교차 확인
        if (Child->GetLooseRegion().IntersectToRay(RayOrigin, RayDirection, OutDistance))
        {
            // 교차하면 자식 노드 탐색
            Child->QueryTree(RayOrigin, RayDirection, OutActors);
        }
        // 교차하지 않으면 이 자식 노드와 그 하위는 탐색 생략
    }
}

void Octree::QueryFrustum(const FCameraFrustum& Frustum, TArray<UStaticMeshComponent*>& OutComponents)
{
    // 느슨한 영역으로 초기 필터링
    if (!Frustum.IntersectMesh(GetLooseRegion()))
    {
        return;
    }

    // Leaf 노드에서만 객체 추가 + 프러스텀 교차 검사
    if (!Children[0])
    {
        for (auto Comp : ActorComps)
        {
            if (Comp && Frustum.IntersectMesh(Comp->GetWorldSpaceBoundingBox()))
            {
                OutComponents.Add(Comp);
            }
        }
        return;
    }

    // 자식 노드 탐색
    for (const auto& Child : Children)
    {
        if (Child)
        {
            Child->QueryFrustum(Frustum, OutComponents);
        }
    }
}

void Octree::ClearTree()
{
    ActorComps.Empty();

    for (auto& Child : Children)
    {
        if (Child)
        {
            Child->ClearTree();
            Child.reset();
        }
    }

    ActiveNodeMask = 0;
}
