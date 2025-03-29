#include "Octree.h"

#include "Components/PrimitiveComponent.h"
#include "Math/JungleMath.h"

std::queue<AActor*> Octree::PendingInsertion;
bool Octree::bReadyTree = false;
bool Octree::bBuildTree = false;
uint32 Octree::Capacity = 2;

Octree::Octree() : Parent(nullptr), ActiveNodeMask(0)
{
    Children.SetNum(8);
}

Octree::Octree(const FBoundingBox& InRegion) : Region(InRegion), Parent(nullptr), ActiveNodeMask(0)
{
    Children.SetNum(8);
}

Octree::Octree(const FBoundingBox& InRegion, const TArray<AActor*>& InActors) : Region(InRegion), Actors(InActors), Parent(nullptr), ActiveNodeMask(0)
{
    Children.SetNum(8);
}

int Octree::GetOctant(const FVector& Center, const FVector& HalfSize) const
{
    int Octant = 0;
    if (Center.x >= Region.min.x + HalfSize.x) Octant |= 1;
    if (Center.y >= Region.min.y + HalfSize.y) Octant |= 2;
    if (Center.z >= Region.min.z + HalfSize.z) Octant |= 4;
    return Octant;
}

FBoundingBox Octree::CalculateActorBoundingBox(const AActor* Actor) const
{
    FBoundingBox BoundingBox(FVector(FLT_MAX, FLT_MAX, FLT_MAX), FVector(-FLT_MAX, -FLT_MAX, -FLT_MAX));
    for (auto& Component : Actor->GetComponents())
    {
        if (UPrimitiveComponent* PrimitiveComp = Cast<UPrimitiveComponent>(Component))
        {
            // 로컬 바운딩 박스를 가져옵니다.
            FBoundingBox LocalBB = PrimitiveComp->GetBoundingBox();

            // 모델 행렬을 생성합니다.
            FMatrix Model = JungleMath::CreateModelMatrix(
                PrimitiveComp->GetWorldLocation(),
                PrimitiveComp->GetWorldRotation(),
                PrimitiveComp->GetWorldScale()
            );

            // 로컬 AABB의 8개 코너를 계산합니다.
            TArray<FVector> Corners;
            Corners.Add(FVector(LocalBB.min.x, LocalBB.min.y, LocalBB.min.z));
            Corners.Add(FVector(LocalBB.max.x, LocalBB.min.y, LocalBB.min.z));
            Corners.Add(FVector(LocalBB.min.x, LocalBB.max.y, LocalBB.min.z));
            Corners.Add(FVector(LocalBB.min.x, LocalBB.min.y, LocalBB.max.z));
            Corners.Add(FVector(LocalBB.max.x, LocalBB.max.y, LocalBB.min.z));
            Corners.Add(FVector(LocalBB.max.x, LocalBB.min.y, LocalBB.max.z));
            Corners.Add(FVector(LocalBB.min.x, LocalBB.max.y, LocalBB.max.z));
            Corners.Add(FVector(LocalBB.max.x, LocalBB.max.y, LocalBB.max.z));

            // 각 코너를 월드 좌표로 변환하고, ActorBoundingBox를 갱신합니다.
            for (const FVector& LocalCorner : Corners)
            {
                FVector WorldCorner = Model.TransformPosition(LocalCorner);
                BoundingBox.min.x = std::min(BoundingBox.min.x, WorldCorner.x);
                BoundingBox.min.y = std::min(BoundingBox.min.y, WorldCorner.y);
                BoundingBox.min.z = std::min(BoundingBox.min.z, WorldCorner.z);

                BoundingBox.max.x = std::max(BoundingBox.max.x, WorldCorner.x);
                BoundingBox.max.y = std::max(BoundingBox.max.y, WorldCorner.y);
                BoundingBox.max.z = std::max(BoundingBox.max.z, WorldCorner.z);
            }
        }
    }
    
    return BoundingBox;
}

void Octree::Insert(AActor* Actor)
{
    FBoundingBox ActorBoundingBox = CalculateActorBoundingBox(Actor);
    
    // 현재 Actor의 바운딩 박스가 영역에 포함되는지 확인
    if (!Region.Contains(ActorBoundingBox))
    {
        return;
    }

    // 현재 노드가 Leaf-node
    if (Children.IsEmpty())
    {
        Actors.Add(Actor);
        if (Actors.Num() > Capacity && Region.Size() > MinSize)
        {
            BuildTree();
        }
    }
    else // 자식 노드가 있다면, 적절한 위치에 삽입
    {
        FVector actorCenter = (ActorBoundingBox.min + ActorBoundingBox.max) * 0.5f;
        FVector BoxSize = ActorBoundingBox.max - ActorBoundingBox.min;
        FVector HalfSize = BoxSize * 0.5f;
        
        int octant = GetOctant(actorCenter, HalfSize);
        Children[octant]->Insert(Actor);
    }
}


void Octree::UpdateTree()
{
    if (!bBuildTree)
    {
        while (PendingInsertion.size() != 0) // Lazy initialization
        {
            Actors.Add(PendingInsertion.front());
            PendingInsertion.pop();
            BuildTree();
        }
    }
    else
    {
        while (PendingInsertion.size() != 0) // After initialization
        {
            Insert(PendingInsertion.front());
            PendingInsertion.pop();
        }
    }
    bReadyTree = true;
}

void Octree::BuildTree()
{
    // 현재 노드가 이미 분할되어 있다면 리턴
    if (Children.IsEmpty())
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

    TArray<AActor*> RemainingActors;
    for (AActor* Actor : Actors)
    {
        FBoundingBox ActorBoundingBox = CalculateActorBoundingBox(Actor);
        bool Inserted = false;
        for (uint32 i = 0; i < 8; ++i)
        {
            if (Children[i]->Region.Contains(ActorBoundingBox))
            {
                Children[i]->Actors.Add(Actor);
                ActiveNodeMask |= (1 << i);
                Inserted = true;
                break;
            }
        }
        if (!Inserted)
        {
            RemainingActors.Add(Actor);
        }
    }
    Actors = RemainingActors;
}
