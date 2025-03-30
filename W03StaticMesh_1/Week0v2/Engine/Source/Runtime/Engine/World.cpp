#include "Engine/Source/Runtime/Engine/World.h"

#include "Actors/Player.h"
#include "BaseGizmos/GizmoActor.h"
#include "Camera/CameraComponent.h"
#include "LevelEditor/SLevelEditor.h"
#include "Engine/FLoaderOBJ.h"
#include "Classes/Components/StaticMeshComponent.h"
#include "Engine/StaticMeshActor.h"
#include "Components/SkySphereComponent.h"
#include "Math/JungleMath.h"
#include "UnrealEd/PrimitiveBatch.h"


void UWorld::Initialize()
{
    CreateBaseObject();
}

void UWorld::CreateBaseObject()
{
    if (EditorPlayer == nullptr)
    {
        EditorPlayer = FObjectFactory::ConstructObject<AEditorPlayer>();;
    }

    if (camera == nullptr)
    {
        camera = FObjectFactory::ConstructObject<UCameraComponent>();
        camera->SetLocation(FVector(8.0f, 8.0f, 8.f));
        camera->SetRotation(FVector(0.0f, 45.0f, -135.0f));
    }

    if (LocalGizmo == nullptr)
    {
        LocalGizmo = FObjectFactory::ConstructObject<AGizmoActor>();
    }

    if (RootOctree == nullptr)
    {
        RootOctree = new Octree();
    }
}

void UWorld::ReleaseBaseObject()
{
    if (LocalGizmo)
    {
        DestroyActor(LocalGizmo);
        LocalGizmo = nullptr;
    }
    
    if (camera)
    {
        GUObjectArray.MarkRemoveObject(camera);
        camera = nullptr;
    }

    if (EditorPlayer)
    {
        GUObjectArray.MarkRemoveObject(EditorPlayer);
        EditorPlayer = nullptr;
    }

}

void UWorld::Tick(float DeltaTime)
{
	// camera->TickComponent(DeltaTime);
	EditorPlayer->Tick(DeltaTime);
	LocalGizmo->Tick(DeltaTime);

    // SpawnActor()에 의해 Actor가 생성된 경우, 여기서 BeginPlay 호출
    for (AActor* Actor : PendingBeginPlayActors)
    {
        Actor->BeginPlay();
    }
    PendingBeginPlayActors.Empty();

    // 매 틱마다 Actor->Tick(...) 호출
	// for (AActor* Actor : ActorsArray)
	// {
	//     Actor->Tick(DeltaTime);
	// }
}

void UWorld::Release()
{
	for (AActor* Actor : ActorsArray)
	{
		Actor->EndPlay(EEndPlayReason::WorldTransition);
        TSet<UActorComponent*> Components = Actor->GetComponents();
	    for (UActorComponent* Component : Components)
	    {
	        GUObjectArray.MarkRemoveObject(Component);
	    }
	    GUObjectArray.MarkRemoveObject(Actor);
	}
    ActorsArray.Empty();

	pickingGizmo = nullptr;
	ReleaseBaseObject();

    GUObjectArray.ProcessPendingDestroyObjects();
}

bool UWorld::DestroyActor(AActor* ThisActor)
{
    if (ThisActor->GetWorld() == nullptr)
    {
        return false;
    }

    if (ThisActor->IsActorBeingDestroyed())
    {
        return true;
    }

    // 액터의 Destroyed 호출
    ThisActor->Destroyed();

    if (ThisActor->GetOwner())
    {
        ThisActor->SetOwner(nullptr);
    }

    TSet<UActorComponent*> Components = ThisActor->GetComponents();
    for (UActorComponent* Component : Components)
    {
        Component->DestroyComponent();
    }

    // World에서 제거
    ActorsArray.Remove(ThisActor);

    // 제거 대기열에 추가
    GUObjectArray.MarkRemoveObject(ThisActor);
    return true;
}

void UWorld::SetPickingGizmo(UObject* Object)
{
	pickingGizmo = Cast<USceneComponent>(Object);
}

void UWorld::ComputeWorldExtents()
{
    FVector WorldMin = FVector(FLT_MAX, FLT_MAX, FLT_MAX);
    FVector WolrdMax = FVector(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    for (const auto& Actor : ActorsArray)
    {
        auto Component = Cast<UStaticMeshComponent>(Actor->GetRootComponent());
        if (!Component) continue;

        FBoundingBox BoundingBox = Component->GetWorldSpaceBoundingBox();
        
        WorldMin.x = std::min(WorldMin.x, BoundingBox.min.x);
        WorldMin.y = std::min(WorldMin.y, BoundingBox.min.y);
        WorldMin.z = std::min(WorldMin.z, BoundingBox.min.z);

        WolrdMax.x = std::max(WolrdMax.x, BoundingBox.max.x);
        WolrdMax.y = std::max(WolrdMax.y, BoundingBox.max.y);
        WolrdMax.z = std::max(WolrdMax.z, BoundingBox.max.z);
    }

    RootOctree->Region.min = WorldMin;
    RootOctree->Region.max = WolrdMax;
}