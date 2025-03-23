﻿#include "Actor.h"

void AActor::BeginPlay()
{
    for (UActorComponent* Comp : OwnedComponents)
    {
        Comp->BeginPlay();
    }
}

void AActor::Tick(float DeltaTime)
{
    // TODO: 임시로 Actor에서 Tick 돌리기
    for (UActorComponent* Comp : OwnedComponents)
    {
        Comp->TickComponent(DeltaTime);
    }
}

void AActor::Destroyed()
{
    // World->Tick에서 컴포넌트 제거
    // for (UActorComponent* Comp : OwnedComponents)
    // {
    //     Comp->DestroyComponent();
    // }
}

// TODO: 추후 제거해야 함
void AActor::Render()
{
    for (UActorComponent* Comp : OwnedComponents)
    {
        if (USceneComponent* SceneComp = Cast<USceneComponent>(Comp))
        {
            SceneComp->Render();
        }
    }
}

bool AActor::Destroy()
{
    if (UWorld* World = GetWorld())
    {
        World->DestroyActor(this);
    }

    return true;
}

void AActor::RemoveOwnedComponent(UActorComponent* Component)
{
    OwnedComponents.Remove(Component);
}

void AActor::SetRootComponent(USceneComponent* NewRootComponent)
{
    if (NewRootComponent != nullptr || NewRootComponent->GetOwner() == this)
    {
        if (RootComponent != NewRootComponent)
        {
            USceneComponent* OldRootComponent = RootComponent;
            RootComponent = NewRootComponent;

            OldRootComponent->SetupAttachment(RootComponent);
        }
    }
}
