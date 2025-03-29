﻿#pragma once
#include <queue>

#include "Define.h"
#include "GameFramework/Actor.h"

class Octree
{
public:
    // 기본 생성자: 영역이 지정되지 않은 경우 (나중에 설정)
    Octree();

    // 지정된 영역(InRegion)으로 Octree를 초기화하는 생성자
    Octree(const FBoundingBox& InRegion);

    // 소멸자: 동적 할당된 자식 노드들의 메모리를 해제합니다.
    ~Octree() = default;

    // 새로운 액터를 Octree에 삽입합니다.
    void Insert(AActor* Actor);

    // 대기 중인 삽입 객체를 처리하는 등 트리를 업데이트합니다.
    void UpdateTree();

    // 현재까지의 객체들을 기반으로 트리를 구축합니다.
    void BuildTree();

private:
    // 내부용 생성자: 특정 영역과 액터 목록(InActors)을 사용해 Octree를 초기화합니다.
    Octree(const FBoundingBox& InRegion, const TArray<AActor*>& InActors);

    int GetOctant(const FVector& Center, const FVector& HalfSize) const;

    FBoundingBox CalculateActorBoundingBox(const AActor* Actor) const;
public:
    // 현재 노드가 담당하는 영역(직육면체)
    FBoundingBox Region;

    // 이 노드에 포함된 액터들
    TArray<AActor*> Actors;
 
    // 부모 노드(루트 노드라면 nullptr)
    Octree* Parent;

    // 8개의 자식 노드. 자식 노드가 없으면 nullptr (리프 노드)
    TArray<std::unique_ptr<Octree>> Children;

    // 활성화된 자식 노드를 비트 마스크 형태로 표현 (어떤 자식이 사용 중인지 표시)
    uint8_t ActiveNodeMask;

    // 한 공간이 가지는 오브젝트 최대 크기
    static uint32 Capacity;
    
    // 최소 영역 크기: 영역이 이 크기보다 작으면 더 이상 분할하지 않음 (예: 1x1x1 큐브)
    static constexpr int MinSize = 1;

    // 삽입 대기 중인 액터들을 저장하는 큐 (트리 구축 전이나 업데이트 시 사용)
    static std::queue<AActor*> PendingInsertion;

    // 트리에 삽입해야 할 객체들이 남아있어 트리가 완전하지 않음을 표시
    static bool bReadyTree;

    // 아직 구축된 트리가 없음을 표시 (최초 구축 여부)
    static bool bBuildTree;
};
