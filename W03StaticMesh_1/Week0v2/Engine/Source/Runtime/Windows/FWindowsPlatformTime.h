#pragma once

#include "Runtime/Core/Container/Map.h"
#include "Runtime/Core/Container/String.h"

//-------------------------------------------------------------------------------------------------
// FWindowsPlatformTime
//-------------------------------------------------------------------------------------------------
class FWindowsPlatformTime
{
public:
    // 측정 단위 (초/사이클) – 초기값은 0
    static double GSecondsPerCycle;
    static bool bInitialized;

    static void InitTiming();
    static float GetSecondsPerCycle();
    static uint64_t GetFrequency();
    static double ToMilliseconds(uint64_t CycleDiff);
    static uint64_t Cycles64();

    static TMap<FString, double> TimeMap;
    static int PickTime;
    static double AccumulatedTime;
};

struct TStatId
{
    FString Name;

    // 기본 생성자: 빈 문자열을 사용
    TStatId() : Name(TEXT("")) {}

    // FString을 받아서 초기화하는 생성자
    TStatId(const FString& InName) : Name(InName) {}
};

//-------------------------------------------------------------------------------------------------
// FScopeCycleCounter
// 범위 기반 성능 측정용 타이머 클래스
// 생성 시 타이머 시작, Finish() 또는 소멸자에서 경과 사이클을 계산
//-------------------------------------------------------------------------------------------------
class FScopeCycleCounter
{
public:
    FScopeCycleCounter(FString Id);
    ~FScopeCycleCounter();

    uint64_t GetStartCycles() const { return StartCycles; }
    uint64_t Finish();

private:
    uint64_t StartCycles;
    TStatId UsedStatId;
};