#pragma once

#include <Windows.h>
#include <cstdint>
#include <cassert>

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
};

typedef FWindowsPlatformTime FPlatformTime;

struct TStatId
{
    
};

//-------------------------------------------------------------------------------------------------
// FScopeCycleCounter
// 범위 기반 성능 측정용 타이머 클래스
// 생성 시 타이머 시작, Finish() 또는 소멸자에서 경과 사이클을 계산
//-------------------------------------------------------------------------------------------------
class FScopeCycleCounter
{
public:
    FScopeCycleCounter(TStatId StatId);
    ~FScopeCycleCounter();

    // 필요 시 통계에 추가하는 로직 추가 가능
    uint64_t Finish();

private:
    uint64_t StartCycles;
    TStatId UsedStatId;
};