#include "Windows/FWindowsPlatformTime.h"
#include "Windows/FThreadStats.h"
#include <sstream>

double FWindowsPlatformTime::GSecondsPerCycle = 0.0;
bool FWindowsPlatformTime::bInitialized = false;

void FWindowsPlatformTime::InitTiming()
{
    if (!bInitialized)
    {
        bInitialized = true;
        double Frequency = static_cast<double>(GetFrequency());
        if (Frequency <= 0.0)
        {
            Frequency = 1.0;
        }
        GSecondsPerCycle = 1.0 / Frequency;
    }
}

float FWindowsPlatformTime::GetSecondsPerCycle()
{
    if (!bInitialized)
    {
        InitTiming();
    }
    return static_cast<float>(GSecondsPerCycle);
}

uint64_t FWindowsPlatformTime::GetFrequency()
{
    LARGE_INTEGER Frequency;
    QueryPerformanceFrequency(&Frequency);
    return static_cast<uint64_t>(Frequency.QuadPart);
}

double FWindowsPlatformTime::ToMilliseconds(uint64_t CycleDiff)
{
    double Ms = static_cast<double>(CycleDiff) * GetSecondsPerCycle() * 1000.0;
    return Ms;
}

uint64_t FWindowsPlatformTime::Cycles64()
{
    LARGE_INTEGER CycleCount;
    QueryPerformanceCounter(&CycleCount);
    return static_cast<uint64_t>(CycleCount.QuadPart);
}

//-------------------------------------------------------------------------------------------------
// FScopeCycleCounter 구현
//-------------------------------------------------------------------------------------------------
FScopeCycleCounter::FScopeCycleCounter(TStatId StatId)
    : StartCycles(FPlatformTime::Cycles64())
    , UsedStatId(StatId)
{
    
}

FScopeCycleCounter::~FScopeCycleCounter()
{
    // 소멸 시 Finish()를 호출해 자동 측정
    Finish();
}

uint64_t FScopeCycleCounter::Finish()
{
    const uint64_t EndCycles = FPlatformTime::Cycles64();
    const uint64_t CycleDiff = EndCycles - StartCycles;
    double elapsedMs = FPlatformTime::ToMilliseconds(CycleDiff);

    FThreadStats::SetPickingTime(elapsedMs);
    FThreadStats::AddAccumulatedTime(elapsedMs);
    FThreadStats::IncrementNumAttempts();

    return CycleDiff;
}
