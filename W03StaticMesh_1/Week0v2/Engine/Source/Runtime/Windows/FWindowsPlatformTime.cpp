#include "Windows/FWindowsPlatformTime.h"


double FWindowsPlatformTime::GSecondsPerCycle = 0.0;
bool FWindowsPlatformTime::bInitialized = false;
TMap<FString, double> FWindowsPlatformTime::TimeMap;
int FWindowsPlatformTime::PickTime = 0;
double FWindowsPlatformTime::AccumulatedTime = 0.0;


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
FScopeCycleCounter::FScopeCycleCounter(FString Id)
    : StartCycles(FWindowsPlatformTime::Cycles64())
    , UsedStatId(TStatId(Id))
{
}

FScopeCycleCounter::~FScopeCycleCounter()
{
    Finish();
}

uint64_t FScopeCycleCounter::Finish()
{
    const uint64_t EndCycles = FWindowsPlatformTime::Cycles64();
    const uint64_t CycleDiff = EndCycles - StartCycles;
    double elapsedMs = FWindowsPlatformTime::ToMilliseconds(CycleDiff);

    if (FWindowsPlatformTime::TimeMap.Find(UsedStatId.Name) == nullptr)
    {
        FWindowsPlatformTime::TimeMap[UsedStatId.Name] = 0.0;
    }
    else
    {
        if (UsedStatId.Name == "Picking")
        {
            FWindowsPlatformTime::PickTime++;
            FWindowsPlatformTime::AccumulatedTime += elapsedMs;
        }
        FWindowsPlatformTime::TimeMap[UsedStatId.Name] = elapsedMs;
    }
        
    return CycleDiff;
}
