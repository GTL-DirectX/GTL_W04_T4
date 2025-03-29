#include "FThreadStats.h"

std::mutex FThreadStats::Mutex;
float FThreadStats::FPS = 0.0f;
int FThreadStats::FrameMs = 0;
double FThreadStats::PickingTime = 0.0;
int FThreadStats::NumAttempts = 0;
double FThreadStats::AccumulatedTime = 0.0;

void FThreadStats::SetFPS(float fps, int frameMs) {
    std::lock_guard<std::mutex> lock(Mutex);
    FPS = fps;
    FrameMs = frameMs;
}

void FThreadStats::SetPickingTime(double pickingTime) {
    std::lock_guard<std::mutex> lock(Mutex);
    PickingTime = pickingTime;
}

void FThreadStats::IncrementNumAttempts() {
    std::lock_guard<std::mutex> lock(Mutex);
    ++NumAttempts;
}

void FThreadStats::AddAccumulatedTime(double timeMs) {
    std::lock_guard<std::mutex> lock(Mutex);
    AccumulatedTime += timeMs;
}

void FThreadStats::GetStats(float& fps, int& frameMs, double& pickingTime, int& numAttempts, double& accumulatedTime) {
    std::lock_guard<std::mutex> lock(Mutex);
    fps = FPS;
    frameMs = FrameMs;
    pickingTime = PickingTime;
    numAttempts = NumAttempts;
    accumulatedTime = AccumulatedTime;
}

void FThreadStats::ResetStats() {
    std::lock_guard<std::mutex> lock(Mutex);
    FPS = 0;
    FrameMs = 0;
    PickingTime = 0;
    NumAttempts = 0;
    AccumulatedTime = 0;
}
