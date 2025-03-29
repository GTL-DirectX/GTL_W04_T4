#pragma once
#include <mutex>

class FThreadStats {
public:
    // FPS와 프레임 시간을 설정
    static void SetFPS(float fps, int frameMs);

    // 최근 picking 시간을 설정 (단일 측정 값)
    static void SetPickingTime(double pickingTime);

    // picking 시도 횟수를 1 증가
    static void IncrementNumAttempts();

    // 누적 picking 시간에 현재 측정 값을 추가
    static void AddAccumulatedTime(double timeMs);

    // 현재 통계 값을 출력용으로 가져옴
    static void GetStats(float& fps, int& frameMs, double& pickingTime, int& numAttempts, double& accumulatedTime);

    // 통계 값을 초기화 (필요 시 호출)
    static void ResetStats();

private:
    static std::mutex Mutex;
    static float FPS;            
    static int FrameMs;         
    static double PickingTime;     // 마지막 측정한 picking 시간 (ms)
    static int NumAttempts;        // picking 시도 횟수
    static double AccumulatedTime; // 누적된 picking 시간 (ms)
};