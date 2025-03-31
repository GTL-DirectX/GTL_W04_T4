#include "Core/HAL/PlatformType.h"
#include "EngineLoop.h"

FEngineLoop GEngineLoop;


int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nShowCmd)
{
    //AllocConsole(); // 콘솔 창 생성
    //FILE* stream;
    //freopen_s(&stream, "CONOUT$", "w", stdout); // std::cout을 콘솔에 연결
    // 사용 안하는 파라미터들
    UNREFERENCED_PARAMETER(hPrevInstance);
    UNREFERENCED_PARAMETER(lpCmdLine);
    UNREFERENCED_PARAMETER(nShowCmd);

    GEngineLoop.Init(hInstance);
    GEngineLoop.Tick();
    GEngineLoop.Exit();

    return 0;
}
