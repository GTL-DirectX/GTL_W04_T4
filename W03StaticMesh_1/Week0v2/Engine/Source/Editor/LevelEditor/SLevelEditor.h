#pragma once
#include "Define.h"

class SSplitterH;
class SSplitterV;
class UWorld;
class FEditorViewportClient;
class SLevelEditor
{
public:
    SLevelEditor();
    ~SLevelEditor();
    void Initialize();
    void Tick(double deltaTime);
    void Input();
    void Release();
    
    void SelectViewport(POINT point);

private:
    SSplitterH* HSplitter;
    SSplitterV* VSplitter;
    UWorld* World;
    std::shared_ptr<FEditorViewportClient> viewportClients[4];
    std::shared_ptr<FEditorViewportClient> ActiveViewportClient;

    bool bLRButtonDown = false;
    POINT lastMousePos;

public:
    std::shared_ptr<FEditorViewportClient>* GetViewports() { return viewportClients; }
    std::shared_ptr<FEditorViewportClient> GetActiveViewportClient() const
    {
        return ActiveViewportClient;
    }
    void SetViewportClient(std::shared_ptr<FEditorViewportClient> viewportClient)
    {
        ActiveViewportClient = viewportClient;
    }
    void SetViewportClient(int index)
    {
        ActiveViewportClient = viewportClients[index];
    }
};

