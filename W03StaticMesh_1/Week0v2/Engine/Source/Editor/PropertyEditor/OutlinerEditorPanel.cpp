﻿#include "OutlinerEditorPanel.h"
#include "EngineLoop.h"
#include "Components/SceneComponent.h"

void OutlinerEditorPanel::Render()
{
    /* Pre Setup */
    ImGuiIO& io = ImGui::GetIO();
    
    float PanelWidth = (Width) * 0.2f - 6.0f;
    float PanelHeight = (Height) * 0.3f;

    float PanelPosX = (Width) * 0.8f + 5.0f;
    float PanelPosY = 5.0f;

    ImVec2 MinSize(140, 100);
    ImVec2 MaxSize(FLT_MAX, 500);
    
    /* Min, Max Size */
    ImGui::SetNextWindowSizeConstraints(MinSize, MaxSize);
    
    /* Panel Position */
    ImGui::SetNextWindowPos(ImVec2(PanelPosX, PanelPosY), ImGuiCond_Always);

    /* Panel Size */
    ImGui::SetNextWindowSize(ImVec2(PanelWidth, PanelHeight), ImGuiCond_Always);

    /* Panel Flags */
    ImGuiWindowFlags PanelFlags = ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse;
    
    /* Render Start */
    ImGui::Begin("Outliner", nullptr, PanelFlags);

    if (ImGui::TreeNode("Primitives")) // 트리 노드 생성
    {
        static int selected = -1; // 선택된 항목 저장용 변수
        
        // 오브젝트 리스트
        for (int32 i = 0; i < GEngineLoop.GetWorld()->GetObjectArr().Num();i++)
        {
            if(!GEngineLoop.GetWorld()->GetObjectArr()[i]->IsA(USceneComponent::StaticClass()))
                continue;
            // 선택 가능 항목 (Selectable)
            if (ImGui::Selectable(*GEngineLoop.GetWorld()->GetObjectArr()[i]->GetName(), selected == i))
            {
                selected = i; // 선택된 아이템 업데이트
                GEngineLoop.GetWorld()->SetPickingObj(GEngineLoop.GetWorld()->GetObjectArr()[i]);
            }
        }
        ImGui::TreePop(); // 트리 닫기
    }
    ImGui::End();
    
}
    
void OutlinerEditorPanel::OnResize(HWND hWnd)
{
    RECT clientRect;
    GetClientRect(hWnd, &clientRect);
    Width = clientRect.right - clientRect.left;
    Height = clientRect.bottom - clientRect.top;
}
