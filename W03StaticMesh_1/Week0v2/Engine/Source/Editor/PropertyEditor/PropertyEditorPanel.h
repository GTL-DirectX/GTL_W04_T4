﻿#pragma once
#include "Components/ActorComponent.h"
#include "UnrealEd/EditorPanel.h"

class PropertyEditorPanel : public UEditorPanel
{
public:
    virtual void Render() override;
    virtual void OnResize(HWND hWnd) override;


private:
    void RGBToHSV(float r, float g, float b, float& h, float& s, float& v);
    void HSVToRGB(float h, float s, float v, float& r, float& g, float& b);
private:
    float Width = 0, Height = 0;
    FVector Location = FVector(0, 0, 0);
    FVector Rotation = FVector(0, 0, 0);
    FVector Scale = FVector(0, 0, 0);
};
