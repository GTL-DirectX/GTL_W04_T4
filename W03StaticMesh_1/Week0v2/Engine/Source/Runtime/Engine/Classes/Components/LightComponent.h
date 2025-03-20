#pragma once
#include "PrimitiveComponent.h"
#include "Define.h"
class UBillboardComponent;

class ULightComponentBase : public USceneComponent
{
    DECLARE_CLASS(ULightComponentBase, USceneComponent)

public:
    ULightComponentBase();
    virtual ~ULightComponentBase() override;

    virtual void Render() override;
    virtual void Update(double deltaTim) override;
    virtual int CheckRayIntersection(FVector& rayOrigin, FVector& rayDirection, float& pfNearHitDistance);
    void InitializeLight();
    void SetColor(FVector4 newColor);
    FVector4 GetColor() const;
    float GetRadius() const;
    void SetRadius(float r);

private:
    FVector4 color;
    float radius;
    FBoundingBox AABB;
    UBillboardComponent* texture2D;
};
