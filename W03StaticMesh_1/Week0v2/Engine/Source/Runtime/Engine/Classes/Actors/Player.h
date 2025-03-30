#pragma once
#include <mutex>

#include "GameFramework/Actor.h"
#include "UObject/ObjectMacros.h"
#include "UObject/ObjectTypes.h"
class UGizmoBaseComponent;
class UGizmoArrowComponent;
class USceneComponent;
class UPrimitiveComponent;

class AEditorPlayer : public AActor
{
    DECLARE_CLASS(AEditorPlayer, AActor)

    AEditorPlayer();

    virtual void Tick(float DeltaTime) override;

    void Input();
    bool PickGizmo(FVector& rayOrigin);
    void PickActor(const FVector& pickPosition, const FVector& PickDirection);
    void PickActor(const FRay& Ray); // Octree
    void AddControlMode();
    void AddCoordiMode();

private:
    int RayIntersectsObject(const FVector& pickPosition, USceneComponent* obj, float& hitDistance, int& intersectCount);
    inline int RayIntersectsObject(const FRay& Ray, USceneComponent* obj, float& hitDistance, int& intersectCount); // Octree
    void ScreenToViewSpace(int screenX, int screenY, const FMatrix& viewMatrix, const FMatrix& projectionMatrix, FVector& rayOrigin);
    FRay GetMouseRay(int screenX, int screenY); // Octree
    void PickedObjControl();
    void ControlRotation(USceneComponent* pObj, UGizmoBaseComponent* Gizmo, int32 deltaX, int32 deltaY);
    void ControlTranslation(USceneComponent* pObj, UGizmoBaseComponent* Gizmo, int32 deltaX, int32 deltaY);
    void ControlScale(USceneComponent* pObj, UGizmoBaseComponent* Gizmo, int32 deltaX, int32 deltaY);
    bool bLeftMouseDown = false;
    bool bRightMouseDown = false;
    bool bSpaceDown = false;

    POINT m_LastMousePos;
    ControlMode cMode = CM_TRANSLATION;
    CoordiMode cdMode = CDM_WORLD;
    std::mutex ResultMutex;

public:
    void SetMode(ControlMode _Mode) { cMode = _Mode; }
    ControlMode GetControlMode() const { return cMode; }
    CoordiMode GetCoordiMode() const { return cdMode; }
};
