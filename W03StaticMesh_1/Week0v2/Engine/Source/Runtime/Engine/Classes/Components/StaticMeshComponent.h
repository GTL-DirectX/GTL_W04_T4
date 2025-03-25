#pragma once
#include "Components/MeshComponent.h"
#include "Mesh/StaticMesh.h"

class UStaticMeshComponent : public UMeshComponent
{
    DECLARE_CLASS(UStaticMeshComponent, USceneComponent)

public:
    UStaticMeshComponent();
    ~UStaticMeshComponent() {}

    virtual void Initialize() override {}
    virtual void Update(double deltaTime) override {}
    virtual void Release() override {}
    virtual void Render() override;

    virtual uint32 GetNumMaterials() const override ;
    virtual UMaterial* GetMaterial(uint32 ElementIndex) const override;
    virtual uint32 GetMaterialIndex(FName MaterialSlotName) const override;
    virtual TArray<FName> GetMaterialSlotNames() const override;
    virtual void GetUsedMaterials(TArray<UMaterial*> Out) const override;

    virtual int CheckRayIntersection(FVector& rayOrigin, FVector& rayDirection, float& pfNearHitDistance) override;

    UStaticMesh* GetStaticMesh() { return staticMesh; }
    void SetStaticMesh(UStaticMesh* value)
    { 
        staticMesh = value;
        OverrideMaterials.SetNum(value->GetMaterials().Num());
    }

protected:
    UStaticMesh* staticMesh = nullptr;
    int selectedSubMeshIndex = -1;
};