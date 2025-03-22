#pragma once
#include "PrimitiveComponent.h"
#include "Material/Material.h"

class UMeshComponent : public UPrimitiveComponent
{
public:
    virtual void Initialize() override;
    virtual void Update(double deltaTime) override;
    virtual void Release() override;
    virtual void Render() override;

#pragma region Material
    virtual uint32 GetNumMaterials() const { return 0; }
    virtual UMaterial* GetMaterial(uint32 ElementIndex) const;
    virtual uint32 GetMaterialIndex(FName MaterialSlotName) const;
    virtual UMaterial* GetMaterialByName(FName MaterialSlotName) const;
    virtual TArray<FName> GetMaterialSlotNames() const;
    virtual void SetMaterial(uint32 ElementIndex, UMaterial* Material);
    virtual void SetMaterialByName(FName MaterialSlotName, class UMaterial* Material);
    virtual void GetUsedMaterials(TArray<UMaterial*> Out) const;
#pragma endregion
protected:
    TArray<UMaterial*> OverrideMaterials;
};

