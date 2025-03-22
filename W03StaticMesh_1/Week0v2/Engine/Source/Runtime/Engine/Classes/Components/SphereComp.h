#pragma once
#include "PrimitiveComponent.h"
class USphereComp :public UPrimitiveComponent
{
    DECLARE_CLASS(USphereComp, UPrimitiveComponent)

public:
    USphereComp();
    virtual ~USphereComp() override;

    virtual void		Initialize()	override;
    virtual void		Update(double deltaTime)		override;
    virtual void		Release()					override;
    virtual	void		Render()					override;

};

