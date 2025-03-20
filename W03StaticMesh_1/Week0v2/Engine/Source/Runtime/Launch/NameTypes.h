#pragma once
#include "Define.h"
#include <ostream>
#include <iostream>

#include "Container/Map.h"
#include "Core/Container/Array.h"
#include "Core/HAL/PlatformType.h"

class FNameEntry
{
public:
    FNameEntry(const FString name) : Name(name) {}

    FString Name;
    const FString& GetName() { return Name; }
};

class FNamePool
{
private:
    FNamePool() {}
    TArray<FNameEntry*> Entries;
    TMap<FString , int32> NameToIndex;

public:
    static FString ToLowerCase(const FString& Str)
    {
        FString LowerStr = Str;
        std::transform(LowerStr.begin(), LowerStr.end(), LowerStr.begin(),
            [](unsigned char c) { return std::tolower(c); });
        return LowerStr;
    }
    static FNamePool& GetInstance()
    {
        static FNamePool instance;
        return instance;
    }

    int AddEntry(const FString& Name)
    {
        FString lowName = ToLowerCase(Name);
        if (const int32* Value = NameToIndex.Find(Name))
        {
            return *Value;
        }

        int32 newid = static_cast<int32>(Entries.Num());
        Entries.Add(new FNameEntry(Name));
        NameToIndex[Name] = newid;
        return newid;
    }
    const FString& GetEntry(int32 idx)
    {
        return Entries[idx]->GetName();
    }
};

class FName
{
private:
    int32 index;

public:
    FName(const FString& Name)
    {
        index = FNamePool::GetInstance().AddEntry(Name);
    }
    const FString& ToString() const 
    {
        return FNamePool::GetInstance().GetEntry(index);
    }
    bool operator==(const FName& Other) const
    {
        return index == Other.index;
    }

    bool operator!=(const FName& Other) const
    {
        return index != Other.index;
    }
    friend std::ostream& operator<<(std::ostream& Os, const FName& Name)
    {
        Os << Name.ToString();
        return Os;
    }
};
