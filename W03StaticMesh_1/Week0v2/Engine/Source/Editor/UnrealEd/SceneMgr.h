#pragma once
#include "Define.h"
#include "Container/Map.h"

class UObject;
struct SceneData {
    int32 NextUUID;
    TMap<int32, UObject*> Primitives;
};
class FSceneMgr
{
public:
    static bool ParseSceneData(const FString& jsonStr);
    static FString LoadSceneFromFile(const FString& filename);
    static std::string SerializeSceneData(const SceneData& sceneData);
    static bool SaveSceneToFile(const FString& filename, const SceneData& sceneData);
};

