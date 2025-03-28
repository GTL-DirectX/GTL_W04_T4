// Gamejam version 
#include "UnrealEd/SceneMgr.h"
#include "JSON/json.hpp"
#include "UObject/Object.h"
#include "Components/CubeComp.h"
#include <fstream>

#include "EditorViewportClient.h"
#include "World.h"
#include "Engine/FLoaderOBJ.h"
#include "Engine/StaticMeshActor.h"
#include "GameFramework/Actor.h"
#include "LevelEditor/SLevelEditor.h"

using json = nlohmann::json;

bool FSceneMgr::ParseSceneData(const FString& jsonStr)
{
    try {
        json j = json::parse(*jsonStr);

        // 버전과 NextUUID 읽기
        
        // Primitives 처리 (C++14 스타일)
        auto primitives = j["Primitives"];
        for (auto it = primitives.begin(); it != primitives.end(); ++it) {
            int id = std::stoi(it.key());  // Key는 문자열, 숫자로 변환
            const json& value = it.value();

            AStaticMeshActor* obj = nullptr;
            
            if (value.contains("Type"))
            {
                const FString TypeName = value["Type"].get<std::string>();
                if (TypeName == "StaticMeshComp")
                {
                     obj = GEngineLoop.GetWorld()->SpawnActor<AStaticMeshActor>();
                }
                // else if (TypeName == UCubeComp::StaticClass()->GetName())
                // {
                //     obj = FObjectFactory::ConstructObject<UCubeComp>();
                // }
                // else if (TypeName == UGizmoArrowComponent::StaticClass()->GetName())
                // {
                //     obj = FObjectFactory::ConstructObject<UGizmoArrowComponent>();
                // }
                // else if (TypeName == UBillboardComponent::StaticClass()->GetName())
                // {
                //     obj = FObjectFactory::ConstructObject<UBillboardComponent>();
                // }
                // else if (TypeName == ULightComponentBase::StaticClass()->GetName())
                // {
                //     obj = FObjectFactory::ConstructObject<ULightComponentBase>();
                // }
                // else if (TypeName == USkySphereComponent::StaticClass()->GetName())
                // {
                //     obj = FObjectFactory::ConstructObject<USkySphereComponent>();
                //     USkySphereComponent* skySphere = static_cast<USkySphereComponent*>(obj);
                // }
            }
            
            if (value.contains("ObjStaticMeshAsset"))
            {
                const FString ObjStaticMeshAssetPath = value["ObjStaticMeshAsset"].get<std::string>();
                UStaticMesh* NewMesh = FManagerOBJ::CreateStaticMesh(ObjStaticMeshAssetPath);
                obj->GetStaticMeshComponent()->SetStaticMesh(NewMesh);
            }

            if (value.contains("Location"))
            {
                obj->SetActorLocation(FVector(
                value["Location"].get<std::vector<float>>()[0],
                value["Location"].get<std::vector<float>>()[1],
                value["Location"].get<std::vector<float>>()[2]));
            }
            if (value.contains("Rotation"))
            {
                obj->SetActorRotation(FVector(
                value["Rotation"].get<std::vector<float>>()[0],
                value["Rotation"].get<std::vector<float>>()[1],
                value["Rotation"].get<std::vector<float>>()[2]));
            }
            if (value.contains("Scale"))
            {
                obj->SetActorScale(FVector(
                value["Scale"].get<std::vector<float>>()[0],
                value["Scale"].get<std::vector<float>>()[1],
                value["Scale"].get<std::vector<float>>()[2]));
            }
            // if (value.contains("Type"))
            // {
            //     if (UPrimitiveComponent* primitiveComp = Cast<UPrimitiveComponent>(SceneComp))
            //     {
            //         primitiveComp->SetType(value["Type"].get<std::string>());
            //     }
            //     else
            //     {
            //         std::string name = value["Type"].get<std::string>();
            //         // obj->NamePrivate = name.c_str();
            //     }
            // }
        }

        auto perspectiveCamera = j["PerspectiveCamera"];
        FViewportCameraTransform& Camera = GEngineLoop.GetLevelEditor()->GetActiveViewportClient()->ViewTransformPerspective;

        if (perspectiveCamera.contains("Location"))
        {
            auto loc = perspectiveCamera["Location"].get<std::vector<float>>();
            Camera.SetLocation(FVector(loc[0], loc[1], loc[2]));
        }

        if (perspectiveCamera.contains("Rotation"))
        {
            auto rot = perspectiveCamera["Rotation"].get<std::vector<float>>();
            Camera.SetRotation(FVector(rot[0], rot[1], rot[2]));
        }

        if (perspectiveCamera.contains("FOV"))
        {
            // FOV 값이 배열 형태이므로 첫 번째 요소 사용
            GEngineLoop.GetLevelEditor()->GetActiveViewportClient()->ViewFOV = perspectiveCamera["FOV"].get<std::vector<float>>()[0];
        }

        if (perspectiveCamera.contains("NearClip"))
        {
            // NearClip 값이 배열 형태이므로 첫 번째 요소 사용
            GEngineLoop.GetLevelEditor()->GetActiveViewportClient()->nearPlane = perspectiveCamera["NearClip"].get<std::vector<float>>()[0];
        }

        if (perspectiveCamera.contains("FarClip"))
        {
            // FarClip 값이 배열 형태이므로 첫 번째 요소 사용
            GEngineLoop.GetLevelEditor()->GetActiveViewportClient()->farPlane = perspectiveCamera["FarClip"].get<std::vector<float>>()[0];
        }
        
    }
    catch (const std::exception& e) {
        FString errorMessage = "Error parsing JSON: ";
        errorMessage += e.what();

        UE_LOG(LogLevel::Error, "No Json file");
        return false;
    }
    
    return true;
}

FString FSceneMgr::LoadSceneFromFile(const FString& filename)
{
    std::ifstream inFile(*filename);
    if (!inFile) {
        UE_LOG(LogLevel::Error, "Failed to open file for reading: %s", *filename);
        return FString();
    }

    json j;
    try {
        inFile >> j; // JSON 파일 읽기
    }
    catch (const std::exception& e) {
        UE_LOG(LogLevel::Error, "Error parsing JSON: %s", e.what());
        return FString();
    }

    inFile.close();

    return j.dump(4);
}

std::string FSceneMgr::SerializeSceneData(const SceneData& sceneData)
{
    // json j;
    //
    // // NextUUID 저장
    // j["NextUUID"] = sceneData.NextUUID;
    //
    // // Primitives 처리 (C++17 스타일)
    // for (const auto& [Id, Obj] : sceneData.Primitives)
    // {
    //     USceneComponent* primitive = static_cast<USceneComponent*>(Obj);
    //     std::vector<float> Location = { primitive->GetWorldLocation().x,primitive->GetWorldLocation().y,primitive->GetWorldLocation().z };
    //     std::vector<float> Rotation = { primitive->GetWorldRotation().x,primitive->GetWorldRotation().y,primitive->GetWorldRotation().z };
    //     std::vector<float> Scale = { primitive->GetWorldScale().x,primitive->GetWorldScale().y,primitive->GetWorldScale().z };
    //
    //     std::string primitiveName = *primitive->GetName();
    //     size_t pos = primitiveName.rfind('_');
    //     if (pos != INDEX_NONE) {
    //         primitiveName = primitiveName.substr(0, pos);
    //     }
    //     j["Primitives"][std::to_string(Id)] = {
    //         {"Location", Location},
    //         {"Rotation", Rotation},
    //         {"Scale", Scale},
    //         {"Type",primitiveName}
    //     };
    // }
    //
    // for (const auto& [id, camera] : sceneData)
    // {
    //     UCameraComponent* cameraComponent = static_cast<UCameraComponent*>(camera);
    //     TArray<float> Location = { cameraComponent->GetWorldLocation().x, cameraComponent->GetWorldLocation().y, cameraComponent->GetWorldLocation().z };
    //     TArray<float> Rotation = { 0.0f, cameraComponent->GetWorldRotation().y, cameraComponent->GetWorldRotation().z };
    //     float FOV = cameraComponent->GetFOV();
    //     float nearClip = cameraComponent->GetNearClip();
    //     float farClip = cameraComponent->GetFarClip();
    //
    //     //
    //     j["PerspectiveCamera"][std::to_string(id)] = {
    //         {"Location", Location},
    //         {"Rotation", Rotation},
    //         {"FOV", FOV},
    //         {"NearClip", nearClip},
    //         {"FarClip", farClip}
    //     };
    // }
    //
    //
    // return j.dump(4); // 4는 들여쓰기 수준
    return std::string();
}

bool FSceneMgr::SaveSceneToFile(const FString& filename, const SceneData& sceneData)
{
    // std::ofstream outFile(*filename);
    // if (!outFile) {
    //     FString errorMessage = "Failed to open file for writing: ";
    //     MessageBoxA(NULL, *errorMessage, "Error", MB_OK | MB_ICONERROR);
    //     return false;
    // }
    //
    // std::string jsonData = SerializeSceneData(sceneData);
    // outFile << jsonData;
    // outFile.close();

    return true;
}

