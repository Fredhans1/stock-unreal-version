#pragma once

#include "CoreMinimal.h"
#include "Components/PrimitiveComponent.h"
#include "Interfaces/Interface_CollisionDataProvider.h"
#include "PhysicsEngine/BodySetup.h"

#include "OctreeTerrainComponent.generated.h"

class FOctreeTerrainSceneProxy;

UCLASS(ClassGroup = Rendering, meta = (BlueprintSpawnableComponent))
class RMCTEST_API UOctreeTerrainComponent final
	: public UPrimitiveComponent
	, public IInterface_CollisionDataProvider
{
	GENERATED_BODY()

public:
	UOctreeTerrainComponent();

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "100.0"))
	float BaseChunkSize = 1000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "4", ClampMax = "64"))
	int32 GridResolution = 16;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "1", ClampMax = "8"))
	int32 NumLODLevels = 4;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "3", ClampMax = "21"))
	int32 ChunksPerSide = 11;

	// Retained for compatibility with existing assets. The stock implementation
	// renders a heightfield and therefore does not allocate vertical chunk layers.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "1", ClampMax = "13"))
	int32 ChunksVertical = 1;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "0.0"))
	float UpdateThreshold = 100.0f;

	// Retained for compatibility with existing assets. CPU mesh sizing is
	// derived from GridResolution.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "1024"))
	int32 MaxVertsPerChunk = 2048;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "6144"))
	int32 MaxIndicesPerChunk = 12288;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain")
	TObjectPtr<UMaterialInterface> TerrainMaterial;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.0"))
	float HeightScale = 8000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.000001"))
	float NoiseFrequency = 0.0001f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "1", ClampMax = "8"))
	int32 NoiseOctaves = 5;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.0"))
	float CaveNoiseFrequency = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.0"))
	float CaveNoiseAmplitude = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Collision", meta = (ClampMin = "1", ClampMax = "32"))
	int32 CollisionDownsampleFactor = 2;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Rendering", meta = (ClampMin = "0.0"))
	float SkirtDepth = 1000.0f;

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void ApplySphereEdit(FVector Center, float Radius, bool bSubtract);

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void ClearAllEdits();

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void RebuildTerrain();

	virtual void BeginPlay() override;
	virtual void TickComponent(
		float DeltaTime,
		ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	virtual FPrimitiveSceneProxy* CreateSceneProxy() override;
	virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;
	virtual void GetUsedMaterials(
		TArray<UMaterialInterface*>& OutMaterials,
		bool bGetDebugMaterials = false) const override;
	virtual UBodySetup* GetBodySetup() override;

	virtual bool GetPhysicsTriMeshData(
		FTriMeshCollisionData* CollisionData,
		bool InUseAllTriData) override;
	virtual bool ContainsPhysicsTriMeshData(bool InUseAllTriData) const override;
	virtual bool WantsNegXTriMesh() override { return false; }

private:
	friend class FOctreeTerrainSceneProxy;

	struct FSphereEdit
	{
		FVector Center = FVector::ZeroVector;
		float Radius = 0.0f;
		bool bSubtract = true;
	};

	FVector GetPlayerPosition() const;
	float SampleHeight(double WorldX, double WorldY) const;
	FVector3f SampleNormal(double WorldX, double WorldY, double Step) const;
	void BuildTerrainMesh(const FVector& PlayerPosition);
	void AppendChunk(
		int32 LOD,
		int32 ChunkX,
		int32 ChunkY,
		const FVector2D& ChunkOrigin,
		float ChunkSize,
		int32 Resolution,
		bool bBuildCollision);
	void AppendSkirts(
		int32 FirstVertex,
		int32 Resolution,
		float Depth);
	void RebuildCollision();

	TArray<FSphereEdit> SphereEdits;

	TArray<FVector3f> MeshPositions;
	TArray<FVector3f> MeshNormals;
	TArray<FVector2f> MeshUVs;
	TArray<FColor> MeshColors;
	TArray<uint32> MeshIndices;

	TArray<FVector3f> CollisionVertices;
	TArray<FTriIndices> CollisionTriangles;

	UPROPERTY(Transient)
	TObjectPtr<UBodySetup> CollisionBodySetup;

	FBoxSphereBounds LocalBounds = FBoxSphereBounds(FVector::ZeroVector, FVector(1000.0), 1000.0);
	FIntVector LastTerrainOrigin = FIntVector(MAX_int32, MAX_int32, MAX_int32);
	FVector LastBuildPlayerPosition = FVector(MAX_flt, MAX_flt, MAX_flt);
	bool bRebuildRequested = true;
	bool bCollisionCookInProgress = false;
};
