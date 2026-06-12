#pragma once

#include <atomic>

#include "CoreMinimal.h"
#include "Components/PrimitiveComponent.h"
#include "Interfaces/Interface_CollisionDataProvider.h"
#include "PhysicsEngine/BodySetup.h"
#include "RHIGPUReadback.h"
#include "RenderGraphUtils.h"

#include "OctreeTerrainComponent.generated.h"

class FOctreeTerrainSceneProxy;

USTRUCT(BlueprintType)
struct FOctreeNoiseLayer
{
	GENERATED_BODY()

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Noise")
	float Frequency = 0.0001f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Noise")
	float Amplitude = 8000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Noise")
	float OffsetX = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Noise")
	float OffsetY = 0.0f;
};

// GPU-resident voxel terrain.
//
// A 3D clipmap of marching-cubes chunks is generated and kept entirely on the
// GPU: the CPU only decides which chunks are dirty (ring shifts, sphere edits,
// LOD carve-boundary changes) and dispatches compute. Every chunk allocates an
// exact-size range in one shared vertex pool (count pass -> bump allocator ->
// mesh pass), so chunk vertex counts vary freely with the landscape and no VRAM
// is spent on padding or empty chunks. Draws are per-chunk DrawIndexedIndirect
// with a GPU-written BaseVertexLocation and a shared identity index buffer.
//
// Sphere edits (add/subtract) re-mesh only the chunks they touch, so they run
// comfortably at per-frame rates. Collision for line traces / physics is a
// GPU-gathered readback of the LOD 0 chunks around the player, cooked async.
//
// The component expects to sit at the world origin (vertices are generated in
// world space, like VoxelMeshComponent).
UCLASS(ClassGroup = Rendering, meta = (BlueprintSpawnableComponent))
class RMCTEST_API UOctreeTerrainComponent final
	: public UPrimitiveComponent
	, public IInterface_CollisionDataProvider
{
	GENERATED_BODY()

public:
	UOctreeTerrainComponent();

	// ── Grid configuration ─────────────────────────────────────────────────

	// World size of a LOD 0 chunk (cm). Cell size at LOD 0 = BaseChunkSize / GridResolution.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "100.0"))
	float BaseChunkSize = 1000.0f;

	// Marching-cubes cells per chunk axis. Rounded to a multiple of 8.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "8", ClampMax = "64"))
	int32 GridResolution = 32;

	// Each LOD doubles the chunk size. The finer LOD's ring volume is carved
	// out of the coarser LOD, clipmap style.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "1", ClampMax = "10"))
	int32 NumLODLevels = 7;

	// Horizontal chunks per ring side (per LOD).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "3", ClampMax = "21"))
	int32 ChunksPerSide = 9;

	// Vertical chunk layers per LOD, centered on the player.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "1", ClampMax = "13"))
	int32 ChunksVertical = 4;

	// Player movement (cm) before the clipmap re-centers.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "0.0"))
	float UpdateThreshold = 500.0f;

	// Initial shared vertex-pool capacity. The pool compacts and grows
	// automatically when the landscape needs more, so this is just a hint.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Grid", meta = (ClampMin = "65536"))
	int32 InitialPoolVertexCapacity = 4000000;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain")
	TObjectPtr<UMaterialInterface> TerrainMaterial;

	// ── Landscape definition ────────────────────────────────────────────────

	// 2D perlin height layers (filled with defaults at BeginPlay when empty).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise")
	TArray<FOctreeNoiseLayer> NoiseLayers;

	// Optional 3D noise carved into the height field (caves / overhang detail).
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.0"))
	float CaveNoiseFrequency = 0.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Noise", meta = (ClampMin = "0.0"))
	float CaveNoiseAmplitude = 0.0f;

	// UV = world XY * this scale.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Rendering", meta = (ClampMin = "0.000001"))
	float WorldUVScale = 0.0005f;

	// ── Collision ──────────────────────────────────────────────────────────

	// LOD 0 chunk radius (chebyshev) around the player gathered into the
	// physics tri-mesh.
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "OctreeTerrain|Collision", meta = (ClampMin = "1", ClampMax = "8"))
	int32 CollisionRadiusChunks = 2;

	// ── Edit API ───────────────────────────────────────────────────────────

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void ApplySphereEdit(FVector Center, float Radius, bool bSubtract);

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void ClearAllEdits();

	UFUNCTION(BlueprintCallable, Category = "OctreeTerrain")
	void RebuildTerrain();

	// ── UPrimitiveComponent ────────────────────────────────────────────────

	virtual void BeginPlay() override;
	virtual void EndPlay(const EEndPlayReason::Type EndPlayReason) override;
	virtual void TickComponent(float DeltaTime, ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;
	virtual FPrimitiveSceneProxy* CreateSceneProxy() override;
	virtual FBoxSphereBounds CalcBounds(const FTransform& LocalToWorld) const override;
	virtual void GetUsedMaterials(TArray<UMaterialInterface*>& OutMaterials,
		bool bGetDebugMaterials = false) const override;
	virtual UBodySetup* GetBodySetup() override;

	// ── IInterface_CollisionDataProvider ───────────────────────────────────

	virtual bool GetPhysicsTriMeshData(FTriMeshCollisionData* CollisionData, bool InUseAllTriData) override;
	virtual bool ContainsPhysicsTriMeshData(bool InUseAllTriData) const override;
	virtual bool WantsNegXTriMesh() override { return false; }

private:
	friend class FOctreeTerrainSceneProxy;

	// ── Derived configuration (computed at BeginPlay) ──────────────────────
	int32  CellsN = 32;                 // sanitized GridResolution
	int32  ChunksPerLOD = 0;            // ChunksPerSide^2 * ChunksVertical
	int32  MaxChunks = 0;               // ChunksPerLOD * NumLODLevels
	uint32 MaxVertsPerChunkClamp = 0;   // per-chunk vertex count safety cap
	uint32 CurrentCapacity = 0;         // live vertex-pool capacity
	double ZBandMin = 0.0;              // conservative surface height bounds
	double ZBandMax = 0.0;
	bool   bGPUInitialized = false;

	double ChunkSizeForLOD(int32 LOD) const;
	FVector GetPlayerPosition() const;

	void SanitizeConfig();
	void EnsureNoiseDefaults();

	// ── CPU mirror of the GPU chunk grid ───────────────────────────────────
	// Slot index = LOD * ChunksPerLOD + (sz * CPS + sy) * CPS + sx, with
	// toroidal coord->slot mapping so chunks keep their slot while inside the
	// ring (only chunks entering the ring get re-meshed on movement).
	TArray<FIntVector> SlotCoord;
	TArray<uint8>      SlotActive;
	TArray<FIntVector> RingOrigin;      // per LOD, in that LOD's chunk units
	bool               bForceAllDirty = true;
	bool               bUpdateRequested = false;

	bool ChunkMayHaveSurface(const FIntVector& Coord, int32 LOD) const;
	FBox ChunkBox(const FIntVector& Coord, int32 LOD) const;
	FBox FinerRingBox(const FIntVector& Origin, int32 FinerLOD) const;

	// Builds the dirty set + upload payload and enqueues the GPU update graph.
	void BuildAndEnqueueUpdate(const FVector& PlayerPos);

	// ── Edits ──────────────────────────────────────────────────────────────
	// xyz = center, w = +radius (add) / -radius (subtract); applied in order.
	TArray<FVector4f> Edits;
	int32             NumPendingEdits = 0;

	// ── Pool management (driven by the AllocState readback) ───────────────
	bool   bWantCompact = false;
	uint32 PendingNewCapacity = 0;
	uint32 LastPoolChangeSerial = 0;   // update serial of the last pool init/compact
	uint32 StateSerialInFlight = 0;    // update serial the in-flight state readback observed
	void HandleAllocState(uint32 Head, uint32 Overflow, uint32 TotalLive, uint32 Serial);

	// ── GPU-resident buffers (created on the render thread, refs cached here
	//    so they survive scene-proxy recreation) ─────────────────────────────
	TRefCountPtr<FRDGPooledBuffer> PoolPos;
	TRefCountPtr<FRDGPooledBuffer> PoolTan;
	TRefCountPtr<FRDGPooledBuffer> PoolUV;
	TRefCountPtr<FRDGPooledBuffer> PoolColor;
	TRefCountPtr<FRDGPooledBuffer> PoolIdentity;
	TRefCountPtr<FRDGPooledBuffer> PoolArgs;
	TRefCountPtr<FRDGPooledBuffer> PoolAlloc;
	TRefCountPtr<FRDGPooledBuffer> PoolState;

	// Active slots from the latest update; the proxy only emits draw calls for
	// these (inactive slots cannot contain surface).
	TArray<uint32> CachedActiveSlots;

	// ── AllocState readback (head / overflow / live counters) ──────────────
	FRHIGPUBufferReadback* StateReadback = nullptr;
	bool                   bStatePending = false;
	std::atomic<bool>      bStateArmed{ false };

	// ── Ray tracing: per-chunk BLAS with exact triangle counts ─────────────
	// Exact per-chunk pool offsets / vertex counts come from a readback of the
	// GPU alloc table. Each readback is tagged with the update serial it
	// observed; a slot's BLAS is only (re)built from table data at least as new
	// as the slot's last re-mesh, so the BLAS always matches the pool contents.
	uint32               UpdateSerial = 0;
	TArray<uint32>       SlotDirtySerial;    // serial of each slot's last re-mesh
	TSet<uint32>         BlasPending;        // slots whose BLAS needs (re)building
	TArray<FUintVector4> LastAllocTable;     // latest alloc-table readback
	uint32               LastAllocTableSerial = 0;

	FRHIGPUBufferReadback* TableReadback = nullptr;
	bool                   bTablePending = false;
	std::atomic<bool>      bTableArmed{ false };
	uint32                 TableSerialInFlight = 0;

	void HandleAllocTable(TArray<FUintVector4>&& Table, uint32 Serial);
	void DrainBlasPending();

	// ── Collision (GPU gather -> readback -> async cook) ───────────────────
	FRHIGPUBufferReadback* CollMetaReadback = nullptr;
	FRHIGPUBufferReadback* CollPosReadback = nullptr;
	bool                   bCollPending = false;
	std::atomic<bool>      bCollArmed{ false };
	bool                   bCollRegionDirty = true;
	bool                   bCollisionCookInProgress = false;
	FIntVector             LastCollCenter = FIntVector(MAX_int32, MAX_int32, MAX_int32);
	uint32                 LastCollSlotCount = 0;

	UPROPERTY(Transient)
	TObjectPtr<UBodySetup> CollisionBodySetup;

	TArray<FVector3f>   CollisionVertices;
	TArray<FTriIndices> CollisionTriIndices;

	void RebuildCollision();
	void PollReadbacks();

	FVector LastUpdatePlayerPos = FVector(MAX_dbl, MAX_dbl, MAX_dbl);
};