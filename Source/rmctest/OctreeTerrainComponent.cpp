#include "OctreeTerrainComponent.h"

#include "Async/Async.h"
#include "Engine/Engine.h"
#include "Engine/World.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "GlobalShader.h"
#include "LocalVertexFactory.h"
#include "Materials/Material.h"
#include "Materials/MaterialInterface.h"
#include "Materials/MaterialRenderProxy.h"
#include "PrimitiveSceneProxy.h"
#include "PrimitiveUniformShaderParameters.h"
#include "PrimitiveUniformShaderParametersBuilder.h"
#include "RHICommandList.h"
#include "RayTracingGeometry.h"
#include "RayTracingInstance.h"
#include "RenderGraphBuilder.h"
#include "RenderGraphUtils.h"
#include "RenderUtils.h"
#include "RenderingThread.h"
#include "SceneManagement.h"
#include "ShaderParameterStruct.h"

// ============================================================================
// Tunables / batch payload
// ============================================================================

namespace OctreeTerrain
{
	constexpr uint32 MaxPoolVerts   = 16u << 20;   // hard pool-growth ceiling (~1 GB of streams)
	constexpr uint32 CollMaxVerts   = 393216;      // collision gather vertex cap
	constexpr int32  MaxEdits       = 65536;
	constexpr int32  MaxLODs        = 10;
	constexpr uint32 FlagEmpty      = 1u;

	// One BLAS (re)build request for the scene proxy: the chunk's exact range
	// in the vertex pool, taken from an alloc-table readback.
	struct FBlasBuild
	{
		uint32 Slot = 0;
		uint32 Offset = 0;
		uint32 VertCount = 0;
	};

	// Everything the render-thread update graph needs, snapshotted on the game
	// thread so the GPU pipeline never depends on mutable component state.
	struct FUpdateBatch
	{
		bool bInit = false;
		bool bCompact = false;
		bool bCollision = false;
		bool bStateReadback = false;
		bool bTableReadback = false;
		bool bActiveChanged = false;

		uint32 Capacity = 0;
		uint32 MaxChunks = 0;
		uint32 GroupsPerAxis = 0;
		uint32 CellsPerAxis = 0;
		uint32 MaxVertsPerChunkClamp = 0;

		uint32    NumNoiseLayers = 0;
		FVector4f Noise[8];
		float     CaveFreq = 0.0f;
		float     CaveAmp = 0.0f;
		float     WorldUVScale = 0.0f;
		float     ZBandMin = 0.0f;
		float     ZBandMax = 0.0f;

		TArray<FVector4f>    FinerMin;      // per LOD carve volume
		TArray<FVector4f>    FinerMax;
		TArray<FVector4f>    DirtyA;        // xyz = chunk origin, w = cell size
		TArray<FUintVector4> DirtyB;        // x = slot, y = edit start, z = edit count, w = flags | lod<<8
		TArray<uint32>       SlotToDirty;   // compaction only
		TArray<FVector4f>    EditData;
		TArray<uint32>       EditIdx;
		TArray<uint32>       CollSlots;
		TArray<uint32>       ActiveSlots;
		TArray<uint32>       BlasReleases;  // slots that became empty this update
	};

	static int32 PosMod(int32 A, int32 M)
	{
		const int32 R = A % M;
		return R < 0 ? R + M : R;
	}
}

// ============================================================================
// Compute shaders
// ============================================================================

BEGIN_SHADER_PARAMETER_STRUCT(FOctreeSDFParams, )
	SHADER_PARAMETER(uint32, NumNoiseLayers)
	SHADER_PARAMETER_ARRAY(FVector4f, NoiseLayers, [8])
	SHADER_PARAMETER(float, CaveNoiseFrequency)
	SHADER_PARAMETER(float, CaveNoiseAmplitude)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, EditSpheres)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, EditIndexList)
END_SHADER_PARAMETER_STRUCT()

BEGIN_SHADER_PARAMETER_STRUCT(FOctreeChunkParams, )
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, DirtyInfoA)
	SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, DirtyInfoB)
	SHADER_PARAMETER(uint32, DirtyCount)
	SHADER_PARAMETER(uint32, DirtyBase)
	SHADER_PARAMETER(uint32, CellsPerAxis)
	SHADER_PARAMETER(uint32, GroupsPerAxis)
	SHADER_PARAMETER_ARRAY(FVector4f, FinerVolMin, [10])
	SHADER_PARAMETER_ARRAY(FVector4f, FinerVolMax, [10])
END_SHADER_PARAMETER_STRUCT()

class FOctreeInitPoolCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeInitPoolCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeInitPoolCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, IdentityIndices)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint4>, ChunkAlloc)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, DrawArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, AllocState)
		SHADER_PARAMETER(uint32, IdentityCount)
		SHADER_PARAMETER(uint32, InitStride)
		SHADER_PARAMETER(uint32, ClearControl)
		SHADER_PARAMETER(uint32, MaxChunks)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeInitPoolCS, "/Project/OctreeTerrainCS.usf", "InitPoolCS", SF_Compute);

class FOctreeCountCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeCountCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeCountCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FOctreeSDFParams, SDF)
		SHADER_PARAMETER_STRUCT_INCLUDE(FOctreeChunkParams, Chunk)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, Counts)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeCountCS, "/Project/OctreeTerrainCS.usf", "CountCS", SF_Compute);

class FOctreeAllocCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeAllocCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeAllocCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, DirtyInfoB)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CountsRO)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint4>, ChunkAlloc)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, DrawArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, AllocState)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, MeshSkip)
		SHADER_PARAMETER(uint32, DirtyCount)
		SHADER_PARAMETER(uint32, CapacityVerts)
		SHADER_PARAMETER(uint32, MaxVertsPerChunkClamp)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeAllocCS, "/Project/OctreeTerrainCS.usf", "AllocCS", SF_Compute);

class FOctreeCompactAllocCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeCompactAllocCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeCompactAllocCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, DirtyInfoB)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CountsRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SlotToDirty)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint4>, ChunkAlloc)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, DrawArgs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, AllocState)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, MeshSkip)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, OldOffsets)
		SHADER_PARAMETER(uint32, MaxChunks)
		SHADER_PARAMETER(uint32, CapacityVerts)
		SHADER_PARAMETER(uint32, MaxVertsPerChunkClamp)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeCompactAllocCS, "/Project/OctreeTerrainCS.usf", "CompactAllocCS", SF_Compute);

class FOctreeCopyChunksCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeCopyChunksCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeCopyChunksCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SlotToDirty)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, OldOffsetsRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, ChunkAllocRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SrcPositions)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SrcTangents)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float2>, SrcUVs)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, SrcColors)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, OutPositions)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, OutTangents)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float2>, OutUVs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, OutColors)
		SHADER_PARAMETER(uint32, MaxChunks)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeCopyChunksCS, "/Project/OctreeTerrainCS.usf", "CopyChunksCS", SF_Compute);

class FOctreeMeshCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeMeshCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeMeshCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_STRUCT_INCLUDE(FOctreeSDFParams, SDF)
		SHADER_PARAMETER_STRUCT_INCLUDE(FOctreeChunkParams, Chunk)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, ChunkAllocRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, MeshSkipRO)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, Cursors)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, OutPositions)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, OutTangents)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float2>, OutUVs)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, OutColors)
		SHADER_PARAMETER(float, WorldUVScale)
		SHADER_PARAMETER(float, ZBandMin)
		SHADER_PARAMETER(float, ZBandMax)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeMeshCS, "/Project/OctreeTerrainCS.usf", "MeshCS", SF_Compute);

class FOctreeCollisionPrefixCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeCollisionPrefixCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeCollisionPrefixCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CollSlots)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, ChunkAllocRO)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<uint>, CollMeta)
		SHADER_PARAMETER(uint32, CollCount)
		SHADER_PARAMETER(uint32, CollMaxVerts)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeCollisionPrefixCS, "/Project/OctreeTerrainCS.usf", "CollisionPrefixCS", SF_Compute);

class FOctreeCollisionCopyCS : public FGlobalShader
{
public:
	DECLARE_GLOBAL_SHADER(FOctreeCollisionCopyCS);
	SHADER_USE_PARAMETER_STRUCT(FOctreeCollisionCopyCS, FGlobalShader);

	BEGIN_SHADER_PARAMETER_STRUCT(FParameters, )
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CollSlots)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint4>, ChunkAllocRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<uint>, CollMetaRO)
		SHADER_PARAMETER_RDG_BUFFER_SRV(Buffer<float4>, SrcPositions)
		SHADER_PARAMETER_RDG_BUFFER_UAV(RWBuffer<float4>, CollOutPos)
		SHADER_PARAMETER(uint32, CollCount)
	END_SHADER_PARAMETER_STRUCT()
};
IMPLEMENT_GLOBAL_SHADER(FOctreeCollisionCopyCS, "/Project/OctreeTerrainCS.usf", "CollisionCopyCS", SF_Compute);

// ============================================================================
// Scene proxy
// ============================================================================

class FOctreeTerrainSceneProxy final : public FPrimitiveSceneProxy
{
public:
	SIZE_T GetTypeHash() const override
	{
		static SIZE_T UniquePointer;
		return reinterpret_cast<SIZE_T>(&UniquePointer);
	}

	explicit FOctreeTerrainSceneProxy(UOctreeTerrainComponent* InComponent)
		: FPrimitiveSceneProxy(InComponent)
	{
		Material = InComponent->TerrainMaterial
			? static_cast<UMaterialInterface*>(InComponent->TerrainMaterial)
			: UMaterial::GetDefaultMaterial(MD_Surface);
		MaterialRelevance = Material->GetRelevance_Concurrent(GetScene().GetShaderPlatform());
		// GPU-Scene must stay OFF: this proxy's ray tracing instances are
		// dynamic (per-chunk BLAS via GetDynamicRayTracingInstances) and have
		// no GPU-Scene instance entries - with GPU-Scene enabled, path-traced
		// hit shading resolves per-primitive data through garbage instance ids
		// (terrain shades black in PT). The raster path uses per-frame
		// primitive uniform buffers either way.
		bSupportsGPUScene = false;
	}

	virtual ~FOctreeTerrainSceneProxy() override
	{
		if (VertexFactory)
		{
			VertexFactory->ReleaseResource();
			delete VertexFactory;
			VertexFactory = nullptr;
		}
		PositionBufWrap.ReleaseResource();
		TangentBufWrap.ReleaseResource();
		UVBufWrap.ReleaseResource();
		ColorBufWrap.ReleaseResource();
		IdentityIndexBufWrap.ReleaseResource();

#if RHI_RAYTRACING
		for (TUniquePtr<FRayTracingGeometry>& Geo : ChunkBLAS)
		{
			if (Geo.IsValid())
			{
				if (Geo->IsInitialized())
				{
					Geo->ReleaseResource();
				}
				Geo.Reset();
			}
		}
#endif
	}

	// (Re)points the vertex factory at the current pool buffers. Called on init
	// and after every pool compaction / growth.
	void UpdateBuffers_RenderThread(
		TRefCountPtr<FRDGPooledBuffer> InPos,
		TRefCountPtr<FRDGPooledBuffer> InTan,
		TRefCountPtr<FRDGPooledBuffer> InUV,
		TRefCountPtr<FRDGPooledBuffer> InColor,
		TRefCountPtr<FRDGPooledBuffer> InIdentity,
		TRefCountPtr<FRDGPooledBuffer> InArgs,
		uint32 InCapacity,
		uint32 InMaxChunks)
	{
		check(IsInRenderingThread());
		if (!InPos.IsValid() || !InIdentity.IsValid() || !InArgs.IsValid())
		{
			return;
		}

		FRHICommandListImmediate& RHICmdList = FRHICommandListExecutor::GetImmediateCommandList();

		CurrentPos      = InPos;
		CurrentTan      = InTan;
		CurrentUV       = InUV;
		CurrentColor    = InColor;
		CurrentIdentity = InIdentity;
		CurrentArgs     = InArgs;
		Capacity        = InCapacity;
		MaxChunks       = InMaxChunks;

		if (!PositionBufWrap.IsInitialized())      PositionBufWrap.InitResource(RHICmdList);
		if (!TangentBufWrap.IsInitialized())       TangentBufWrap.InitResource(RHICmdList);
		if (!UVBufWrap.IsInitialized())            UVBufWrap.InitResource(RHICmdList);
		if (!ColorBufWrap.IsInitialized())         ColorBufWrap.InitResource(RHICmdList);
		if (!IdentityIndexBufWrap.IsInitialized()) IdentityIndexBufWrap.InitResource(RHICmdList);

		if (!VertexFactory)
		{
			VertexFactory = new FLocalVertexFactory(GetScene().GetFeatureLevel(), "FOctreeTerrainSceneProxy");
		}

		FLocalVertexFactory::FDataType Data;
		Data.NumTexCoords = 1;
		Data.LightMapCoordinateIndex = 0;
		Data.PositionComponent = FVertexStreamComponent(&PositionBufWrap, 0, 16, VET_Float3);
		Data.TangentBasisComponents[0] = FVertexStreamComponent(&TangentBufWrap, 0, 32, VET_Float4);
		Data.TangentBasisComponents[1] = FVertexStreamComponent(&TangentBufWrap, 16, 32, VET_Float4);
		Data.TextureCoordinates.Add(FVertexStreamComponent(&UVBufWrap, 0, 8, VET_Float2));
		Data.ColorComponent = FVertexStreamComponent(&ColorBufWrap, 0, 4, VET_Color);

		FRHIViewDesc::FBufferSRV::FInitializer PosSRVInit = FRHIViewDesc::CreateBufferSRV();
		PosSRVInit.SetType(FRHIViewDesc::EBufferType::Typed);
		PosSRVInit.SetFormat(PF_A32B32G32R32F);
		PosSRVInit.SetNumElements(Capacity);
		Data.PositionComponentSRV = RHICmdList.CreateShaderResourceView(InPos->GetRHI(), PosSRVInit);

		FRHIViewDesc::FBufferSRV::FInitializer TanSRVInit = FRHIViewDesc::CreateBufferSRV();
		TanSRVInit.SetType(FRHIViewDesc::EBufferType::Typed);
		TanSRVInit.SetFormat(PF_A32B32G32R32F);
		TanSRVInit.SetNumElements(Capacity * 2);
		Data.TangentsSRV = RHICmdList.CreateShaderResourceView(InTan->GetRHI(), TanSRVInit);

		FRHIViewDesc::FBufferSRV::FInitializer UVSRVInit = FRHIViewDesc::CreateBufferSRV();
		UVSRVInit.SetType(FRHIViewDesc::EBufferType::Typed);
		UVSRVInit.SetFormat(PF_G32R32F);
		UVSRVInit.SetNumElements(Capacity);
		Data.TextureCoordinatesSRV = RHICmdList.CreateShaderResourceView(InUV->GetRHI(), UVSRVInit);

		FRHIViewDesc::FBufferSRV::FInitializer ColSRVInit = FRHIViewDesc::CreateBufferSRV();
		ColSRVInit.SetType(FRHIViewDesc::EBufferType::Typed);
		ColSRVInit.SetFormat(PF_R8G8B8A8);
		ColSRVInit.SetNumElements(Capacity);
		Data.ColorComponentsSRV = RHICmdList.CreateShaderResourceView(InColor->GetRHI(), ColSRVInit);

		VertexFactory->SetData(RHICmdList, Data);

		// Pointer swaps + factory init happen on the RHI timeline so in-flight
		// draws never see half-updated stream bindings (VoxelMeshComponent's
		// proven pattern).
		bBuffersReady = false;
		RHICmdList.EnqueueLambda(
			[this,
			 PosRHI = InPos->GetRHI(),
			 TanRHI = InTan->GetRHI(),
			 UVRHI = InUV->GetRHI(),
			 ColRHI = InColor->GetRHI(),
			 IdentRHI = InIdentity->GetRHI()](FRHICommandListImmediate& CmdList)
			{
				PositionBufWrap.SetBuffer(PosRHI);
				TangentBufWrap.SetBuffer(TanRHI);
				UVBufWrap.SetBuffer(UVRHI);
				ColorBufWrap.SetBuffer(ColRHI);
				IdentityIndexBufWrap.SetBuffer(IdentRHI);

				if (!VertexFactory->IsInitialized())
				{
					VertexFactory->InitResource(CmdList);
				}
				else
				{
					VertexFactory->UpdateRHI(CmdList);
				}

				bBuffersReady = true;
			});
	}

	void SetActiveSlots_RenderThread(TArray<uint32> InActiveSlots)
	{
		check(IsInRenderingThread());
		ActiveSlots = MoveTemp(InActiveSlots);
	}

#if RHI_RAYTRACING
	// (Re)builds the BLAS of the given chunks against the current vertex pool
	// and releases the BLAS of chunks that became empty. Each BLAS is built
	// indexed through the whole-pool identity IB at the chunk's byte offset, so
	// it covers exactly the chunk's triangles and the index values stay global
	// vertex ids (the same trick the raster path uses for attribute fetch).
	void UpdateChunkBLAS_RenderThread(const TArray<OctreeTerrain::FBlasBuild>& Builds, const TArray<uint32>& Releases)
	{
		check(IsInRenderingThread());

		if (ChunkBLAS.Num() != int32(MaxChunks))
		{
			ChunkBLAS.SetNum(MaxChunks);
			BlasOffset.SetNumZeroed(MaxChunks);
			BlasVertCount.SetNumZeroed(MaxChunks);
		}

		auto ReleaseSlot = [this](uint32 Slot)
		{
			if (Slot < uint32(ChunkBLAS.Num()) && ChunkBLAS[Slot].IsValid())
			{
				if (ChunkBLAS[Slot]->IsInitialized())
				{
					ChunkBLAS[Slot]->ReleaseResource();
				}
				ChunkBLAS[Slot].Reset();
				BlasVertCount[Slot] = 0;
			}
		};

		for (uint32 Slot : Releases)
		{
			ReleaseSlot(Slot);
		}

		if (Builds.Num() == 0)
		{
			return;
		}
		// IsRayTracingAllowed (static), NOT IsRayTracingEnabled: with on-demand
		// dynamic ray tracing the enable toggle flips at runtime, and builds
		// dropped while it is off would leave chunks without BLAS forever
		// (their slots were already removed from the pending set).
		if (!IsRayTracingAllowed() || !CurrentPos.IsValid() || !CurrentIdentity.IsValid())
		{
			return;
		}

		FRHIBuffer* PosRHI = CurrentPos->GetRHI();
		FRHIBuffer* IdentRHI = CurrentIdentity->GetRHI();
		if (!PosRHI || !IdentRHI)
		{
			return;
		}

		FRHICommandListImmediate& RHICmdList = FRHICommandListExecutor::GetImmediateCommandList();

		for (const OctreeTerrain::FBlasBuild& Bld : Builds)
		{
			if (Bld.Slot >= uint32(ChunkBLAS.Num()))
			{
				continue;
			}
			ReleaseSlot(Bld.Slot);

			if (Bld.VertCount < 3 || Bld.Offset + Bld.VertCount > Capacity)
			{
				continue;
			}
			// D3D12 requires BLAS index-buffer offsets to be 16-byte aligned;
			// the GPU allocator advances in 4-vertex steps to guarantee this.
			if ((Bld.Offset & 3u) != 0)
			{
				continue;
			}
			const uint32 NumTris = Bld.VertCount / 3;

			TUniquePtr<FRayTracingGeometry> Geo = MakeUnique<FRayTracingGeometry>();

			FRayTracingGeometryInitializer Initializer;
			Initializer.IndexBuffer = IdentRHI;
			Initializer.IndexBufferOffset = Bld.Offset * sizeof(uint32);
			Initializer.GeometryType = RTGT_Triangles;
			Initializer.bFastBuild = true;     // chunks re-mesh at edit rates
			Initializer.bAllowUpdate = false;  // exact-fit ranges move, so always full rebuilds
			Initializer.DebugName = FName("OctreeChunkBLAS");
			Initializer.TotalPrimitiveCount = NumTris;

			FRayTracingGeometrySegment Segment;
			Segment.VertexBuffer = PosRHI;
			Segment.VertexBufferOffset = 0;
			Segment.VertexBufferStride = 16;   // float4 positions, xyz read
			Segment.VertexBufferElementType = VET_Float3;
			Segment.MaxVertices = Bld.Offset + Bld.VertCount;
			Segment.NumPrimitives = NumTris;
			Segment.FirstPrimitive = 0;
			Initializer.Segments.Add(Segment);

			Geo->SetInitializer(Initializer);
			Geo->InitResource(RHICmdList);

			ChunkBLAS[Bld.Slot] = MoveTemp(Geo);
			BlasOffset[Bld.Slot] = Bld.Offset;
			BlasVertCount[Bld.Slot] = Bld.VertCount;
		}
	}

	// The base implementation gates on IsDrawnInGame() and static-mesh-style
	// caching; returning Dynamic routes this proxy through
	// GetDynamicRayTracingInstances every frame (VoxelMeshComponent's proven
	// bypass).
	virtual ERayTracingPrimitiveFlags GetCachedRayTracingInstance(FRayTracingInstance& OutRayTracingInstance) override
	{
		// Static check only: this result is cached, and with on-demand dynamic
		// ray tracing the runtime toggle may be off at caching time.
		if (!IsRayTracingAllowed())
		{
			return ERayTracingPrimitiveFlags::Exclude;
		}
		return ERayTracingPrimitiveFlags::Dynamic;
	}

	virtual void GetDynamicRayTracingInstances(FRayTracingInstanceCollector& Collector) override
	{
		if (!bBuffersReady || !VertexFactory || !VertexFactory->IsInitialized())
		{
			return;
		}

		FMaterialRenderProxy* MaterialProxy = Material->GetRenderProxy();
		if (!MaterialProxy)
		{
			return;
		}

		// Real primitive data for hit shading (identity UB lacks valid scene
		// metadata and path-traced shading goes black with it). All chunks
		// share the proxy and its transform, so one buffer serves every batch.
		FDynamicPrimitiveUniformBuffer& DynamicPrimitiveUniformBuffer = Collector.AllocateOneFrameResource<FDynamicPrimitiveUniformBuffer>();
		{
			FPrimitiveUniformShaderParametersBuilder Builder;
			BuildUniformShaderParameters(Builder);
			DynamicPrimitiveUniformBuffer.Set(Collector.GetRHICommandList(), Builder);
		}

		// One instance per chunk that has a built BLAS. FirstIndex uses the
		// offset/count the BLAS was committed with (not the live draw args), so
		// geometry and attribute fetch stay consistent even while a re-meshed
		// chunk waits for its BLAS rebuild.
		for (int32 Slot = 0; Slot < ChunkBLAS.Num(); Slot++)
		{
			FRayTracingGeometry* Geo = ChunkBLAS[Slot].Get();
			if (!Geo || !Geo->IsInitialized() || BlasVertCount[Slot] < 3)
			{
				continue;
			}

			FRayTracingInstance RayTracingInstance;
			RayTracingInstance.Geometry = Geo;
			RayTracingInstance.InstanceTransforms.Add(GetLocalToWorld());

			FMeshBatch MeshBatch;
			MeshBatch.VertexFactory = VertexFactory;
			MeshBatch.SegmentIndex = 0;
			MeshBatch.MaterialRenderProxy = MaterialProxy;
			MeshBatch.Type = PT_TriangleList;
			MeshBatch.DepthPriorityGroup = SDPG_World;
			MeshBatch.bCanApplyViewModeOverrides = false;
			MeshBatch.bDisableBackfaceCulling = true;
			MeshBatch.CastRayTracedShadow = true;
			MeshBatch.ReverseCulling = IsLocalToWorldDeterminantNegative();

			FMeshBatchElement& BatchElement = MeshBatch.Elements[0];
			BatchElement.IndexBuffer = &IdentityIndexBufWrap;
			BatchElement.FirstIndex = BlasOffset[Slot];
			BatchElement.NumPrimitives = BlasVertCount[Slot] / 3;
			BatchElement.BaseVertexIndex = 0;
			BatchElement.MinVertexIndex = 0;
			BatchElement.MaxVertexIndex = Capacity > 0 ? Capacity - 1 : 0;
			BatchElement.PrimitiveUniformBufferResource = &DynamicPrimitiveUniformBuffer.UniformBuffer;

			RayTracingInstance.Materials.Add(MeshBatch);
			Collector.AddRayTracingInstance(0, MoveTemp(RayTracingInstance));
		}
	}

	virtual bool IsRayTracingRelevant() const override { return true; }
	virtual bool IsRayTracingStaticRelevant() const override { return false; }
	virtual bool HasRayTracingRepresentation() const override { return true; }
#endif // RHI_RAYTRACING

	virtual void GetDynamicMeshElements(const TArray<const FSceneView*>& Views,
		const FSceneViewFamily& ViewFamily, uint32 VisibilityMap,
		FMeshElementCollector& Collector) const override
	{
		if (!bBuffersReady || !VertexFactory || !VertexFactory->IsInitialized() ||
			!CurrentArgs.IsValid() || ActiveSlots.Num() == 0)
		{
			return;
		}

		FMaterialRenderProxy* MaterialProxy = Material->GetRenderProxy();
		if (!MaterialProxy)
		{
			return;
		}

		for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ViewIndex++)
		{
			if (!(VisibilityMap & (1 << ViewIndex)))
			{
				continue;
			}

			FMeshBatch& Mesh = Collector.AllocateMesh();
			Mesh.bWireframe = false;
			Mesh.VertexFactory = VertexFactory;
			Mesh.MaterialRenderProxy = MaterialProxy;
			Mesh.ReverseCulling = IsLocalToWorldDeterminantNegative();
			Mesh.Type = PT_TriangleList;
			Mesh.DepthPriorityGroup = GetDepthPriorityGroup(Views[ViewIndex]);
			Mesh.bCanApplyViewModeOverrides = true;
			Mesh.bDisableBackfaceCulling = true;

			// One indirect draw per active chunk. Inactive / empty chunks have
			// IndexCount 0 in their args and cost nothing on the GPU.
			for (int32 i = 0; i < ActiveSlots.Num(); i++)
			{
				FMeshBatchElement& BatchElement = (i == 0) ? Mesh.Elements[0] : Mesh.Elements.AddDefaulted_GetRef();
				BatchElement.IndexBuffer = &IdentityIndexBufWrap;
				BatchElement.IndirectArgsBuffer = CurrentArgs->GetRHI();
				BatchElement.IndirectArgsOffset = ActiveSlots[i] * 8 * sizeof(uint32);
				BatchElement.FirstIndex = 0;
				BatchElement.NumPrimitives = 0;   // must be 0 when using IndirectArgsBuffer
				BatchElement.BaseVertexIndex = 0; // args carry StartIndexLocation; identity IB maps index -> global vertex id
				BatchElement.MinVertexIndex = 0;
				BatchElement.MaxVertexIndex = Capacity > 0 ? Capacity - 1 : 0;
			}

			Collector.AddMesh(ViewIndex, Mesh);
		}
	}

	virtual FPrimitiveViewRelevance GetViewRelevance(const FSceneView* View) const override
	{
		FPrimitiveViewRelevance Result;
		Result.bDrawRelevance = IsShown(View);
		Result.bShadowRelevance = IsShadowCast(View);
		Result.bDynamicRelevance = true;
		Result.bStaticRelevance = false;
		Result.bRenderInMainPass = ShouldRenderInMainPass();
		Result.bUsesLightingChannels = GetLightingChannelMask() != GetDefaultLightingChannelMask();
		Result.bRenderCustomDepth = ShouldRenderCustomDepth();
		MaterialRelevance.SetPrimitiveViewRelevance(Result);
		return Result;
	}

	virtual uint32 GetMemoryFootprint() const override { return sizeof(*this) + GetAllocatedSize(); }

private:
	friend class UOctreeTerrainComponent;

	class FPooledVertexBuffer : public FVertexBuffer
	{
	public:
		virtual void InitRHI(FRHICommandListBase&) override {}
		void SetBuffer(FRHIBuffer* InBuffer) { VertexBufferRHI = InBuffer; }
	};
	class FPooledIndexBuffer : public FIndexBuffer
	{
	public:
		virtual void InitRHI(FRHICommandListBase&) override {}
		void SetBuffer(FRHIBuffer* InBuffer) { IndexBufferRHI = InBuffer; }
	};

	bool bBuffersReady = false;

	TRefCountPtr<FRDGPooledBuffer> CurrentPos;
	TRefCountPtr<FRDGPooledBuffer> CurrentTan;
	TRefCountPtr<FRDGPooledBuffer> CurrentUV;
	TRefCountPtr<FRDGPooledBuffer> CurrentColor;
	TRefCountPtr<FRDGPooledBuffer> CurrentIdentity;
	TRefCountPtr<FRDGPooledBuffer> CurrentArgs;

	uint32 Capacity = 0;
	uint32 MaxChunks = 0;
	TArray<uint32> ActiveSlots;

	FPooledVertexBuffer PositionBufWrap;
	FPooledVertexBuffer TangentBufWrap;
	FPooledVertexBuffer UVBufWrap;
	FPooledVertexBuffer ColorBufWrap;
	FPooledIndexBuffer  IdentityIndexBufWrap;

#if RHI_RAYTRACING
	// Per-chunk BLAS plus the pool range each was committed with. A stale BLAS
	// keeps its source pool alive through the FBufferRHIRef in its initializer,
	// so chunks remain traceable across pool swaps until their rebuild lands.
	TArray<TUniquePtr<FRayTracingGeometry>> ChunkBLAS;
	TArray<uint32> BlasOffset;
	TArray<uint32> BlasVertCount;
#endif

	FLocalVertexFactory* VertexFactory = nullptr;
	UMaterialInterface*  Material = nullptr;
	FMaterialRelevance   MaterialRelevance;
};

// ============================================================================
// Component: setup
// ============================================================================

UOctreeTerrainComponent::UOctreeTerrainComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;

	CastShadow = true;
	bCastDynamicShadow = true;
	bCastStaticShadow = false;
	bUseAsOccluder = false;
	bVisibleInRayTracing = true;
	SetCullDistance(0.0f);

	SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	SetCollisionObjectType(ECC_WorldStatic);
	SetCollisionResponseToAllChannels(ECR_Block);
}

double UOctreeTerrainComponent::ChunkSizeForLOD(int32 LOD) const
{
	return double(BaseChunkSize) * double(1 << LOD);
}

void UOctreeTerrainComponent::SanitizeConfig()
{
	GridResolution = FMath::Clamp(GridResolution, 8, 64);
	CellsN = (GridResolution / 8) * 8;            // multiple of 8 (also keeps N even for carve alignment)
	NumLODLevels = FMath::Clamp(NumLODLevels, 1, OctreeTerrain::MaxLODs);
	ChunksPerSide = FMath::Clamp(ChunksPerSide, 3, 21);
	ChunksVertical = FMath::Clamp(ChunksVertical, 1, 13);

	ChunksPerLOD = ChunksPerSide * ChunksPerSide * ChunksVertical;
	MaxChunks = ChunksPerLOD * NumLODLevels;

	const uint64 WorstVerts = uint64(CellsN) * CellsN * CellsN * 15;
	MaxVertsPerChunkClamp = uint32(FMath::Min<uint64>(WorstVerts, 4 * 1024 * 1024));

	CurrentCapacity = FMath::Clamp<uint32>(uint32(InitialPoolVertexCapacity), 65536u, OctreeTerrain::MaxPoolVerts);
}

void UOctreeTerrainComponent::EnsureNoiseDefaults()
{
	if (NoiseLayers.Num() == 0)
	{
		auto AddLayer = [this](float Frequency, float Amplitude)
		{
			FOctreeNoiseLayer& Layer = NoiseLayers.AddDefaulted_GetRef();
			Layer.Frequency = Frequency;
			Layer.Amplitude = Amplitude;
		};
		// Keep frequency x amplitude (= slope) moderate: steeper than ~1 turns
		// into cliff/needle fields that only the finest LOD can sample.
		AddLayer(0.00002f, 25000.0f);
		AddLayer(0.0001f, 6000.0f);
		AddLayer(0.0008f, 600.0f);
		AddLayer(0.005f, 60.0f);
	}
	while (NoiseLayers.Num() > 8)
	{
		NoiseLayers.Pop();
	}

	double AmpSum = double(CaveNoiseAmplitude);
	for (const FOctreeNoiseLayer& L : NoiseLayers)
	{
		AmpSum += FMath::Abs(L.Amplitude);
	}
	const double Margin = 4.0 * ChunkSizeForLOD(NumLODLevels - 1) / double(CellsN);
	ZBandMax = AmpSum + Margin;
	ZBandMin = -ZBandMax;
}

void UOctreeTerrainComponent::BeginPlay()
{
	Super::BeginPlay();

	SanitizeConfig();
	EnsureNoiseDefaults();

	SlotCoord.Init(FIntVector(MAX_int32, MAX_int32, MAX_int32), MaxChunks);
	SlotActive.Init(0, MaxChunks);
	RingOrigin.Init(FIntVector::ZeroValue, NumLODLevels);
	bForceAllDirty = true;
	bCollRegionDirty = true;

	UpdateSerial = 0;
	SlotDirtySerial.Init(0, MaxChunks);
	BlasPending.Empty();
	LastAllocTable.Empty();
	LastAllocTableSerial = 0;

	UE_LOG(LogTemp, Log,
		TEXT("OctreeTerrain: %d slots (%d LODs x %d), %d cells/axis, pool %u verts (%.1f MB streams)"),
		MaxChunks, NumLODLevels, ChunksPerLOD, CellsN, CurrentCapacity,
		double(CurrentCapacity) * 60.0 / (1024.0 * 1024.0));

	const FVector PlayerPos = GetPlayerPosition();
	LastUpdatePlayerPos = PlayerPos;
	BuildAndEnqueueUpdate(PlayerPos);
}

void UOctreeTerrainComponent::EndPlay(const EEndPlayReason::Type EndPlayReason)
{
	FlushRenderingCommands();

	delete StateReadback;    StateReadback = nullptr;
	delete CollMetaReadback; CollMetaReadback = nullptr;
	delete CollPosReadback;  CollPosReadback = nullptr;
	delete TableReadback;    TableReadback = nullptr;
	bStatePending = false;
	bCollPending = false;
	bTablePending = false;

	Super::EndPlay(EndPlayReason);
}

FVector UOctreeTerrainComponent::GetPlayerPosition() const
{
	if (UWorld* World = GetWorld())
	{
		if (APlayerController* PC = World->GetFirstPlayerController())
		{
			if (APawn* Pawn = PC->GetPawn())
			{
				return Pawn->GetActorLocation();
			}
		}
	}
	return FVector::ZeroVector;
}

FBoxSphereBounds UOctreeTerrainComponent::CalcBounds(const FTransform& LocalToWorld) const
{
	// Geometry follows the player far from the component origin; use generous
	// static bounds like VoxelMeshComponent so chunks never get culled away.
	return FBoxSphereBounds(FVector::ZeroVector, FVector(5000000.0f), 5000000.0f);
}

void UOctreeTerrainComponent::GetUsedMaterials(TArray<UMaterialInterface*>& OutMaterials, bool bGetDebugMaterials) const
{
	Super::GetUsedMaterials(OutMaterials, bGetDebugMaterials);
	if (TerrainMaterial)
	{
		OutMaterials.Add(TerrainMaterial);
	}
}

FPrimitiveSceneProxy* UOctreeTerrainComponent::CreateSceneProxy()
{
	if (!TerrainMaterial)
	{
		// The MD_Surface default (WorldGridMaterial) is a special engine
		// material whose ray tracing hit shading returns garbage attributes
		// (terrain shades black in the path tracer). Fall back to a regular
		// engine material so RT/PT work without a user-assigned material.
		TerrainMaterial = LoadObject<UMaterialInterface>(nullptr,
			TEXT("/Engine/BasicShapes/BasicShapeMaterial.BasicShapeMaterial"));
	}

	FOctreeTerrainSceneProxy* Proxy = new FOctreeTerrainSceneProxy(this);

	// If the GPU buffers already exist (proxy recreation after a render-state
	// change), hand them straight to the new proxy.
	if (PoolPos.IsValid() && PoolArgs.IsValid() && PoolIdentity.IsValid())
	{
		ENQUEUE_RENDER_COMMAND(OctreeRestoreBuffers)(
			[Proxy,
			 Pos = PoolPos, Tan = PoolTan, UV = PoolUV, Col = PoolColor,
			 Ident = PoolIdentity, Args = PoolArgs,
			 Capacity = CurrentCapacity, NumChunks = uint32(MaxChunks),
			 Slots = CachedActiveSlots](FRHICommandListImmediate& RHICmdList) mutable
			{
				Proxy->UpdateBuffers_RenderThread(Pos, Tan, UV, Col, Ident, Args, Capacity, NumChunks);
				Proxy->SetActiveSlots_RenderThread(MoveTemp(Slots));
			});
	}

#if RHI_RAYTRACING
	// The fresh proxy has no BLAS; queue every active chunk for a rebuild
	// (their alloc-table data is still current, so the next drain handles it).
	if (IsRayTracingAllowed())
	{
		for (uint32 Slot : CachedActiveSlots)
		{
			BlasPending.Add(Slot);
		}
	}
#endif

	return Proxy;
}

// ============================================================================
// Component: edit API
// ============================================================================

void UOctreeTerrainComponent::ApplySphereEdit(FVector Center, float Radius, bool bSubtract)
{
	if (Radius <= 0.0f)
	{
		return;
	}
	if (Edits.Num() >= OctreeTerrain::MaxEdits)
	{
		UE_LOG(LogTemp, Warning, TEXT("OctreeTerrain: edit limit (%d) reached, ignoring edit"), OctreeTerrain::MaxEdits);
		return;
	}

	Edits.Add(FVector4f(float(Center.X), float(Center.Y), float(Center.Z), bSubtract ? -Radius : Radius));
	NumPendingEdits++;
}

void UOctreeTerrainComponent::ClearAllEdits()
{
	Edits.Empty();
	NumPendingEdits = 0;
	bForceAllDirty = true;
}

void UOctreeTerrainComponent::RebuildTerrain()
{
	bForceAllDirty = true;
}

// ============================================================================
// Component: tick / dirty tracking
// ============================================================================

void UOctreeTerrainComponent::TickComponent(float DeltaTime, ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	PollReadbacks();

	if (!bGPUInitialized)
	{
		return;
	}

	// A collision refresh may still be owed from a tick where the cook or a
	// readback was busy.
	if (bCollRegionDirty && !bCollPending && !bCollisionCookInProgress)
	{
		bUpdateRequested = true;
	}

#if RHI_RAYTRACING
	if (IsRayTracingAllowed())
	{
		// Build BLAS for slots whose table data is already fresh enough
		// (budget-limited leftovers from a previous drain), then request a new
		// alloc-table snapshot if stale slots remain.
		DrainBlasPending();
		if (BlasPending.Num() > 0 && !bTablePending)
		{
			bUpdateRequested = true;
		}
	}
#endif

	const FVector PlayerPos = GetPlayerPosition();
	const bool bMoved = FVector::Dist(PlayerPos, LastUpdatePlayerPos) > UpdateThreshold;

	if (bMoved || NumPendingEdits > 0 || bForceAllDirty || bUpdateRequested)
	{
		bUpdateRequested = false;
		LastUpdatePlayerPos = PlayerPos;
		BuildAndEnqueueUpdate(PlayerPos);
	}
}

FBox UOctreeTerrainComponent::ChunkBox(const FIntVector& Coord, int32 LOD) const
{
	const double S = ChunkSizeForLOD(LOD);
	const FVector Min(Coord.X * S, Coord.Y * S, Coord.Z * S);
	return FBox(Min, Min + FVector(S));
}

FBox UOctreeTerrainComponent::FinerRingBox(const FIntVector& Origin, int32 FinerLOD) const
{
	const double S = ChunkSizeForLOD(FinerLOD);
	const FVector Min(Origin.X * S, Origin.Y * S, Origin.Z * S);
	return FBox(Min, Min + FVector(ChunksPerSide * S, ChunksPerSide * S, ChunksVertical * S));
}

bool UOctreeTerrainComponent::ChunkMayHaveSurface(const FIntVector& Coord, int32 LOD) const
{
	const double S = ChunkSizeForLOD(LOD);
	const double Z0 = Coord.Z * S;
	if (Z0 + S >= ZBandMin && Z0 <= ZBandMax)
	{
		return true;
	}

	// Outside the noise height band - only an edit can put surface here.
	const FBox Box = ChunkBox(Coord, LOD).ExpandBy(4.0 * S / double(CellsN));
	for (const FVector4f& E : Edits)
	{
		const double R = FMath::Abs(double(E.W));
		const FVector C(E.X, E.Y, E.Z);
		if (Box.Intersect(FBox(C - FVector(R), C + FVector(R))))
		{
			return true;
		}
	}
	return false;
}

void UOctreeTerrainComponent::HandleAllocState(uint32 Head, uint32 Overflow, uint32 TotalLive, uint32 Serial)
{
	// A snapshot taken before the latest pool init/compaction describes the old
	// pool; acting on its head/overflow would double-grow or re-compact.
	if (Serial < LastPoolChangeSerial)
	{
		return;
	}

	const uint64 Cap = CurrentCapacity;

	if (Overflow)
	{
		// Some chunks were skipped (or dropped during a compaction). Grow the
		// pool and rebuild everything - rare, self-healing path.
		bWantCompact = true;
		PendingNewCapacity = uint32(FMath::Min<uint64>(OctreeTerrain::MaxPoolVerts,
			FMath::Max<uint64>(Cap * 2, uint64(TotalLive) * 3 / 2)));
		bForceAllDirty = true;
		bUpdateRequested = true;
		UE_LOG(LogTemp, Warning, TEXT("OctreeTerrain: vertex pool overflow (live=%u cap=%llu) - growing to %u"),
			TotalLive, Cap, PendingNewCapacity);
	}
	else if (TotalLive > Cap * 7 / 10 && Cap < OctreeTerrain::MaxPoolVerts)
	{
		bWantCompact = true;
		PendingNewCapacity = uint32(FMath::Min<uint64>(OctreeTerrain::MaxPoolVerts, Cap * 2));
		bUpdateRequested = true;
		UE_LOG(LogTemp, Log, TEXT("OctreeTerrain: pool growth (live=%u cap=%llu -> %u)"),
			TotalLive, Cap, PendingNewCapacity);
	}
	else if (Head > Cap * 85 / 100)
	{
		// Bump-allocator garbage from re-meshed chunks is piling up - compact in
		// place (same capacity).
		bWantCompact = true;
		PendingNewCapacity = 0;
		bUpdateRequested = true;
	}
}

// ============================================================================
// Component: update graph
// ============================================================================

void UOctreeTerrainComponent::BuildAndEnqueueUpdate(const FVector& PlayerPos)
{
	using namespace OctreeTerrain;

	const int32 CPS = ChunksPerSide;
	const int32 CV = ChunksVertical;

	FUpdateBatch B;
	B.bInit = !bGPUInitialized;
	B.bCompact = bWantCompact && !B.bInit;
	const bool bForceAll = bForceAllDirty || B.bInit;

	// Every update gets a serial; readbacks are tagged with the serial current
	// when they were requested so consumers can detect pre-pool-swap snapshots.
	++UpdateSerial;
	if (B.bInit || B.bCompact)
	{
		LastPoolChangeSerial = UpdateSerial;
	}

	if (B.bCompact && PendingNewCapacity > CurrentCapacity)
	{
		CurrentCapacity = FMath::Min(PendingNewCapacity, MaxPoolVerts);
	}
	bWantCompact = false;
	PendingNewCapacity = 0;
	bForceAllDirty = false;

	// ── 1. Ring origins per LOD (snapped to that LOD's chunk grid) ─────────
	TArray<FIntVector> NewRing;
	NewRing.SetNum(NumLODLevels);
	for (int32 L = 0; L < NumLODLevels; L++)
	{
		const double S = ChunkSizeForLOD(L);
		NewRing[L] = FIntVector(
			FMath::FloorToInt32(PlayerPos.X / S) - CPS / 2,
			FMath::FloorToInt32(PlayerPos.Y / S) - CPS / 2,
			FMath::FloorToInt32(PlayerPos.Z / S) - CV / 2);
	}

	// ── 2. Per-slot coords, activation, base dirty set ─────────────────────
	// DirtyMask: 0 = clean, 1 = re-mesh, 2 = became empty (zero its allocation)
	TArray<uint8> DirtyMask;
	DirtyMask.SetNumZeroed(MaxChunks);

	for (int32 L = 0; L < NumLODLevels; L++)
	{
		const FIntVector O = NewRing[L];
		for (int32 sz = 0; sz < CV; sz++)
		for (int32 sy = 0; sy < CPS; sy++)
		for (int32 sx = 0; sx < CPS; sx++)
		{
			const int32 Slot = L * ChunksPerLOD + (sz * CPS + sy) * CPS + sx;
			const FIntVector Coord(
				O.X + PosMod(sx - PosMod(O.X, CPS), CPS),
				O.Y + PosMod(sy - PosMod(O.Y, CPS), CPS),
				O.Z + PosMod(sz - PosMod(O.Z, CV), CV));

			if (Coord != SlotCoord[Slot] || bForceAll)
			{
				const bool bActive = ChunkMayHaveSurface(Coord, L);
				if (bActive)
				{
					DirtyMask[Slot] = 1;
				}
				else if (SlotActive[Slot] && !B.bInit)
				{
					DirtyMask[Slot] = 2;
				}
				SlotCoord[Slot] = Coord;
				SlotActive[Slot] = bActive ? 1 : 0;
			}
		}
	}

	// ── 3. Carve-boundary changes: when the finer ring moves, coarse chunks
	//       whose overlap with it changed must re-mesh ───────────────────────
	if (!bForceAll)
	{
		for (int32 L = 1; L < NumLODLevels; L++)
		{
			if (NewRing[L - 1] == RingOrigin[L - 1])
			{
				continue;
			}
			const FBox OldFiner = FinerRingBox(RingOrigin[L - 1], L - 1);
			const FBox NewFiner = FinerRingBox(NewRing[L - 1], L - 1);

			for (int32 i = 0; i < ChunksPerLOD; i++)
			{
				const int32 Slot = L * ChunksPerLOD + i;
				if (!SlotActive[Slot] || DirtyMask[Slot] != 0)
				{
					continue;
				}
				const FBox CB = ChunkBox(SlotCoord[Slot], L);
				if (!CB.Overlap(OldFiner).Equals(CB.Overlap(NewFiner), 1.0))
				{
					DirtyMask[Slot] = 1;
				}
			}
		}
	}

	// ── 4. Freshly applied edits: activate + dirty every chunk they touch ──
	if (NumPendingEdits > 0 && !bForceAll)
	{
		for (int32 e = Edits.Num() - NumPendingEdits; e < Edits.Num(); e++)
		{
			const FVector4f& E = Edits[e];
			const double R = FMath::Abs(double(E.W));

			for (int32 L = 0; L < NumLODLevels; L++)
			{
				const double S = ChunkSizeForLOD(L);
				const double Margin = 4.0 * S / double(CellsN);
				const FIntVector MinC(
					FMath::FloorToInt32((E.X - R - Margin) / S),
					FMath::FloorToInt32((E.Y - R - Margin) / S),
					FMath::FloorToInt32((E.Z - R - Margin) / S));
				const FIntVector MaxC(
					FMath::FloorToInt32((E.X + R + Margin) / S),
					FMath::FloorToInt32((E.Y + R + Margin) / S),
					FMath::FloorToInt32((E.Z + R + Margin) / S));

				const FIntVector Lo(
					FMath::Max(MinC.X, NewRing[L].X), FMath::Max(MinC.Y, NewRing[L].Y), FMath::Max(MinC.Z, NewRing[L].Z));
				const FIntVector Hi(
					FMath::Min(MaxC.X, NewRing[L].X + CPS - 1), FMath::Min(MaxC.Y, NewRing[L].Y + CPS - 1),
					FMath::Min(MaxC.Z, NewRing[L].Z + CV - 1));

				for (int32 cz = Lo.Z; cz <= Hi.Z; cz++)
				for (int32 cy = Lo.Y; cy <= Hi.Y; cy++)
				for (int32 cx = Lo.X; cx <= Hi.X; cx++)
				{
					const int32 Slot = L * ChunksPerLOD +
						(PosMod(cz, CV) * CPS + PosMod(cy, CPS)) * CPS + PosMod(cx, CPS);
					if (SlotCoord[Slot] == FIntVector(cx, cy, cz))
					{
						SlotActive[Slot] = 1;
						DirtyMask[Slot] = 1;
					}
				}
			}
		}
	}
	NumPendingEdits = 0;
	RingOrigin = NewRing;

	// ── 5. Compose the GPU payload ──────────────────────────────────────────
	B.Capacity = CurrentCapacity;
	B.MaxChunks = uint32(MaxChunks);
	B.CellsPerAxis = uint32(CellsN);
	B.GroupsPerAxis = uint32(CellsN / 8);
	B.MaxVertsPerChunkClamp = MaxVertsPerChunkClamp;
	B.WorldUVScale = WorldUVScale;
	B.ZBandMin = float(ZBandMin);
	B.ZBandMax = float(ZBandMax);
	B.CaveFreq = CaveNoiseFrequency;
	B.CaveAmp = CaveNoiseAmplitude;
	B.NumNoiseLayers = uint32(FMath::Min(NoiseLayers.Num(), 8));
	for (uint32 i = 0; i < B.NumNoiseLayers; i++)
	{
		B.Noise[i] = FVector4f(NoiseLayers[i].Frequency, NoiseLayers[i].Amplitude,
			NoiseLayers[i].OffsetX, NoiseLayers[i].OffsetY);
	}

	B.FinerMin.SetNum(NumLODLevels);
	B.FinerMax.SetNum(NumLODLevels);
	for (int32 L = 0; L < NumLODLevels; L++)
	{
		if (L == 0)
		{
			B.FinerMin[L] = FVector4f(1e30f, 1e30f, 1e30f, 0);
			B.FinerMax[L] = FVector4f(-1e30f, -1e30f, -1e30f, 0);
		}
		else
		{
			const FBox F = FinerRingBox(NewRing[L - 1], L - 1);
			B.FinerMin[L] = FVector4f(float(F.Min.X), float(F.Min.Y), float(F.Min.Z), 0);
			B.FinerMax[L] = FVector4f(float(F.Max.X), float(F.Max.Y), float(F.Max.Z), 0);
		}
	}

	B.EditData = Edits;
	for (int32 Slot = 0; Slot < MaxChunks; Slot++)
	{
		if (DirtyMask[Slot] == 0)
		{
			continue;
		}

		const int32 L = Slot / ChunksPerLOD;
		const double S = ChunkSizeForLOD(L);
		const double Cell = S / double(CellsN);
		const FIntVector Coord = SlotCoord[Slot];

		uint32 Flags = uint32(L) << 8;
		uint32 EditStart = uint32(B.EditIdx.Num());
		uint32 EditCount = 0;

		if (DirtyMask[Slot] == 2)
		{
			Flags |= FlagEmpty;
		}
		else
		{
			const FBox Box = ChunkBox(Coord, L).ExpandBy(4.0 * Cell);
			for (int32 j = 0; j < Edits.Num(); j++)
			{
				const double R = FMath::Abs(double(Edits[j].W));
				const FVector C(Edits[j].X, Edits[j].Y, Edits[j].Z);
				if (Box.Intersect(FBox(C - FVector(R), C + FVector(R))))
				{
					B.EditIdx.Add(uint32(j));
					EditCount++;
				}
			}
		}

		B.DirtyA.Add(FVector4f(float(Coord.X * S), float(Coord.Y * S), float(Coord.Z * S), float(Cell)));
		B.DirtyB.Add(FUintVector4(uint32(Slot), EditStart, EditCount, Flags));
	}

	if (B.bCompact)
	{
		B.SlotToDirty.Init(0xFFFFFFFFu, MaxChunks);
		for (int32 i = 0; i < B.DirtyB.Num(); i++)
		{
			B.SlotToDirty[B.DirtyB[i].X] = uint32(i);
		}
	}

	// Active slot list for the proxy's draw loop.
	TArray<uint32> Active;
	for (int32 Slot = 0; Slot < MaxChunks; Slot++)
	{
		if (SlotActive[Slot])
		{
			Active.Add(uint32(Slot));
		}
	}
	B.bActiveChanged = (Active != CachedActiveSlots) || B.bInit;
	CachedActiveSlots = Active;
	B.ActiveSlots = MoveTemp(Active);

#if RHI_RAYTRACING
	// ── 5b. BLAS bookkeeping ────────────────────────────────────────────────
	// Re-meshed chunks need a BLAS rebuild once this update's exact counts can
	// be read back; emptied chunks release theirs right away.
	// IsRayTracingAllowed, not IsRayTracingEnabled: with on-demand dynamic ray
	// tracing the runtime toggle can be off during this update (e.g. the init
	// update that dirties every chunk), and slots skipped here would never get
	// a BLAS - chunks only re-enter this path when they happen to re-mesh.
	if (IsRayTracingAllowed())
	{
		const uint32 Serial = UpdateSerial;
		for (int32 Slot = 0; Slot < MaxChunks; Slot++)
		{
			if (DirtyMask[Slot] == 1)
			{
				SlotDirtySerial[Slot] = Serial;
				BlasPending.Add(uint32(Slot));
			}
			else if (DirtyMask[Slot] == 2)
			{
				SlotDirtySerial[Slot] = Serial;
				BlasPending.Remove(uint32(Slot));
				B.BlasReleases.Add(uint32(Slot));
			}
		}
		if (B.bCompact)
		{
			// Compaction moves every chunk's pool range, so every active chunk
			// needs a rebuild from this update's table (clean ones included).
			for (uint32 Slot : B.ActiveSlots)
			{
				SlotDirtySerial[Slot] = Serial;
				BlasPending.Add(Slot);
			}
		}
	}
#endif

	// ── 6. Collision request (LOD 0 region around the player) ──────────────
	{
		const double S0 = ChunkSizeForLOD(0);
		const FIntVector Center(
			FMath::FloorToInt32(PlayerPos.X / S0),
			FMath::FloorToInt32(PlayerPos.Y / S0),
			FMath::FloorToInt32(PlayerPos.Z / S0));

		bool bAnyLOD0Dirty = false;
		for (int32 i = 0; i < ChunksPerLOD && !bAnyLOD0Dirty; i++)
		{
			bAnyLOD0Dirty = DirtyMask[i] != 0;
		}
		if (bAnyLOD0Dirty || Center != LastCollCenter)
		{
			bCollRegionDirty = true;
		}

		if (bCollRegionDirty && !bCollPending && !bCollisionCookInProgress)
		{
			for (int32 i = 0; i < ChunksPerLOD; i++)
			{
				if (!SlotActive[i])
				{
					continue;
				}
				const FIntVector D = SlotCoord[i] - Center;
				if (FMath::Max3(FMath::Abs(D.X), FMath::Abs(D.Y), FMath::Abs(D.Z)) <= CollisionRadiusChunks)
				{
					B.CollSlots.Add(uint32(i));
				}
			}

			if (B.CollSlots.Num() > 0)
			{
				if (!CollMetaReadback)
				{
					CollMetaReadback = new FRHIGPUBufferReadback(TEXT("OctreeCollMetaReadback"));
					CollPosReadback = new FRHIGPUBufferReadback(TEXT("OctreeCollPosReadback"));
				}
				B.bCollision = true;
				bCollPending = true;
				bCollArmed.store(false, std::memory_order_release);
				bCollRegionDirty = false;
				LastCollCenter = Center;
				LastCollSlotCount = uint32(B.CollSlots.Num());
			}
		}
	}

	// ── 7. AllocState readback (one in flight) ─────────────────────────────
	if (!bStatePending)
	{
		if (!StateReadback)
		{
			StateReadback = new FRHIGPUBufferReadback(TEXT("OctreeAllocStateReadback"));
		}
		B.bStateReadback = true;
		bStatePending = true;
		bStateArmed.store(false, std::memory_order_release);
		StateSerialInFlight = UpdateSerial;
	}

#if RHI_RAYTRACING
	// ── 7b. Alloc-table readback for pending BLAS builds (one in flight) ───
	if (IsRayTracingAllowed() && !bTablePending && BlasPending.Num() > 0)
	{
		if (!TableReadback)
		{
			TableReadback = new FRHIGPUBufferReadback(TEXT("OctreeAllocTableReadback"));
		}
		B.bTableReadback = true;
		bTablePending = true;
		bTableArmed.store(false, std::memory_order_release);
		TableSerialInFlight = UpdateSerial;
	}
#endif

	const bool bAnyWork = B.bInit || B.bCompact || B.bCollision || B.bActiveChanged ||
		B.bTableReadback || B.DirtyB.Num() > 0 || B.BlasReleases.Num() > 0;
	if (!bAnyWork)
	{
		if (B.bStateReadback)
		{
			bStatePending = false;
		}
		return;
	}

	bGPUInitialized = true;

	// ── 8. Render-thread update graph ───────────────────────────────────────
	ENQUEUE_RENDER_COMMAND(OctreeTerrainUpdate)(
		[this, B = MoveTemp(B)](FRHICommandListImmediate& RHICmdList) mutable
		{
			FRDGBuilder GraphBuilder(RHICmdList);

			const uint32 DirtyCount = uint32(B.DirtyB.Num());
			const bool bSwapPools = B.bInit || B.bCompact;

			// --- pool buffers (old + target) ---------------------------------
			FRDGBufferRef OldPosBuf = nullptr, OldTanBuf = nullptr, OldUVBuf = nullptr, OldColBuf = nullptr;
			if (!B.bInit)
			{
				OldPosBuf = GraphBuilder.RegisterExternalBuffer(PoolPos);
				OldTanBuf = GraphBuilder.RegisterExternalBuffer(PoolTan);
				OldUVBuf = GraphBuilder.RegisterExternalBuffer(PoolUV);
				OldColBuf = GraphBuilder.RegisterExternalBuffer(PoolColor);
			}

			FRDGBufferRef PosBuf, TanBuf, UVBuf, ColBuf;
			if (bSwapPools)
			{
				FRDGBufferDesc PosDesc = FRDGBufferDesc::CreateBufferDesc(16, B.Capacity);
				PosDesc.Usage |= BUF_VertexBuffer | BUF_ShaderResource | BUF_UnorderedAccess;
				PosBuf = GraphBuilder.CreateBuffer(PosDesc, TEXT("OctreePosPool"));

				FRDGBufferDesc TanDesc = FRDGBufferDesc::CreateBufferDesc(16, B.Capacity * 2);
				TanDesc.Usage |= BUF_VertexBuffer | BUF_ShaderResource | BUF_UnorderedAccess;
				TanBuf = GraphBuilder.CreateBuffer(TanDesc, TEXT("OctreeTanPool"));

				FRDGBufferDesc UVDesc = FRDGBufferDesc::CreateBufferDesc(8, B.Capacity);
				UVDesc.Usage |= BUF_VertexBuffer | BUF_ShaderResource | BUF_UnorderedAccess;
				UVBuf = GraphBuilder.CreateBuffer(UVDesc, TEXT("OctreeUVPool"));

				FRDGBufferDesc ColDesc = FRDGBufferDesc::CreateBufferDesc(4, B.Capacity);
				ColDesc.Usage |= BUF_VertexBuffer | BUF_ShaderResource | BUF_UnorderedAccess;
				ColBuf = GraphBuilder.CreateBuffer(ColDesc, TEXT("OctreeColPool"));
			}
			else
			{
				PosBuf = OldPosBuf;
				TanBuf = OldTanBuf;
				UVBuf = OldUVBuf;
				ColBuf = OldColBuf;
			}

			// --- control buffers ---------------------------------------------
			FRDGBufferRef ArgsBuf, AllocBuf, StateBuf, IdentBuf;
			if (B.bInit)
			{
				FRDGBufferDesc ArgsDesc = FRDGBufferDesc::CreateBufferDesc(4, B.MaxChunks * 8);
				ArgsDesc.Usage |= BUF_DrawIndirect | BUF_UnorderedAccess | BUF_ShaderResource;
				ArgsBuf = GraphBuilder.CreateBuffer(ArgsDesc, TEXT("OctreeDrawArgs"));

				FRDGBufferDesc AllocDesc = FRDGBufferDesc::CreateBufferDesc(16, B.MaxChunks);
				AllocDesc.Usage |= BUF_SourceCopy;   // read back for per-chunk BLAS counts
				AllocBuf = GraphBuilder.CreateBuffer(AllocDesc, TEXT("OctreeChunkAlloc"));

				FRDGBufferDesc StateDesc = FRDGBufferDesc::CreateBufferDesc(4, 8);
				StateDesc.Usage |= BUF_SourceCopy;
				StateBuf = GraphBuilder.CreateBuffer(StateDesc, TEXT("OctreeAllocState"));
			}
			else
			{
				ArgsBuf = GraphBuilder.RegisterExternalBuffer(PoolArgs);
				AllocBuf = GraphBuilder.RegisterExternalBuffer(PoolAlloc);
				StateBuf = GraphBuilder.RegisterExternalBuffer(PoolState);
			}

			// The identity index buffer spans the whole vertex pool (the index
			// value IS the global vertex id), so it is rebuilt at pool size
			// whenever the pool itself is (re)created.
			if (bSwapPools)
			{
				FRDGBufferDesc IdentDesc = FRDGBufferDesc::CreateBufferDesc(4, B.Capacity);
				IdentDesc.Usage |= BUF_IndexBuffer | BUF_UnorderedAccess | BUF_ShaderResource;
				IdentBuf = GraphBuilder.CreateBuffer(IdentDesc, TEXT("OctreeIdentityIB"));
			}
			else
			{
				IdentBuf = GraphBuilder.RegisterExternalBuffer(PoolIdentity);
			}

			// --- transient uploads --------------------------------------------
			auto Upload = [&GraphBuilder](const TCHAR* Name, uint32 Stride, const void* Data, uint32 Count)
			{
				FRDGBufferRef Buf = GraphBuilder.CreateBuffer(
					FRDGBufferDesc::CreateBufferDesc(Stride, FMath::Max(Count, 1u)), Name);
				if (Count > 0)
				{
					GraphBuilder.QueueBufferUpload(Buf, Data, uint64(Stride) * Count);
				}
				else
				{
					// Empty lists still get a dummy element: the buffer may be
					// referenced as an SRV (e.g. zero edits), and RDG requires
					// referenced resources to be produced.
					void* Zero = GraphBuilder.Alloc(Stride, 16);
					FMemory::Memzero(Zero, Stride);
					GraphBuilder.QueueBufferUpload(Buf, Zero, Stride);
				}
				return Buf;
			};

			FRDGBufferRef DirtyABuf = Upload(TEXT("OctreeDirtyA"), 16, B.DirtyA.GetData(), DirtyCount);
			FRDGBufferRef DirtyBBuf = Upload(TEXT("OctreeDirtyB"), 16, B.DirtyB.GetData(), DirtyCount);
			FRDGBufferRef EditBuf = Upload(TEXT("OctreeEdits"), 16, B.EditData.GetData(), uint32(B.EditData.Num()));
			FRDGBufferRef EditIdxBuf = Upload(TEXT("OctreeEditIdx"), 4, B.EditIdx.GetData(), uint32(B.EditIdx.Num()));

			FRDGBufferRef CountsBuf = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateBufferDesc(4, FMath::Max(DirtyCount, 1u)), TEXT("OctreeCounts"));
			FRDGBufferRef CursorsBuf = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateBufferDesc(4, FMath::Max(DirtyCount, 1u)), TEXT("OctreeCursors"));
			FRDGBufferRef MeshSkipBuf = GraphBuilder.CreateBuffer(
				FRDGBufferDesc::CreateBufferDesc(4, FMath::Max(DirtyCount, 1u)), TEXT("OctreeMeshSkip"));

			FRDGBufferRef SlotToDirtyBuf = nullptr;
			FRDGBufferRef OldOffsetsBuf = nullptr;
			if (B.bCompact)
			{
				SlotToDirtyBuf = Upload(TEXT("OctreeSlotToDirty"), 4, B.SlotToDirty.GetData(), B.MaxChunks);
				OldOffsetsBuf = GraphBuilder.CreateBuffer(
					FRDGBufferDesc::CreateBufferDesc(4, B.MaxChunks), TEXT("OctreeOldOffsets"));
			}

			auto FillSDF = [&](FOctreeSDFParams& S)
			{
				S.NumNoiseLayers = B.NumNoiseLayers;
				for (int32 i = 0; i < 8; i++)
				{
					S.NoiseLayers[i] = (i < int32(B.NumNoiseLayers)) ? B.Noise[i] : FVector4f(0, 0, 0, 0);
				}
				S.CaveNoiseFrequency = B.CaveFreq;
				S.CaveNoiseAmplitude = B.CaveAmp;
				S.EditSpheres = GraphBuilder.CreateSRV(EditBuf, PF_A32B32G32R32F);
				S.EditIndexList = GraphBuilder.CreateSRV(EditIdxBuf, PF_R32_UINT);
			};

			auto FillChunk = [&](FOctreeChunkParams& C, uint32 DirtyBase)
			{
				C.DirtyInfoA = GraphBuilder.CreateSRV(DirtyABuf, PF_A32B32G32R32F);
				C.DirtyInfoB = GraphBuilder.CreateSRV(DirtyBBuf, PF_R32G32B32A32_UINT);
				C.DirtyCount = DirtyCount;
				C.DirtyBase = DirtyBase;
				C.CellsPerAxis = B.CellsPerAxis;
				C.GroupsPerAxis = B.GroupsPerAxis;
				for (int32 i = 0; i < 10; i++)
				{
					C.FinerVolMin[i] = (i < B.FinerMin.Num()) ? B.FinerMin[i] : FVector4f(1e30f, 1e30f, 1e30f, 0);
					C.FinerVolMax[i] = (i < B.FinerMax.Num()) ? B.FinerMax[i] : FVector4f(-1e30f, -1e30f, -1e30f, 0);
				}
			};

			// D3D caps dispatches at 65535 groups per dimension; a force-all
			// rebuild can exceed that on the Z axis, so the per-chunk cell
			// passes split the dirty list into batches.
			const uint32 MaxChunksPerDispatch = FMath::Max(1u, 65535u / FMath::Max(B.GroupsPerAxis, 1u));

			const auto ShaderMap = GetGlobalShaderMap(GMaxRHIFeatureLevel);

			// --- PASS: pool init (identity IB fill; allocator reset on first init)
			if (bSwapPools)
			{
				TShaderMapRef<FOctreeInitPoolCS> CS(ShaderMap);
				auto* P = GraphBuilder.AllocParameters<FOctreeInitPoolCS::FParameters>();
				P->IdentityIndices = GraphBuilder.CreateUAV(IdentBuf, PF_R32_UINT);
				P->ChunkAlloc = GraphBuilder.CreateUAV(AllocBuf, PF_R32G32B32A32_UINT);
				P->DrawArgs = GraphBuilder.CreateUAV(ArgsBuf, PF_R32_UINT);
				P->AllocState = GraphBuilder.CreateUAV(StateBuf, PF_R32_UINT);
				P->IdentityCount = B.Capacity;
				P->ClearControl = B.bInit ? 1 : 0;
				P->MaxChunks = B.MaxChunks;
				const uint32 MaxItems = FMath::Max(B.Capacity, B.MaxChunks);
				const uint32 NumGroups = FMath::Min(FMath::DivideAndRoundUp(MaxItems, 64u), 16384u);
				P->InitStride = NumGroups * 64u;
				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeInitPool"), CS, P,
					FIntVector(int32(NumGroups), 1, 1));
			}

			// --- PASS: count --------------------------------------------------
			// A compact pass reads CountsRO even with zero dirty chunks (it
			// only dereferences dirty slots, but RDG requires every referenced
			// resource to be produced), so clear it for compacts too.
			if (DirtyCount > 0 || B.bCompact)
			{
				AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CountsBuf, PF_R32_UINT), 0u);
			}
			if (DirtyCount > 0)
			{
				AddClearUAVPass(GraphBuilder, GraphBuilder.CreateUAV(CursorsBuf, PF_R32_UINT), 0u);

				TShaderMapRef<FOctreeCountCS> CS(ShaderMap);
				for (uint32 Base = 0; Base < DirtyCount; Base += MaxChunksPerDispatch)
				{
					const uint32 Num = FMath::Min(MaxChunksPerDispatch, DirtyCount - Base);
					auto* P = GraphBuilder.AllocParameters<FOctreeCountCS::FParameters>();
					FillSDF(P->SDF);
					FillChunk(P->Chunk, Base);
					P->Counts = GraphBuilder.CreateUAV(CountsBuf, PF_R32_UINT);
					FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeCount"), CS, P,
						FIntVector(B.GroupsPerAxis, B.GroupsPerAxis, B.GroupsPerAxis * Num));
				}
			}

			// --- PASS: allocation ---------------------------------------------
			if (B.bCompact)
			{
				TShaderMapRef<FOctreeCompactAllocCS> CS(ShaderMap);
				auto* P = GraphBuilder.AllocParameters<FOctreeCompactAllocCS::FParameters>();
				P->DirtyInfoB = GraphBuilder.CreateSRV(DirtyBBuf, PF_R32G32B32A32_UINT);
				P->CountsRO = GraphBuilder.CreateSRV(CountsBuf, PF_R32_UINT);
				P->SlotToDirty = GraphBuilder.CreateSRV(SlotToDirtyBuf, PF_R32_UINT);
				P->ChunkAlloc = GraphBuilder.CreateUAV(AllocBuf, PF_R32G32B32A32_UINT);
				P->DrawArgs = GraphBuilder.CreateUAV(ArgsBuf, PF_R32_UINT);
				P->AllocState = GraphBuilder.CreateUAV(StateBuf, PF_R32_UINT);
				P->MeshSkip = GraphBuilder.CreateUAV(MeshSkipBuf, PF_R32_UINT);
				P->OldOffsets = GraphBuilder.CreateUAV(OldOffsetsBuf, PF_R32_UINT);
				P->MaxChunks = B.MaxChunks;
				P->CapacityVerts = B.Capacity;
				P->MaxVertsPerChunkClamp = B.MaxVertsPerChunkClamp;
				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeCompactAlloc"), CS, P, FIntVector(1, 1, 1));

				// Copy surviving clean chunks into the new pool.
				TShaderMapRef<FOctreeCopyChunksCS> CopyCS(ShaderMap);
				auto* CP = GraphBuilder.AllocParameters<FOctreeCopyChunksCS::FParameters>();
				CP->SlotToDirty = GraphBuilder.CreateSRV(SlotToDirtyBuf, PF_R32_UINT);
				CP->OldOffsetsRO = GraphBuilder.CreateSRV(OldOffsetsBuf, PF_R32_UINT);
				CP->ChunkAllocRO = GraphBuilder.CreateSRV(AllocBuf, PF_R32G32B32A32_UINT);
				CP->SrcPositions = GraphBuilder.CreateSRV(OldPosBuf, PF_A32B32G32R32F);
				CP->SrcTangents = GraphBuilder.CreateSRV(OldTanBuf, PF_A32B32G32R32F);
				CP->SrcUVs = GraphBuilder.CreateSRV(OldUVBuf, PF_G32R32F);
				CP->SrcColors = GraphBuilder.CreateSRV(OldColBuf, PF_R32_UINT);
				CP->OutPositions = GraphBuilder.CreateUAV(PosBuf, PF_A32B32G32R32F);
				CP->OutTangents = GraphBuilder.CreateUAV(TanBuf, PF_A32B32G32R32F);
				CP->OutUVs = GraphBuilder.CreateUAV(UVBuf, PF_G32R32F);
				CP->OutColors = GraphBuilder.CreateUAV(ColBuf, PF_R32_UINT);
				CP->MaxChunks = B.MaxChunks;
				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeCopyChunks"), CopyCS, CP,
					FIntVector(B.MaxChunks, 1, 1));
			}
			else if (DirtyCount > 0)
			{
				TShaderMapRef<FOctreeAllocCS> CS(ShaderMap);
				auto* P = GraphBuilder.AllocParameters<FOctreeAllocCS::FParameters>();
				P->DirtyInfoB = GraphBuilder.CreateSRV(DirtyBBuf, PF_R32G32B32A32_UINT);
				P->CountsRO = GraphBuilder.CreateSRV(CountsBuf, PF_R32_UINT);
				P->ChunkAlloc = GraphBuilder.CreateUAV(AllocBuf, PF_R32G32B32A32_UINT);
				P->DrawArgs = GraphBuilder.CreateUAV(ArgsBuf, PF_R32_UINT);
				P->AllocState = GraphBuilder.CreateUAV(StateBuf, PF_R32_UINT);
				P->MeshSkip = GraphBuilder.CreateUAV(MeshSkipBuf, PF_R32_UINT);
				P->DirtyCount = DirtyCount;
				P->CapacityVerts = B.Capacity;
				P->MaxVertsPerChunkClamp = B.MaxVertsPerChunkClamp;
				FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeAlloc"), CS, P, FIntVector(1, 1, 1));
			}

			// --- PASS: mesh -----------------------------------------------------
			if (DirtyCount > 0)
			{
				TShaderMapRef<FOctreeMeshCS> CS(ShaderMap);
				for (uint32 Base = 0; Base < DirtyCount; Base += MaxChunksPerDispatch)
				{
					const uint32 Num = FMath::Min(MaxChunksPerDispatch, DirtyCount - Base);
					auto* P = GraphBuilder.AllocParameters<FOctreeMeshCS::FParameters>();
					FillSDF(P->SDF);
					FillChunk(P->Chunk, Base);
					P->ChunkAllocRO = GraphBuilder.CreateSRV(AllocBuf, PF_R32G32B32A32_UINT);
					P->MeshSkipRO = GraphBuilder.CreateSRV(MeshSkipBuf, PF_R32_UINT);
					P->Cursors = GraphBuilder.CreateUAV(CursorsBuf, PF_R32_UINT);
					P->OutPositions = GraphBuilder.CreateUAV(PosBuf, PF_A32B32G32R32F);
					P->OutTangents = GraphBuilder.CreateUAV(TanBuf, PF_A32B32G32R32F);
					P->OutUVs = GraphBuilder.CreateUAV(UVBuf, PF_G32R32F);
					P->OutColors = GraphBuilder.CreateUAV(ColBuf, PF_R32_UINT);
					P->WorldUVScale = B.WorldUVScale;
					P->ZBandMin = B.ZBandMin;
					P->ZBandMax = B.ZBandMax;
					FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeMesh"), CS, P,
						FIntVector(B.GroupsPerAxis, B.GroupsPerAxis, B.GroupsPerAxis * Num));
				}
			}

			// --- PASS: collision gather ----------------------------------------
			TRefCountPtr<FRDGPooledBuffer> CollMetaPooled, CollPosPooled;
			const uint32 CollCount = uint32(B.CollSlots.Num());
			if (B.bCollision && CollCount > 0)
			{
				FRDGBufferRef CollSlotsBuf = Upload(TEXT("OctreeCollSlots"), 4, B.CollSlots.GetData(), CollCount);

				FRDGBufferDesc MetaDesc = FRDGBufferDesc::CreateBufferDesc(4, 2 * CollCount + 1);
				MetaDesc.Usage |= BUF_SourceCopy;
				FRDGBufferRef CollMetaBuf = GraphBuilder.CreateBuffer(MetaDesc, TEXT("OctreeCollMeta"));

				FRDGBufferDesc PosOutDesc = FRDGBufferDesc::CreateBufferDesc(16, OctreeTerrain::CollMaxVerts);
				PosOutDesc.Usage |= BUF_SourceCopy;
				FRDGBufferRef CollPosBuf = GraphBuilder.CreateBuffer(PosOutDesc, TEXT("OctreeCollPos"));

				{
					TShaderMapRef<FOctreeCollisionPrefixCS> CS(ShaderMap);
					auto* P = GraphBuilder.AllocParameters<FOctreeCollisionPrefixCS::FParameters>();
					P->CollSlots = GraphBuilder.CreateSRV(CollSlotsBuf, PF_R32_UINT);
					P->ChunkAllocRO = GraphBuilder.CreateSRV(AllocBuf, PF_R32G32B32A32_UINT);
					P->CollMeta = GraphBuilder.CreateUAV(CollMetaBuf, PF_R32_UINT);
					P->CollCount = CollCount;
					P->CollMaxVerts = OctreeTerrain::CollMaxVerts;
					FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeCollPrefix"), CS, P, FIntVector(1, 1, 1));
				}
				{
					TShaderMapRef<FOctreeCollisionCopyCS> CS(ShaderMap);
					auto* P = GraphBuilder.AllocParameters<FOctreeCollisionCopyCS::FParameters>();
					P->CollSlots = GraphBuilder.CreateSRV(CollSlotsBuf, PF_R32_UINT);
					P->ChunkAllocRO = GraphBuilder.CreateSRV(AllocBuf, PF_R32G32B32A32_UINT);
					P->CollMetaRO = GraphBuilder.CreateSRV(CollMetaBuf, PF_R32_UINT);
					P->SrcPositions = GraphBuilder.CreateSRV(PosBuf, PF_A32B32G32R32F);
					P->CollOutPos = GraphBuilder.CreateUAV(CollPosBuf, PF_A32B32G32R32F);
					P->CollCount = CollCount;
					FComputeShaderUtils::AddPass(GraphBuilder, RDG_EVENT_NAME("OctreeCollCopy"), CS, P,
						FIntVector(CollCount, 1, 1));
				}

				GraphBuilder.QueueBufferExtraction(CollMetaBuf, &CollMetaPooled, ERHIAccess::CopySrc);
				GraphBuilder.QueueBufferExtraction(CollPosBuf, &CollPosPooled, ERHIAccess::CopySrc);
			}

			// --- extraction ------------------------------------------------------
			if (bSwapPools)
			{
				GraphBuilder.QueueBufferExtraction(PosBuf, &PoolPos, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(TanBuf, &PoolTan, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(UVBuf, &PoolUV, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(ColBuf, &PoolColor, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(IdentBuf, &PoolIdentity, ERHIAccess::UAVCompute);
			}
			if (B.bInit)
			{
				GraphBuilder.QueueBufferExtraction(ArgsBuf, &PoolArgs, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(AllocBuf, &PoolAlloc, ERHIAccess::UAVCompute);
				GraphBuilder.QueueBufferExtraction(StateBuf, &PoolState, ERHIAccess::UAVCompute);
			}

			GraphBuilder.Execute();

			// --- proxy + readbacks ------------------------------------------------
			if (FOctreeTerrainSceneProxy* Proxy = static_cast<FOctreeTerrainSceneProxy*>(SceneProxy))
			{
				if (bSwapPools)
				{
					Proxy->UpdateBuffers_RenderThread(PoolPos, PoolTan, PoolUV, PoolColor,
						PoolIdentity, PoolArgs, B.Capacity, B.MaxChunks);
				}
				if (B.bActiveChanged)
				{
					Proxy->SetActiveSlots_RenderThread(B.ActiveSlots);
				}
#if RHI_RAYTRACING
				if (B.BlasReleases.Num() > 0)
				{
					Proxy->UpdateChunkBLAS_RenderThread(TArray<OctreeTerrain::FBlasBuild>(), B.BlasReleases);
				}
#endif
			}

			if (B.bStateReadback && StateReadback && PoolState.IsValid())
			{
				StateReadback->EnqueueCopy(RHICmdList, PoolState->GetRHI(), 16);
				bStateArmed.store(true, std::memory_order_release);
			}

#if RHI_RAYTRACING
			if (B.bTableReadback && TableReadback && PoolAlloc.IsValid())
			{
				TableReadback->EnqueueCopy(RHICmdList, PoolAlloc->GetRHI(), B.MaxChunks * sizeof(FUintVector4));
				bTableArmed.store(true, std::memory_order_release);
			}
#endif

			if (B.bCollision && CollMetaPooled.IsValid() && CollPosPooled.IsValid())
			{
				CollMetaReadback->EnqueueCopy(RHICmdList, CollMetaPooled->GetRHI(), (2 * CollCount + 1) * sizeof(uint32));
				CollPosReadback->EnqueueCopy(RHICmdList, CollPosPooled->GetRHI(), OctreeTerrain::CollMaxVerts * sizeof(FVector4f));
				bCollArmed.store(true, std::memory_order_release);
			}
		});
}

// ============================================================================
// Component: readbacks + collision cooking
// ============================================================================

void UOctreeTerrainComponent::PollReadbacks()
{
	// --- alloc state (pool occupancy, overflow) ---------------------------
	if (bStatePending && bStateArmed.load(std::memory_order_acquire) && StateReadback && StateReadback->IsReady())
	{
		bStatePending = false;
		bStateArmed.store(false, std::memory_order_release);

		FRHIGPUBufferReadback* Readback = StateReadback;
		const uint32 Serial = StateSerialInFlight;
		TWeakObjectPtr<UOctreeTerrainComponent> WeakThis(this);
		ENQUEUE_RENDER_COMMAND(OctreeReadAllocState)(
			[Readback, Serial, WeakThis](FRHICommandListImmediate&)
			{
				const uint32* Data = static_cast<const uint32*>(Readback->Lock(16));
				if (!Data)
				{
					return;
				}
				const uint32 Head = Data[0];
				const uint32 Overflow = Data[1];
				const uint32 Live = Data[2];
				Readback->Unlock();

				AsyncTask(ENamedThreads::GameThread, [WeakThis, Head, Overflow, Live, Serial]()
				{
					if (UOctreeTerrainComponent* Comp = WeakThis.Get())
					{
						Comp->HandleAllocState(Head, Overflow, Live, Serial);
					}
				});
			});
	}

#if RHI_RAYTRACING
	// --- alloc table (exact per-chunk offsets/counts for BLAS builds) -------
	if (bTablePending && bTableArmed.load(std::memory_order_acquire) && TableReadback && TableReadback->IsReady())
	{
		bTablePending = false;
		bTableArmed.store(false, std::memory_order_release);

		FRHIGPUBufferReadback* Readback = TableReadback;
		const int32 NumChunks = MaxChunks;
		const uint32 Serial = TableSerialInFlight;
		TWeakObjectPtr<UOctreeTerrainComponent> WeakThis(this);
		ENQUEUE_RENDER_COMMAND(OctreeReadAllocTable)(
			[Readback, NumChunks, Serial, WeakThis](FRHICommandListImmediate&)
			{
				const uint32 Bytes = uint32(NumChunks) * sizeof(FUintVector4);
				const FUintVector4* Data = static_cast<const FUintVector4*>(Readback->Lock(Bytes));
				if (!Data)
				{
					return;
				}
				TArray<FUintVector4> Table(Data, NumChunks);
				Readback->Unlock();

				AsyncTask(ENamedThreads::GameThread,
					[WeakThis, Table = MoveTemp(Table), Serial]() mutable
					{
						if (UOctreeTerrainComponent* Comp = WeakThis.Get())
						{
							Comp->HandleAllocTable(MoveTemp(Table), Serial);
						}
					});
			});
	}
#endif

	// --- collision geometry -------------------------------------------------
	if (bCollPending && bCollArmed.load(std::memory_order_acquire) &&
		CollMetaReadback && CollPosReadback &&
		CollMetaReadback->IsReady() && CollPosReadback->IsReady())
	{
		bCollPending = false;
		bCollArmed.store(false, std::memory_order_release);

		FRHIGPUBufferReadback* MetaRB = CollMetaReadback;
		FRHIGPUBufferReadback* PosRB = CollPosReadback;
		const uint32 SlotCount = LastCollSlotCount;
		TWeakObjectPtr<UOctreeTerrainComponent> WeakThis(this);

		ENQUEUE_RENDER_COMMAND(OctreeReadCollision)(
			[MetaRB, PosRB, SlotCount, WeakThis](FRHICommandListImmediate&)
			{
				const uint32 MetaBytes = (2 * SlotCount + 1) * sizeof(uint32);
				const uint32* Meta = static_cast<const uint32*>(MetaRB->Lock(MetaBytes));
				if (!Meta)
				{
					return;
				}
				const uint32 Total = FMath::Min(Meta[2 * SlotCount], OctreeTerrain::CollMaxVerts);
				MetaRB->Unlock();

				TArray<FVector3f> NewVerts;
				if (Total >= 3)
				{
					const FVector4f* Pos = static_cast<const FVector4f*>(PosRB->Lock(Total * sizeof(FVector4f)));
					if (Pos)
					{
						const uint32 UsableTotal = Total - (Total % 3);
						NewVerts.Reserve(UsableTotal);
						for (uint32 i = 0; i < UsableTotal; i++)
						{
							NewVerts.Add(FVector3f(Pos[i].X, Pos[i].Y, Pos[i].Z));
						}
						PosRB->Unlock();
					}
				}

				if (NewVerts.Num() < 3)
				{
					return;
				}

				// The GPU soup's native MC order is what DXR treats as
				// front-facing, but Chaos uses the opposite convention and its
				// trimesh queries are one-sided - flip here so traces hit the
				// terrain from outside.
				TArray<FTriIndices> NewTris;
				NewTris.Reserve(NewVerts.Num() / 3);
				for (int32 i = 0; i + 2 < NewVerts.Num(); i += 3)
				{
					FTriIndices T;
					T.v0 = i;
					T.v1 = i + 2;
					T.v2 = i + 1;
					NewTris.Add(T);
				}

				AsyncTask(ENamedThreads::GameThread,
					[WeakThis, Verts = MoveTemp(NewVerts), Tris = MoveTemp(NewTris)]() mutable
					{
						if (UOctreeTerrainComponent* Comp = WeakThis.Get())
						{
							Comp->CollisionVertices = MoveTemp(Verts);
							Comp->CollisionTriIndices = MoveTemp(Tris);
							Comp->RebuildCollision();
						}
					});
			});
	}
}

void UOctreeTerrainComponent::HandleAllocTable(TArray<FUintVector4>&& Table, uint32 Serial)
{
	LastAllocTable = MoveTemp(Table);
	LastAllocTableSerial = Serial;
	DrainBlasPending();
}

void UOctreeTerrainComponent::DrainBlasPending()
{
#if RHI_RAYTRACING
	if (BlasPending.Num() == 0 || LastAllocTable.Num() != MaxChunks || !SceneProxy)
	{
		return;
	}

	// Budgeted: a force-all rebuild queues every chunk, and thousands of BLAS
	// builds in one frame would hitch. Leftovers drain on subsequent ticks.
	constexpr int32 MaxBuildsPerDrain = 256;
	TArray<OctreeTerrain::FBlasBuild> Builds;
	Builds.Reserve(FMath::Min(BlasPending.Num(), MaxBuildsPerDrain));

	for (auto It = BlasPending.CreateIterator(); It; ++It)
	{
		const uint32 Slot = *It;
		if (Slot >= uint32(LastAllocTable.Num()))
		{
			It.RemoveCurrent();
			continue;
		}
		// The table snapshot predates this slot's last re-mesh; wait for a
		// fresher readback (the tick loop requests one).
		if (LastAllocTableSerial < SlotDirtySerial[Slot])
		{
			continue;
		}

		OctreeTerrain::FBlasBuild Bld;
		Bld.Slot = Slot;
		Bld.Offset = LastAllocTable[Slot].X;
		Bld.VertCount = LastAllocTable[Slot].Y;   // 0 -> proxy releases the BLAS
		Builds.Add(Bld);
		It.RemoveCurrent();

		if (Builds.Num() >= MaxBuildsPerDrain)
		{
			break;
		}
	}

	if (Builds.Num() > 0)
	{
		UE_LOG(LogTemp, Verbose, TEXT("OctreeTerrain: BLAS drain - %d builds enqueued, %d still pending"),
			Builds.Num(), BlasPending.Num());
		ENQUEUE_RENDER_COMMAND(OctreeBuildChunkBLAS)(
			[this, Builds = MoveTemp(Builds)](FRHICommandListImmediate&)
			{
				if (FOctreeTerrainSceneProxy* Proxy = static_cast<FOctreeTerrainSceneProxy*>(SceneProxy))
				{
					Proxy->UpdateChunkBLAS_RenderThread(Builds, TArray<uint32>());
				}
			});
	}
#endif
}

void UOctreeTerrainComponent::RebuildCollision()
{
	if (CollisionVertices.Num() == 0 || CollisionTriIndices.Num() == 0 || bCollisionCookInProgress)
	{
		return;
	}

	if (!CollisionBodySetup)
	{
		CollisionBodySetup = NewObject<UBodySetup>(this, NAME_None, RF_Transient);
		CollisionBodySetup->CollisionTraceFlag = CTF_UseComplexAsSimple;
	}

	bCollisionCookInProgress = true;
	CollisionBodySetup->InvalidatePhysicsData();

	TWeakObjectPtr<UOctreeTerrainComponent> WeakThis(this);
	CollisionBodySetup->CreatePhysicsMeshesAsync(
		FOnAsyncPhysicsCookFinished::CreateLambda([WeakThis](bool bSuccess)
		{
			AsyncTask(ENamedThreads::GameThread, [WeakThis, bSuccess]()
			{
				if (UOctreeTerrainComponent* Comp = WeakThis.Get())
				{
					Comp->bCollisionCookInProgress = false;
					if (bSuccess)
					{
						Comp->RecreatePhysicsState();
					}
				}
			});
		}));
}

UBodySetup* UOctreeTerrainComponent::GetBodySetup()
{
	return CollisionBodySetup;
}

bool UOctreeTerrainComponent::GetPhysicsTriMeshData(FTriMeshCollisionData* CollisionData, bool InUseAllTriData)
{
	if (CollisionVertices.Num() == 0 || CollisionTriIndices.Num() == 0)
	{
		return false;
	}

	CollisionData->Vertices = CollisionVertices;
	CollisionData->Indices = CollisionTriIndices;
	CollisionData->MaterialIndices.SetNumZeroed(CollisionTriIndices.Num());
	CollisionData->bFastCook = true;
	return true;
}

bool UOctreeTerrainComponent::ContainsPhysicsTriMeshData(bool InUseAllTriData) const
{
	return CollisionVertices.Num() > 0 && CollisionTriIndices.Num() > 0;
}