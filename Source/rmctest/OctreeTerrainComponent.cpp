#include "OctreeTerrainComponent.h"

#include "Async/Async.h"
#include "DynamicMeshBuilder.h"
#include "Engine/CollisionProfile.h"
#include "GameFramework/Pawn.h"
#include "GameFramework/PlayerController.h"
#include "Materials/Material.h"
#include "Materials/MaterialInterface.h"
#include "PhysicsEngine/BodySetup.h"
#include "PrimitiveViewRelevance.h"
#include "SceneManagement.h"

class FOctreeTerrainSceneProxy final : public FPrimitiveSceneProxy
{
public:
	explicit FOctreeTerrainSceneProxy(const UOctreeTerrainComponent* Component)
		: FPrimitiveSceneProxy(Component)
		, Material(Component->TerrainMaterial
			? Component->TerrainMaterial
			: UMaterial::GetDefaultMaterial(MD_Surface))
		, MaterialRelevance(Material->GetRelevance_Concurrent(GetScene().GetShaderPlatform()))
		, Positions(Component->MeshPositions)
		, Normals(Component->MeshNormals)
		, UVs(Component->MeshUVs)
		, Colors(Component->MeshColors)
		, Indices(Component->MeshIndices)
	{
	}

	virtual SIZE_T GetTypeHash() const override
	{
		static size_t UniquePointer;
		return reinterpret_cast<SIZE_T>(&UniquePointer);
	}

	virtual void GetDynamicMeshElements(
		const TArray<const FSceneView*>& Views,
		const FSceneViewFamily& ViewFamily,
		uint32 VisibilityMap,
		FMeshElementCollector& Collector) const override
	{
		if (Positions.IsEmpty() || Indices.IsEmpty())
		{
			return;
		}

		for (int32 ViewIndex = 0; ViewIndex < Views.Num(); ++ViewIndex)
		{
			if ((VisibilityMap & (1u << ViewIndex)) == 0)
			{
				continue;
			}

			FDynamicMeshBuilder MeshBuilder(ViewFamily.GetFeatureLevel());
			MeshBuilder.ReserveVertices(Positions.Num());
			MeshBuilder.ReserveTriangles(Indices.Num() / 3);

			for (int32 VertexIndex = 0; VertexIndex < Positions.Num(); ++VertexIndex)
			{
				const FVector3f Normal = Normals.IsValidIndex(VertexIndex)
					? Normals[VertexIndex]
					: FVector3f::UpVector;
				FVector3f TangentX = FVector3f::CrossProduct(FVector3f::YAxisVector, Normal);
				if (!TangentX.Normalize())
				{
					TangentX = FVector3f::XAxisVector;
				}
				const FVector3f TangentY =
					FVector3f::CrossProduct(Normal, TangentX).GetSafeNormal();

				MeshBuilder.AddVertex(
					Positions[VertexIndex],
					UVs.IsValidIndex(VertexIndex) ? UVs[VertexIndex] : FVector2f::ZeroVector,
					TangentX,
					TangentY,
					Normal,
					Colors.IsValidIndex(VertexIndex) ? Colors[VertexIndex] : FColor::White);
			}

			MeshBuilder.AddTriangles(Indices);
			MeshBuilder.GetMesh(
				GetLocalToWorld(),
				Material->GetRenderProxy(),
				GetDepthPriorityGroup(Views[ViewIndex]),
				false,
				true,
				ViewIndex,
				Collector);
		}
	}

	virtual FPrimitiveViewRelevance GetViewRelevance(const FSceneView* View) const override
	{
		FPrimitiveViewRelevance Result;
		Result.bDrawRelevance = IsShown(View);
		Result.bShadowRelevance = IsShadowCast(View);
		Result.bDynamicRelevance = true;
		Result.bRenderInMainPass = ShouldRenderInMainPass();
		Result.bUsesLightingChannels =
			GetLightingChannelMask() != GetDefaultLightingChannelMask();
		Result.bRenderCustomDepth = ShouldRenderCustomDepth();
		MaterialRelevance.SetPrimitiveViewRelevance(Result);
		return Result;
	}

	virtual uint32 GetMemoryFootprint() const override
	{
		return sizeof(*this) + GetAllocatedSize();
	}

	uint32 GetAllocatedSize() const
	{
		return static_cast<uint32>(
			Positions.GetAllocatedSize()
			+ Normals.GetAllocatedSize()
			+ UVs.GetAllocatedSize()
			+ Colors.GetAllocatedSize()
			+ Indices.GetAllocatedSize());
	}

private:
	UMaterialInterface* Material = nullptr;
	FMaterialRelevance MaterialRelevance;
	TArray<FVector3f> Positions;
	TArray<FVector3f> Normals;
	TArray<FVector2f> UVs;
	TArray<FColor> Colors;
	TArray<uint32> Indices;
};

UOctreeTerrainComponent::UOctreeTerrainComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
	PrimaryComponentTick.TickGroup = TG_PrePhysics;

	SetCollisionProfileName(UCollisionProfile::BlockAll_ProfileName);
	SetCollisionEnabled(ECollisionEnabled::QueryAndPhysics);
	SetCollisionObjectType(ECC_WorldStatic);

	CastShadow = true;
	bCastDynamicShadow = true;
	bCastStaticShadow = false;
	bUseAsOccluder = true;
	bAffectDynamicIndirectLighting = true;
	bAffectDistanceFieldLighting = false;
}

void UOctreeTerrainComponent::BeginPlay()
{
	Super::BeginPlay();
	RebuildTerrain();
}

void UOctreeTerrainComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);

	const FVector PlayerPosition = GetPlayerPosition();
	const float SmallestChunk = FMath::Max(100.0f, BaseChunkSize);
	const FIntVector TerrainOrigin(
		FMath::FloorToInt(PlayerPosition.X / SmallestChunk),
		FMath::FloorToInt(PlayerPosition.Y / SmallestChunk),
		0);

	const bool bMovedChunk = TerrainOrigin != LastTerrainOrigin;
	const bool bMovedThreshold =
		FVector::DistSquared2D(PlayerPosition, LastBuildPlayerPosition)
		>= FMath::Square(UpdateThreshold);

	if (bRebuildRequested || bMovedChunk || bMovedThreshold)
	{
		BuildTerrainMesh(PlayerPosition);
		LastTerrainOrigin = TerrainOrigin;
		LastBuildPlayerPosition = PlayerPosition;
		bRebuildRequested = false;
		MarkRenderStateDirty();
		RebuildCollision();
	}
}

void UOctreeTerrainComponent::ApplySphereEdit(
	FVector Center,
	float Radius,
	bool bSubtract)
{
	if (Radius <= UE_KINDA_SMALL_NUMBER)
	{
		return;
	}

	FSphereEdit& Edit = SphereEdits.AddDefaulted_GetRef();
	Edit.Center = Center;
	Edit.Radius = Radius;
	Edit.bSubtract = bSubtract;
	bRebuildRequested = true;
}

void UOctreeTerrainComponent::ClearAllEdits()
{
	SphereEdits.Reset();
	bRebuildRequested = true;
}

void UOctreeTerrainComponent::RebuildTerrain()
{
	bRebuildRequested = true;
}

FVector UOctreeTerrainComponent::GetPlayerPosition() const
{
	if (const UWorld* World = GetWorld())
	{
		if (const APlayerController* Controller = World->GetFirstPlayerController())
		{
			if (const APawn* Pawn = Controller->GetPawn())
			{
				return Pawn->GetActorLocation();
			}
		}
	}
	return GetComponentLocation();
}

float UOctreeTerrainComponent::SampleHeight(double WorldX, double WorldY) const
{
	const float Frequency = FMath::Max(NoiseFrequency, 0.000001f);
	float Amplitude = HeightScale;
	float CurrentFrequency = Frequency;
	float Height = 0.0f;
	float Weight = 0.0f;

	for (int32 Octave = 0; Octave < FMath::Clamp(NoiseOctaves, 1, 8); ++Octave)
	{
		const FVector2D Point(WorldX * CurrentFrequency, WorldY * CurrentFrequency);
		Height += FMath::PerlinNoise2D(Point) * Amplitude;
		Weight += Amplitude;
		CurrentFrequency *= 2.03f;
		Amplitude *= 0.5f;
	}

	if (Weight > UE_KINDA_SMALL_NUMBER)
	{
		Height *= HeightScale / Weight;
	}

	if (CaveNoiseFrequency > 0.0f && CaveNoiseAmplitude > 0.0f)
	{
		Height += FMath::PerlinNoise2D(FVector2D(
			WorldX * CaveNoiseFrequency,
			WorldY * CaveNoiseFrequency)) * CaveNoiseAmplitude;
	}

	for (const FSphereEdit& Edit : SphereEdits)
	{
		const double DX = WorldX - Edit.Center.X;
		const double DY = WorldY - Edit.Center.Y;
		const double HorizontalDistanceSquared = DX * DX + DY * DY;
		const double RadiusSquared = FMath::Square(static_cast<double>(Edit.Radius));
		if (HorizontalDistanceSquared >= RadiusSquared)
		{
			continue;
		}

		const float VerticalExtent = static_cast<float>(
			FMath::Sqrt(RadiusSquared - HorizontalDistanceSquared));
		const float SphereTop = Edit.Center.Z + VerticalExtent;
		const float SphereBottom = Edit.Center.Z - VerticalExtent;
		Height = Edit.bSubtract
			? FMath::Min(Height, SphereBottom)
			: FMath::Max(Height, SphereTop);
	}

	return Height;
}

FVector3f UOctreeTerrainComponent::SampleNormal(
	double WorldX,
	double WorldY,
	double Step) const
{
	const float HeightLeft = SampleHeight(WorldX - Step, WorldY);
	const float HeightRight = SampleHeight(WorldX + Step, WorldY);
	const float HeightDown = SampleHeight(WorldX, WorldY - Step);
	const float HeightUp = SampleHeight(WorldX, WorldY + Step);
	return FVector3f(
		HeightLeft - HeightRight,
		HeightDown - HeightUp,
		static_cast<float>(2.0 * Step)).GetSafeNormal();
}

void UOctreeTerrainComponent::BuildTerrainMesh(const FVector& PlayerPosition)
{
	MeshPositions.Reset();
	MeshNormals.Reset();
	MeshUVs.Reset();
	MeshColors.Reset();
	MeshIndices.Reset();
	CollisionVertices.Reset();
	CollisionTriangles.Reset();

	const int32 Resolution = FMath::Clamp(GridResolution, 4, 64);
	const int32 SideCount = FMath::Clamp(ChunksPerSide, 3, 21) | 1;
	const int32 HalfSide = SideCount / 2;
	const int32 LODCount = FMath::Clamp(NumLODLevels, 1, 8);
	const int32 CollisionStride = FMath::Max(1, CollisionDownsampleFactor);

	FBox TerrainBounds(EForceInit::ForceInit);

	for (int32 LOD = 0; LOD < LODCount; ++LOD)
	{
		const float ChunkSize = BaseChunkSize * static_cast<float>(1 << LOD);
		const int32 CenterChunkX = FMath::FloorToInt(PlayerPosition.X / ChunkSize);
		const int32 CenterChunkY = FMath::FloorToInt(PlayerPosition.Y / ChunkSize);

		for (int32 LocalY = -HalfSide; LocalY <= HalfSide; ++LocalY)
		{
			for (int32 LocalX = -HalfSide; LocalX <= HalfSide; ++LocalX)
			{
				if (LOD > 0)
				{
					const int32 InnerHalf = FMath::Max(1, HalfSide / 2);
					if (FMath::Abs(LocalX) <= InnerHalf && FMath::Abs(LocalY) <= InnerHalf)
					{
						continue;
					}
				}

				const int32 ChunkX = CenterChunkX + LocalX;
				const int32 ChunkY = CenterChunkY + LocalY;
				const FVector2D Origin(ChunkX * ChunkSize, ChunkY * ChunkSize);
				const bool bBuildCollision =
					LOD == 0
					&& (LocalX % CollisionStride) == 0
					&& (LocalY % CollisionStride) == 0;

				const int32 StartVertex = MeshPositions.Num();
				AppendChunk(
					LOD,
					ChunkX,
					ChunkY,
					Origin,
					ChunkSize,
					Resolution,
					bBuildCollision);

				for (int32 VertexIndex = StartVertex; VertexIndex < MeshPositions.Num(); ++VertexIndex)
				{
					TerrainBounds += FVector(MeshPositions[VertexIndex]);
				}
			}
		}
	}

	if (TerrainBounds.IsValid)
	{
		LocalBounds = FBoxSphereBounds(TerrainBounds);
	}
	else
	{
		LocalBounds = FBoxSphereBounds(FVector::ZeroVector, FVector(1000.0), 1000.0);
	}

	UpdateBounds();
}

void UOctreeTerrainComponent::AppendChunk(
	int32 LOD,
	int32 ChunkX,
	int32 ChunkY,
	const FVector2D& ChunkOrigin,
	float ChunkSize,
	int32 Resolution,
	bool bBuildCollision)
{
	const int32 FirstVertex = MeshPositions.Num();
	const int32 VerticesPerSide = Resolution + 1;
	const double Step = ChunkSize / static_cast<double>(Resolution);
	const FTransform ComponentTransform = GetComponentTransform();

	for (int32 Y = 0; Y <= Resolution; ++Y)
	{
		for (int32 X = 0; X <= Resolution; ++X)
		{
			const double WorldX = ChunkOrigin.X + X * Step;
			const double WorldY = ChunkOrigin.Y + Y * Step;
			const float WorldZ = SampleHeight(WorldX, WorldY);
			const FVector WorldPosition(WorldX, WorldY, WorldZ);
			const FVector LocalPosition =
				ComponentTransform.InverseTransformPosition(WorldPosition);
			const FVector WorldNormal = FVector(SampleNormal(WorldX, WorldY, Step));
			const FVector LocalNormal =
				ComponentTransform.InverseTransformVectorNoScale(WorldNormal).GetSafeNormal();

			MeshPositions.Add(FVector3f(LocalPosition));
			MeshNormals.Add(FVector3f(LocalNormal));
			MeshUVs.Add(FVector2f(
				static_cast<float>(WorldX / BaseChunkSize),
				static_cast<float>(WorldY / BaseChunkSize)));

			const float NormalizedHeight = FMath::Clamp(
				(WorldZ / FMath::Max(HeightScale, 1.0f) + 1.0f) * 0.5f,
				0.0f,
				1.0f);
			MeshColors.Add(FColor(
				static_cast<uint8>(FMath::RoundToInt(NormalizedHeight * 255.0f)),
				static_cast<uint8>(LOD * 32),
				128,
				255));
		}
	}

	for (int32 Y = 0; Y < Resolution; ++Y)
	{
		for (int32 X = 0; X < Resolution; ++X)
		{
			const uint32 I0 = FirstVertex + Y * VerticesPerSide + X;
			const uint32 I1 = I0 + 1;
			const uint32 I2 = I0 + VerticesPerSide;
			const uint32 I3 = I2 + 1;
			MeshIndices.Append({ I0, I3, I1, I0, I2, I3 });

			if (bBuildCollision)
			{
				const int32 CollisionBase = CollisionVertices.Num();
				CollisionVertices.Append({
					MeshPositions[I0],
					MeshPositions[I1],
					MeshPositions[I2],
					MeshPositions[I3]
				});

				FTriIndices TriangleA;
				TriangleA.v0 = CollisionBase;
				TriangleA.v1 = CollisionBase + 3;
				TriangleA.v2 = CollisionBase + 1;
				CollisionTriangles.Add(TriangleA);

				FTriIndices TriangleB;
				TriangleB.v0 = CollisionBase;
				TriangleB.v1 = CollisionBase + 2;
				TriangleB.v2 = CollisionBase + 3;
				CollisionTriangles.Add(TriangleB);
			}
		}
	}

	if (SkirtDepth > 0.0f)
	{
		AppendSkirts(FirstVertex, Resolution, SkirtDepth * static_cast<float>(1 << LOD));
	}
}

void UOctreeTerrainComponent::AppendSkirts(
	int32 FirstVertex,
	int32 Resolution,
	float Depth)
{
	const int32 VerticesPerSide = Resolution + 1;
	TArray<int32> Boundary;
	Boundary.Reserve(Resolution * 4);

	for (int32 X = 0; X < Resolution; ++X)
	{
		Boundary.Add(FirstVertex + X);
	}
	for (int32 Y = 0; Y < Resolution; ++Y)
	{
		Boundary.Add(FirstVertex + Y * VerticesPerSide + Resolution);
	}
	for (int32 X = Resolution; X > 0; --X)
	{
		Boundary.Add(FirstVertex + Resolution * VerticesPerSide + X);
	}
	for (int32 Y = Resolution; Y > 0; --Y)
	{
		Boundary.Add(FirstVertex + Y * VerticesPerSide);
	}

	const int32 SkirtStart = MeshPositions.Num();
	for (const int32 SourceIndex : Boundary)
	{
		FVector3f Position = MeshPositions[SourceIndex];
		const FVector3f Normal = MeshNormals[SourceIndex];
		const FVector2f UV = MeshUVs[SourceIndex];
		const FColor Color = MeshColors[SourceIndex];
		Position.Z -= Depth;
		MeshPositions.Add(Position);
		MeshNormals.Add(Normal);
		MeshUVs.Add(UV);
		MeshColors.Add(Color);
	}

	for (int32 Index = 0; Index < Boundary.Num(); ++Index)
	{
		const int32 Next = (Index + 1) % Boundary.Num();
		const uint32 Top0 = Boundary[Index];
		const uint32 Top1 = Boundary[Next];
		const uint32 Bottom0 = SkirtStart + Index;
		const uint32 Bottom1 = SkirtStart + Next;
		MeshIndices.Append({ Top0, Top1, Bottom1, Top0, Bottom1, Bottom0 });
	}
}

FPrimitiveSceneProxy* UOctreeTerrainComponent::CreateSceneProxy()
{
	return MeshPositions.IsEmpty()
		? nullptr
		: new FOctreeTerrainSceneProxy(this);
}

FBoxSphereBounds UOctreeTerrainComponent::CalcBounds(
	const FTransform& LocalToWorld) const
{
	return LocalBounds.TransformBy(LocalToWorld);
}

void UOctreeTerrainComponent::GetUsedMaterials(
	TArray<UMaterialInterface*>& OutMaterials,
	bool bGetDebugMaterials) const
{
	if (TerrainMaterial)
	{
		OutMaterials.Add(TerrainMaterial);
	}
}

UBodySetup* UOctreeTerrainComponent::GetBodySetup()
{
	return CollisionBodySetup;
}

bool UOctreeTerrainComponent::GetPhysicsTriMeshData(
	FTriMeshCollisionData* CollisionData,
	bool InUseAllTriData)
{
	if (CollisionVertices.IsEmpty() || CollisionTriangles.IsEmpty())
	{
		return false;
	}

	CollisionData->Vertices = CollisionVertices;
	CollisionData->Indices = CollisionTriangles;
	CollisionData->MaterialIndices.SetNumZeroed(CollisionTriangles.Num());
	CollisionData->bFastCook = true;
	CollisionData->bFlipNormals = false;
	return true;
}

bool UOctreeTerrainComponent::ContainsPhysicsTriMeshData(
	bool InUseAllTriData) const
{
	return !CollisionVertices.IsEmpty() && !CollisionTriangles.IsEmpty();
}

void UOctreeTerrainComponent::RebuildCollision()
{
	if (CollisionVertices.IsEmpty()
		|| CollisionTriangles.IsEmpty()
		|| bCollisionCookInProgress)
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
		FOnAsyncPhysicsCookFinished::CreateLambda(
			[WeakThis](bool bSuccess)
			{
				AsyncTask(
					ENamedThreads::GameThread,
					[WeakThis, bSuccess]()
					{
						if (UOctreeTerrainComponent* Component = WeakThis.Get())
						{
							Component->bCollisionCookInProgress = false;
							if (bSuccess)
							{
								Component->RecreatePhysicsState();
							}
						}
					});
			}));
}
