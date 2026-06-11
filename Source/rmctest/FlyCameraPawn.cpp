// Fill out your copyright notice in the Description page of Project Settings.

#include "FlyCameraPawn.h"
#include "Camera/CameraComponent.h"
#include "GameFramework/FloatingPawnMovement.h"
#include "EnhancedInputComponent.h"
#include "EnhancedInputSubsystems.h"
#include "OctreeTerrainComponent.h"
#include "GameFramework/PlayerController.h"
#include "Engine/World.h"
#include "EngineUtils.h"
#include "CollisionQueryParams.h"
#include "DrawDebugHelpers.h"

AFlyCameraPawn::AFlyCameraPawn()
{
	PrimaryActorTick.bCanEverTick = true;

	// Create a camera component
	CameraComponent = CreateDefaultSubobject<UCameraComponent>(TEXT("CameraComponent"));
	RootComponent = CameraComponent;
    CameraComponent->bUsePawnControlRotation = true;

	// Create a floating pawn movement component
	MovementComponent = CreateDefaultSubobject<UFloatingPawnMovement>(TEXT("MovementComponent"));
	MovementComponent->SetUpdatedComponent(CameraComponent);
	MovementComponent->MaxSpeed = 500000.f; // FLY REALLY FAST
	MovementComponent->Acceleration = 20000.f;
	MovementComponent->Deceleration = 5000.f;

    bUseControllerRotationPitch = true;
    bUseControllerRotationYaw = true;

    // Terrain edit defaults: 500-unit diameter sphere, add mode.
    EditRadius   = 250.0f;
    bSubtractMode = false;
}

void AFlyCameraPawn::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	if (UEnhancedInputComponent* EnhancedInputComponent = Cast<UEnhancedInputComponent>(PlayerInputComponent))
	{
		// Moving
		if (MoveAction)
		{
			EnhancedInputComponent->BindAction(MoveAction, ETriggerEvent::Triggered, this, &AFlyCameraPawn::Move);
		}

		// Looking
		if (LookAction)
		{
			EnhancedInputComponent->BindAction(LookAction, ETriggerEvent::Triggered, this, &AFlyCameraPawn::Look);
		}

        // Up/Down
        if (MoveUpAction)
        {
            EnhancedInputComponent->BindAction(MoveUpAction, ETriggerEvent::Triggered, this, &AFlyCameraPawn::MoveUp);
        }
	}

    // Physical-key fallback. This does not require any entries in
    // DefaultInput.ini or any Enhanced Input assets on the spawned pawn.
    PlayerInputComponent->BindAxisKey(EKeys::W, this, &AFlyCameraPawn::MoveForward);
    PlayerInputComponent->BindAxisKey(EKeys::S, this, &AFlyCameraPawn::MoveBackward);
    PlayerInputComponent->BindAxisKey(EKeys::D, this, &AFlyCameraPawn::MoveRight);
    PlayerInputComponent->BindAxisKey(EKeys::A, this, &AFlyCameraPawn::MoveLeft);
    PlayerInputComponent->BindAxisKey(EKeys::SpaceBar, this, &AFlyCameraPawn::MoveUpDirect);
    PlayerInputComponent->BindAxisKey(EKeys::LeftControl, this, &AFlyCameraPawn::MoveDown);
    PlayerInputComponent->BindAxisKey(EKeys::MouseX, this, &AFlyCameraPawn::LookYaw);
    PlayerInputComponent->BindAxisKey(EKeys::MouseY, this, &AFlyCameraPawn::LookPitch);

    // Hardcoded terrain edit keys (bypass Enhanced Input mapping context).
    PlayerInputComponent->BindKey(EKeys::Q,                 IE_Pressed, this, &AFlyCameraPawn::ToggleEditMode);
    PlayerInputComponent->BindKey(EKeys::T,                 IE_Pressed, this, &AFlyCameraPawn::GrowEditRadius);
    PlayerInputComponent->BindKey(EKeys::LeftMouseButton,   IE_Pressed, this, &AFlyCameraPawn::PerformTerrainEdit);
}

void AFlyCameraPawn::ToggleEditMode()
{
    bSubtractMode = !bSubtractMode;
    UE_LOG(LogTemp, Log, TEXT("TerrainEdit: mode=%s"),
        bSubtractMode ? TEXT("SUBTRACT") : TEXT("ADD"));
}

void AFlyCameraPawn::GrowEditRadius()
{
    EditRadius *= 1.5f;
    UE_LOG(LogTemp, Log, TEXT("TerrainEdit: radius=%.1f (diameter=%.1f)"),
        EditRadius, EditRadius * 2.0f);
}

void AFlyCameraPawn::PerformTerrainEdit()
{
    UWorld* World = GetWorld();
    if (!World) return;

    // Ray from camera forward. Use the player controller's view point so the
    // trace follows the actual rendered camera (includes controller rotation).
    FVector CamLoc;
    FRotator CamRot;
    if (APlayerController* PC = Cast<APlayerController>(GetController()))
    {
        PC->GetPlayerViewPoint(CamLoc, CamRot);
    }
    else
    {
        CamLoc = CameraComponent->GetComponentLocation();
        CamRot = CameraComponent->GetComponentRotation();
    }

    const FVector TraceStart = CamLoc;
    const FVector TraceEnd   = CamLoc + CamRot.Vector() * 1000000.0f;

    FCollisionQueryParams Params(SCENE_QUERY_STAT(TerrainEdit), /*bTraceComplex*/ true);
    Params.AddIgnoredActor(this);

    FHitResult Hit;
    const bool bHit = World->LineTraceSingleByChannel(
        Hit, TraceStart, TraceEnd, ECC_WorldStatic, Params);

    // Debug visualization: draw the line trace (green on hit, red on miss)
    // and a sphere at the hit / radius preview. Persists for a few seconds.
    const FColor LineColor = bHit ? FColor::Green : FColor::Red;
    const FVector VisEnd   = bHit ? Hit.ImpactPoint : TraceEnd;
    DrawDebugLine(World, TraceStart, VisEnd, LineColor, /*bPersistent*/ false,
                  /*LifeTime*/ 3.0f, /*DepthPriority*/ 0, /*Thickness*/ 2.0f);
    if (bHit)
    {
        const FColor SphereColor = bSubtractMode ? FColor::Red : FColor::Cyan;
        DrawDebugSphere(World, Hit.ImpactPoint, EditRadius, /*Segments*/ 16,
                        SphereColor, /*bPersistent*/ false, /*LifeTime*/ 3.0f,
                        /*DepthPriority*/ 0, /*Thickness*/ 1.5f);
    }

    if (!bHit) return;

    // Prefer the directly hit component, fall back to any OctreeTerrain on the actor.
    UOctreeTerrainComponent* Terrain = Cast<UOctreeTerrainComponent>(Hit.GetComponent());
    if (!Terrain)
    {
        if (AActor* HitActor = Hit.GetActor())
        {
            Terrain = HitActor->FindComponentByClass<UOctreeTerrainComponent>();
        }
    }
    if (!Terrain) return;

    Terrain->ApplySphereEdit(Hit.ImpactPoint, EditRadius, bSubtractMode);
}

void AFlyCameraPawn::MoveForward(float Value)
{
    if (Value != 0.0f)
    {
        const FRotator Rotation = GetControlRotation();
        const FVector Direction = FRotationMatrix(Rotation).GetUnitAxis(EAxis::X);
        AddMovementInput(Direction, Value);
    }
}

void AFlyCameraPawn::MoveBackward(float Value)
{
	MoveForward(-Value);
}

void AFlyCameraPawn::MoveRight(float Value)
{
    if (Value != 0.0f)
    {
        const FRotator Rotation = GetControlRotation();
        const FVector Direction = FRotationMatrix(Rotation).GetUnitAxis(EAxis::Y);
        AddMovementInput(Direction, Value);
    }
}

void AFlyCameraPawn::MoveLeft(float Value)
{
	MoveRight(-Value);
}

void AFlyCameraPawn::MoveUpDirect(float Value)
{
    if (Value != 0.0f)
    {
        AddMovementInput(FVector::UpVector, Value);
    }
}

void AFlyCameraPawn::MoveDown(float Value)
{
	MoveUpDirect(-Value);
}

void AFlyCameraPawn::LookYaw(float Value)
{
    AddControllerYawInput(Value);
}

void AFlyCameraPawn::LookPitch(float Value)
{
    AddControllerPitchInput(-Value);
}

void AFlyCameraPawn::Move(const FInputActionValue& Value)
{
	FVector2D MovementVector = Value.Get<FVector2D>();

	if (Controller != nullptr)
	{
		// Find out which way is forward
		const FRotator Rotation = Controller->GetControlRotation();
		const FRotator YawRotation(0, Rotation.Yaw, 0);

		// Get forward vector
		const FVector ForwardDirection = FRotationMatrix(Rotation).GetUnitAxis(EAxis::X);
		
		// Get right vector 
		const FVector RightDirection = FRotationMatrix(YawRotation).GetUnitAxis(EAxis::Y);

		// Add movement
		AddMovementInput(ForwardDirection, MovementVector.Y);
		AddMovementInput(RightDirection, MovementVector.X);
	}
}

void AFlyCameraPawn::Look(const FInputActionValue& Value)
{
	FVector2D LookAxisVector = Value.Get<FVector2D>();

	if (Controller != nullptr)
	{
		AddControllerYawInput(LookAxisVector.X);
		AddControllerPitchInput(LookAxisVector.Y);
	}
}

void AFlyCameraPawn::MoveUp(const FInputActionValue& Value)
{
    float MoveUpValue = Value.Get<float>();
    AddMovementInput(FVector::UpVector, MoveUpValue);
}
