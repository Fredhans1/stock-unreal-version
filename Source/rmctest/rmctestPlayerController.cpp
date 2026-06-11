// Copyright Epic Games, Inc. All Rights Reserved.

#include "rmctestPlayerController.h"

#include "Blueprint/UserWidget.h"
#include "Engine/LocalPlayer.h"
#include "Engine/World.h"
#include "EnhancedInputSubsystems.h"
#include "FlyCameraPawn.h"
#include "InputMappingContext.h"
#include "rmctest.h"
#include "Widgets/Input/SVirtualJoystick.h"

void ArmctestPlayerController::BeginPlay()
{
	Super::BeginPlay();

	if (ShouldUseTouchControls()
		&& IsLocalPlayerController()
		&& MobileControlsWidgetClass)
	{
		MobileControlsWidget =
			CreateWidget<UUserWidget>(this, MobileControlsWidgetClass);

		if (MobileControlsWidget)
		{
			MobileControlsWidget->AddToPlayerScreen(0);
		}
		else
		{
			UE_LOG(Logrmctest, Error, TEXT("Could not spawn mobile controls widget."));
		}
	}
}

void ArmctestPlayerController::SetupInputComponent()
{
	Super::SetupInputComponent();

	InputComponent->BindKey(
		EKeys::F,
		IE_Pressed,
		this,
		&ArmctestPlayerController::ToggleFlyPawn);

	if (!IsLocalPlayerController())
	{
		return;
	}

	if (UEnhancedInputLocalPlayerSubsystem* Subsystem =
		ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(
			GetLocalPlayer()))
	{
		for (UInputMappingContext* Context : DefaultMappingContexts)
		{
			if (Context)
			{
				Subsystem->AddMappingContext(Context, 0);
			}
		}

		if (!ShouldUseTouchControls())
		{
			for (UInputMappingContext* Context : MobileExcludedMappingContexts)
			{
				if (Context)
				{
					Subsystem->AddMappingContext(Context, 0);
				}
			}
		}
	}
}

void ArmctestPlayerController::ToggleFlyPawn()
{
	if (!GetWorld())
	{
		return;
	}

	if (Cast<AFlyCameraPawn>(GetPawn()))
	{
		if (IsValid(PreviousPawn))
		{
			Possess(PreviousPawn);
		}
		return;
	}

	PreviousPawn = GetPawn();

	FVector CameraLocation;
	FRotator CameraRotation;
	GetPlayerViewPoint(CameraLocation, CameraRotation);

	if (!IsValid(FlyCamPawn))
	{
		FActorSpawnParameters SpawnParameters;
		SpawnParameters.Owner = this;
		SpawnParameters.SpawnCollisionHandlingOverride =
			ESpawnActorCollisionHandlingMethod::AlwaysSpawn;

		FlyCamPawn = GetWorld()->SpawnActor<AFlyCameraPawn>(
			AFlyCameraPawn::StaticClass(),
			CameraLocation,
			CameraRotation,
			SpawnParameters);
	}
	else
	{
		FlyCamPawn->SetActorLocationAndRotation(
			CameraLocation,
			CameraRotation,
			false,
			nullptr,
			ETeleportType::TeleportPhysics);
	}

	if (IsValid(FlyCamPawn))
	{
		SetControlRotation(CameraRotation);
		Possess(FlyCamPawn);
	}
}

bool ArmctestPlayerController::ShouldUseTouchControls() const
{
	return SVirtualJoystick::ShouldDisplayTouchInterface()
		|| bForceTouchControls;
}
