// Copyright Epic Games, Inc. All Rights Reserved.


#include "rmctestPlayerController.h"
#include "EnhancedInputSubsystems.h"
#include "Engine/LocalPlayer.h"
#include "InputMappingContext.h"
#include "Blueprint/UserWidget.h"
#include "rmctest.h"
#include "Widgets/Input/SVirtualJoystick.h"

void ArmctestPlayerController::BeginPlay()
{
	Super::BeginPlay();

	// Capture the default character pawn
	SpawnedCharacter = Cast<ACharacter>(GetPawn());

	// only spawn touch controls on local player controllers
	if (ShouldUseTouchControls() && IsLocalPlayerController())
	{
		// spawn the mobile controls widget
		MobileControlsWidget = CreateWidget<UUserWidget>(this, MobileControlsWidgetClass);

		if (MobileControlsWidget)
		{
			// add the controls to the player screen
			MobileControlsWidget->AddToPlayerScreen(0);

		} else {

			UE_LOG(Logrmctest, Error, TEXT("Could not spawn mobile controls widget."));

		}

	}
}

void ArmctestPlayerController::SetupInputComponent()
{
	Super::SetupInputComponent();

	InputComponent->BindKey(EKeys::F, IE_Pressed, this, &ArmctestPlayerController::TogglePawn);

	// only add IMCs for local player controllers
	if (IsLocalPlayerController())
	{
		// Add Input Mapping Contexts
		if (UEnhancedInputLocalPlayerSubsystem* Subsystem = ULocalPlayer::GetSubsystem<UEnhancedInputLocalPlayerSubsystem>(GetLocalPlayer()))
		{
			for (UInputMappingContext* CurrentContext : DefaultMappingContexts)
			{
				Subsystem->AddMappingContext(CurrentContext, 0);
			}

			// only add these IMCs if we're not using mobile touch input
			if (!ShouldUseTouchControls())
			{
				for (UInputMappingContext* CurrentContext : MobileExcludedMappingContexts)
				{
					Subsystem->AddMappingContext(CurrentContext, 0);
				}
			}
		}
	}
}

void ArmctestPlayerController::TogglePawn()
{
	if (!bInFlyCam)
	{
		// Currently in character — switch to fly cam
		SpawnedCharacter = Cast<ACharacter>(GetPawn()); // keep ref up to date

		FVector SpawnLoc = SpawnedCharacter ? SpawnedCharacter->GetActorLocation() + FVector(0, 0, 100) : FVector::ZeroVector;

		if (!FlyCamPawn || !IsValid(FlyCamPawn))
		{
			FActorSpawnParameters Params;
			Params.SpawnCollisionHandlingOverride = ESpawnActorCollisionHandlingMethod::AlwaysSpawn;
			FlyCamPawn = GetWorld()->SpawnActor<AFlyCameraPawn>(AFlyCameraPawn::StaticClass(), SpawnLoc, GetControlRotation(), Params);
		}

		if (FlyCamPawn && IsValid(FlyCamPawn))
		{
			UnPossess();
			Possess(FlyCamPawn);
			bInFlyCam = true;
		}
	}
	else
	{
		// Currently in fly cam — switch back to character
		if (SpawnedCharacter && IsValid(SpawnedCharacter))
		{
			FVector CamLoc = FlyCamPawn ? FlyCamPawn->GetActorLocation() : SpawnedCharacter->GetActorLocation();
			SpawnedCharacter->SetActorLocation(CamLoc, false, nullptr, ETeleportType::TeleportPhysics);
			UnPossess();
			Possess(SpawnedCharacter);
			bInFlyCam = false;
		}
	}
}

bool ArmctestPlayerController::ShouldUseTouchControls() const
{
	// are we on a mobile platform? Should we force touch?
	return SVirtualJoystick::ShouldDisplayTouchInterface() || bForceTouchControls;
}
