// Copyright Epic Games, Inc. All Rights Reserved.

#include "rmctestGameMode.h"
#include "FlyCameraPawn.h"

ArmctestGameMode::ArmctestGameMode()
{
	DefaultPawnClass = AFlyCameraPawn::StaticClass();
	PlayerControllerClass = APlayerController::StaticClass();
	UE_LOG(LogTemp, Warning, TEXT("ArmctestGameMode initialized with DefaultPawnClass: %s"), *DefaultPawnClass->GetName());
}
