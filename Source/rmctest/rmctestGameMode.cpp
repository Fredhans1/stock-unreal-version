// Copyright Epic Games, Inc. All Rights Reserved.

#include "rmctestGameMode.h"

#include "rmctestPlayerController.h"

ArmctestGameMode::ArmctestGameMode()
{
	PlayerControllerClass = ArmctestPlayerController::StaticClass();
}
