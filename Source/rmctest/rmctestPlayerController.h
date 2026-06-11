// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/PlayerController.h"
#include "rmctestPlayerController.generated.h"

class AFlyCameraPawn;
class UInputMappingContext;
class UUserWidget;

UCLASS()
class ArmctestPlayerController : public APlayerController
{
	GENERATED_BODY()

protected:
	UPROPERTY(EditAnywhere, Category = "Input|Input Mappings")
	TArray<TObjectPtr<UInputMappingContext>> DefaultMappingContexts;

	UPROPERTY(EditAnywhere, Category = "Input|Input Mappings")
	TArray<TObjectPtr<UInputMappingContext>> MobileExcludedMappingContexts;

	UPROPERTY(EditAnywhere, Category = "Input|Touch Controls")
	TSubclassOf<UUserWidget> MobileControlsWidgetClass;

	UPROPERTY()
	TObjectPtr<UUserWidget> MobileControlsWidget;

	UPROPERTY(EditAnywhere, Config, Category = "Input|Touch Controls")
	bool bForceTouchControls = false;

	virtual void BeginPlay() override;
	virtual void SetupInputComponent() override;

private:
	bool ShouldUseTouchControls() const;
	void ToggleFlyPawn();

	UPROPERTY()
	TObjectPtr<AFlyCameraPawn> FlyCamPawn;

	UPROPERTY()
	TObjectPtr<APawn> PreviousPawn;
};
