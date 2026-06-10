// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Pawn.h"
#include "FlyCameraPawn.generated.h"

class UCameraComponent;
class UFloatingPawnMovement;
class UInputAction;
struct FInputActionValue;

UCLASS()
class RMCTEST_API AFlyCameraPawn : public APawn
{
	GENERATED_BODY()

public:
	AFlyCameraPawn();

protected:
	virtual void BeginPlay() override;

public:	
	virtual void Tick(float DeltaTime) override;

	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	UCameraComponent* CameraComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Movement")
	UFloatingPawnMovement* MovementComponent;

	/** Move Input Action */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	UInputAction* MoveAction;

	/** Look Input Action */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	UInputAction* LookAction;

    /** Up/Down Move Input Action */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	UInputAction* MoveUpAction;

	/** Current sphere-edit radius (diameter/2). */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Terrain Edit")
	float EditRadius;

	/** If true, LMB subtracts (digs); otherwise adds (fills). Toggled by Q. */
	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Terrain Edit")
	bool bSubtractMode;

protected:
	void Move(const FInputActionValue& Value);
	void Look(const FInputActionValue& Value);
    void MoveUp(const FInputActionValue& Value);

    // Legacy Input Handlers for hardcoded keys
    void MoveForward(float Value);
    void MoveRight(float Value);
    void MoveUpLegacy(float Value);
    void LookYaw(float Value);
    void LookPitch(float Value);

    // Terrain edit inputs
    void ToggleEditMode();
    void GrowEditRadius();
    void PerformTerrainEdit();

};
