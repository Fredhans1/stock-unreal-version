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

	virtual void SetupPlayerInputComponent(UInputComponent* PlayerInputComponent) override;

protected:
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Camera")
	TObjectPtr<UCameraComponent> CameraComponent;

	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Movement")
	TObjectPtr<UFloatingPawnMovement> MovementComponent;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	TObjectPtr<UInputAction> MoveAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	TObjectPtr<UInputAction> LookAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Input")
	TObjectPtr<UInputAction> MoveUpAction;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Terrain Edit")
	float EditRadius = 250.0f;

	UPROPERTY(EditAnywhere, BlueprintReadOnly, Category = "Terrain Edit")
	bool bSubtractMode = false;

private:
	void Move(const FInputActionValue& Value);
	void Look(const FInputActionValue& Value);
	void MoveUp(const FInputActionValue& Value);

	void MoveForward(float Value);
	void MoveBackward(float Value);
	void MoveRight(float Value);
	void MoveLeft(float Value);
	void MoveUpDirect(float Value);
	void MoveDown(float Value);
	void LookYaw(float Value);
	void LookPitch(float Value);

	void ToggleEditMode();
	void GrowEditRadius();
	void PerformTerrainEdit();
};
