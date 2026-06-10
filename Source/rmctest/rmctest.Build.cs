// Copyright Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;

public class rmctest : ModuleRules
{
	public rmctest(ReadOnlyTargetRules Target) : base(Target)
	{
		PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

		PublicDependencyModuleNames.AddRange(new[]
		{
			"Core",
			"CoreUObject",
			"Engine",
			"InputCore",
			"EnhancedInput",
			"AIModule",
			"StateTreeModule",
			"GameplayStateTreeModule",
			"UMG",
			"Slate",
			"SlateCore"
		});

		PrivateDependencyModuleNames.AddRange(new[]
		{
			"RenderCore",
			"RHI",
			"Renderer",
			"PhysicsCore"
		});
	}
}
