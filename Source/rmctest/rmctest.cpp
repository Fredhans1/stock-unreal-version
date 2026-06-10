#include "rmctest.h"
#include "Modules/ModuleManager.h"

// We do NOT need a custom StartupModule anymore because UE5 maps /Project/ automatically.
IMPLEMENT_PRIMARY_GAME_MODULE( FDefaultGameModuleImpl, rmctest, "rmctest" );


DEFINE_LOG_CATEGORY(Logrmctest)