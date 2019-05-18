#include <src/wasmpp/builtins/system-builtins.h>
#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace wasmpp {
using namespace wabt;

SystemBuiltins::SystemBuiltins(ModuleManager *module_manager, ModuleManagerOptions* options) :
    Builtins(module_manager, options) {
#define BUILD_FUNC(var, name) \
  if(options->system.var) \
    var##_ = Build##name();
  SYSTEM_BUILTINS(BUILD_FUNC)
#undef BUILD_FUNC
}

#define BUILD_PRINT_FUNC(t1, t2)                                                     \
Var SystemBuiltins::BuildPrint##t1() {                                               \
  return module_manager_->MakeFuncImport("System", "print", {{Type::t1}, {}});  \
}                                                                                    \
Var SystemBuiltins::Print##t1() const {                                              \
  assert(options_->system.print_##t2);                                               \
  return print_##t2##_;                                                              \
}
  BUILD_PRINT_FUNC(I32, i32);
  BUILD_PRINT_FUNC(I64, i64);
  BUILD_PRINT_FUNC(F32, f32);
  BUILD_PRINT_FUNC(F64, f64);

} // namespace wasmpp