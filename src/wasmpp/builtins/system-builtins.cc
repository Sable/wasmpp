#include <src/wasmpp/builtins/system-builtins.h>
#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace wasmpp {
using namespace wabt;

void SystemBuiltins::InitImports() {
  if(options_->system.print_i32)
    print_i32_ = BuildPrintI32();
  if(options_->system.print_i64)
    print_i64_ = BuildPrintI64();
  if(options_->system.print_f32)
    print_f32_ = BuildPrintF32();
  if(options_->system.print_f64)
    print_f64_ = BuildPrintF64();
}

void SystemBuiltins::InitDefinitions() {}

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