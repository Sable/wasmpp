#ifndef WASM_WASM_SYSTEM_BUILTINS_H_
#define WASM_WASM_SYSTEM_BUILTINS_H_

#include <src/wasmpp/builtins/builtins.h>

namespace wasmpp {

// Order is important
#define SYSTEM_BUILTINS(V) \
  V(print_i32, PrintI32) \
  V(print_i64, PrintI64) \
  V(print_f32, PrintF32) \
  V(print_f64, PrintF64) \

class SystemBuiltins : public Builtins {
private:
#define CREATE_VAR(var, name) \
  wabt::Var var##_;
    SYSTEM_BUILTINS(CREATE_VAR)
#undef CREATE_VAR

#define CREATE_BUILD_FUNC(var, name) \
  wabt::Var Build##name();
    SYSTEM_BUILTINS(CREATE_BUILD_FUNC)
#undef CREATE_BUILD_FUNC

public:
  SystemBuiltins(ModuleManager* moduleManager, ModuleManagerOptions* options) : Builtins(moduleManager, options) {}
  void InitImports();
  void InitDefinitions();

#define CREATE_GET_VAR(var, name) \
  wabt::Var name() const;
  SYSTEM_BUILTINS(CREATE_GET_VAR)
#undef CREATE_VAR
};

} // namespace wasmpp

#endif
