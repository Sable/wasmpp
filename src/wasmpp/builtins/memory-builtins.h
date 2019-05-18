#ifndef WASM_WASM_MEMORY_BUILTINS_H_
#define WASM_WASM_MEMORY_BUILTINS_H_

#include <src/wasmpp/builtins/builtins.h>

namespace wasmpp {

// Order is important
#define MEMORY_BUILTINS(V) \
  V(fill_i32, FillI32) \
  V(fill_i64, FillI64) \
  V(fill_f32, FillF32) \
  V(fill_f64, FillF64)

class MemoryBuiltins : public Builtins {
private:
#define CREATE_VAR(var, name) \
  wabt::Var var##_;
    MEMORY_BUILTINS(CREATE_VAR)
#undef CREATE_VAR

#define CREATE_BUILD_FUNC(var, name) \
  wabt::Var Build##name();
    MEMORY_BUILTINS(CREATE_BUILD_FUNC)
#undef CREATE_BUILD_FUNC

public:
  MemoryBuiltins(ModuleManager* moduleManager, ModuleManagerOptions* options);

#define CREATE_GET_VAR(var, name) \
  wabt::Var name() const;
  MEMORY_BUILTINS(CREATE_GET_VAR)
#undef CREATE_VAR
};

} // namespace wasmpp

#endif
