#ifndef WASM_WASM_MATH_BUILTINS_H_
#define WASM_WASM_MATH_BUILTINS_H_

#include <src/wasmpp/builtins/builtins.h>

namespace wasmpp {

// Order is important
#define MATH_BUILTINS(V) \
  V(exp, Exp) \
  V(random, Random) \
  V(sigmoid, Sigmoid) \

class MathBuiltins : public Builtins {
private:
#define CREATE_VAR(var, name) \
  wabt::Var var##_;
    MATH_BUILTINS(CREATE_VAR)
#undef CREATE_VAR

#define CREATE_BUILD_FUNC(var, name) \
  wabt::Var Build##name();
    MATH_BUILTINS(CREATE_BUILD_FUNC)
#undef CREATE_BUILD_FUNC

public:
  MathBuiltins(ModuleManager* moduleManager, ModuleManagerOptions* options);

#define CREATE_GET_VAR(var, name) \
  wabt::Var name() const;
  MATH_BUILTINS(CREATE_GET_VAR)
#undef CREATE_VAR
};

} // namespace wasmpp

#endif
