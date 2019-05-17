#include <src/wasmpp/builtins/math-builtins.h>
#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace wasmpp {
using namespace wabt;

MathBuiltins::MathBuiltins(ModuleManager *module_manager, ModuleManagerOptions* options) :
    Builtins(module_manager, options) {
#define BUILD_FUNC(var, name) \
  if(options->math.var) \
    var##_ = Build##name();
  MATH_BUILTINS(BUILD_FUNC)
#undef BUILD_FUNC
}

Var MathBuiltins::BuildSigmoid() {
  return module_manager_->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f, std::vector<Var> params,
                                                                                    std::vector<Var> locals) {
    auto val = MakeLocalGet(params[0]);
    auto negVal = MakeUnary(Opcode::F64Neg, &val);
    auto exp = MakeCall(Exp(), {&negVal});
    auto one1 = MakeI32Const(1);
    auto one2 = MakeI32Const(1);
    auto add = MakeBinary(Opcode::I32Add, &one1, &exp);
    auto div = MakeBinary(Opcode::F64Div, &one2, &add);
    f.Insert(&div);
  });
}

Var MathBuiltins::BuildExp() {
  return module_manager_->MakeFuncImport("Math", "exp", {{Type::F64}, {Type::F64}});
}

Var MathBuiltins::Sigmoid() const {
  assert(options_->math.sigmoid);
  assert(options_->math.exp);
  return sigmoid_;
}

Var MathBuiltins::Exp() const {
  assert(options_->math.exp);
  return exp_;
}

} // namespace wasmpp