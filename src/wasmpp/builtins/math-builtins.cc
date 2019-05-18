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
  return module_manager_->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f, std::vector<Var> params,
                                                                                    std::vector<Var> locals) {
    auto exp = MakeCall(Exp(), {MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0]))});
    auto div = MakeBinary(Opcode::F64Div, MakeF64Const(1), MakeBinary(Opcode::F64Add, MakeF64Const(1), exp));
    f.Insert(div);
  });
}

Var MathBuiltins::BuildExp() {
  return module_manager_->MakeFuncImport("Math", "exp", {{Type::F64}, {Type::F64}});
}

Var MathBuiltins::BuildRandom() {
  return module_manager_->MakeFuncImport("Math", "random", {{}, {Type::F64}});
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

Var MathBuiltins::Random() const {
  assert(options_->math.random);
  return random_;
}

} // namespace wasmpp