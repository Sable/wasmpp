#include <src/apps/nn-builder/src/builtins/activation.h>
#include <src/apps/nn-builder/src/arch/model.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Activation::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

void Activation::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);

  // Sigmoid function
  // -  f(x) = 1 / (1 + e(-x))
  // - df(x) = f(x) * (1 - f(x))
  sigmoid_.function = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto denom = MakeBinary(Opcode::F64Add, MakeF64Const(1),
        MakeCall(model->Builtins().math.Exp(), {MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F64Div, MakeF64Const(1), denom);
    f.Insert(div);
  });
  sigmoid_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto sig = MakeLocalSet(params[0], MakeCall(sigmoid_.function, {MakeLocalGet(params[0])}));
    f.Insert(sig);
    auto sub = MakeBinary(Opcode::F64Sub, MakeF64Const(1), MakeLocalGet(params[0]));
    auto mul = MakeBinary(Opcode::F64Mul, MakeLocalGet(params[0]), sub);
    f.Insert(mul);
  });
}

} // namespace builtins
} // namespace nn
