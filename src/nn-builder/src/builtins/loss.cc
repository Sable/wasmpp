#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/arch/model.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Loss::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

void Loss::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);

  // Mean Squared Error function
  // ! Note:  `y` is true label, `yHat` is prediction value
  // -  f(y, yHat) = (1/2) * ((yHat - y)^2)
  // - df(y, yHat) = yHat - y
  mean_squared_error_.function = module_manager->MakeFunction(nullptr, {{Type::F64, Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto sub = MakeBinary(Opcode::F64Sub, MakeLocalGet(params[1]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(params[0], sub));
    auto pow = MakeBinary(Opcode::F64Mul, MakeLocalGet(params[0]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(params[0], pow));
    auto mul = MakeBinary(Opcode::F64Mul, MakeF64Const(0.5), MakeLocalGet(params[0]));
    f.Insert(mul);
  });
  mean_squared_error_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64, Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeBinary(Opcode::F64Sub, MakeLocalGet(params[1]), MakeLocalGet(params[0])));
  });

  // Cross-Entropy function
  // ! Note:  `y` is true label, `yHat` is prediction value
  // -  f(y, yHat) = y == 1 ? -log(yHat) : -log(1 - yHat)
  // - df(y, yHat) = -(y / yHat) + ((1 - y) / (1 - yHat))
  cross_entropy_.function = module_manager->MakeFunction(nullptr, {{Type::F64, Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F64Eq, MakeLocalGet(params[0]), MakeF64Const(1));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F64}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeUnary(Opcode::F64Neg, MakeCall(model->Builtins().math.Log(), {MakeLocalGet(params[1])})));
    }, [&](BlockBody false_block){
      auto log = MakeCall(model->Builtins().math.Log(), {MakeBinary(Opcode::F64Sub, MakeF64Const(1), MakeLocalGet(params[1]))});
      false_block.Insert(MakeUnary(Opcode::F64Neg, log));
    }));
  });
  cross_entropy_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64, Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto left_div = MakeUnary(Opcode::F64Neg, MakeBinary(Opcode::F64Div, MakeLocalGet(params[0]), MakeLocalGet(params[1])));
    auto right_nom = MakeBinary(Opcode::F64Sub, MakeF64Const(1), MakeLocalGet(params[0]));
    auto right_den = MakeBinary(Opcode::F64Sub, MakeF64Const(1), MakeLocalGet(params[1]));
    auto right_div = MakeBinary(Opcode::F64Div, right_nom, right_den);
    f.Insert(MakeBinary(Opcode::F64Add, left_div, right_div));
  });
}

} // namespace builtins
} // namespace nn
