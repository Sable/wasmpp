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
  mean_squared_error_.function = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto sub = MakeBinary(Opcode::F32Sub, MakeLocalGet(params[1]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(params[0], sub));
    auto pow = MakeBinary(Opcode::F32Mul, MakeLocalGet(params[0]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(params[0], pow));
    auto mul = MakeBinary(Opcode::F32Mul, MakeF32Const(0.5), MakeLocalGet(params[0]));
    f.Insert(mul);
  });
  mean_squared_error_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeBinary(Opcode::F32Sub, MakeLocalGet(params[1]), MakeLocalGet(params[0])));
  });

  // Cross-Entropy function
  // ! Note:  `y` is true label, `yHat` is prediction value
  // -  f(y, yHat) = y == 1 ? -log(yHat) : -log(1 - yHat)
  // - df(y, yHat) = -(y / yHat) + ((1 - y) / (1 - yHat))
  cross_entropy_.function = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Eq, MakeLocalGet(params[0]), MakeF32Const(1));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeUnary(Opcode::F32Neg, MakeCall(model->Builtins().math.Log(), {MakeLocalGet(params[1])})));
    }, [&](BlockBody false_block){
      auto log = MakeCall(model->Builtins().math.Log(), {MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(params[1]))});
      false_block.Insert(MakeUnary(Opcode::F32Neg, log));
    }));
  });
  cross_entropy_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto left_div = MakeUnary(Opcode::F32Neg, MakeBinary(Opcode::F32Div, MakeLocalGet(params[0]), MakeLocalGet(params[1])));
    auto right_nom = MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(params[0]));
    auto right_den = MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(params[1]));
    auto right_div = MakeBinary(Opcode::F32Div, right_nom, right_den);
    f.Insert(MakeBinary(Opcode::F32Add, left_div, right_div));
  });
}

} // namespace builtins
} // namespace nn
