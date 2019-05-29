#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/model.h>
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

  // ReLU function
  // -  f(x) = x > 0 ? x : 0
  // - df(x) = x > 0 ? 1 : 0
  relu_.function = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F64Gt, MakeLocalGet(params[0]), MakeF64Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{}, {Type::F64}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeLocalGet(params[0]));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF64Const(0));
    }));
  });
  relu_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F64Gt, MakeLocalGet(params[0]), MakeF64Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F64}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeF64Const(1));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF64Const(0));
    }));
  });

  // Leaky ReLU function
  // -  f(x) = x > 0 ? x : alpha * x
  // - df(x) = x > 0 ? 1 : alpha
  leaky_relu_.function = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F64Gt, MakeLocalGet(params[0]), MakeF64Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{}, {Type::F64}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeLocalGet(params[0]));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeBinary(Opcode::F64Mul, MakeF64Const(options_.leaky_relu_slope), MakeLocalGet(params[0])));
    }));
  });
  leaky_relu_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F64Gt, MakeLocalGet(params[0]), MakeF64Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F64}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeF64Const(1));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF64Const(options_.leaky_relu_slope));
    }));
  });
}

} // namespace builtins
} // namespace nn
