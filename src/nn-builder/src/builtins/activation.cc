#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/model.h>
#include <src/wasmpp/wasm-instructions-gen.h>
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
    auto denom = MakeBinary(Opcode::F64Add, MakeF64Const(1),
                            MakeCall(model->Builtins().math.Exp(), {MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F64Div, MakeF64Const(1), denom);
    f.Insert(MakeLocalSet(params[0], div));
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

  // Tanh function
  // -  f(x) = (e(x) - e(-x)) / (e(x) + e(-x))
  // - df(x) = 1 - tanh(x)^2
  tanh_.function = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {Type::F64}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeLocalSet(locals[0], MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])})));
    f.Insert(MakeLocalSet(params[0], MakeCall(model->Builtins().math.Exp(),
                                              { MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0])) })));
    auto nom = MakeBinary(Opcode::F64Sub, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    auto den = MakeBinary(Opcode::F64Add, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    f.Insert(MakeBinary(Opcode::F64Div, nom, den));
  });
  tanh_.derivative = module_manager->MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {Type::F64}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeLocalSet(locals[0], MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])})));
    f.Insert(MakeLocalSet(params[0], MakeCall(model->Builtins().math.Exp(),
                                              { MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0])) })));
    auto nom = MakeBinary(Opcode::F64Sub, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    auto den = MakeBinary(Opcode::F64Add, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(locals[0], MakeBinary(Opcode::F64Div, nom, den)));
    f.Insert(GenerateCompoundAssignment(locals[0], Opcode::F64Mul, MakeLocalGet(locals[0])));
    f.Insert(MakeBinary(Opcode::F64Sub, MakeF64Const(1), MakeLocalGet(locals[0])));
  });
}

} // namespace builtins
} // namespace nn
