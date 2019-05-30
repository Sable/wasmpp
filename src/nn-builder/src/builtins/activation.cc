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
  sigmoid_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto denom = MakeBinary(Opcode::F32Add, MakeF32Const(1),
        MakeCall(model->Builtins().math.Exp(), {MakeUnary(Opcode::F32Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F32Div, MakeF32Const(1), denom);
    f.Insert(div);
  });
  sigmoid_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto denom = MakeBinary(Opcode::F32Add, MakeF32Const(1),
                            MakeCall(model->Builtins().math.Exp(), {MakeUnary(Opcode::F32Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F32Div, MakeF32Const(1), denom);
    f.Insert(MakeLocalSet(params[0], div));
    auto sub = MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(params[0]));
    auto mul = MakeBinary(Opcode::F32Mul, MakeLocalGet(params[0]), sub);
    f.Insert(mul);
  });

  // ReLU function
  // -  f(x) = x > 0 ? x : 0
  // - df(x) = x > 0 ? 1 : 0
  relu_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{}, {Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeLocalGet(params[0]));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF32Const(0));
    }));
  });
  relu_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeF32Const(1));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF32Const(0));
    }));
  });

  // Leaky ReLU function
  // -  f(x) = x > 0 ? x : alpha * x
  // - df(x) = x > 0 ? 1 : alpha
  leaky_relu_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{}, {Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeLocalGet(params[0]));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeBinary(Opcode::F32Mul, MakeF32Const(options_.leaky_relu_slope), MakeLocalGet(params[0])));
    }));
  });
  leaky_relu_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeF32Const(1));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeF32Const(options_.leaky_relu_slope));
    }));
  });

  // Tanh function
  // -  f(x) = (e(x) - e(-x)) / (e(x) + e(-x))
  // - df(x) = 1 - tanh(x)^2
  tanh_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {Type::F32}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeLocalSet(locals[0], MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])})));
    f.Insert(MakeLocalSet(params[0], MakeCall(model->Builtins().math.Exp(),
                                              { MakeUnary(Opcode::F32Neg, MakeLocalGet(params[0])) })));
    auto nom = MakeBinary(Opcode::F32Sub, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    auto den = MakeBinary(Opcode::F32Add, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    f.Insert(MakeBinary(Opcode::F32Div, nom, den));
  });
  tanh_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {Type::F32}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeLocalSet(locals[0], MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])})));
    f.Insert(MakeLocalSet(params[0], MakeCall(model->Builtins().math.Exp(),
                                              { MakeUnary(Opcode::F32Neg, MakeLocalGet(params[0])) })));
    auto nom = MakeBinary(Opcode::F32Sub, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    auto den = MakeBinary(Opcode::F32Add, MakeLocalGet(locals[0]), MakeLocalGet(params[0]));
    f.Insert(MakeLocalSet(locals[0], MakeBinary(Opcode::F32Div, nom, den)));
    f.Insert(GenerateCompoundAssignment(locals[0], Opcode::F32Mul, MakeLocalGet(locals[0])));
    f.Insert(MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(locals[0])));
  });

  // Linear function
  // -  f(x) = alpha * x
  // - df(x) = alpha
  linear_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeBinary(Opcode::F32Mul, MakeLocalGet(params[0]), MakeF32Const(options_.linear_slope)));
  });
  linear_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeLocalGet(params[0]));
  });

  // ELU function
  // -  f(x) = x > 0 ? x : alpha * (e(x) - 1)
  // - df(x) = x > 0 ? 1 : alpha * e(x)
  elu_.function = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{}, {Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeLocalGet(params[0]));
    }, [&](BlockBody false_block){
      auto sub = MakeBinary(Opcode::F32Sub, MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])}), MakeF32Const(1));
      false_block.Insert(MakeBinary(Opcode::F32Mul, MakeF32Const(options_.elu_slope), sub));
    }));
  });
  elu_.derivative = module_manager->MakeFunction(nullptr, {{Type::F32}, {Type::F32}}, {}, [&](FuncBody f,
      std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Gt, MakeLocalGet(params[0]), MakeF32Const(0));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeF32Const(1));
    }, [&](BlockBody false_block){
      false_block.Insert(MakeBinary(Opcode::F32Mul, MakeF32Const(options_.elu_slope),
                                    MakeCall(model->Builtins().math.Exp(), {MakeLocalGet(params[0])})));
    }));
  });
}

} // namespace builtins
} // namespace nn
