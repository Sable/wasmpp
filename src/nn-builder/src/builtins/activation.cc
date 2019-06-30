#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/model.h>
#include <src/wasmpp/wasm-instructions-gen.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

bool ActivationFunction::operator==(const nn::builtins::ActivationFunction &func) const {
  return type == func.type;
}

bool ActivationFunction::operator!=(const nn::builtins::ActivationFunction &func) const {
  return !(operator==(func));
}

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
  sigmoid_.type = ActivationFunction::SIGMOID;
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
  relu_.type = ActivationFunction::RELU;
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
  leaky_relu_.type = ActivationFunction::LEAKY_RELU;
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
  tanh_.type = ActivationFunction::TANH;
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
  linear_.type = ActivationFunction::LINEAR;
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
  elu_.type = ActivationFunction::ELU;
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

  // Softmax function
  // ! Note: softmax is a little different becasue it takes/returns a vector as input/output
  // -  f(X) = (exp(X[i]) - MAX(X)) / SUM(exp(X))
  // - df(X) = not defined - check backward algorithm for details
  softmax_.type = ActivationFunction::SOFTMAX;
  softmax_.function = module_manager->MakeFunction(nullptr, {{Type::I32, Type::I32, Type::I32, Type::I32}, {}},
      {Type::I32, Type::I32, Type::I32, Type::F32, Type::F32, Type::I32, Type::I32},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {

    assert(params.size() == 4);
    auto src_begin = params[0];
    auto dst_begin = params[1];
    auto row_count = params[2];
    auto col_count = params[3];

    assert(locals.size() == 7);
    auto row = locals[0];
    auto col = locals[1];
    auto cell_addr = locals[2];
    auto total_exp = locals[3];
    auto cell_exp = locals[4];
    auto width_bytes = locals[5];
    auto height_bytes = locals[6];

    uint32_t type_size = TypeSize(Type::F32);

    f.Insert(MakeLocalSet(width_bytes, MakeBinary(Opcode::I32Mul, MakeLocalGet(col_count), MakeI32Const(type_size))));
    f.Insert(MakeLocalSet(height_bytes, MakeBinary(Opcode::I32Mul, MakeLocalGet(row_count), MakeLocalGet(width_bytes))));
    f.Insert(GenerateRangeLoop(f.Label(), col, 0, width_bytes, type_size, {}, [&](BlockBody* b1) {
      b1->Insert(MakeLocalSet(total_exp, MakeF32Const(0)));
      // TODO Find max and subtract

      // First compute the e(x[i]) column wise
      // store the result of each cell in the destination matrix
      // keep track of the total SUM(exp(x[i]))
      b1->Insert(GenerateRangeLoop(f.Label(), row, 0, height_bytes, width_bytes, {}, [&](BlockBody* b2) {
        // Compute src current cell address
        auto src_curr_addr = MakeBinary(Opcode::I32Add,
                                        MakeLocalGet(row), MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeLocalGet(src_begin)));
        // Compute dst current cell address
        auto dst_curr_addr = MakeBinary(Opcode::I32Add,
                                        MakeLocalGet(row), MakeBinary(Opcode::I32Add, MakeLocalGet(col),MakeLocalGet(dst_begin)));
        // Compute exp(x) and store it in a local
        b2->Insert(MakeLocalSet(cell_exp, MakeCall(model->Builtins().math.Exp(), {
            MakeF32Load(src_curr_addr)
        })));
        // Store exp(x) at the destination cell
        b2->Insert(MakeF32Store(dst_curr_addr, MakeLocalGet(cell_exp)));
        // Add exp(x) to the local storing the total
        b2->Insert(GenerateCompoundAssignment(total_exp, Opcode::F32Add, MakeLocalGet(cell_exp)));
      }));

      // Now that we have all the exp(x[i]) for a column
      // in the destination matrix, divide them by
      // SUM(exp(e[i]))
      b1->Insert(GenerateRangeLoop(f.Label(), row, 0, height_bytes, width_bytes, {}, [&](BlockBody* b2) {
        // Compute dst current cell address and cache it in a local
        b2->Insert(MakeLocalSet(cell_addr, MakeBinary(Opcode::I32Add,
                                                      MakeLocalGet(row), MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeLocalGet(dst_begin)))));
        // Divide value by SUM(exp(x[i]))
        b2->Insert(MakeF32Store(MakeLocalGet(cell_addr),
                                MakeBinary(Opcode::F32Div, MakeF32Load(MakeLocalGet(cell_addr)),
                                           MakeLocalGet(total_exp))));
      }));
    }));
  });
}

} // namespace builtins
} // namespace nn
