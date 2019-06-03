#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/arch/model.h>
#include <src/wasmpp/wasm-instructions-gen.h>
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
  // ! Note:  `Y` is matrix of true labels, `Y_Hat` is matrix of predicted values
  // -  f(Y, Y_Hat) = (1/nm) * ((Y_Hat - Y)^2)
  // - df(Y, Y_Hat) = Y_Hat - Y
  mean_squared_error_.cost = module_manager->MakeFunction(nullptr,
      {{Type::I32, Type::I32, Type::I32, Type::I32}, {Type::F32}}, {Type::I32, Type::F32, Type::F32},
          [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto y_begin = params[0];
    auto y_hat_begin = params[1];
    auto rows = params[2];
    auto cols = params[3];

    auto y_hat_end = locals[0];
    auto tmp_res = locals[1];
    auto cost_val = locals[2];

    // Compute end of y_hat
    auto y_hat_end_rel_addr = MakeBinary(Opcode::I32Shl, MakeBinary(Opcode::I32Mul, MakeLocalGet(rows), MakeLocalGet(cols)),
        MakeI32Const(TypeShiftLeft(Type::F32)));
    f.Insert(MakeLocalSet(y_hat_end, MakeBinary(Opcode::I32Add, MakeLocalGet(y_hat_begin), y_hat_end_rel_addr)));
    // Element wise iteration
    f.Insert(GenerateDoWhileLoop(f.Label(), y_hat_begin, y_hat_end, TypeSize(Type::F32), {}, [&](BlockBody* b){
      auto sub = MakeBinary(Opcode::F32Sub, MakeF32Load(MakeLocalGet(y_hat_begin)), MakeF32Load(MakeLocalGet(y_begin)));
      b->Insert(MakeLocalSet(tmp_res, sub));
      auto pow = MakeBinary(Opcode::F32Mul, MakeLocalGet(tmp_res), MakeLocalGet(tmp_res));
      b->Insert(GenerateCompoundAssignment(cost_val, Opcode::F32Add, pow));
      // Increment address
      b->Insert(GenerateCompoundAssignment(y_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
    }));
    // Multiply cost by 1/(rows*cols)
    auto scalar = MakeBinary(Opcode::F32Div,
        MakeF32Const(1.0f),
        MakeBinary(Opcode::F32Mul,
                   MakeUnary(Opcode::F32ConvertI32U, MakeLocalGet(rows)),
                   MakeUnary(Opcode::F32ConvertI32U, MakeLocalGet(cols))));
    f.Insert(GenerateCompoundAssignment(cost_val, Opcode::F32Mul, scalar));
    f.Insert(MakeLocalGet(cost_val));
  });

  mean_squared_error_.loss = module_manager->MakeFunction(nullptr,
      {{Type::I32, Type::I32, Type::I32, Type::I32, Type::I32}, {}}, {Type::I32},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto y_begin = params[0];
    auto y_hat_begin = params[1];
    auto dst_begin = params[2];
    auto rows = params[3];
    auto cols = params[4];

    auto dst_end = locals[0];

    // Compute end of dst
    auto dst_end_rel_addr = MakeBinary(Opcode::I32Shl, MakeBinary(Opcode::I32Mul, MakeLocalGet(rows), MakeLocalGet(cols)),
        MakeI32Const(TypeShiftLeft(Type::F32)));
    f.Insert(MakeLocalSet(dst_end, MakeBinary(Opcode::I32Add, MakeLocalGet(dst_begin), dst_end_rel_addr)));
    // Element wise iteration
    f.Insert(GenerateDoWhileLoop(f.Label(), dst_begin, dst_end, TypeSize(Type::F32), {}, [&](BlockBody* b){
      auto sub = MakeBinary(Opcode::F32Sub, MakeF32Load(MakeLocalGet(y_hat_begin)), MakeF32Load(MakeLocalGet(y_begin)));
      b->Insert(MakeF32Store(MakeLocalGet(dst_begin), sub));
      // Increment addresses
      b->Insert(GenerateCompoundAssignment(y_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
      b->Insert(GenerateCompoundAssignment(y_hat_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
    }));
  });

  // Cross-Entropy function
  // ! Note:  `y` is true label, `yHat` is prediction value
  // -  f(y, yHat) = y == 1 ? -log(yHat) : -log(1 - yHat)
  // - df(y, yHat) = -(y / yHat) + ((1 - y) / (1 - yHat))
  cross_entropy_.cost = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto cond = MakeBinary(Opcode::F32Eq, MakeLocalGet(params[0]), MakeF32Const(1));
    f.Insert(MakeIf(f.Label(), cond, {{},{Type::F32}}, [&](BlockBody true_block, Var label){
      true_block.Insert(MakeUnary(Opcode::F32Neg, MakeCall(model->Builtins().math.Log(), {MakeLocalGet(params[1])})));
    }, [&](BlockBody false_block){
      auto log = MakeCall(model->Builtins().math.Log(), {MakeBinary(Opcode::F32Sub, MakeF32Const(1), MakeLocalGet(params[1]))});
      false_block.Insert(MakeUnary(Opcode::F32Neg, log));
    }));
  });
  cross_entropy_.loss = module_manager->MakeFunction(nullptr, {{Type::F32, Type::F32}, {Type::F32}}, {},
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
