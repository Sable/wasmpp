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
  // - cost(Y, Y_Hat, rows, cols)      : return (1/(|rows|*|cols|)) * SUM((Y_Hat - Y)^2)
  // - loss(Y, Y_Hat, DST, rows, cols) : DST = Y_Hat - Y
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

  // Sigmoid Cross-Entropy function
  // ! Note:  `Y` is matrix of true labels, `Y_Hat` is matrix of predicted values
  // - cost(Y, Y_Hat, rows, cols)      : - 1/|cols| * SUM(Y * log(Y_HAT))
  // - loss(Y, Y_Hat, DST, rows, cols) : DST =  ((1 - Y) / (1 - Y_Hat)) - (Y / Y_Hat)
  sigmoid_cross_entropy_.cost = module_manager->MakeFunction(nullptr,
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
      auto mul = MakeBinary(Opcode::F32Mul, MakeF32Load(MakeLocalGet(y_begin)),
          MakeCall(model->Builtins().math.Log(), {MakeF32Load(MakeLocalGet(y_hat_begin))}));
      b->Insert(GenerateCompoundAssignment(cost_val, Opcode::F32Add, mul));
      // Increment address
      b->Insert(GenerateCompoundAssignment(y_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
    }));
    // Multiply cost by -1/cols
    auto scalar = MakeBinary(Opcode::F32Div, MakeF32Const(1.0f),MakeUnary(Opcode::F32ConvertI32U, MakeLocalGet(cols)));
    f.Insert(GenerateCompoundAssignment(cost_val, Opcode::F32Mul, MakeUnary(Opcode::F32Neg, scalar)));
    f.Insert(MakeLocalGet(cost_val));
  });
  sigmoid_cross_entropy_.loss = module_manager->MakeFunction(nullptr,
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
      auto left_nom = MakeBinary(Opcode::F32Sub, MakeF32Const(1.0f), MakeF32Load(MakeLocalGet(y_begin)));
      auto left_den = MakeBinary(Opcode::F32Sub, MakeF32Const(1.0f), MakeF32Load(MakeLocalGet(y_hat_begin)));
      auto left_div = MakeBinary(Opcode::F32Div_Opcode, left_nom, left_den);
      auto right_div = MakeBinary(Opcode::F32Div, MakeF32Load(MakeLocalGet(y_begin)),
                                  MakeF32Load(MakeLocalGet(y_hat_begin)));
      b->Insert(MakeF32Store(MakeLocalGet(dst_begin), MakeBinary(Opcode::F32Sub, left_div, right_div)));
      // Increment addresses
      b->Insert(GenerateCompoundAssignment(y_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
      b->Insert(GenerateCompoundAssignment(y_hat_begin, Opcode::I32Add, MakeI32Const(TypeSize(Type::F32))));
    }));
  });
}

} // namespace builtins
} // namespace nn
