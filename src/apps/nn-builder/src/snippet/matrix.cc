#include <src/apps/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;

#define MATRIX_CHECK(x) \
  ERROR_UNLESS(x != nullptr, #x " cannot be null"); \
  ERROR_UNLESS(x->Shape().size() == 2, #x " is expected to be a 2D matrix");

wabt::ExprList* MatrixDot(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 7);

  auto row = locals[0];
  auto col = locals[1];
  auto col_row = locals[2];
  auto row_n = locals[3];
  auto row_p = locals[4];
  auto col_row_p = locals[5];
  auto res_cell = locals[6];

  wabt::ExprList* e = new wabt::ExprList();
  uint32_t shift_f64 = 3; // 2^3
  Merge(e, MakeLocalSet(row_n, MakeI32Const(0)));
  Merge(e, MakeLocalSet(row_p, MakeI32Const(0)));
  auto loopX = GenerateRangeLoop(label_manager, row, 0, lhs->Shape()[0], 1, [&](BlockBody* bX) {
    auto loopY = GenerateRangeLoop(label_manager, col, 0, rhs->Shape()[1], 1, [&](BlockBody* bY) {
      bY->Insert(MakeLocalSet(res_cell, MakeF64Const(0)));
      bY->Insert(MakeLocalSet(col_row_p, MakeI32Const(0)));
      auto loopZ = GenerateRangeLoop(label_manager, col_row, 0, rhs->Shape()[0], 1, [&](BlockBody* bZ) {
        // LHS cell
        auto lhs_cell_rel_addr = MakeBinary(Opcode::I32Shl,
            MakeBinary(Opcode::I32Add, MakeLocalGet(row_n), MakeLocalGet(col_row)), MakeI32Const(shift_f64));
        auto lhs_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->GetLinearIndex({0,0})), lhs_cell_rel_addr);
        auto lhs_cell = MakeF64Load(lhs_cell_abs_addr);
        // RHS cell
        auto rhs_cell_rel_addr = MakeBinary(Opcode::I32Shl,
            MakeBinary(Opcode::I32Add, MakeLocalGet(col_row_p), MakeLocalGet(col)), MakeI32Const(shift_f64));
        auto rhs_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->GetLinearIndex({0,0})), rhs_cell_rel_addr);
        auto rhs_cell = MakeF64Load(rhs_cell_abs_addr);
        // ACC local
        auto mul_cells = MakeBinary(Opcode::F64Mul, lhs_cell, rhs_cell);
        auto update_res_cell = MakeLocalSet(res_cell, MakeBinary(Opcode::F64Add, MakeLocalGet(res_cell), mul_cells));
        bZ->Insert(update_res_cell);

        auto acc_col_row_p = MakeBinary(Opcode::I32Add, MakeLocalGet(col_row_p), MakeI32Const(rhs->Shape()[1]));
        bZ->Insert(MakeLocalSet(col_row_p, acc_col_row_p));
      });
      bY->Insert(loopZ);
      // DST cell
      auto dst_cell_rel_addr = MakeBinary(Opcode::I32Shl,
          MakeBinary(Opcode::I32Add, MakeLocalGet(row_p), MakeLocalGet(col)), MakeI32Const(shift_f64));
      auto dst_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->GetLinearIndex({0,0})), dst_cell_rel_addr);
      auto update_dst_cell = MakeF64Store(dst_cell_abs_addr, MakeLocalGet(res_cell));
      bY->Insert(update_dst_cell);;
    });
    bX->Insert(loopY);
    auto acc_row_n = MakeBinary(Opcode::I32Add, MakeLocalGet(row_n), MakeI32Const(rhs->Shape()[0]));
    bX->Insert(MakeLocalSet(row_n, acc_row_n));
    auto acc_row_p = MakeBinary(Opcode::I32Add, MakeLocalGet(row_p), MakeI32Const(rhs->Shape()[1]));
    bX->Insert(MakeLocalSet(row_p, acc_row_p));
  });
  Merge(e, loopX);
  return e;
}

wabt::ExprList* MatrixAddition(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                               std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape() == rhs->Shape(), "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(rhs->Shape() == dst->Shape(), "rhs and dst matrices are not compatible");
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, lhs->Shape()[0] * lhs->Shape()[1], 1, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto lhs_cell = MakeF64Load(lhs_addr);
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto rhs_cell = MakeF64Load(rhs_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeBinary(Opcode::F64Add, lhs_cell, rhs_cell)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixScalar(LabelManager* label_manager, ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                             std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(src->Shape() == dst->Shape(), "src and dst matrices are not compatible");
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, src->Shape()[0] * src->Shape()[1], 1, [&](BlockBody* b) {
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto src_cell = MakeF64Load(src_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeBinary(Opcode::F64Mul, src_cell, scalar)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixActivation(LabelManager* label_manager, ds::NDArray* src, builtins::ActivationFunction func,
                                 ds::NDArray* dst, std::vector<Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(src->Shape() == dst->Shape(), "src and dst matrices are not compatible");
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, src->Shape()[0] * src->Shape()[1], 1, [&](BlockBody* b) {
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto src_cell = MakeF64Load(src_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->GetLinearIndex({0, 0})), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeCall(func.function, {src_cell})));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixLoss(wasmpp::LabelManager* label_manager, ds::NDArray* prediction, ds::NDArray* target,
                           builtins::LossFunction func, ds::NDArray* dst, std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(prediction)
  MATRIX_CHECK(target)
  MATRIX_CHECK(dst)
  ERROR_UNLESS(prediction->Shape() == target->Shape(), "prediction and target matrices are not compatible");
  ERROR_UNLESS(target->Shape() == dst->Shape(), "prediction and dst matrices are not compatible");
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, dst->Shape()[0] * dst->Shape()[1], 1, [&](BlockBody* b) {
    auto pre_addr = MakeBinary(Opcode::I32Add, MakeI32Const(prediction->Memory()->Begin()), MakeLocalGet(addr));
    auto pre_cell = MakeF64Load(pre_addr);
    auto tar_addr = MakeBinary(Opcode::I32Add, MakeI32Const(target->Memory()->Begin()), MakeLocalGet(addr));
    auto tar_cell = MakeF64Load(tar_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeCall(func.derivative, {tar_cell, pre_cell})));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

} // namespace snippet
} // namespace nn
