#include <src/apps/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;

#define MATRIX_CHECK(x) \
  ERROR_UNLESS((x) != nullptr, #x " cannot be null"); \
  ERROR_UNLESS((x)->Shape().size() == 2, #x " is expected to be a 2D matrix");

wabt::ExprList* MatrixDot(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 6);

  auto rhs_col = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];

  uint32_t type_size = TypeSize(Type::F64);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_width_bytes, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager, rhs_col, 0, rhs_width_bytes, type_size, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF64Const(0)));
      b2->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs->Memory()->Begin())));
      b2->Insert(GenerateRangeLoop(label_manager, lhs_col_rhs_rows, 0, rhs_height_bytes, type_size, [&](BlockBody* bZ) {
        auto lhs_cell = MakeF64Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        auto rhs_cell = MakeF64Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(rhs_col)));
        bZ->Insert(GenerateCompoundAssignment(res_cell, Opcode::F64Add, MakeBinary(Opcode::F64Mul, lhs_cell, rhs_cell)));
        bZ->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
      }));
      auto dst_cell_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col));
      b2->Insert(MakeF64Store(dst_cell_addr, MakeLocalGet(res_cell)));
    }));
    b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
  }));
  return e;
}

wabt::ExprList* MatrixDotRT(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals) {
  assert(0);
}

wabt::ExprList* ElementWiseOperation(wabt::Opcode op, LabelManager* label_manager, ds::NDArray* lhs,
                                           ds::NDArray* rhs, ds::NDArray* dst, std::vector<wabt::Var> locals) {
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
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto lhs_cell = MakeF64Load(lhs_addr);
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_cell = MakeF64Load(rhs_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeBinary(op, lhs_cell, rhs_cell)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixAddition(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                               std::vector<wabt::Var> locals) {
  return ElementWiseOperation(Opcode::F64Add, label_manager, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixMultiplication(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                                     ds::NDArray* dst, std::vector<wabt::Var> locals) {
  return ElementWiseOperation(Opcode::F64Mul, label_manager, lhs, rhs, dst, locals);
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
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    auto src_cell = MakeF64Load(src_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeBinary(Opcode::F64Mul, src_cell, scalar)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixActivation(LabelManager* label_manager, ds::NDArray* src, builtins::ActivationFunction func,
                                 ds::NDArray* dst, std::vector<Var> locals, bool prime) {
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
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    auto src_cell = MakeF64Load(src_addr);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeCall(prime ? func.derivative : func.function, {src_cell})));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixLoss(wasmpp::LabelManager* label_manager, ds::NDArray* prediction, ds::NDArray* target,
                           builtins::LossFunction func, ds::NDArray* dst, std::vector<wabt::Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(prediction);
  MATRIX_CHECK(target);
  MATRIX_CHECK(dst);
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

wabt::ExprList* MatrixCopy(wasmpp::LabelManager* label_manager, ds::NDArray* src, ds::NDArray* dst,
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
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF64Store(dst_addr, MakeF64Load(src_addr)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

wabt::ExprList* MatrixBiasBroadcast(wasmpp::LabelManager* label_manager, ds::NDArray* bias, std::vector<Var> locals) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  MATRIX_CHECK(bias);
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, bias->Shape()[0] * bias->Shape()[1], 1, [&](BlockBody* b) {
    auto row_bytes = bias->Shape()[0] * TypeSize(Type::F64);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(bias->Memory()->Begin()), MakeLocalGet(addr));
    auto src_rel_addr = MakeBinary(Opcode::I32Mul,
        MakeBinary(Opcode::I32DivU, MakeLocalGet(addr), MakeI32Const(row_bytes)), MakeI32Const(row_bytes));
    auto src_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(bias->Memory()->Begin()), src_rel_addr);
    b->Insert(MakeF64Store(dst_addr, MakeF64Load(src_abs_addr)));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(Type::F64)))));
  });
  Merge(e, loopRC);
  return e;
}

} // namespace snippet
} // namespace nn
