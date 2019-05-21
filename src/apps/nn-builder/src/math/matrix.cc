#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace math {

using namespace wasmpp;
using namespace wabt;

template<Type type>
exprs_sptr Multiply2DArrays(LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs, ds::NDArray dst,
                            std::vector<wabt::Var> locals) {
  assert(label_manager != nullptr);
  assert(lhs.Shape().size() == 2);
  assert(rhs.Shape().size() == 2);
  assert(dst.Shape().size() == 2);
  assert(lhs.Shape()[1] == rhs.Shape()[0]);
  assert(dst.Shape()[0] == lhs.Shape()[0]);
  assert(dst.Shape()[1] == rhs.Shape()[1]);
  assert(locals.size() == 7);
  auto row = locals[0];
  auto col = locals[1];
  auto col_row = locals[2];
  auto res_cell = locals[3];
  auto row_n = locals[4];
  auto row_p = locals[5];
  auto col_row_p = locals[6];

  // Type based variables
  auto load_func = MakeI32Load;
  auto store_func = MakeI32Store;
  auto op_add = Opcode::I32Add;
  auto op_mul = Opcode::I32Mul;
  exprs_sptr reset_res_cell = nullptr;
  uint32_t shift = 2;

  switch (type) {
    case Type::I32:
      // default values
      reset_res_cell = MakeI32Const(0);
      break;
    case Type::I64:
      load_func = MakeI64Load;
      store_func = MakeI64Store;
      op_add = Opcode::I64Add;
      op_mul = Opcode::I64Mul;
      reset_res_cell = MakeI64Const(0);
      shift = 3;
      break;
    case Type::F32:
      load_func = MakeF32Load;
      store_func = MakeF32Store;
      op_add = Opcode::F32Add;
      op_mul = Opcode::F32Mul;
      reset_res_cell = MakeF32Const(0);
      shift = 2;
      break;
    case Type::F64:
      load_func = MakeF64Load;
      store_func = MakeF64Store;
      op_add = Opcode::F64Add;
      op_mul = Opcode::F64Mul;
      reset_res_cell = MakeF64Const(0);
      shift = 3;
      break;
    default:
      assert(!"Matrix multiplication type not supported");
  }

  //        n            p
  //   +--------+   +--------+
  // m |        | n |        |
  //   |        |   |        |
  //   +--------+   +--------+
  exprs_sptr e = CreateExprList();
  Merge(e, MakeLocalSet(row_n, MakeI32Const(0)));
  Merge(e, MakeLocalSet(row_p, MakeI32Const(0)));
  auto loopX = GenerateRangeLoop(label_manager, row, 0, lhs.Shape()[0], 1, [&](BlockBody* bX) {
    auto loopY = GenerateRangeLoop(label_manager, col, 0, rhs.Shape()[1], 1, [&](BlockBody* bY) {
      bY->Insert(MakeLocalSet(res_cell, reset_res_cell));
      bY->Insert(MakeLocalSet(col_row_p, MakeI32Const(0)));
      auto loopZ = GenerateRangeLoop(label_manager, col_row, 0, rhs.Shape()[0], 1, [&](BlockBody* bZ) {
        auto lhs_cell_rel_addr = MakeBinary(Opcode::I32Shl,
            MakeBinary(Opcode::I32Add, MakeLocalGet(row_n), MakeLocalGet(col_row)), MakeI32Const(shift));
        auto lhs_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs.GetLinearIndex({0,0})), lhs_cell_rel_addr);
        auto lhs_cell = load_func(lhs_cell_abs_addr, wabt::WABT_USE_NATURAL_ALIGNMENT, 0);

        auto rhs_cell_rel_addr = MakeBinary(Opcode::I32Shl,
            MakeBinary(Opcode::I32Add, MakeLocalGet(col_row_p), MakeLocalGet(col)), MakeI32Const(shift));
        auto rhs_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs.GetLinearIndex({0,0})), rhs_cell_rel_addr);
        auto rhs_cell = load_func(rhs_cell_abs_addr, wabt::WABT_USE_NATURAL_ALIGNMENT, 0);

        auto mul_cells = MakeBinary(op_mul, lhs_cell, rhs_cell);
        auto update_res_cell = MakeLocalSet(res_cell, MakeBinary(op_add, MakeLocalGet(res_cell), mul_cells));
        bZ->Insert(update_res_cell);

        auto acc_col_row_p = MakeBinary(Opcode::I32Add, MakeLocalGet(col_row_p), MakeI32Const(rhs.Shape()[1]));
        bZ->Insert(MakeLocalSet(col_row_p, acc_col_row_p));
      });
      bY->Insert(loopZ);

      auto dst_cell_rel_addr = MakeBinary(Opcode::I32Shl,
          MakeBinary(Opcode::I32Add, MakeLocalGet(row_p), MakeLocalGet(col)), MakeI32Const(shift));
      auto dst_cell_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst.GetLinearIndex({0,0})), dst_cell_rel_addr);
      auto update_dst_cell = store_func(dst_cell_abs_addr, MakeLocalGet(res_cell), wabt::WABT_USE_NATURAL_ALIGNMENT, 0);
      bY->Insert(update_dst_cell);;
    });
    bX->Insert(loopY);
    auto acc_row_n = MakeBinary(Opcode::I32Add, MakeLocalGet(row_n), MakeI32Const(rhs.Shape()[0]));
    bX->Insert(MakeLocalSet(row_n, acc_row_n));
    auto acc_row_p = MakeBinary(Opcode::I32Add, MakeLocalGet(row_p), MakeI32Const(rhs.Shape()[1]));
    bX->Insert(MakeLocalSet(row_p, acc_row_p));
  });
  Merge(e, loopX);
  return e;
}

#define EXPLICIT_INSTANTIATION(t) \
template exprs_sptr Multiply2DArrays<t>(LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs,  \
    ds::NDArray dst, std::vector<Var> locals);
EXPLICIT_INSTANTIATION(Type::I32)
EXPLICIT_INSTANTIATION(Type::I64)
EXPLICIT_INSTANTIATION(Type::F32)
EXPLICIT_INSTANTIATION(Type::F64)
#undef EXPLICIT_INSTANTIATION


template<Type type>
exprs_sptr Add2DArrays(LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs, ds::NDArray dst,
                          std::vector<wabt::Var> locals) {
  assert(label_manager != nullptr);
  assert(lhs.Shape().size() == 2);
  assert(rhs.Shape().size() == 2);
  assert(dst.Shape().size() == 2);
  assert(lhs.Shape()[0] == rhs.Shape()[0]);
  assert(lhs.Shape()[1] == rhs.Shape()[1]);
  assert(rhs.Shape()[0] == dst.Shape()[0]);
  assert(rhs.Shape()[1] == dst.Shape()[1]);
  assert(locals.size() == 2);

  auto row_col = locals[0];
  auto addr = locals[1];

  // Type based variables
  auto store_func = MakeI32Store;
  auto load_func = MakeI32Load;
  auto op_add = Opcode::I32Add;
  switch (type) {
    case Type::I32:
      // default values
      break;
    case Type::I64:
      store_func = MakeI64Store;
      load_func = MakeI64Load;
      op_add = Opcode::I64Add;
      break;
    case Type::F32:
      store_func = MakeF32Store;
      load_func = MakeF32Load;
      op_add = Opcode::F32Add;
      break;
    case Type::F64:
      store_func = MakeF64Store;
      load_func = MakeF64Load;
      op_add = Opcode::F64Add;
      break;
    default:
      assert(!"Matrix addition type not supported");
  }

  exprs_sptr e = CreateExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  auto loopRC = GenerateRangeLoop(label_manager, row_col, 0, lhs.Shape()[0] * lhs.Shape()[1], 1, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs.GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto lhs_cell = load_func(lhs_addr, WABT_USE_NATURAL_ALIGNMENT, 0);
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs.GetLinearIndex({0, 0})), MakeLocalGet(addr));
    auto rhs_cell = load_func(rhs_addr, WABT_USE_NATURAL_ALIGNMENT, 0);
    auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst.GetLinearIndex({0, 0})), MakeLocalGet(addr));
    b->Insert(store_func(dst_addr, MakeBinary(op_add, lhs_cell, rhs_cell), WABT_USE_NATURAL_ALIGNMENT, 0));
    b->Insert(MakeLocalSet(addr, MakeBinary(Opcode::I32Add, MakeLocalGet(addr), MakeI32Const(TypeSize(type)))));
  });
  Merge(e, loopRC);
  return e;
}

#define EXPLICIT_INSTANTIATION(t) \
template exprs_sptr Add2DArrays<t>(LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs,  \
    ds::NDArray dst, std::vector<Var> locals);
EXPLICIT_INSTANTIATION(Type::I32)
EXPLICIT_INSTANTIATION(Type::I64)
EXPLICIT_INSTANTIATION(Type::F32)
EXPLICIT_INSTANTIATION(Type::F64)
#undef EXPLICIT_INSTANTIATION


} // namespace math
} // namespace nn
