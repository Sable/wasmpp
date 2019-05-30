#include <src/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;

#define MATRIX_CHECK(x) \
  ERROR_UNLESS((x) != nullptr, #x " cannot be null"); \
  ERROR_UNLESS((x)->Shape().size() == 2, #x " is expected to be a 2D matrix");

#define MATRIX_SAME_SHAPE(x, y) \
  ERROR_UNLESS((x)->Shape() == (y)->Shape(), #x " and " #y " matrices are not compatible");

#define LABEL_CHECK(l) \
  ERROR_UNLESS((l) != nullptr, "label manager cannot be null");

wabt::ExprList* MatrixDot(LabelManager* label_manager, ds::NDArray* lhs, Mat rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 6);

  auto rhs_col = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager, rhs_col, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      if(rhs.HasBeginVar()) {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
      } else {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
      }
      b2->Insert(GenerateRangeLoop(label_manager, lhs_col_rhs_rows, 0, rhs_height_bytes, type_size, {}, [&](BlockBody* b3) {
        auto lhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(rhs_col)));
        b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));
        b3->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
      }));
      auto dst_cell_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col));
      b2->Insert(MakeF32Store(dst_cell_addr, MakeLocalGet(res_cell)));
    }));
    b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
  }));
  return e;
}

wabt::ExprList* MatrixDotLT(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                            std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[0] == rhs->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[1], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 6);

  auto lhs_col = locals[0];
  auto rhs_col = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_width_bytes = rhs->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_col, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager, rhs_col, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      b2->Insert(MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
      b2->Insert(GenerateRangeLoop(label_manager, rhs_row_offset, rhs->Memory()->Begin(), rhs->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b3) {
        auto lhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col)));
        auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(rhs_col)));
        b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));
        b3->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
      }));
      auto dst_cell_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col));
      b2->Insert(MakeF32Store(dst_cell_addr, MakeLocalGet(res_cell)));
    }));
    b1->Insert(GenerateCompoundAssignment(lhs_col, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixDotRT(LabelManager* label_manager, ds::NDArray* lhs, Mat rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[1], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[0], "dst and rhs matrices are not compatible");
  assert(locals.size() == 6);

  auto rhs_rows = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_height_bytes, {}, [&](BlockBody* b1) {
    if(rhs.HasBeginVar()) {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
    } else {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
    }
    b1->Insert(GenerateRangeLoop(label_manager, rhs_rows, 0, rhs_height_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      b2->Insert(GenerateRangeLoop(label_manager, lhs_col_rhs_rows, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b3) {
        auto lhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));
      }));
      auto dst_cell_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_rows));
      b2->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
      b2->Insert(MakeF32Store(dst_cell_addr, MakeLocalGet(res_cell)));
    }));
    b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
  }));
  return e;
}

wabt::ExprList* ElementWiseBinaryOperation(wabt::Opcode op, LabelManager* label_manager, ds::NDArray* lhs,
                                           ds::NDArray* rhs, ds::NDArray* dst, std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(lhs, rhs);
  MATRIX_SAME_SHAPE(rhs, dst);
  assert(locals.size() == 2);

  auto dst_addr = locals[0];
  auto addr = locals[1];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(op, MakeF32Load(lhs_addr), MakeF32Load(rhs_addr))));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixAddition(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                               std::vector<wabt::Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Add, label_manager, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixSubtraction(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                  std::vector<wabt::Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Sub, label_manager, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixMultiplication(LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                                     ds::NDArray* dst, std::vector<wabt::Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Mul, label_manager, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixScalar(LabelManager* label_manager, ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                             std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(src, dst);
  assert(locals.size() == 2);

  auto dst_addr = locals[0];
  auto addr = locals[1];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(Opcode::F32Mul, MakeF32Load(src_addr), scalar)));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixMean(wasmpp::LabelManager* label_manager, ds::NDArray* src,
                           std::vector<wabt::Var> locals, wabt::Var result) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(src);
  assert(locals.size() == 1);

  auto src_addr = locals[0];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(result, MakeF32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, src_addr, src->Memory()->Begin(), src->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    b->Insert(GenerateCompoundAssignment(result, Opcode::F32Add, MakeF32Load(MakeLocalGet(src_addr))));
  }));
  Merge(e, MakeLocalSet(result, MakeBinary(Opcode::F32Div, MakeLocalGet(result),
                                           MakeF32Const(src->Shape()[0]*src->Shape()[1]))));
  return e;
}

wabt::ExprList* ElementWiseFunction(LabelManager* label_manager, std::vector<Mat> args, Var func, ds::NDArray* dst,
                                    std::vector<Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(dst);
  for(auto arg : args) {
    MATRIX_CHECK(arg.Array());
    MATRIX_SAME_SHAPE(arg.Array(), dst);
  }
  assert(locals.size() == 2);

  auto dst_addr = locals[0];
  auto addr = locals[1];

  std::vector<wabt::ExprList*> args_expr;
  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {},
                             [&](BlockBody* b) {
    for(auto arg : args) {
      ExprList* arg_addr = nullptr;
      if(arg.HasBeginVar()) {
        arg_addr = MakeLocalGet(arg.Var());
      } else {
        arg_addr = MakeI32Const(arg.Array()->Memory()->Begin());
      }
      args_expr.push_back(MakeF32Load(MakeBinary(Opcode::I32Add, arg_addr, MakeLocalGet(addr))));
    }
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeCall(func, args_expr)));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixActivation(LabelManager* label_manager, Mat src, builtins::ActivationFunction func,
                                 ds::NDArray* dst, std::vector<Var> locals, bool prime) {
  return ElementWiseFunction(label_manager, {src}, prime ? func.derivative : func.function, dst, locals);
}

wabt::ExprList* MatrixLoss(wasmpp::LabelManager* label_manager, Mat target, Mat prediction,
                           builtins::LossFunction func, ds::NDArray* dst, std::vector<wabt::Var> locals, bool prime) {
  return ElementWiseFunction(label_manager, {target, prediction}, prime ? func.derivative : func.function, dst, locals);
}

wabt::ExprList* MatrixCopy(wasmpp::LabelManager* label_manager, ds::NDArray* src, ds::NDArray* dst,
                           std::vector<wabt::Var> locals) {
  return MatrixScalar(label_manager, src, MakeF32Const(1), dst, locals);
}

// TODO Optimize this function
wabt::ExprList* MatrixColumnArgmax(wasmpp::LabelManager* label_manager, ds::NDArray* src, ds::NDArray* dst,
                                   std::vector<wabt::Var> locals) {
  LABEL_CHECK(label_manager)
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(src, dst);
  assert(locals.size() == 3);

  auto row = locals[0];
  auto col = locals[1];
  auto max = locals[2];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateRangeLoop(label_manager, col, 0, dst->Shape()[1], 1, {}, [&](BlockBody* b1) {
    b1->Insert(MakeLocalSet(max, MakeI32Const(0)));
    b1->Insert(GenerateRangeLoop(label_manager, row, 1, dst->Shape()[0], 1, {}, [&](BlockBody* b2) {
      auto max_row = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()),
                                MakeBinary(Opcode::I32Mul, MakeLocalGet(max),
                                           MakeI32Const(src->Shape()[1] * type_size)));
      auto max_index = MakeBinary(Opcode::I32Add, max_row,
                                  MakeBinary(Opcode::I32Mul, MakeLocalGet(col), MakeI32Const(type_size)));
      auto cur_row = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()),
                                MakeBinary(Opcode::I32Mul, MakeLocalGet(row),
                                           MakeI32Const(src->Shape()[1] * type_size)));
      auto cur_index = MakeBinary(Opcode::I32Add, cur_row,
                                  MakeBinary(Opcode::I32Mul, MakeLocalGet(col), MakeI32Const(type_size)));
      auto cond = MakeBinary(Opcode::F32Ge, MakeF32Load(cur_index), MakeF32Load(max_index));
      auto comp = MakeIf(label_manager, cond, {}, [&](BlockBody t, Var label) {
        t.Insert(MakeLocalSet(max, MakeLocalGet(row)));
      });
      b2->Insert(comp);
    }));

    b1->Insert(GenerateRangeLoop(label_manager, row, 0, dst->Shape()[0], 1, {}, [&](BlockBody* b2) {
      auto cond = MakeBinary(Opcode::I32Eq, MakeLocalGet(max), MakeLocalGet(row));
      auto comp = MakeIf(label_manager, cond, {}, [&](BlockBody t, Var label) {
        auto cur_row = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()),
                                  MakeBinary(Opcode::I32Mul, MakeLocalGet(row),
                                             MakeI32Const(src->Shape()[1] * type_size)));
        auto cur_index = MakeBinary(Opcode::I32Add, cur_row,
                                    MakeBinary(Opcode::I32Mul, MakeLocalGet(col), MakeI32Const(type_size)));
        t.Insert(MakeF32Store(cur_index, MakeF32Const(1)));
      }, [&](BlockBody f) {
        auto cur_row = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()),
                                  MakeBinary(Opcode::I32Mul, MakeLocalGet(row),
                                             MakeI32Const(src->Shape()[1] * type_size)));
        auto cur_index = MakeBinary(Opcode::I32Add, cur_row,
                                    MakeBinary(Opcode::I32Mul, MakeLocalGet(col), MakeI32Const(type_size)));
        f.Insert(MakeF32Store(cur_index, MakeF32Const(0)));
      });
      b2->Insert(comp);
    }));
  }));
  return e;
}

wabt::ExprList* MatrixBiasBroadcast(wasmpp::LabelManager* label_manager, ds::NDArray* bias, std::vector<Var> locals) {
  LABEL_CHECK(label_manager);
  MATRIX_CHECK(bias);
  assert(locals.size() == 2);

  auto dst_addr = locals[0];
  auto addr = locals[1];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t bias_width_bytes = bias->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager, dst_addr, bias->Memory()->Begin(), bias->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto src_rel_addr = MakeBinary(Opcode::I32Mul,
        MakeBinary(Opcode::I32DivU, MakeLocalGet(addr), MakeI32Const(bias_width_bytes)), MakeI32Const(bias_width_bytes));
    auto src_abs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(bias->Memory()->Begin()), src_rel_addr);
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeF32Load(src_abs_addr)));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

} // namespace snippet
} // namespace nn
