#include <src/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/nn-builder/src/arch/model.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;
using namespace ds;

wabt::ExprList* MatrixSnippet::MatrixDot(NDArray* lhs, RelocMat rhs, NDArray* dst, std::vector<Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 7);

  auto rhs_col = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto used_by_simd = locals[6];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_col, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      if(rhs.HasBeginVar()) {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
      } else {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
      }
      b2->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, rhs_height_bytes, type_size, {}, [&](BlockBody* b3) {
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

wabt::ExprList* MatrixSnippet::MatrixDotLT(NDArray* lhs, NDArray* rhs, NDArray* dst, std::vector<Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[0] == rhs->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[1], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs->Shape()[1], "dst and rhs matrices are not compatible");
  assert(locals.size() == 7);

  auto lhs_col = locals[0];
  auto rhs_col = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto used_by_simd = locals[6];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_width_bytes = rhs->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_col, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_col, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      b2->Insert(MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
      b2->Insert(GenerateRangeLoop(label_manager_, rhs_row_offset, rhs->Memory()->Begin(), rhs->Memory()->End(), rhs_width_bytes, {}, [&](BlockBody* b3) {
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

wabt::ExprList* MatrixSnippet::MatrixDotRT(NDArray* lhs, RelocMat rhs, NDArray* dst, std::vector<Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[1], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[0], "dst and rhs matrices are not compatible");
  assert(locals.size() == 7);

  auto rhs_rows = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto used_by_simd = locals[6];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_height_bytes, {}, [&](BlockBody* b1) {
    if(rhs.HasBeginVar()) {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
    } else {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
    }
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_rows, 0, rhs_height_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));
      b2->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, rhs_width_bytes, type_size, {}, [&](BlockBody* b3) {
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

wabt::ExprList* MatrixSnippet::ElementWiseBinaryOperation(Opcode op, NDArray* lhs, NDArray* rhs, NDArray* dst,
                                                          std::vector<Var> locals) {
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
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(op, MakeF32Load(lhs_addr), MakeF32Load(rhs_addr))));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixAddition(NDArray* lhs, NDArray* rhs, NDArray* dst, std::vector<Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Add, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixSnippet::MatrixSubtraction(NDArray* lhs, NDArray* rhs, NDArray* dst, std::vector<Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Sub, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixSnippet::MatrixMultiplication(NDArray* lhs, NDArray* rhs, NDArray* dst, std::vector<Var> locals) {
  return ElementWiseBinaryOperation(Opcode::F32Mul, lhs, rhs, dst, locals);
}

wabt::ExprList* MatrixSnippet::MatrixVectorAddition(NDArray* matrix, NDArray* vector, NDArray* dst_matrix,
                                                    std::vector<Var> locals) {
  return MatrixVectorBinaryOperation(Opcode::F32Add, matrix, vector, dst_matrix, locals);
}

wabt::ExprList* MatrixSnippet::MatrixVectorBinaryOperation(Opcode op, NDArray *matrix, NDArray *vector,
                                                           NDArray *dst_matrix, std::vector<Var> locals) {
  MATRIX_CHECK(matrix);
  VECTOR_CHECK(vector);
  MATRIX_CHECK(dst_matrix);
  MATRIX_SAME_SHAPE(matrix, dst_matrix);

  assert(locals.size() == 4);

  auto row = locals[0];
  auto col = locals[1];
  auto vec_row_offset = locals[2];
  auto addr = locals[3];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t dst_width_bytes = dst_matrix->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(vec_row_offset, MakeI32Const(vector->Memory()->Begin())));
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, row, 0, dst_matrix->Memory()->Bytes(), dst_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(GenerateRangeLoop(label_manager_, col, 0, dst_width_bytes, type_size, {}, [&](BlockBody* b2){
      auto mat_addr = MakeBinary(Opcode::I32Add, MakeI32Const(matrix->Memory()->Begin()), MakeLocalGet(addr));
      auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst_matrix->Memory()->Begin()), MakeLocalGet(addr));
      auto vec_addr = MakeLocalGet(vec_row_offset);
      auto result = MakeBinary(op, MakeF32Load(mat_addr), MakeF32Load(vec_addr));
      b2->Insert(MakeF32Store(dst_addr, result));
      b2->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
    }));
    b1->Insert(GenerateCompoundAssignment(vec_row_offset, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixScalar(NDArray* src, ExprList* scalar, NDArray* dst, std::vector<Var> locals) {
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(src, dst);
  assert(locals.size() == 3);

  auto dst_addr = locals[0];
  auto addr = locals[1];
  auto used_by_simd = locals[2];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(Opcode::F32Mul, MakeF32Load(src_addr), scalar)));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::ElementWiseFunction(std::vector<RelocMat> args, Var func, NDArray* dst,
                                                   std::vector<Var> locals) {
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
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {},
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

wabt::ExprList* MatrixSnippet::MatrixActivation(RelocMat src, builtins::ActivationFunction func, NDArray* dst,
                                                std::vector<Var> locals, bool prime) {
  return ElementWiseFunction({src}, prime ? func.derivative : func.function, dst, locals);
}

wabt::ExprList* MatrixSnippet::MatrixColumnHardmax(NDArray* src, NDArray* dst, std::vector<Var> locals) {
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(src, dst);
  assert(locals.size() == 5);

  auto row = locals[0];
  auto col = locals[1];
  auto src_max_addr = locals[2];
  auto dst_max_addr = locals[3];
  auto curr_addr = locals[4];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t width_bytes = src->Shape()[1] * type_size;
  uint32_t height_bytes = src->Shape()[0] * width_bytes;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateRangeLoop(label_manager_, col, 0, width_bytes, type_size, {}, [&](BlockBody* b1) {
    b1->Insert(MakeLocalSet(src_max_addr, MakeBinary(Opcode::I32Add, MakeI32Const(src->Begin()), MakeLocalGet(col))));
    b1->Insert(MakeLocalSet(dst_max_addr, MakeBinary(Opcode::I32Add, MakeI32Const(dst->Begin()), MakeLocalGet(col))));
    // Find max
    b1->Insert(GenerateRangeLoop(label_manager_, row, 0, height_bytes, width_bytes, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(curr_addr,
                              MakeBinary(Opcode::I32Add, MakeLocalGet(row),
                                         MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeI32Const(src->Begin())))));
      auto cond = MakeBinary(Opcode::F32Ge, MakeF32Load(MakeLocalGet(curr_addr)), MakeF32Load(MakeLocalGet(src_max_addr)));
      b2->Insert(MakeIf(label_manager_, cond, {}, [&](BlockBody t, Var label) {
        t.Insert(MakeLocalSet(src_max_addr, MakeLocalGet(curr_addr)));
        t.Insert(MakeLocalSet(dst_max_addr,
                                MakeBinary(Opcode::I32Add, MakeLocalGet(row),
                                           MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeI32Const(dst->Begin())))));
      }));
    }));

    // Place 1 in max and 0 in rest
    b1->Insert(GenerateRangeLoop(label_manager_, row, 0, height_bytes, width_bytes, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(curr_addr,
                              MakeBinary(Opcode::I32Add, MakeLocalGet(row),
                                         MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeI32Const(dst->Begin())))));
      auto cond = MakeBinary(Opcode::I32Eq, MakeLocalGet(curr_addr), MakeLocalGet(dst_max_addr));
      b2->Insert(MakeIf(label_manager_, cond, {}, [&](BlockBody t, Var label) {
        t.Insert(MakeF32Store(MakeLocalGet(curr_addr), MakeF32Const(1)));
      }, [&](BlockBody f) {
        f.Insert(MakeF32Store(MakeLocalGet(curr_addr), MakeF32Const(0)));
      }));
    }));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixHorizontalSum(nn::ds::NDArray *matrix, nn::ds::NDArray *dst_vector,
                                                   std::vector<wabt::Var> locals) {

  MATRIX_CHECK(matrix);
  VECTOR_CHECK(dst_vector);
  assert(locals.size() == 5);

  auto mat_row_offset = locals[0];
  auto col = locals[1];
  auto vec_row_offset = locals[2];
  auto res = locals[3];
  auto used_by_simd = locals[4];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t matrix_width_bytes = matrix->Shape()[1] * type_size;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(vec_row_offset, MakeI32Const(dst_vector->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, mat_row_offset, matrix->Memory()->Begin(), matrix->Memory()->End(), matrix_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(MakeLocalSet(res, MakeF32Const(0)));
    b1->Insert(GenerateRangeLoop(label_manager_, col, 0, matrix_width_bytes, type_size, {}, [&](BlockBody* b2){
      auto mat_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(mat_row_offset), MakeLocalGet(col));
      b2->Insert(GenerateCompoundAssignment(res, Opcode::F32Add, MakeF32Load(mat_addr)));
    }));
    b1->Insert(MakeF32Store(MakeLocalGet(vec_row_offset), MakeLocalGet(res)));
    b1->Insert(GenerateCompoundAssignment(vec_row_offset, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixAbsSum(nn::ds::NDArray *matrix, wabt::Var result, std::vector<wabt::Var> locals) {
  MATRIX_CHECK(matrix);

  assert(locals.size() == 1);
  auto dst_addr = locals[0];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(result, MakeF32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, matrix->Begin(), matrix->End(), type_size, {}, [&](BlockBody* b){
    b->Insert(GenerateCompoundAssignment(result, Opcode::F32Add, MakeUnary(Opcode::F32Abs, MakeF32Load(MakeLocalGet(dst_addr)))));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixSquareSum(nn::ds::NDArray *matrix, wabt::Var result, std::vector<wabt::Var> locals) {
  MATRIX_CHECK(matrix);

  assert(locals.size() == 2);
  auto dst_addr = locals[0];
  auto cache = locals[1];

  uint32_t type_size = TypeSize(Type::F32);

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(result, MakeF32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, matrix->Begin(), matrix->End(), type_size, {}, [&](BlockBody* b){
    b->Insert(MakeLocalSet(cache, MakeF32Load(MakeLocalGet(dst_addr))));
    b->Insert(GenerateCompoundAssignment(result, Opcode::F32Add, MakeBinary(Opcode::F32Mul,
                                                                            MakeLocalGet(cache), MakeLocalGet(cache))));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixAddRightScale(nn::ds::NDArray *lhs, nn::ds::NDArray *rhs, nn::ds::NDArray *dst,
                                                   float scale, std::vector<wabt::Var> locals) {
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
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(Opcode::F32Add, MakeF32Load(lhs_addr),
                                                              MakeBinary(Opcode::F32Mul,
                                                                         MakeF32Load(rhs_addr), MakeF32Const(scale)))));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixAddRightSignScale(nn::ds::NDArray *lhs, nn::ds::NDArray *rhs, nn::ds::NDArray *dst,
                                                       float scale, std::vector<wabt::Var> locals) {
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
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    // Compute right scale sign addition
    auto rhs_sign = MakeBinary(Opcode::F32Copysign, MakeF32Const(1), MakeF32Load(rhs_addr));
    auto rhs_val = MakeBinary(Opcode::F32Mul, rhs_sign, MakeF32Const(scale));
    b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(Opcode::F32Add, MakeF32Load(lhs_addr), rhs_val)));
    // Move to next element
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippetSimd::ElementWiseBinaryOperation(Opcode op, NDArray *lhs, NDArray *rhs, NDArray *dst,
                                                              std::vector<Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(lhs, rhs);
  MATRIX_SAME_SHAPE(rhs, dst);
  assert(locals.size() == 2);

  // Cannot optimize
  if(lhs->Memory()->Bytes() < WASMPP_V128_SIZE) {
    return MatrixSnippet::ElementWiseBinaryOperation(op, lhs, rhs, dst, locals);
  }

  auto dst_addr = locals[0];
  auto addr = locals[1];

  uint32_t simd_type_size = TypeSize(Type::V128);
  auto remainder = dst->Memory()->Bytes() % WASMPP_V128_SIZE;
  auto dst_simd_end = dst->Memory()->End() - remainder;

  // Use SIMD while possible
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst_simd_end, simd_type_size, {}, [&](BlockBody* b) {
    auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
    auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeV128Store(MakeLocalGet(dst_addr),
                            MakeBinary(OpcodeToSimdOpcode(op), MakeV128Load(lhs_addr), MakeV128Load(rhs_addr))));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(simd_type_size)));
  }));

  // Fallback to regular computation
  if(remainder > 0) {
    auto type_size = TypeSize(Type::F32);
    Merge(e, GenerateDoWhileLoop(label_manager_, dst_addr, dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
      auto lhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Memory()->Begin()), MakeLocalGet(addr));
      auto rhs_addr = MakeBinary(Opcode::I32Add, MakeI32Const(rhs->Memory()->Begin()), MakeLocalGet(addr));
      b->Insert(MakeF32Store(MakeLocalGet(dst_addr),
                              MakeBinary(op, MakeF32Load(lhs_addr), MakeF32Load(rhs_addr))));
      b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
    }));
  }
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixScalar(nn::ds::NDArray *src, wabt::ExprList *scalar, nn::ds::NDArray *dst,
                                                std::vector<wabt::Var> locals) {
  MATRIX_CHECK(src);
  MATRIX_CHECK(dst);
  MATRIX_SAME_SHAPE(src, dst);
  assert(locals.size() == 3);

  // Cannot optimize
  if(src->Memory()->Bytes() < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixScalar(src, scalar, dst, locals);
  }

  auto dst_addr = locals[0];
  auto addr = locals[1];
  auto scalar_val = locals[2];

  uint32_t simd_type_size = TypeSize(Type::V128);
  auto remainder = dst->Memory()->Bytes() % WASMPP_V128_SIZE;
  auto dst_simd_end = dst->Memory()->End() - remainder;

  // Use SIMD while possible
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(scalar_val, scalar));
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_addr, dst->Memory()->Begin(), dst_simd_end, simd_type_size, {}, [&](BlockBody* b) {
    auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
    b->Insert(MakeV128Store(MakeLocalGet(dst_addr),
                            MakeBinary(Opcode::F32X4Mul, MakeV128Load(src_addr),
                                       MakeUnary(Opcode::F32X4Splat, MakeLocalGet(scalar_val)))));
    b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(simd_type_size)));
  }));

  // Fallback to regular computation
  if(remainder > 0) {
    auto type_size = TypeSize(Type::F32);
    Merge(e, GenerateDoWhileLoop(label_manager_, dst_addr, dst->Memory()->End(), type_size, {}, [&](BlockBody* b) {
      auto src_addr = MakeBinary(Opcode::I32Add, MakeI32Const(src->Memory()->Begin()), MakeLocalGet(addr));
      b->Insert(MakeF32Store(MakeLocalGet(dst_addr), MakeBinary(Opcode::F32Mul, MakeF32Load(src_addr),
                                                                MakeLocalGet(scalar_val))));
      b->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
    }));
  }
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixVectorBinaryOperation(Opcode op, NDArray *matrix, NDArray *vector,
                                                               NDArray *dst_matrix, std::vector<Var> locals) {
  MATRIX_CHECK(matrix);
  VECTOR_CHECK(vector);
  MATRIX_CHECK(dst_matrix);
  MATRIX_SAME_SHAPE(matrix, dst_matrix);
  assert(locals.size() == 4);

  uint32_t simd_type_size = TypeSize(Type::V128);
  uint32_t type_size = TypeSize(Type::F32);
  uint32_t dst_width_bytes = dst_matrix->Shape()[1] * type_size;
  auto width_remainder = dst_width_bytes % WASMPP_V128_SIZE;
  auto dst_simd_width_bytes = dst_width_bytes - width_remainder;

  // Cannot optimize if matrix width bytes is too small
  if(dst_width_bytes < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixVectorBinaryOperation(op, matrix, vector, dst_matrix, locals);
  }

  auto row = locals[0];
  auto col = locals[1];
  auto vec_row_offset = locals[2];
  auto addr = locals[3];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(vec_row_offset, MakeI32Const(vector->Memory()->Begin())));
  Merge(e, MakeLocalSet(addr, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, row, 0, dst_matrix->Memory()->Bytes(), dst_width_bytes, {}, [&](BlockBody* b1) {

    // Use SIMD while possible
    b1->Insert(GenerateRangeLoop(label_manager_, col, 0, dst_simd_width_bytes, simd_type_size, {}, [&](BlockBody* b2){
      auto mat_addr = MakeBinary(Opcode::I32Add, MakeI32Const(matrix->Memory()->Begin()), MakeLocalGet(addr));
      auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst_matrix->Memory()->Begin()), MakeLocalGet(addr));
      auto vec_addr = MakeLocalGet(vec_row_offset);
      auto result = MakeBinary(OpcodeToSimdOpcode(op), MakeV128Load(mat_addr), MakeUnary(Opcode::F32X4Splat, MakeF32Load(vec_addr)));
      b2->Insert(MakeV128Store(dst_addr, result));
      b2->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(simd_type_size)));
    }));

    // Fallback to regular compuation
    if(width_remainder > 0) {
      b1->Insert(GenerateDoWhileLoop(label_manager_, col, dst_width_bytes, type_size, {}, [&](BlockBody* b2){
        auto mat_addr = MakeBinary(Opcode::I32Add, MakeI32Const(matrix->Memory()->Begin()), MakeLocalGet(addr));
        auto dst_addr = MakeBinary(Opcode::I32Add, MakeI32Const(dst_matrix->Memory()->Begin()), MakeLocalGet(addr));
        auto vec_addr = MakeLocalGet(vec_row_offset);
        auto result = MakeBinary(op, MakeF32Load(mat_addr), MakeF32Load(vec_addr));
        b2->Insert(MakeF32Store(dst_addr, result));
        b2->Insert(GenerateCompoundAssignment(addr, Opcode::I32Add, MakeI32Const(type_size)));
      }));
    }

    b1->Insert(GenerateCompoundAssignment(vec_row_offset, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixHorizontalSum(nn::ds::NDArray *matrix, nn::ds::NDArray *dst_vector,
                                                std::vector<wabt::Var> locals) {
  MATRIX_CHECK(matrix);
  VECTOR_CHECK(dst_vector);
  assert(locals.size() == 5);

  uint32_t simd_type_size = TypeSize(Type::V128);
  uint32_t type_size = TypeSize(Type::F32);
  uint32_t matrix_width_bytes = matrix->Shape()[1] * type_size;
  auto width_remainder = matrix_width_bytes % WASMPP_V128_SIZE;
  auto matrix_simd_width_bytes = matrix_width_bytes - width_remainder;

  // Cannot optimize if matrix width bytes is too small
  if(matrix_width_bytes < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixHorizontalSum(matrix, dst_vector, locals);
  }

  auto mat_row_offset = locals[0];
  auto col = locals[1];
  auto vec_row_offset = locals[2];
  auto res = locals[3];
  auto res_128 = locals[4];

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(vec_row_offset, MakeI32Const(dst_vector->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, mat_row_offset, matrix->Memory()->Begin(), matrix->Memory()->End(), matrix_width_bytes, {}, [&](BlockBody* b1) {
    b1->Insert(MakeLocalSet(res_128, MakeUnary(Opcode::F32X4Splat, MakeF32Const(0))));

    // Use SIMD while possible
    b1->Insert(GenerateRangeLoop(label_manager_, col, 0, matrix_simd_width_bytes, simd_type_size, {}, [&](BlockBody* b2){
      auto mat_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(mat_row_offset), MakeLocalGet(col));
      b2->Insert(GenerateCompoundAssignment(res_128, Opcode::F32X4Add, MakeV128Load(mat_addr)));
    }));
    b1->Insert(MakeLocalSet(res, GenerateF32X4HorizontalLTRSum(res_128)));

    // Fallback to regular computation
    if(width_remainder > 0) {
      b1->Insert(GenerateDoWhileLoop(label_manager_, col, matrix_width_bytes, type_size, {}, [&](BlockBody* b2){
        auto mat_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(mat_row_offset), MakeLocalGet(col));
        b2->Insert(GenerateCompoundAssignment(res, Opcode::F32Add, MakeF32Load(mat_addr)));
      }));
    }
    b1->Insert(MakeF32Store(MakeLocalGet(vec_row_offset), MakeLocalGet(res)));
    b1->Insert(GenerateCompoundAssignment(vec_row_offset, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixDotLT(ds::NDArray *lhs, ds::NDArray *rhs, ds::NDArray *dst,
                                               std::vector<Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs);
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[0] == rhs->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[1], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs->Shape()[1], "dst and rhs matrices are not compatible");

  assert(locals.size() == 7);
  auto lhs_col = locals[0];
  auto rhs_col = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto res_128 = locals[6];

  uint32_t simd_type_size = TypeSize(Type::V128);
  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_width_bytes = rhs->Shape()[1] * type_size;
  uint32_t width_remainder = rhs_width_bytes % WASMPP_V128_SIZE;
  uint32_t simd_width_bytes = rhs_width_bytes - width_remainder;

  // Cannot optimize if rhs width bytes is too small
  if(rhs_width_bytes < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixDotLT(lhs, rhs, dst, locals);
  }

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_col, MakeI32Const(0)));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Begin(), dst->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {

    // Use SIMD while possible
    // Loop on rhs columns in group of 4
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_col, 0, simd_width_bytes, simd_type_size, {}, [&](BlockBody* b2) {
      // Reset lhs pointer to first row and compute the correct column;
      b2->Insert(MakeLocalSet(lhs_row_offset, MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Begin()), MakeLocalGet(lhs_col))));

      // Reset result counter
      b2->Insert(MakeLocalSet(res_128, MakeUnary(Opcode::F32X4Splat, MakeF32Const(0))));

      // Loop vertically on a column group
      b2->Insert(GenerateRangeLoop(label_manager_, rhs_row_offset, rhs->Begin(), rhs->End(), rhs_width_bytes, {}, [&](BlockBody* b3){
        auto lhs_cell = MakeUnary(Opcode::F32X4Splat, MakeF32Load(MakeLocalGet(lhs_row_offset)));
        auto rhs_cell = MakeV128Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(rhs_col)));

        // Compute 4 cells at a time
        b3->Insert(GenerateCompoundAssignment(res_128, Opcode::F32X4Add, MakeBinary(Opcode::F32X4Mul, lhs_cell, rhs_cell)));

        // Move lhs pointer to next row
        b3->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
      }));

      // Store result in destination matrix
      b2->Insert(MakeV128Store(MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col)), MakeLocalGet(res_128)));
    }));

     if(width_remainder > 0) {
       // Fallback to regular computation
       b1->Insert(GenerateDoWhileLoop(label_manager_, rhs_col, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {
         // Reset lhs pointer to first row and compute the correct column;
         b2->Insert(MakeLocalSet(lhs_row_offset, MakeBinary(Opcode::I32Add, MakeI32Const(lhs->Begin()), MakeLocalGet(lhs_col))));

         // Reset result counter
         b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));

         // Loop vertically on a column
         b2->Insert(GenerateRangeLoop(label_manager_, rhs_row_offset, rhs->Begin(), rhs->End(), rhs_width_bytes, {}, [&](BlockBody* b3){
           auto lhs_cell = MakeF32Load(MakeLocalGet(lhs_row_offset));
           auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(rhs_col)));

           // Compute cell
           b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));

           // Move lhs pointer to next row
           b3->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
         }));

         // Store result in destination matrix
         b2->Insert(MakeF32Store(MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col)), MakeLocalGet(res_cell)));
       }));
     }

    // Increment the lhs column counter
    b1->Insert(GenerateCompoundAssignment(lhs_col, Opcode::I32Add, MakeI32Const(type_size)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixDotRT(nn::ds::NDArray *lhs, nn::snippet::RelocMat rhs, nn::ds::NDArray *dst,
                                               std::vector<wabt::Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[1], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[0], "dst and rhs matrices are not compatible");
  assert(locals.size() == 7);

  uint32_t simd_type_size = TypeSize(Type::V128);
  uint32_t type_size = TypeSize(Type::F32);
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;
  uint32_t width_remainder = lhs_width_bytes % WASMPP_V128_SIZE;
  uint32_t simd_width_bytes = lhs_width_bytes - width_remainder;

  auto rhs_rows = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto res_128 = locals[6];

  // Handle special case where the number of columns is 1
  // and rhs has more than 4 elements
  wabt::ExprList* e = new wabt::ExprList();
  if(lhs->Shape()[1] == 1 && rhs_height_bytes >= WASMPP_V128_SIZE) {

    uint32_t height_remainder = rhs_height_bytes % WASMPP_V128_SIZE;
    uint32_t simd_height_bytes = rhs_height_bytes - height_remainder;

    Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
    Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_height_bytes, {}, [&](BlockBody* b1) {
      // Reset rhs pointer to top row
      if(rhs.HasBeginVar()) {
        b1->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
      } else {
        b1->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
      }

      // Apply SIMD while possible
      b1->Insert(GenerateRangeLoop(label_manager_, rhs_rows, 0, simd_height_bytes, simd_type_size, {}, [&](BlockBody* b2) {
        auto lhs_op = MakeUnary(Opcode::F32X4Splat, MakeF32Load(MakeLocalGet(lhs_row_offset)));
        auto rhs_op = MakeV128Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_rows), MakeLocalGet(rhs_row_offset)));
        auto dest_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_rows));
        b2->Insert(MakeV128Store(dest_addr, MakeBinary(Opcode::F32X4Mul, lhs_op, rhs_op)));
      }));

      // Fallback to regular computation
      if(height_remainder > 0) {
        b1->Insert(GenerateDoWhileLoop(label_manager_, rhs_rows, rhs_height_bytes, type_size, {}, [&](BlockBody* b2) {
          auto lhs_op = MakeF32Load(MakeLocalGet(lhs_row_offset));
          auto rhs_op = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_rows), MakeLocalGet(rhs_row_offset)));
          auto dest_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_rows));
          b2->Insert(MakeF32Store(dest_addr, MakeBinary(Opcode::F32Mul, lhs_op, rhs_op)));
        }));
      }

      // Move lhs pointer to next row
      b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
    }));
    return e;
  }

  // Cannot optimize if matrix width bytes is too small
  if(lhs_width_bytes < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixDotRT(lhs, rhs, dst, locals);
  }

  // Optimize for large matrices
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), rhs_height_bytes, {}, [&](BlockBody* b1) {
    if(rhs.HasBeginVar()) {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
    } else {
      b1->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Memory()->Begin())));
    }
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_rows, 0, rhs_height_bytes, type_size, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(res_128, MakeUnary(Opcode::F32X4Splat, MakeF32Const(0))));

      // Use SIMD while possible
      b2->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, simd_width_bytes, simd_type_size, {}, [&](BlockBody* b3) {
        auto lhs_cell = MakeV128Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        auto rhs_cell = MakeV128Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        b3->Insert(GenerateCompoundAssignment(res_128, Opcode::F32X4Add, MakeBinary(Opcode::F32X4Mul, lhs_cell, rhs_cell)));
      }));
      b2->Insert(MakeLocalSet(res_cell, GenerateF32X4HorizontalLTRSum(res_128)));

      // Fallback to regular computation
      if(width_remainder > 0) {
        b2->Insert(GenerateDoWhileLoop(label_manager_, lhs_col_rhs_rows, rhs_width_bytes, type_size, {}, [&](BlockBody* b3) {
          auto lhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(lhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
          auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
          b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));
        }));
      }
      auto dst_cell_addr = MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_rows));
      b2->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
      b2->Insert(MakeF32Store(dst_cell_addr, MakeLocalGet(res_cell)));
    }));
    b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
  }));
  return e;
}

wabt::ExprList* MatrixSnippetSimd::MatrixDot(nn::ds::NDArray *lhs, nn::snippet::RelocMat rhs, nn::ds::NDArray *dst,
                                             std::vector<wabt::Var> locals) {
  MATRIX_CHECK(lhs);
  MATRIX_CHECK(rhs.Array());
  MATRIX_CHECK(dst);
  ERROR_UNLESS(lhs->Shape()[1] == rhs.Array()->Shape()[0], "lhs and rhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[0] == lhs->Shape()[0], "dst and lhs matrices are not compatible");
  ERROR_UNLESS(dst->Shape()[1] == rhs.Array()->Shape()[1], "dst and rhs matrices are not compatible");

  assert(locals.size() == 7);
  auto rhs_col = locals[0];
  auto lhs_col_rhs_rows = locals[1];
  auto lhs_row_offset = locals[2];
  auto rhs_row_offset = locals[3];
  auto dst_row_offset = locals[4];
  auto res_cell = locals[5];
  auto res_128 = locals[6];

  uint32_t simd_type_size = TypeSize(Type::V128);
  uint32_t type_size = TypeSize(Type::F32);
  uint32_t rhs_height_bytes = rhs.Array()->Shape()[0] * type_size;
  uint32_t rhs_width_bytes = rhs.Array()->Shape()[1] * type_size;
  uint32_t lhs_width_bytes = lhs->Shape()[1] * type_size;
  uint32_t width_remainder = rhs_width_bytes % WASMPP_V128_SIZE;
  uint32_t simd_width_bytes = rhs_width_bytes - width_remainder;

  // Handle special case where rhs is a vector
  // and has more than 4 rows
  if(rhs.Array()->Shape()[1] == 1 && rhs_height_bytes >= WASMPP_V128_SIZE) {
    uint32_t height_remainder = rhs_height_bytes % WASMPP_V128_SIZE;
    uint32_t simd_height_bytes = rhs_height_bytes - height_remainder;

    wabt::ExprList* e = new wabt::ExprList();
    Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Memory()->Begin())));
    Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Memory()->Begin(), dst->Memory()->End(), type_size, {}, [&](BlockBody* b1) {
      // Reset result local
      b1->Insert(MakeLocalSet(res_128, MakeUnary(Opcode::F32X4Splat, MakeF32Const(0))));

      // Reset rhs pointer
      if(rhs.HasBeginVar()) {
        b1->Insert(MakeLocalSet(rhs_row_offset, MakeLocalGet(rhs.Var())));
      } else {
        b1->Insert(MakeLocalSet(rhs_row_offset, MakeI32Const(rhs.Array()->Begin())));
      }

      // Use SIMD while possible
      b1->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, simd_height_bytes, simd_type_size, {}, [&](BlockBody* b2) {
        auto lhs_cell = MakeV128Load(MakeLocalGet(lhs_row_offset));
        auto rhs_cell = MakeV128Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
        b2->Insert(GenerateCompoundAssignment(res_128, Opcode::F32X4Add, MakeBinary(Opcode::F32X4Mul, lhs_cell, rhs_cell)));
        b2->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(simd_type_size)));
      }));

      // Compute result
      b1->Insert(MakeLocalSet(res_cell, GenerateF32X4HorizontalLTRSum(res_128)));

      // Fallback to regular computation
      if(height_remainder > 0) {
        b1->Insert(GenerateDoWhileLoop(label_manager_, lhs_col_rhs_rows, rhs_height_bytes, type_size, {}, [&](BlockBody* b2) {
          auto lhs_cell = MakeF32Load(MakeLocalGet(lhs_row_offset));
          auto rhs_cell = MakeF32Load(MakeBinary(Opcode::I32Add, MakeLocalGet(rhs_row_offset), MakeLocalGet(lhs_col_rhs_rows)));
          b2->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));
          b2->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(type_size)));
        }));
      }

      // Store result in destination cell
      b1->Insert(MakeF32Store(MakeLocalGet(dst_row_offset), MakeLocalGet(res_cell)));
    }));
    return e;
  }

  // Cannot optimize if rhs width is too small
  if(rhs_width_bytes < WASMPP_V128_SIZE) {
    return MatrixSnippet::MatrixDot(lhs, rhs, dst, locals);
  }

  // Optimize for large matrices
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(lhs_row_offset, MakeI32Const(lhs->Begin())));
  Merge(e, GenerateRangeLoop(label_manager_, dst_row_offset, dst->Begin(), dst->End(), rhs_width_bytes, {}, [&](BlockBody* b1) {

    // Use SIMD while possible
    // Loop on rhs columns in group of 4
    b1->Insert(GenerateRangeLoop(label_manager_, rhs_col, 0, simd_width_bytes, simd_type_size, {}, [&](BlockBody* b2) {

      // Reset result counter
      b2->Insert(MakeLocalSet(res_128, MakeUnary(Opcode::F32X4Splat, MakeF32Const(0))));

      // Set rhs pointer to next 4 columns
      if(rhs.HasBeginVar()) {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeBinary(Opcode::I32Add, MakeLocalGet(rhs.Var()), MakeLocalGet(rhs_col))));
      } else {
        b2->Insert(MakeLocalSet(rhs_row_offset, MakeBinary(Opcode::I32Add, MakeI32Const(rhs.Array()->Begin()), MakeLocalGet(rhs_col))));
      }

      // Loop vertically on a column group
      b2->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, rhs.Array()->Memory()->Bytes(), rhs_width_bytes, {}, [&](BlockBody* b3){
        auto lhs_cell = MakeUnary(Opcode::F32X4Splat, MakeF32Load(MakeLocalGet(lhs_row_offset)));
        auto rhs_cell = MakeV128Load(MakeLocalGet(rhs_row_offset));

        // Compute 4 cells at a time
        b3->Insert(GenerateCompoundAssignment(res_128, Opcode::F32X4Add, MakeBinary(Opcode::F32X4Mul, lhs_cell, rhs_cell)));

        // Move lhs pointer to next column
        b3->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(type_size)));

        // Move rhs pointer to next row
        b3->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
      }));

      // Reset lhs pointer to beginning of row
      b2->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Sub, MakeI32Const(lhs_width_bytes)));

      // Store result in destination matrix
      b2->Insert(MakeV128Store(MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col)), MakeLocalGet(res_128)));
    }));

    if(width_remainder > 0) {
      // Fallback to regular computation
      // Loop on remaining rhs columns
      b1->Insert(GenerateDoWhileLoop(label_manager_, rhs_col, rhs_width_bytes, type_size, {}, [&](BlockBody* b2) {

        // Reset result counter
        b2->Insert(MakeLocalSet(res_cell, MakeF32Const(0)));

        // Set rhs pointer to next columns
        if(rhs.HasBeginVar()) {
          b2->Insert(MakeLocalSet(rhs_row_offset, MakeBinary(Opcode::I32Add, MakeLocalGet(rhs.Var()), MakeLocalGet(rhs_col))));
        } else {
          b2->Insert(MakeLocalSet(rhs_row_offset, MakeBinary(Opcode::I32Add, MakeI32Const(rhs.Array()->Begin()), MakeLocalGet(rhs_col))));
        }

        // Loop vertically on a column
        b2->Insert(GenerateRangeLoop(label_manager_, lhs_col_rhs_rows, 0, rhs.Array()->Memory()->Bytes(), rhs_width_bytes, {}, [&](BlockBody* b3){
          auto lhs_cell = MakeF32Load(MakeLocalGet(lhs_row_offset));
          auto rhs_cell = MakeF32Load(MakeLocalGet(rhs_row_offset));

          // Compute cell
          b3->Insert(GenerateCompoundAssignment(res_cell, Opcode::F32Add, MakeBinary(Opcode::F32Mul, lhs_cell, rhs_cell)));

          // Move lhs pointer to next column
          b3->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(type_size)));

          // Move rhs pointer to next row
          b3->Insert(GenerateCompoundAssignment(rhs_row_offset, Opcode::I32Add, MakeI32Const(rhs_width_bytes)));
        }));

        // Reset lhs pointer to beginning of row
        b2->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Sub, MakeI32Const(lhs_width_bytes)));

        // Store result in destination matrix
        b2->Insert(MakeF32Store(MakeBinary(Opcode::I32Add, MakeLocalGet(dst_row_offset), MakeLocalGet(rhs_col)), MakeLocalGet(res_cell)));
      }));
    }

    // Move lhs offset to next row
    b1->Insert(GenerateCompoundAssignment(lhs_row_offset, Opcode::I32Add, MakeI32Const(lhs_width_bytes)));
  }));
  return e;

}

} // namespace snippet
} // namespace nn
