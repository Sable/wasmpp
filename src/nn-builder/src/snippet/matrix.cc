#include <src/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace snippet {

using namespace wasmpp;
using namespace wabt;
using namespace ds;

#define MATRIX_CHECK(x) \
  ERROR_UNLESS((x) != nullptr, #x " cannot be null"); \
  ERROR_UNLESS((x)->Shape().size() == 2, #x " is expected to be a 2D matrix");

#define VECTOR_CHECK(x) \
  MATRIX_CHECK(x) \
  ERROR_UNLESS((x)->Shape()[1] == 1, #x" is expected to be a vector");

#define MATRIX_SAME_SHAPE(x, y) \
  ERROR_UNLESS((x)->Shape() == (y)->Shape(), #x " and " #y " matrices are not compatible");

wabt::ExprList* MatrixSnippet::MatrixDot(NDArray* lhs, RelocMat rhs, NDArray* dst, std::vector<Var> locals) {
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
  MATRIX_CHECK(vector);
  MATRIX_CHECK(dst_matrix);
  MATRIX_SAME_SHAPE(matrix, dst_matrix);
  ERROR_UNLESS(vector->Shape()[1] == 1, "vector must be of shape (n, 1)");

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

wabt::ExprList* MatrixSnippet::MatrixColumnArgmax(NDArray* src, std::vector<Var> locals) {
  MATRIX_CHECK(src);
  assert(locals.size() == 4);

  auto row = locals[0];
  auto src_col_offset = locals[1];
  auto max = locals[2];
  auto curr = locals[3];

  uint32_t type_size = TypeSize(Type::F32);
  uint32_t width = src->Shape()[1] * type_size;
  uint32_t height = src->Shape()[0] * width;
  uint32_t col_begin = src->Memory()->Begin();
  uint32_t col_end = col_begin + width;

  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateRangeLoop(label_manager_, src_col_offset, col_begin, col_end, type_size, {}, [&](BlockBody* b1) {
    // Find max
    b1->Insert(MakeLocalSet(max, MakeLocalGet(src_col_offset)));
    b1->Insert(GenerateRangeLoop(label_manager_, row, width, height, width, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(curr, MakeBinary(Opcode::I32Add, MakeLocalGet(row), MakeLocalGet(src_col_offset))));
      auto cond = MakeBinary(Opcode::F32Ge, MakeF32Load(MakeLocalGet(curr)), MakeF32Load(MakeLocalGet(max)));
      auto comp = MakeIf(label_manager_, cond, {}, [&](BlockBody t, Var label) {
        t.Insert(MakeLocalSet(max, MakeLocalGet(curr)));
      });
      b2->Insert(comp);
    }));

    // Place 1 in max and 0 in rest
    b1->Insert(GenerateRangeLoop(label_manager_, row, 0, height, width, {}, [&](BlockBody* b2) {
      b2->Insert(MakeLocalSet(curr, MakeBinary(Opcode::I32Add, MakeLocalGet(row), MakeLocalGet(src_col_offset))));
      auto cond = MakeBinary(Opcode::I32Eq, MakeLocalGet(curr), MakeLocalGet(max));
      auto comp = MakeIf(label_manager_, cond, {}, [&](BlockBody t, Var label) {
        t.Insert(MakeF32Store(MakeLocalGet(curr), MakeF32Const(1)));
      }, [&](BlockBody f) {
        f.Insert(MakeF32Store(MakeLocalGet(curr), MakeF32Const(0)));
      });
      b2->Insert(comp);
    }));
  }));
  return e;
}

wabt::ExprList* MatrixSnippet::MatrixRowSum(nn::ds::NDArray *matrix, nn::ds::NDArray *dst_vector,
                                            std::vector<wabt::Var> locals) {

}

wabt::Opcode OpcodeToSimd(wabt::Opcode op) {
  switch (op) {
    case wabt::Opcode::F32Add:
      return wabt::Opcode::F32X4Add;
    case wabt::Opcode::F32Sub:
      return wabt::Opcode::F32X4Sub;
    case wabt::Opcode::F32Mul:
      return wabt::Opcode::F32X4Mul;
    case wabt::Opcode::F32Div:
      return wabt::Opcode::F32X4Div;
    default:
      assert(!"Opcode to SIMD not implemented");
  }
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
                            MakeBinary(OpcodeToSimd(op), MakeV128Load(lhs_addr), MakeV128Load(rhs_addr))));
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
  MATRIX_CHECK(vector);
  MATRIX_CHECK(dst_matrix);
  MATRIX_SAME_SHAPE(matrix, dst_matrix);
  ERROR_UNLESS(vector->Shape()[1] == 1, "vector must be of shape (n, 1)");
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
      auto result = MakeBinary(OpcodeToSimd(op), MakeV128Load(mat_addr), MakeUnary(Opcode::F32X4Splat, MakeF32Load(vec_addr)));
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

} // namespace snippet
} // namespace nn
