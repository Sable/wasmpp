#include <src/nn-builder/tests/matrix_test.h>
#include <src/nn-builder/src/data_structure/ndarray.h>

namespace nn {
namespace test {

using namespace wabt;
using namespace wasmpp;

#define NEW_MATRIX(array, rows, cols) \
    ds::NDArray* array = new ds::NDArray(module_manager_->Memory().Allocate((rows) * (cols) * TypeSize(Type::F32)), \
                            {rows, cols}, TypeSize(Type::F32));

void MatrixSnippetTest::MatrixAddition_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val + val)));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixAddition(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddition_1", Type::I32, Type::I32);
}

void MatrixSnippetTest::MatrixSubtraction_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val*4)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val*-3)));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixSubtraction(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSubtraction_1", Type::I32, Type::I32);
}

void MatrixSnippetTest::MatrixMultiplication_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val*val)));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixMultiplication(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixMultiplication_1", Type::I32, Type::I32);
}

void MatrixSnippetTest::MatrixScalar_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(src, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float scalar = 0.2;
    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(src->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val * scalar)));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixScalar(src, MakeF32Const(scalar), dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixScalar_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixDot_test_1() {
  NN_TEST() {
    uint32_t lhs_rows = 5;
    uint32_t lhs_cols = 10;
    uint32_t rhs_rows = lhs_cols;
    uint32_t rhs_cols = 7;

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, lhs_rows, rhs_cols);
    NEW_MATRIX(expected, lhs_rows, rhs_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(lhs_rows, std::vector<float>(rhs_cols, 0));
    float val = 1;
    for (uint32_t row = 0; row < lhs_rows; row++) {
      for (uint32_t col = 0; col < lhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat1[row][col] = val;
        val++;
      }
    }
    for (uint32_t row = 0; row < rhs_rows; row++) {
      for (uint32_t col = 0; col < rhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat2[row][col] = val;
        val++;
      }
    }
    for (auto i = 0; i < lhs_rows; ++i) {
      for (auto j = 0; j < rhs_cols; ++j) {
        for (auto k = 0; k <lhs_cols; ++k) {
          res[i][j] += mat1[i][k] * mat2[k][j];
        }
      }
    }
    for (uint32_t row = 0; row < lhs_rows; row++) {
      for (uint32_t col = 0; col < rhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_.MatrixDot(lhs, snippet::RelocMat(rhs), dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDot_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixDotLT_test_1() {
  NN_TEST() {
    uint32_t lhs_rows = 10;
    uint32_t lhs_cols = 5;
    uint32_t rhs_rows = lhs_rows;
    uint32_t rhs_cols = 7;
    uint32_t dst_rows = lhs_cols;
    uint32_t dst_cols = rhs_cols;

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, dst_rows, dst_cols);
    NEW_MATRIX(expected, dst_rows, dst_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(dst_rows, std::vector<float>(dst_cols, 0));
    float val = 1;
    for (uint32_t row = 0; row < lhs_rows; row++) {
      for (uint32_t col = 0; col < lhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat1[row][col] = val;
        val++;
      }
    }
    for (uint32_t row = 0; row < rhs_rows; row++) {
      for (uint32_t col = 0; col < rhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat2[row][col] = val;
        val++;
      }
    }
    for (auto i = 0; i < lhs_cols; ++i) {
      for (auto j = 0; j < rhs_cols; ++j) {
        for (auto k = 0; k <lhs_rows; ++k) {
          res[i][j] += mat1[k][i] * mat2[k][j];
        }
      }
    }
    for (uint32_t row = 0; row < dst_rows; row++) {
      for (uint32_t col = 0; col < dst_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_.MatrixDotLT(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotLT_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixDotRT_test_1() {
  NN_TEST() {
    uint32_t lhs_rows = 5;
    uint32_t lhs_cols = 10;
    uint32_t rhs_rows = 7;
    uint32_t rhs_cols = lhs_cols;
    uint32_t dst_rows = lhs_rows;
    uint32_t dst_cols = rhs_rows;

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, dst_rows, dst_cols);
    NEW_MATRIX(expected, dst_rows, dst_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(dst_rows, std::vector<float>(dst_cols, 0));
    float val = 1;
    for (uint32_t row = 0; row < lhs_rows; row++) {
      for (uint32_t col = 0; col < lhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat1[row][col] = val;
        val++;
      }
    }
    for (uint32_t row = 0; row < rhs_rows; row++) {
      for (uint32_t col = 0; col < rhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        mat2[row][col] = val;
        val++;
      }
    }
    for (auto i = 0; i < lhs_rows; ++i) {
      for (auto j = 0; j < rhs_rows; ++j) {
        for (auto k = 0; k <lhs_cols; ++k) {
          res[i][j] += mat1[i][k] * mat2[j][k];
        }
      }
    }
    for (uint32_t row = 0; row < dst_rows; row++) {
      for (uint32_t col = 0; col < dst_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_.MatrixDotRT(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotRT_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixVectorAddition_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(vector, rows, 1);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float mat_val = 1;
    float vec_val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      f.Insert(MakeF32Store(MakeI32Const(vector->GetLinearIndex({row, 0})), MakeF32Const(vec_val)));
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(mat_val+vec_val)));
        mat_val++;
      }
      vec_val++;
    }

    f.Insert(matrix_snippet_.MatrixVectorAddition(matrix, vector, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixVectorAddition_1", Type::I32, Type::I32, Type::I32, Type::I32);
}

void MatrixSnippetTest::MatrixRowSum_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(dst, rows, 1);
    NEW_MATRIX(expected, rows, 1);

    float mat_val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      float result = 0;
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        result += mat_val;
        mat_val++;
      }
      f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, 0})), MakeF32Const(result)));
    }

    f.Insert(matrix_snippet_.MatrixRowSum(matrix, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixRowSum_1", Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixAdditionSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 100;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val + val)));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixAddition(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAdditionSimd_1", Type::I32, Type::I32);
}

void MatrixSnippetSimdTest::MatrixSubtractionSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 100;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val*4)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val*-3)));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixSubtraction(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSubtractionSimd_1", Type::I32, Type::I32);
}

void MatrixSnippetSimdTest::MatrixMultiplicationSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 100;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val*val)));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixMultiplication(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixMultiplicationSimd_1", Type::I32, Type::I32);
}

void MatrixSnippetSimdTest::MatrixScalarSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 100;

    NEW_MATRIX(src, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float scalar = 0.2;
    float val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(src->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val * scalar)));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixScalar(src, MakeF32Const(scalar), dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixScalarSimd_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetSimdTest::MatrixVectorAdditionSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 101;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(vector, rows, 1);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float mat_val = 1;
    float vec_val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      f.Insert(MakeF32Store(MakeI32Const(vector->GetLinearIndex({row, 0})), MakeF32Const(vec_val)));
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(mat_val+vec_val)));
        mat_val++;
      }
      vec_val++;
    }

    f.Insert(matrix_snippet_simd_.MatrixVectorAddition(matrix, vector, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixVectorAdditionSimd_1", Type::I32, Type::I32, Type::I32, Type::I32);
}

void MatrixSnippetSimdTest::MatrixRowSumSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 100;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(dst, rows, 1);
    NEW_MATRIX(expected, rows, 1);

    float mat_val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      float result = 0;
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        result += mat_val;
        mat_val++;
      }
      f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, 0})), MakeF32Const(result)));
    }

    f.Insert(matrix_snippet_simd_.MatrixRowSum(matrix, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixRowSumSimd_1", Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

} // namespace test
} // namespace nn

