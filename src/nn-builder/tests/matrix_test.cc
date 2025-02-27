#include <src/nn-builder/tests/matrix_test.h>
#include <src/nn-builder/src/data_structure/ndarray.h>
#include <cmath>

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

    float val = 1.2;
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

    float val = 1.2;
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

    float val = 1.2;
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
    float val = 1.2;
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
  ADD_NN_TEST(module_manager_, "MatrixDot_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
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
    float val = 1.2;
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
  ADD_NN_TEST(module_manager_, "MatrixDotLT_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32,
              Type::V128);
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
    float val = 1.2;
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
  ADD_NN_TEST(module_manager_, "MatrixDotRT_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32,
              Type::F32, Type::V128);
}

void MatrixSnippetTest::MatrixVectorAddition_test_1() {
  NN_TEST() {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(vector, rows, 1);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float mat_val = 1.2;
    float vec_val = 1.3;
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

void MatrixSnippetTest::MatrixAbsSum_test_1() {
  NN_TEST() {
    auto vi32_1 = locals[0];
    auto v128_1 = locals[1];
    auto result = locals[2];

    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(matrix, rows, cols);

    float add = 0;
    float mat_val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        auto val = mat_val * (col % 2 == 0 ? 1 : -1);
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(val)));
        add += abs(val);
        mat_val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixAbsSum(matrix, result, {vi32_1, v128_1}));
    f.Insert(MakeCall(test_builtins_->assert_f32_eq, {
      MakeF32Const(add),
      MakeLocalGet(result)
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAbsSum_1", Type::I32, Type::V128, Type::F32);
}

void MatrixSnippetTest::MatrixSquareSum_test_1() {
  NN_TEST() {
    auto vi32_1 = locals[0];
    auto vf32_1 = locals[1];
    auto v128_1 = locals[2];
    auto result = locals[3];

    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(matrix, rows, cols);

    float add = 0;
    float mat_val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        auto val = mat_val * mat_val;
        add += val;
        mat_val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixSquareSum(matrix, result, {vi32_1, vf32_1, v128_1}));
    f.Insert(MakeCall(test_builtins_->assert_f32_eq, {
        MakeF32Const(add),
        MakeLocalGet(result)
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSquareSum_1", Type::I32, Type::F32, Type::V128, Type::F32);
}

void MatrixSnippetTest::MatrixAddRightScale_test_1() {
  NN_TEST() {
    float scale = 0.01234;
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val + (val * scale))));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixAddRightScale(lhs, rhs, dst, MakeF32Const(scale), locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightScale_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixSubRightScale_test_1() {
  NN_TEST() {
    float scale = 0.01234;
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val - (val * scale))));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixSubRightScale(lhs, rhs, dst, MakeF32Const(scale), locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSubRightScale_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetTest::MatrixAddRightSignScaleAddRightScale_test_1() {
  NN_TEST() {
    float scale1 = 0.01234;
    float scale2 = 0.05678;
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        float s_val = val * (col %2 == 0 ? 1 : -1);
        float s_scale = copysignf(1.0, s_val);
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        float add = (s_scale * scale1) + (s_val * scale2);
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(s_val + add)));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixAddRightSignScaleAddRightScale(lhs, rhs, dst, scale1, scale2, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightSignScaleAddRightScale_1", Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetTest::MatrixAddRightSignScale_test_1() {
  NN_TEST() {
    float scale = 0.01234;
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        float s_val = val * (col %2 == 0 ? 1 : -1);
        float s_scale = copysignf(1.0, s_val);
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(s_val + (s_scale * scale))));
        val++;
      }
    }

    f.Insert(matrix_snippet_.MatrixAddRightSignScale(lhs, rhs, dst, scale, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightSignScale_1", Type::I32, Type::I32);
}

void MatrixSnippetTest::MatrixHorizontalSum_test_1() {
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

    f.Insert(matrix_snippet_.MatrixHorizontalSum(matrix, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixHorizontalSum_1", Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixAdditionSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 101;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
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
    uint32_t cols = 101;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
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
    uint32_t cols = 101;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
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
    uint32_t cols = 101;

    NEW_MATRIX(src, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float scalar = 0.2;
    float val = 1.2;
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

    float mat_val = 1.2;
    float vec_val = 1.3;
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

void MatrixSnippetSimdTest::MatrixHorizontalSumSimd_test_1() {
  NN_TEST() {
    uint32_t rows = 57;
    uint32_t cols = 101;

    NEW_MATRIX(matrix, rows, cols);
    NEW_MATRIX(dst, rows, 1);
    NEW_MATRIX(expected, rows, 1);

    // Simulate the same order of addition for the
    // float numbers as the SIMD version of this function
    // otherwise the result will be slightly off
    // Non-SIMD: (((a[0] + a[1]) + a[2]) + ... + a[n])
    // SIMD: ((a[0] + a[4], a[1] + a[5], a[2] + a[6], a[3] + a[7]) + ...)
    float mat_val = 1.2;
    uint32_t simd_remainder = cols % 4;
    uint32_t simd_cols = cols - simd_remainder;
    for (uint32_t row = 0; row < rows; row++) {
      float vec[4] = {0, 0, 0, 0};
      uint32_t col = 0;
      // Simulate SIMD computation
      for (; col < simd_cols; col++) {
        if(col % 4 == 0) {
          vec[0] += mat_val;
          vec[1] += mat_val + 1;
          vec[2] += mat_val + 2;
          vec[3] += mat_val + 3;
        }
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        mat_val++;
      }
      float result = (((vec[0] + vec[1]) + vec[2]) + vec[3]);
      // Simulate regular computation for the remaining values
      for(; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
        result += mat_val;
        mat_val++;
      }
      f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, 0})), MakeF32Const(result)));
    }

    f.Insert(matrix_snippet_simd_.MatrixHorizontalSum(matrix, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixHorizontalSumSimd_1", Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixDotRTSimd_test_1() {
  NN_TEST("Matrix . Matrix^T") {
    uint32_t lhs_rows = 57;
    uint32_t lhs_cols = 101;
    uint32_t rhs_rows = 73;
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
    float val = 1.2;
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

    // Simulate the same order of addition and multiplication
    // for the float numbers as the SIMD version of this function
    // otherwise the result will be slightly off
    uint32_t simd_remainder = lhs_cols % 4;
    uint32_t simd_cols = lhs_cols - simd_remainder;
    for (auto i = 0; i < lhs_rows; ++i) {
      for (auto j = 0; j < rhs_rows; ++j) {
        float vec[4] = {0, 0, 0 ,0};
        uint32_t k = 0;
        for (; k < simd_cols; k+=4) {
          vec[0] += mat1[i][k] * mat2[j][k];
          vec[1] += mat1[i][k+1] * mat2[j][k+1];
          vec[2] += mat1[i][k+2] * mat2[j][k+2];
          vec[3] += mat1[i][k+3] * mat2[j][k+3];
        }
        float res_val = (((vec[0] + vec[1]) + vec[2]) + vec[3]);
        for (; k < lhs_cols; k++) {
          res_val += mat1[i][k] * mat2[j][k];
        }
        res[i][j] = res_val;
      }
    }
    for (uint32_t row = 0; row < dst_rows; row++) {
      for (uint32_t col = 0; col < dst_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixDotRT(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotRTSimd_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32,
              Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixDotRTSimd_test_2() {
  NN_TEST("Vector . Vector^T") {
    uint32_t lhs_rows = 111;
    uint32_t lhs_cols = 1;
    uint32_t rhs_rows = 231;
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
    float val = 1.2;
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
        float res_val = 0;
        for (auto k = 0; k < lhs_cols; k++) {
          res_val += mat1[i][k] * mat2[j][k];
        }
        res[i][j] = res_val;
      }
    }
    for (uint32_t row = 0; row < dst_rows; row++) {
      for (uint32_t col = 0; col < dst_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixDotRT(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotRTSimd_2", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32,
              Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixDotSimd_test_1() {
  NN_TEST("Matrix . Vector") {
    uint32_t lhs_rows = 101;
    uint32_t lhs_cols = 103;
    uint32_t rhs_rows = lhs_cols;
    uint32_t rhs_cols = 1; // rhs is a vector

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, lhs_rows, rhs_cols);
    NEW_MATRIX(expected, lhs_rows, rhs_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(lhs_rows, std::vector<float>(rhs_cols, 0));
    float val = 1.21;
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

    // Simulate the same order of addition and multiplication
    // for the float numbers as the SIMD version of this function
    // otherwise the result will be slightly off
    uint32_t simd_remainder = rhs_rows % 4;
    uint32_t simd_rows = rhs_rows - simd_remainder;
    for (auto i = 0; i < lhs_rows; ++i) {
      for (auto j = 0; j < rhs_cols; ++j) {
        float vec[4] = {0, 0, 0 ,0};
        uint32_t k = 0;
        for (; k < simd_rows; k+=4) {
          vec[0] += mat1[i][k] * mat2[k][j];
          vec[1] += mat1[i][k+1] * mat2[k+1][j];
          vec[2] += mat1[i][k+2] * mat2[k+2][j];
          vec[3] += mat1[i][k+3] * mat2[k+3][j];
        }
        float res_val = (((vec[0] + vec[1]) + vec[2]) + vec[3]);
        for (; k < lhs_cols; k++) {
          res_val += mat1[i][k] * mat2[k][j];
        }
        res[i][j] = res_val;
      }
    }
    for (uint32_t row = 0; row < lhs_rows; row++) {
      for (uint32_t col = 0; col < rhs_cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(res[row][col])));
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixDot(lhs, snippet::RelocMat(rhs), dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotSimd_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixDotSimd_test_2() {
  NN_TEST("Matrix . Matrix") {
    uint32_t lhs_rows = 101;
    uint32_t lhs_cols = 103;
    uint32_t rhs_rows = lhs_cols;
    uint32_t rhs_cols = 107;

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, lhs_rows, rhs_cols);
    NEW_MATRIX(expected, lhs_rows, rhs_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(lhs_rows, std::vector<float>(rhs_cols, 0));
    float val = 1.2;
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

    f.Insert(matrix_snippet_simd_.MatrixDot(lhs, snippet::RelocMat(rhs), dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotSimd_2", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixDotLTSimd_test_1() {
  NN_TEST("Matrix^T . Matrix") {
    uint32_t lhs_rows = 103;
    uint32_t lhs_cols = 101;
    uint32_t rhs_rows = lhs_rows;
    uint32_t rhs_cols = 107;
    uint32_t dst_rows = lhs_cols;
    uint32_t dst_cols = rhs_cols;

    NEW_MATRIX(lhs, lhs_rows, lhs_cols);
    NEW_MATRIX(rhs, rhs_rows, rhs_cols);
    NEW_MATRIX(dst, dst_rows, dst_cols);
    NEW_MATRIX(expected, dst_rows, dst_cols);

    std::vector<std::vector<float>> mat1(lhs_rows, std::vector<float>(lhs_cols, 0));
    std::vector<std::vector<float>> mat2(rhs_rows, std::vector<float>(rhs_cols, 0));
    std::vector<std::vector<float>> res(dst_rows, std::vector<float>(dst_cols, 0));
    float val = 1.2;
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

    f.Insert(matrix_snippet_simd_.MatrixDotLT(lhs, rhs, dst, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixDotLTSimd_1", Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128);
}

void MatrixSnippetSimdTest::MatrixAbsSumSimd_test_1() {
  NN_TEST() {
    auto vi32_1 = locals[0];
    auto v128_1 = locals[1];
    auto result = locals[2];

    uint32_t rows = 21;
    uint32_t cols = 13;

    NEW_MATRIX(matrix, rows, cols);

    float add = 0;
    float mat_val = 1.2;
    auto end = rows * cols;
    auto simd_end = end - end % 4;

    uint32_t index = 0;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        auto val = mat_val * (index % 2 == 0 ? 1 : -1);
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(val)));
        index++;
      }
    }

    uint32_t i = 0;
    index = 0;
    float v128[4] = {0, 0, 0, 0};
    for(; i < simd_end; i+=4) {
      auto val = mat_val * (index++ % 2 == 0 ? 1 : -1);
      v128[0] += abs(val);
      val = mat_val * (index++ % 2 == 0 ? 1 : -1);
      v128[1] += abs(val);
      val = mat_val * (index++ % 2 == 0 ? 1 : -1);
      v128[2] += abs(val);
      val = mat_val * (index++ % 2 == 0 ? 1 : -1);
      v128[3] += abs(val);
    }
    add = v128[0] + v128[1] + v128[2] + v128[3];
    for(; i < end; i++) {
      auto val = mat_val * (index++ % 2 == 0 ? 1 : -1);
      add += abs(val);
    }

    f.Insert(matrix_snippet_simd_.MatrixAbsSum(matrix, result, {vi32_1, v128_1}));
    f.Insert(MakeCall(test_builtins_->assert_f32_eq, {
        MakeF32Const(add),
        MakeLocalGet(result)
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAbsSumSimd_1", Type::I32, Type::V128, Type::F32);
}

void MatrixSnippetSimdTest::MatrixSquareSumSimd_test_1() {
  NN_TEST() {
    auto vi32_1 = locals[0];
    auto vf32_1 = locals[1];
    auto v128_1 = locals[2];
    auto result = locals[3];

    uint32_t rows = 21;
    uint32_t cols = 13;

    NEW_MATRIX(matrix, rows, cols);

    float add = 0;
    float mat_val = 1.2;
    auto end = rows * cols;
    auto simd_end = end - end % 4;

    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(matrix->GetLinearIndex({row, col})), MakeF32Const(mat_val)));
      }
    }

    uint32_t i = 0;
    float v128[4] = {0, 0, 0, 0};
    for(; i < simd_end; i+=4) {
      v128[0] += mat_val * mat_val;
      v128[1] += mat_val * mat_val;
      v128[2] += mat_val * mat_val;
      v128[3] += mat_val * mat_val;
    }
    add = v128[0] + v128[1] + v128[2] + v128[3];
    for(; i < end; i++) {
      add += mat_val * mat_val;
    }

    f.Insert(matrix_snippet_simd_.MatrixSquareSum(matrix, result, {vi32_1, vf32_1, v128_1}));
    f.Insert(MakeCall(test_builtins_->assert_f32_eq, {
        MakeF32Const(add),
        MakeLocalGet(result)
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSquareSumSimd_1", Type::I32, Type::F32, Type::V128, Type::F32);
}

void MatrixSnippetSimdTest::MatrixAddRightScaleSimd_test_1() {
  NN_TEST() {
    float scale = 0.01234;
    uint32_t rows = 13;
    uint32_t cols = 21;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val + (val * scale))));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixAddRightScale(lhs, rhs, dst, MakeF32Const(scale), locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightScaleSimd_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetSimdTest::MatrixSubRightScaleSimd_test_1() {
  NN_TEST() {
    float scale = 0.01234;
    uint32_t rows = 13;
    uint32_t cols = 21;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val - (val * scale))));
        val++;
      }
    }

    f.Insert(matrix_snippet_simd_.MatrixSubRightScale(lhs, rhs, dst, MakeF32Const(scale), locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixSubRightScaleSimd_1", Type::I32, Type::I32, Type::F32);
}

void MatrixSnippetSimdTest::MatrixAddRightSignScaleSimd_test_1() {
  NN_TEST() {
    float scale = 0.01234;

    uint32_t rows = 13;
    uint32_t cols = 21;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    uint32_t index = 0;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        float s_val = val * (index % 2 == 0 ? 1 : -1);
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        val++;
        index++;
      }
    }

    val = 1.2;
    uint32_t i = 0;
    index = 0;
    auto end = rows * cols;
    auto simd_end = end - end % 4;
    for(; i < simd_end; i++) {
      float s_val = val * (index % 2 == 0 ? 1 : -1);
      signed int ge = (s_val >= 0 ? -1 : 0);
      float cnvt = (float) ge;
      float mul = cnvt * (-2*scale);
      float sub = mul - scale;
      f.Insert(MakeF32Store(MakeI32Const(expected->Begin() + i * 4), MakeF32Const(s_val + sub)));
      index++;
      val++;
    }
    for(; i < end; i++) {
      float s_val = val * (index % 2 == 0 ? 1 : -1);
      float s_scale = copysignf(1.0, s_val);
      f.Insert(MakeF32Store(MakeI32Const(expected->Begin() + i * 4), MakeF32Const(s_val + (s_scale * scale))));
      index++;
      val++;
    }

    f.Insert(matrix_snippet_simd_.MatrixAddRightSignScale(lhs, rhs, dst, scale, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightSignScaleSimd_1", Type::I32, Type::I32);
}

void MatrixSnippetSimdTest::MatrixAddRightSignScaleAddRightScale_test_1() {
  NN_TEST() {
    float scale1 = 0.01234;
    float scale2 = 0.05678;

    uint32_t rows = 13;
    uint32_t cols = 21;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    float val = 1.2;
    uint32_t index = 0;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        float s_val = val * (index % 2 == 0 ? 1 : -1);
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(s_val)));
        val++;
        index++;
      }
    }

    val = 1.2;
    uint32_t i = 0;
    index = 0;
    auto end = rows * cols;
    auto simd_end = end - end % 4;
    for(; i < simd_end; i++) {
      float s_val = val * (index % 2 == 0 ? 1 : -1);
      signed int ge = (s_val >= 0 ? -1 : 0);
      float cnvt = (float) ge;
      float mul = cnvt * (-2*scale1);
      float sub = mul - scale1;
      float rhs_scale2 = s_val * scale2;
      float rhs_val = sub + rhs_scale2;
      f.Insert(MakeF32Store(MakeI32Const(expected->Begin() + i * 4), MakeF32Const(s_val + rhs_val)));
      index++;
      val++;
    }
    for(; i < end; i++) {
      float s_val = val * (index % 2 == 0 ? 1 : -1);
      float s_scale = copysignf(1.0, s_val);
      float rhs_val = (s_scale * scale1) + (s_val * scale2);
      f.Insert(MakeF32Store(MakeI32Const(expected->Begin() + i * 4), MakeF32Const(s_val + rhs_val)));
      index++;
      val++;
    }

    f.Insert(matrix_snippet_simd_.MatrixAddRightSignScaleAddRightScale(lhs, rhs, dst, scale1, scale2, locals));
    f.Insert(MakeCall(test_builtins_->assert_matrix_eq, {
        MakeI32Const(dst->Memory()->Begin()),
        MakeI32Const(expected->Memory()->Begin()),
        MakeI32Const(dst->Shape()[0]),
        MakeI32Const(dst->Shape()[1])
    }));
  };
  ADD_NN_TEST(module_manager_, "MatrixAddRightSignScaleAddRightScaleSimd_1", Type::I32, Type::I32, Type::F32, Type::V128);
}

} // namespace test
} // namespace nn

