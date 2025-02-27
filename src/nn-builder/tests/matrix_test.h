#ifndef NN_TESTS_MATRIX_TEST_H_
#define NN_TESTS_MATRIX_TEST_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/tests/test-common.h>

namespace nn {
namespace test {

class MatrixSnippetTest {
private:
  snippet::MatrixSnippet matrix_snippet_;
  wasmpp::ModuleManager* module_manager_;
  TestBuiltins* test_builtins_;
public:
  MatrixSnippetTest(wasmpp::ModuleManager* module_manager, TestBuiltins* test_builtins) :
      module_manager_(module_manager), test_builtins_(test_builtins), matrix_snippet_(&module_manager->Label(), nullptr) {}
  void MatrixAddition_test_1();
  void MatrixSubtraction_test_1();
  void MatrixMultiplication_test_1();
  void MatrixScalar_test_1();
  void MatrixDot_test_1();
  void MatrixDotLT_test_1();
  void MatrixDotRT_test_1();
  void MatrixVectorAddition_test_1();
  void MatrixHorizontalSum_test_1();
  void MatrixAbsSum_test_1();
  void MatrixSquareSum_test_1();
  void MatrixAddRightScale_test_1();
  void MatrixSubRightScale_test_1();
  void MatrixAddRightSignScale_test_1();
  void MatrixAddRightSignScaleAddRightScale_test_1();
};

class MatrixSnippetSimdTest {
private:
  snippet::MatrixSnippetSimd matrix_snippet_simd_;
  wasmpp::ModuleManager* module_manager_;
  TestBuiltins* test_builtins_;
public:
  MatrixSnippetSimdTest(wasmpp::ModuleManager* module_manager, TestBuiltins* test_builtins) :
      module_manager_(module_manager), test_builtins_(test_builtins),
      matrix_snippet_simd_(&module_manager->Label(), nullptr) {}
  void MatrixAdditionSimd_test_1();
  void MatrixSubtractionSimd_test_1();
  void MatrixMultiplicationSimd_test_1();
  void MatrixScalarSimd_test_1();
  void MatrixDotSimd_test_1();
  void MatrixDotSimd_test_2();
  void MatrixDotLTSimd_test_1();
  void MatrixDotRTSimd_test_1();
  void MatrixDotRTSimd_test_2();
  void MatrixVectorAdditionSimd_test_1();
  void MatrixHorizontalSumSimd_test_1();
  void MatrixAbsSumSimd_test_1();
  void MatrixSquareSumSimd_test_1();
  void MatrixAddRightScaleSimd_test_1();
  void MatrixSubRightScaleSimd_test_1();
  void MatrixAddRightSignScaleSimd_test_1();
  void MatrixAddRightSignScaleAddRightScale_test_1();
};

} // namespace test
} // namespace nn

#endif
