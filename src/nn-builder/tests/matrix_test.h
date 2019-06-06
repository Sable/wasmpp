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
      module_manager_(module_manager), test_builtins_(test_builtins), matrix_snippet_(&module_manager->Label()) {}
  void MatrixAddition_test_1();
  void MatrixAddition_test_2();
  void MatrixSubtraction_test_1();
};

} // namespace test
} // namespace nn

#endif
