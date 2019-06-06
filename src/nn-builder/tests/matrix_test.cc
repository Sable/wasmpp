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
module_manager_->MakeFunction("test_MatrixAddition_1", {}, {Type::I32, Type::I32},
                              [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
  uint32_t rows = 5;
  uint32_t cols = 5;

  NEW_MATRIX(lhs, rows, cols);
  NEW_MATRIX(rhs, rows, cols);
  NEW_MATRIX(dst, rows, cols);
  NEW_MATRIX(expected, rows, cols);

  uint32_t val = 1;
  for(uint32_t row = 0; row < rows; row++) {
    for(uint32_t col = 0; col < cols; col++) {
      f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})), MakeF32Const(val)));
      f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})), MakeF32Const(val)));
      f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})), MakeF32Const(val+val)));
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
});
}

void MatrixSnippetTest::MatrixAddition_test_2() {
  module_manager_->MakeFunction("test_MatrixAddition_2", {}, {Type::I32, Type::I32},
                                [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    uint32_t rows = 5;
    uint32_t cols = 10;

    NEW_MATRIX(lhs, rows, cols);
    NEW_MATRIX(rhs, rows, cols);
    NEW_MATRIX(dst, rows, cols);
    NEW_MATRIX(expected, rows, cols);

    uint32_t val = 1;
    for (uint32_t row = 0; row < rows; row++) {
      for (uint32_t col = 0; col < cols; col++) {
        f.Insert(MakeF32Store(MakeI32Const(lhs->GetLinearIndex({row, col})),
                              MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(rhs->GetLinearIndex({row, col})),
                              MakeF32Const(val)));
        f.Insert(MakeF32Store(MakeI32Const(expected->GetLinearIndex({row, col})),
                              MakeF32Const(val + val)));
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
  });
}

} // namespace test
} // namespace nn

