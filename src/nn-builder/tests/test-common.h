#ifndef NN_TESTS_TEST_JS_H_
#define NN_TESTS_TEST_JS_H_

#include <src/ir.h>

namespace nn {
namespace test {

struct TestBuiltins {
  wabt::Var assert_matrix_eq;
};

} // namespace test
} // namespace nn

#endif