#ifndef NN_TESTS_TEST_JS_H_
#define NN_TESTS_TEST_JS_H_

#include <src/ir.h>

namespace nn {
namespace test {

struct TestBuiltins {
  wabt::Var assert_matrix_eq;
  wabt::Var log_start_function;
};

} // namespace test
} // namespace nn

#endif