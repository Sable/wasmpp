#ifndef NN_TESTS_TEST_JS_H_
#define NN_TESTS_TEST_JS_H_

#include <src/ir.h>

namespace nn {
namespace test {

struct TestBuiltins {
  wabt::Var assert_matrix_eq;
};

#define NN_TEST(desc) \
    auto _test_function = [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals)

#define ADD_NN_TEST(module_manager, name, ...) \
    module_manager->MakeFunction("test_" name, {}, {__VA_ARGS__}, _test_function)

} // namespace test
} // namespace nn

#endif