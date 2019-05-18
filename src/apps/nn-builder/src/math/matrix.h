#ifndef NN_MATH_MATRIX_H_
#define NN_MATH_MATRIX_H_

#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace compute {
namespace math {

// Multiply two matrices
template<wabt::Type type>
void Multiply2DArrays(wasmpp::ContentManager* ctn, wasmpp::NDArray lhs, wasmpp::NDArray rhs, wasmpp::NDArray dst,
                      std::vector<wabt::Var> locals);

} // namespace math
} // namespace compute
} // namespace nn

#endif
