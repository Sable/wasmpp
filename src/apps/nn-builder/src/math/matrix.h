#ifndef NN_MATH_MATRIX_H_
#define NN_MATH_MATRIX_H_

#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace compute {
namespace math {

// Multiply two matrices
void Multiply2DArrays(wasmpp::ModuleManager* mm, wasmpp::ContentManager* ctn,
                      wasmpp::NDArray lhs, wasmpp::NDArray rhs, wasmpp::NDArray dst,
                      wabt::Var x, wabt::Var y, wabt::Var z);

} // namespace math
} // namespace compute
} // namespace nn

#endif
