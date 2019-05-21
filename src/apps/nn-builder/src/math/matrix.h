#ifndef NN_MATH_MATRIX_H_
#define NN_MATH_MATRIX_H_

#include <src/apps/nn-builder/src/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace math {

// Multiply two matrices
template<wabt::Type type>
wasmpp::exprs_sptr Multiply2DArrays(wasmpp::LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs,
                                    ds::NDArray dst, std::vector<wabt::Var> locals);

// Add two matrices
template<wabt::Type type>
wasmpp::exprs_sptr Add2DArrays(wasmpp::LabelManager* label_manager, ds::NDArray lhs, ds::NDArray rhs,
                                    ds::NDArray dst, std::vector<wabt::Var> locals);

// Scalar matrix
template<wabt::Type type>
wasmpp::exprs_sptr Scalar2DArrays(wasmpp::LabelManager* label_manager, ds::NDArray src, wasmpp::exprs_sptr scalar,
                                    ds::NDArray dst, std::vector<wabt::Var> locals);

// Apply function to matrix
template<wabt::Type type>
wasmpp::exprs_sptr ApplyFx2DArrays(wasmpp::LabelManager* label_manager, ds::NDArray src, wabt::Var func,
                                    ds::NDArray dst, std::vector<wabt::Var> locals);

} // namespace math
} // namespace nn

#endif
