#ifndef NN_SNIPPET_MATRIX_H_
#define NN_SNIPPET_MATRIX_H_

#include <src/apps/nn-builder/src/data_structure/ndarray.h>
#include <src/apps/nn-builder/src/builtins/activation.h>
#include <src/apps/nn-builder/src/builtins/loss.h>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace snippet {

//        n            p
//   +--------+   +--------+
// m |        | n |        |
//   |        |   |        |
//   +--------+   +--------+
//
// In this implementation
// matrices are of dimensions
// (m,n) and (n,p)

// Multiply two matrices
wabt::ExprList* MatrixDot(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals);

// Add two matrices
wabt::ExprList* MatrixAddition(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                               ds::NDArray* dst, std::vector<wabt::Var> locals);

// Scalar matrix
wabt::ExprList* MatrixScalar(wasmpp::LabelManager* label_manager, ds::NDArray* src, wabt::ExprList* scalar,
                             ds::NDArray* dst, std::vector<wabt::Var> locals);

// Apply activation function to matrix
wabt::ExprList* MatrixActivation(wasmpp::LabelManager* label_manager, ds::NDArray* src,
                                 builtins::ActivationFunction func, ds::NDArray* dst, std::vector<wabt::Var> locals);

// Apply matrix loss function
wabt::ExprList* MatrixLoss(wasmpp::LabelManager* label_manager, ds::NDArray* prediction, ds::NDArray* target,
                           builtins::LossFunction func, ds::NDArray* dst, std::vector<wabt::Var> locals);

// Copy matrix content from src to dst
wabt::ExprList* MatrixCopy(wasmpp::LabelManager* label_manager, ds::NDArray* src, ds::NDArray* dst,
                           std::vector<wabt::Var> locals);

} // namespace snippet
} // namespace nn

#endif
