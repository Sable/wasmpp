#ifndef NN_SNIPPET_MATRIX_H_
#define NN_SNIPPET_MATRIX_H_

#include <src/nn-builder/src/data_structure/ndarray.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace snippet {

// RelocMat is useful for a matrix which can have a hybrid starting address.
// A matrix wrapped in this class should be used carefully because
// a begin address should point to a matrix with the exact shape
// as the passed array, otherwise the program will result in an undefined
// behaviour
class RelocMat {
public:
  RelocMat(ds::NDArray* array) : array_(array), has_begin_var(false) {}
  RelocMat(ds::NDArray* array, wabt::Var var) : array_(array), var_(var), has_begin_var(true) {}
  ds::NDArray* Array() const { return array_; }
  wabt::Var Var() const { assert(has_begin_var); return var_; }
  bool HasBeginVar() const { return has_begin_var; }
private:
  ds::NDArray* array_;
  wabt::Var var_;
  bool has_begin_var = false;
};

//        n            p
//   +--------+   +--------+
// m |        | n |        |
//   |        |   |        |
//   +--------+   +--------+
//
// In this implementation
// matrices are of dimensions
// (m,n) and (n,p)

// Dot product of two matrices
wabt::ExprList* MatrixDot(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst,
                          std::vector<wabt::Var> locals);

// Dot product of two matrices where the left one is treated as transposed
wabt::ExprList* MatrixDotLT(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                            std::vector<wabt::Var> locals);

// Dot product of two matrices where the right one is treated as transposed
wabt::ExprList* MatrixDotRT(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, RelocMat rhs,
                            ds::NDArray* dst, std::vector<wabt::Var> locals);

// Add two matrices
wabt::ExprList* MatrixAddition(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                               ds::NDArray* dst, std::vector<wabt::Var> locals);

// Add two matrices
wabt::ExprList* MatrixSubtraction(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                                  ds::NDArray* dst, std::vector<wabt::Var> locals);

// Add two matrices
wabt::ExprList* MatrixMultiplication(wasmpp::LabelManager* label_manager, ds::NDArray* lhs, ds::NDArray* rhs,
                                   ds::NDArray* dst, std::vector<wabt::Var> locals);
// Scalar matrix
wabt::ExprList* MatrixScalar(wasmpp::LabelManager* label_manager, ds::NDArray* src, wabt::ExprList* scalar,
                             ds::NDArray* dst, std::vector<wabt::Var> locals);

// Apply activation function to matrixa
wabt::ExprList* MatrixActivation(wasmpp::LabelManager* label_manager, RelocMat src, builtins::ActivationFunction func,
                                 ds::NDArray* dst, std::vector<wabt::Var> locals, bool prime);

// Copy matrix content from src to dst
wabt::ExprList* MatrixCopy(wasmpp::LabelManager* label_manager, ds::NDArray* src, ds::NDArray* dst,
                           std::vector<wabt::Var> locals);

// Broadcast bias array
wabt::ExprList* MatrixBiasBroadcast(wasmpp::LabelManager* label_manager, ds::NDArray* bias,
                                    std::vector<wabt::Var> locals);
// Apply argmax on each matrix column
wabt::ExprList* MatrixColumnArgmax(wasmpp::LabelManager* label_manager, ds::NDArray* src,
                                    std::vector<wabt::Var> locals);

// Get the mean of all matrix cells
wabt::ExprList* MatrixMean(wasmpp::LabelManager* label_manager, ds::NDArray* src,
                           std::vector<wabt::Var> locals, wabt::Var result);

} // namespace snippet
} // namespace nn

#endif
