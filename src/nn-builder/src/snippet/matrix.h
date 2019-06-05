#ifndef NN_SNIPPET_MATRIX_H_
#define NN_SNIPPET_MATRIX_H_

#include <src/nn-builder/src/data_structure/ndarray.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/snippet/snippets.h>
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

class MatrixSnippet : public Snippet {
protected:
  // Apply an element wise binary operation
  // e.g. dst[i] = lhs[i] + rhs[i]
  virtual wabt::ExprList* ElementWiseBinaryOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray*
  dst, std::vector<wabt::Var> locals);

  // Apply an element wise function
  // e.g. dst[i] = func(args[0][i], args[1][i], ...)
  virtual wabt::ExprList* ElementWiseFunction(std::vector<RelocMat> args, wabt::Var func, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);
public:
  MatrixSnippet(wasmpp::LabelManager* label_manager) : Snippet(label_manager) {}

  // Dot product of two matrices
  virtual wabt::ExprList* MatrixDot(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst, std::vector<wabt::Var> locals);

  // Dot product of two matrices where the left one is treated as transposed
  virtual wabt::ExprList* MatrixDotLT(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);

  // Dot product of two matrices where the right one is treated as transposed
  virtual wabt::ExprList* MatrixDotRT(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);

  // Add two matrices
  virtual wabt::ExprList* MatrixAddition(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                         std::vector<wabt::Var> locals);

  // Add two matrices
  virtual wabt::ExprList* MatrixSubtraction(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                    std::vector<wabt::Var> locals);

  // Add two matrices
  virtual wabt::ExprList* MatrixMultiplication(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                       std::vector<wabt::Var> locals);

  // Scalar matrix
  virtual wabt::ExprList* MatrixScalar(ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                               std::vector<wabt::Var> locals);

  // Apply activation function to matrixa
  virtual wabt::ExprList* MatrixActivation(RelocMat src, builtins::ActivationFunction func, ds::NDArray* dst,
                                   std::vector<wabt::Var> locals, bool prime);

  // Broadcast bias array
  virtual wabt::ExprList* MatrixBiasBroadcast(ds::NDArray* bias, std::vector<wabt::Var> locals);

  // Apply argmax on each matrix column
  virtual wabt::ExprList* MatrixColumnArgmax(ds::NDArray* src, std::vector<wabt::Var> locals);
};

class MatrixSnippetSimd : public MatrixSnippet {
protected:
  wabt::ExprList* ElementWiseBinaryOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                             std::vector<wabt::Var> locals) override;

public:
  MatrixSnippetSimd(wasmpp::LabelManager* label_manager) : MatrixSnippet(label_manager) {}

  wabt::ExprList* MatrixScalar(ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                               std::vector<wabt::Var> locals) override ;
};


} // namespace snippet
} // namespace nn

#endif
