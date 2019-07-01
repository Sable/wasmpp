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
  virtual wabt::ExprList* ElementWiseBinaryOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray*dst,
                                                     std::vector<wabt::Var> locals);

  // Apply an element wise binary scalar operation
  // e.g. dst[i] = lhs[i] + (scalar * rhs[i])
  virtual wabt::ExprList* ElementWiseBinaryScalarOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs,
                                                           ds::NDArray*dst, wabt::ExprList* scalar,
                                                           std::vector<wabt::Var> locals);

  // Apply an element wise function
  // e.g. dst[i] = func(args[0][i], args[1][i], ...)
  virtual wabt::ExprList* ElementWiseFunction(std::vector<RelocMat> args, wabt::Var func, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);

  // Apply binary operation on matrix and vector as operands
  // e.g. dst[i] = mat[i] + vec[j]
  // +-+-+   +-+   +---+---+
  // |a|c|   |A|   |A+a|A+c|
  // +-+-+ + +-+ = +---+---+
  // |b|d|   |B|   |B+b|B+d|
  // +-+-+   +-+   +---+---+
  virtual wabt::ExprList* MatrixVectorBinaryOperation(wabt::Opcode op, ds::NDArray* matrix, ds::NDArray* vector,
                                                      ds::NDArray* dst_matrix, std::vector<wabt::Var> locals);

public:
  MatrixSnippet(wasmpp::LabelManager* label_manager, arch::BuiltinFunctions* builtins) : Snippet(label_manager, builtins) {}

  // Dot product of two matrices
  virtual wabt::ExprList* MatrixDot(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst, std::vector<wabt::Var> locals);

  // Dot product of two matrices where the left one is treated as transposed
  virtual wabt::ExprList* MatrixDotLT(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);

  // Dot product of two matrices where the right one is treated as transposed
  virtual wabt::ExprList* MatrixDotRT(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst,
                                      std::vector<wabt::Var> locals);

  // Add matrices and vector (vertically)
  virtual wabt::ExprList* MatrixVectorAddition(ds::NDArray* matrix, ds::NDArray* vector, ds::NDArray* dst_matrix,
                                               std::vector<wabt::Var> locals);

  // Add two matrices
  virtual wabt::ExprList* MatrixAddition(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                         std::vector<wabt::Var> locals);

  // Subtract two matrices
  virtual wabt::ExprList* MatrixSubtraction(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                            std::vector<wabt::Var> locals);

  // Multiply two matrices
  virtual wabt::ExprList* MatrixMultiplication(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                               std::vector<wabt::Var> locals);

  // Scalar matrix
  virtual wabt::ExprList* MatrixScalar(ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                               std::vector<wabt::Var> locals);

  // Apply activation function to matrix
  virtual wabt::ExprList* MatrixActivation(RelocMat src, builtins::ActivationFunction func, ds::NDArray* dst,
                                           std::vector<wabt::Var> locals, bool prime);

  // Apply hard-max on each matrix column
  virtual wabt::ExprList* MatrixColumnHardmax(ds::NDArray* src, ds::NDArray* dst, std::vector<wabt::Var> locals);

  // Sum row values and store them in destination vector
  virtual wabt::ExprList* MatrixHorizontalSum(ds::NDArray* matrix, ds::NDArray* dst_vector, std::vector<wabt::Var> locals);

  // Absolute sum all elements in a matrix
  virtual wabt::ExprList* MatrixAbsSum(ds::NDArray* matrix, wabt::Var result, std::vector<wabt::Var> locals);

  // Square sum all elements in a matrix
  virtual wabt::ExprList* MatrixSquareSum(ds::NDArray* matrix, wabt::Var result, std::vector<wabt::Var> locals);

  // Scale right operand then add both operands
  virtual wabt::ExprList* MatrixAddRightScale(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                              wabt::ExprList* scalar, std::vector<wabt::Var> locals);

  // Scale right operand then sub both operands
  virtual wabt::ExprList* MatrixSubRightScale(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                              wabt::ExprList* scalar, std::vector<wabt::Var> locals);

  // Sign right operand then add both operands
  virtual wabt::ExprList* MatrixAddRightSignScale(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst, float scale,
                                             std::vector<wabt::Var> locals);
};

class MatrixSnippetSimd : public MatrixSnippet {
private:
  wabt::ExprList* ElementWiseBinaryOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst,
                                             std::vector<wabt::Var> locals) override;

  wabt::ExprList* ElementWiseBinaryScalarOperation(wabt::Opcode op, ds::NDArray* lhs, ds::NDArray* rhs,
                                                   ds::NDArray*dst, wabt::ExprList* scalar,
                                                   std::vector<wabt::Var> locals) override;

  wabt::ExprList* MatrixVectorBinaryOperation(wabt::Opcode op, ds::NDArray* matrix, ds::NDArray* vector,
                                              ds::NDArray* dst_matrix, std::vector<wabt::Var> locals) override;
public:
  explicit MatrixSnippetSimd(wasmpp::LabelManager* label_manager, arch::BuiltinFunctions* builtins) :
      MatrixSnippet(label_manager, builtins) {}

  wabt::ExprList* MatrixScalar(ds::NDArray* src, wabt::ExprList* scalar, ds::NDArray* dst,
                               std::vector<wabt::Var> locals) override ;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixHorizontalSum(ds::NDArray* matrix, ds::NDArray* dst_vector, std::vector<wabt::Var> locals) override;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixDotRT(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst, std::vector<wabt::Var> locals) override;

  // The SIMD version of this function generates exact results as the non-SIMD
  wabt::ExprList* MatrixDotLT(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst, std::vector<wabt::Var> locals) override ;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixDot(ds::NDArray* lhs, RelocMat rhs, ds::NDArray* dst, std::vector<wabt::Var> locals) override ;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixAbsSum(ds::NDArray* matrix, wabt::Var result, std::vector<wabt::Var> locals) override ;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixSquareSum(ds::NDArray* matrix, wabt::Var result, std::vector<wabt::Var> locals) override ;

  // The SIMD version of this function generates a result slightly different
  // than the non-SIMD one because of the order of float addition
  wabt::ExprList* MatrixAddRightSignScale(ds::NDArray* lhs, ds::NDArray* rhs, ds::NDArray* dst, float scale,
                                                  std::vector<wabt::Var> locals) override ;
};

#define MATRIX_CHECK(x) \
  ERROR_UNLESS((x) != nullptr, #x " cannot be null"); \
  ERROR_UNLESS((x)->Shape().size() == 2, #x " is expected to be a 2D matrix");

#define VECTOR_CHECK(x) \
  MATRIX_CHECK(x) \
  ERROR_UNLESS((x)->Shape()[1] == 1, #x" is expected to be a vector");

#define MATRIX_SAME_SHAPE(x, y) \
  ERROR_UNLESS((x)->Shape() == (y)->Shape(), #x " and " #y " matrices are not compatible");

} // namespace snippet
} // namespace nn

#endif
