#ifndef NN_BUILTINS_LOSS_H_
#define NN_BUILTINS_LOSS_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

struct LossFunction {
  enum Type {
    MSE,
    SIGMOID_CE,
    SOFTMAX_CE
  } type;
  wabt::Var J;
  wabt::Var dJ;
  bool operator==(const LossFunction &loss_function) const;
  bool operator!=(const LossFunction &loss_function) const;
};

class Loss : public Builtin {
private:
  LossFunction mean_squared_error_;
  LossFunction sigmoid_cross_entropy_;
  LossFunction softmax_cross_entropy_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const LossFunction& MeanSquaredError() const { return mean_squared_error_; }
  const LossFunction& SigmoidCrossEntropy() const { return sigmoid_cross_entropy_; }
  const LossFunction& SoftmaxCrossEntropy() const { return softmax_cross_entropy_; }
};

} // namespace builtins
} // namespace nn

#endif
