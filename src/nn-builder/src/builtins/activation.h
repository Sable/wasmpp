#ifndef NN_BUILTINS_ACTIVATION_H_
#define NN_BUILTINS_ACTIVATION_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

struct ActivationFunction {
  enum Type {
    SIGMOID,
    RELU,
    LEAKY_RELU,
    ELU,
    TANH,
    LINEAR,
    SOFTMAX
  } type;
  wabt::Var function;
  wabt::Var derivative;
  bool operator==(const ActivationFunction& func) const;
  bool operator!=(const ActivationFunction& func) const;
};
struct ActivationOptions {
  float linear_slope = 1;
  float leaky_relu_slope = 0.01;
  float elu_slope = 0.01;
};
class Activation : public Builtin {
private:
  ActivationOptions options_;

  ActivationFunction sigmoid_;
  ActivationFunction relu_;
  ActivationFunction leaky_relu_;
  ActivationFunction elu_;
  ActivationFunction tanh_;
  ActivationFunction linear_;
  ActivationFunction softmax_;
public:
  Activation(ActivationOptions options) : options_(options) {}
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override ;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const ActivationFunction& Sigmoid() const { return sigmoid_; }
  const ActivationFunction& ReLU() const { return relu_; }
  const ActivationFunction& LeakyReLU() const { return leaky_relu_; }
  const ActivationFunction& Tanh() const { return tanh_; }
  const ActivationFunction& Linear() const { return linear_; }
  const ActivationFunction& ELU() const { return elu_; }
  const ActivationFunction& Softmax() const { return softmax_; }
};

} // namespace builtins
} // namespace nn

#endif
