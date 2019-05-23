#ifndef NN_BUILTINS_ACTIVATION_H_
#define NN_BUILTINS_ACTIVATION_H_

#include <src/apps/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

struct ActivationFunction {
  wabt::Var function;
  wabt::Var derivative;
};

class Activation : public Builtin {
private:
  ActivationFunction sigmoid_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name);
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager);

  const ActivationFunction& Sigmoid() const { return sigmoid_; }
};

} // namespace builtins
} // namespace nn

#endif
