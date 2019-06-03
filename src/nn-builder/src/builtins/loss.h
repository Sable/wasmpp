#ifndef NN_BUILTINS_LOSS_H_
#define NN_BUILTINS_LOSS_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

struct LossFunction {
  wabt::Var cost;
  wabt::Var loss;
};

class Loss : public Builtin {
private:
  LossFunction mean_squared_error_;
  LossFunction cross_entropy_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const LossFunction& MeanSquaredError() const { return mean_squared_error_; }
  const LossFunction& CrossEntropy() const { return cross_entropy_; }
};

} // namespace builtins
} // namespace nn

#endif
