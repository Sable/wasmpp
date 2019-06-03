#ifndef NN_BUILTINS_MATH_H_
#define NN_BUILTINS_MATH_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

class Math : public Builtin {
private:
  wabt::Var exp_;
  wabt::Var log_;
  wabt::Var random_;
  wabt::Var mask_matrix_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const wabt::Var& Exp() const { return exp_; }
  const wabt::Var& Log() const { return log_; }
  const wabt::Var& Random() const { return random_; }
  const wabt::Var& MaskMatrix() const { return mask_matrix_; }
};

} // namespace builtins
} // namespace nn

#endif
