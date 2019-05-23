#ifndef NN_BUILTINS_MATH_H_
#define NN_BUILTINS_MATH_H_

#include <src/apps/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

class Math : public Builtin {
private:
  wabt::Var exp_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name);
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager);

  const wabt::Var& Exp() const { return exp_; }
};

} // namespace builtins
} // namespace nn

#endif
