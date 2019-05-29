#ifndef NN_BUILTINS_MESSAGE_H_
#define NN_BUILTINS_MESSAGE_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

class Message : public Builtin {
private:
  wabt::Var log_training_time_;
public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const wabt::Var& LogTrainingTime() const { return log_training_time_; }
};

} // namespace builtins
} // namespace nn

#endif
