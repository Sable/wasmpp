#ifndef NN_BUILTINS_MESSAGE_H_
#define NN_BUILTINS_MESSAGE_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

class Message : public Builtin {
private:
  wabt::Var log_testing_error_;
  wabt::Var log_prediction_time_;

public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const wabt::Var& LogTestingError() const { return log_testing_error_; }
  const wabt::Var& LogPredictionTime() const { return log_prediction_time_; }
};

} // namespace builtins
} // namespace nn

#endif
