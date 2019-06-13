#ifndef NN_BUILTINS_MESSAGE_H_
#define NN_BUILTINS_MESSAGE_H_

#include <src/nn-builder/src/builtins/builtins.h>

namespace nn {
namespace builtins {

#define FORWARD_TIME_MESSAGES(V)  \
  V(Time)                         \
  V(A_1)                          \
  V(A_2)                          \
  V(B)

#define BACKWARD_TIME_MESSAGES(V)  \
  V(Time)                          \
  V(A)                             \
  V(B_1)                           \
  V(B_2)                           \
  V(C_1)                           \
  V(C_2)                           \
  V(D_1)                           \
  V(D_2)                           \
  V(E)                             \
  V(F_1)                           \
  V(F_2)                           \
  V(G_1)                           \
  V(G_2)

class Message : public Builtin {
private:
  wabt::Var log_training_accuracy_;
  wabt::Var log_training_time_;
  wabt::Var log_training_error_;
  wabt::Var log_testing_accuracy_;
  wabt::Var log_testing_time_;
  wabt::Var log_testing_error_;
  wabt::Var log_prediction_time_;

#define VAR_NAMES(name) \
  wabt::Var log_forward_##name##_;
FORWARD_TIME_MESSAGES(VAR_NAMES)
#undef VAR_NAMES

#define VAR_NAMES(name) \
  wabt::Var log_backward_##name##_;
BACKWARD_TIME_MESSAGES(VAR_NAMES)
#undef VAR_NAMES

    public:
  void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) override;
  void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) override;

  const wabt::Var& LogTrainingTime() const { return log_training_time_; }
  const wabt::Var& LogTrainingError() const { return log_training_error_; }
  const wabt::Var& LogTrainingAccuracy() const { return log_training_accuracy_; }
  const wabt::Var& LogTestingTime() const { return log_testing_time_; }
  const wabt::Var& LogTestingError() const { return log_testing_error_; }
  const wabt::Var& LogTestingAccuracy() const { return log_testing_accuracy_; }
  const wabt::Var& LogPredictionTime() const { return log_prediction_time_; }

#define GET_VAR_NAMES(name) \
  const wabt::Var& LogForward##name() const { return log_forward_##name##_; }
FORWARD_TIME_MESSAGES(GET_VAR_NAMES)
#undef GET_VAR_NAMES

#define GET_VAR_NAMES(name) \
  const wabt::Var& LogBackward##name() const { return log_backward_##name##_; }
      BACKWARD_TIME_MESSAGES(GET_VAR_NAMES)
#undef GET_VAR_NAMES
    };

} // namespace builtins
} // namespace nn

#endif
