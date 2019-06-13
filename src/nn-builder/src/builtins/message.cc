#include <src/nn-builder/src/builtins/message.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Message::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
  log_training_accuracy_ = module_manager->MakeFuncImport(module_name, "log_training_accuracy", {{Type::I32, Type::F32}, {}});
  log_training_time_ = module_manager->MakeFuncImport(module_name, "log_training_time", {{Type::I32, Type::F64, Type::F64}, {}});
  log_training_error_ = module_manager->MakeFuncImport(module_name, "log_training_error", {{Type::I32, Type::F32}, {}});
  log_testing_accuracy_ = module_manager->MakeFuncImport(module_name, "log_testing_accuracy", {{Type::F32}, {}});
  log_testing_time_ = module_manager->MakeFuncImport(module_name, "log_testing_time", {{Type::F64}, {}});
  log_testing_error_ = module_manager->MakeFuncImport(module_name, "log_testing_error", {{Type::F32}, {}});
  log_prediction_time_ = module_manager->MakeFuncImport(module_name, "log_prediction_time", {{Type::F64}, {}});

#define LOAD_TIME_MESSAGES(name) \
  log_forward_##name##_ = module_manager->MakeFuncImport(module_name, "log_forward_" #name, {{Type::F64}, {}});
FORWARD_TIME_MESSAGES(LOAD_TIME_MESSAGES)
#undef LOAD_TIME_MESSAGES

#define LOAD_TIME_MESSAGES(name) \
  log_backward_##name##_ = module_manager->MakeFuncImport(module_name, "log_backward_" #name, {{Type::F64}, {}});
BACKWARD_TIME_MESSAGES(LOAD_TIME_MESSAGES)
#undef LOAD_TIME_MESSAGES
}

void Message::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

} // namespace builtins
} // namespace nn
