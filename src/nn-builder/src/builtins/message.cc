#include <src/nn-builder/src/builtins/message.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Message::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
  log_testing_accuracy_ = module_manager->MakeFuncImport(module_name, "log_testing_accuracy", {{Type::F32}, {}});
  log_testing_time_ = module_manager->MakeFuncImport(module_name, "log_testing_time", {{Type::F64}, {}});
  log_testing_error_ = module_manager->MakeFuncImport(module_name, "log_testing_error", {{Type::F32}, {}});
  log_prediction_time_ = module_manager->MakeFuncImport(module_name, "log_prediction_time", {{Type::F64}, {}});
}

void Message::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

} // namespace builtins
} // namespace nn
