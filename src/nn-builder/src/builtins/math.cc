#include <src/nn-builder/src/builtins/math.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void Math::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
  exp_ = module_manager->MakeFuncImport(module_name, "exp", {{Type::F64}, {Type::F64}});
  log_ = module_manager->MakeFuncImport(module_name, "log", {{Type::F64}, {Type::F64}});
}

void Math::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

} // namespace builtins
} // namespace nn
