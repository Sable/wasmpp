#include <src/apps/nn-builder/src/builtins/system.h>
#include <cassert>

namespace nn {
namespace builtins {

using namespace wabt;
using namespace wasmpp;

void System::InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
  print_i32_ = module_manager->MakeFuncImport(module_name, "print", {{Type::I32}, {}});
  print_i64_ = module_manager->MakeFuncImport(module_name, "print", {{Type::I64}, {}});
  print_f32_ = module_manager->MakeFuncImport(module_name, "print", {{Type::F32}, {}});
  print_f64_ = module_manager->MakeFuncImport(module_name, "print", {{Type::F64}, {}});
  print_table_f64_ = module_manager->MakeFuncImport(module_name, "print_table_f64",
      {{Type::I32, Type::I32, Type::I32}, {}});
}

void System::InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) {
  assert(model != nullptr);
  assert(module_manager != nullptr);
}

} // namespace builtins
} // namespace nn
