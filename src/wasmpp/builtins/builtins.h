#ifndef WASM_WASM_BUILTINS_H_
#define WASM_WASM_BUILTINS_H_

#include <src/ir.h>

namespace wasmpp {
struct ModuleManagerOptions;
class ModuleManager;
class Builtins {
protected:
  ModuleManager* module_manager_;
  ModuleManagerOptions* options_;
public:
  Builtins(ModuleManager* module_manager, ModuleManagerOptions* options) :
      module_manager_(module_manager),
      options_(options) {}
  virtual void InitImports() = 0;
  virtual void InitDefinitions() = 0;
};

} // namespace wasmpp

#endif
