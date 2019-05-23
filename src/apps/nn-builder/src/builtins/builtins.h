#ifndef NN_BUILTINS_BUILTINS_H_
#define NN_BUILTINS_BUILTINS_H_

#include <string>
#include <src/wasmpp/wasm-manager.h>

namespace nn {
namespace arch { class Model; }
namespace builtins {

class Builtin {
public:
  virtual void InitImports(arch::Model* model, wasmpp::ModuleManager* module_manager, std::string module_name) = 0;
  virtual void InitDefinitions(arch::Model* model, wasmpp::ModuleManager* module_manager) = 0;
};

} // namespace builtins
} // namespace nn

#endif
