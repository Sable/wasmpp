#ifndef NN_MODEL_H_
#define NN_MODEL_H_

#include <src/wasmpp/wasm-manager.h>

namespace nn {

class Model {
private:
  wasmpp::ModuleManager module_manager_;

public:
  Model(wasmpp::ModuleManagerOptions options) : module_manager_(options) {}
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; };
};

} // namespace nn

#endif
