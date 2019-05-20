#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/apps/nn-builder/src/arch/layer.h>
#include <memory>
#include <utility>

namespace nn {
namespace arch {

class Model {
private:
  wasmpp::ModuleManager module_manager_;
  std::vector<layer_sptr> layers_;

  // Builtin functions
  struct {
    // system functions
    wabt::Var print_i32;
    wabt::Var print_i64;
    wabt::Var print_f32;
    wabt::Var print_f64;

    // math functions
    wabt::Var exp;
    wabt::Var sigmoid;
  } builtins;

  // Initalize all import functions
  void InitImports();
  // Initialize all defined functions
  void InitDefinitions();
public:
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<layer_sptr> layers) { layers_ = std::move(layers); }
  void AddLayer(layer_sptr layer);
  bool RemoveLayer(uint32_t index);
  layer_sptr GetLayer(uint32_t index) const;
  void Setup();
  bool Validate();
};

} // namespace arch
} // namespace nn

#endif
