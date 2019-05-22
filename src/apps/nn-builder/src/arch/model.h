#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/apps/nn-builder/src/arch/layer.h>
#include <memory>
#include <utility>

namespace nn {
namespace arch {

struct LayerMeta;
class Model {
private:
  wasmpp::ModuleManager module_manager_;
  std::vector<LayerMeta*> layers_;

  // Builtin functions
  struct Builtins {
    // system functions
    wabt::Var print_i32;
    wabt::Var print_i64;
    wabt::Var print_f32;
    wabt::Var print_f64;

    // math functions
    wabt::Var exp;
    wabt::Var sigmoid;
    wabt::Var dsigmoid;
  } builtins;

  // Initalize all import functions
  void InitImports();
  // Initialize all defined functions
  void InitDefinitions();
  // Setup layers
  void SetupLayers();
  // Generate feed-forward algorithm
  wabt::Var GenerateFeedForward();
public:
  ~Model();
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<Layer*> layers);
  void AddLayer(Layer* layer);
  bool RemoveLayer(uint32_t index);
  Layer* GetLayer(uint32_t index) const;
  void Setup();
  bool Validate();
  const Builtins& Builtins() const { return builtins; }
};

} // namespace arch
} // namespace nn

#endif
