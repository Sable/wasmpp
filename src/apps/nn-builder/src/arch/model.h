#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/apps/nn-builder/src/arch/layer.h>
#include <src/apps/nn-builder/src/builtins/activation.h>
#include <src/apps/nn-builder/src/builtins/math.h>
#include <src/apps/nn-builder/src/builtins/system.h>
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
    builtins::Activation activation;
    builtins::Math math;
    builtins::System system;
  } builtins_;

  // Initalize functions
  void InitImports();
  void InitDefinitions();
  // Setup layers
  void SetupLayers();
  // Generate neural network algorithms
  wabt::Var GenerateFeedForward();
  wabt::Var GenerateBackpropagation();
public:
  ~Model();
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<Layer*> layers);
  void AddLayer(Layer* layer);
  bool RemoveLayer(uint32_t index);
  Layer* GetLayer(uint32_t index) const;
  void Setup();
  bool Validate();
  const Builtins& Builtins() const { return builtins_; }
};

} // namespace arch
} // namespace nn

#endif
