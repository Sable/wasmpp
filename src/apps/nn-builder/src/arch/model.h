#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/apps/nn-builder/src/arch/layer.h>
#include <src/apps/nn-builder/src/builtins/activation.h>
#include <src/apps/nn-builder/src/builtins/loss.h>
#include <src/apps/nn-builder/src/builtins/math.h>
#include <src/apps/nn-builder/src/builtins/system.h>
#include <src/apps/nn-builder/src/data_structure/ndarray.h>
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
    builtins::Loss loss;
    builtins::Math math;
    builtins::System system;
  } builtins_;

  // Initalize functions
  void InitImports();
  void InitDefinitions();
  // Setup layers
  void SetupLayers(uint32_t batch_size);
  // Generate neural network algorithms
  wabt::Var GenerateFeedForward();
  wabt::Var GenerateBackpropagation();
public:
  Model();
  ~Model();
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<Layer*> layers);
  void AddLayer(Layer* layer);
  bool RemoveLayer(uint32_t index);
  Layer* GetLayer(uint32_t index) const;
  void Train(uint32_t batch_size, std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels);
  bool Validate();
  const Builtins& Builtins() const { return builtins_; }
};

} // namespace arch
} // namespace nn

#endif
