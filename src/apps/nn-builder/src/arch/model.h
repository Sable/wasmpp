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
struct ModelMeta;
class Model {
private:
  wasmpp::ModuleManager module_manager_;
  wabt::Var memory_;
  std::vector<LayerMeta*> layers_;
  std::vector<ds::NDArray*> training_;
  std::vector<ds::NDArray*> labels_;
  uint32_t batch_size_;
  double learning_rate_;

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
  // Model functions
  void AllocateLayers();
  void AllocateInput(std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels);
  void MakeInputData(std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels);
  void MakeWeightData();
  void MakeBiasData();
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
  void Setup(uint32_t batch_size, double learning_rate, std::vector<std::vector<double>> input,
             std::vector<std::vector<double>> labels);
  void Train();
  bool Validate();
  const Builtins& Builtins() const { return builtins_; }
};

} // namespace arch
} // namespace nn

#endif
