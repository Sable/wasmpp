#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/arch/layer.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/math.h>
#include <src/nn-builder/src/builtins/system.h>
#include <src/nn-builder/src/builtins/message.h>
#include <src/nn-builder/src/data_structure/ndarray.h>
#include <memory>
#include <utility>

namespace nn {
namespace arch {

struct LayerMeta;
struct ModelMeta;
struct ModelOptions {
  bool log_training_error = false;
  bool log_training_time = false;
  builtins::ActivationOptions activation_options;
};
class Model {
private:
  ModelOptions options_;
  wasmpp::ModuleManager module_manager_;
  wabt::Var memory_;
  std::vector<LayerMeta*> layers_;
  std::vector<ds::NDArray*> training_;
  std::vector<ds::NDArray*> labels_;
  uint32_t epochs_;
  uint32_t batch_size_;
  double learning_rate_;

  // Builtin functions
  struct Builtins {
    Builtins(builtins::ActivationOptions activation_options) : activation(activation_options) {}
    builtins::Activation activation;
    builtins::Loss loss;
    builtins::Math math;
    builtins::System system;
    builtins::Message message;
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
  Model(ModelOptions options);
  ~Model();
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<Layer*> layers);
  void AddLayer(Layer* layer);
  bool RemoveLayer(uint32_t index);
  Layer* GetLayer(uint32_t index) const;
  void Setup(uint32_t epochs, uint32_t batch_size, double learning_rate, std::vector<std::vector<double>> input,
             std::vector<std::vector<double>> labels);
  void Train();
  bool Validate();
  const Builtins& Builtins() const { return builtins_; }
};

} // namespace arch
} // namespace nn

#endif
