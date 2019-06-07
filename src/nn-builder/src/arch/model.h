#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/arch/layer.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/math.h>
#include <src/nn-builder/src/builtins/system.h>
#include <src/nn-builder/src/builtins/message.h>
#include <src/nn-builder/src/data_structure/ndarray.h>
#include <memory>
#include <utility>
#include <src/nn-builder/src/snippet/matrix.h>
#include "initializers.h"

namespace nn {
namespace arch {

struct LayerMeta;
struct ModelOptions {
  bool log_training_error = false;
  bool log_training_time = false;
  bool log_testing_error = false;
  bool log_testing_time = false;
  bool log_testing_confusion_matrix = false;
  bool use_simd = false;
  builtins::ActivationOptions activation_options;
  WeightDistributionOptions weights_options;
};
class Model {
private:
  ModelOptions options_;
  wasmpp::ModuleManager module_manager_;
  std::vector<LayerMeta*> layers_;
  uint32_t batch_size_;
  builtins::LossFunction loss_;

  // Model members
  wasmpp::Memory* learning_rate = nullptr;
  wabt::ExprList* SetLearningRate(wabt::ExprList* val);
  wabt::ExprList* GetLearningRate();
  void AllocateMembers();

  // Model components variables
  wabt::Var forward_;
  wabt::Var backward_;
  wabt::Var confusion_matrix_func_;

  // Model feature arrays
  ds::NDArray* true_matrix_ = nullptr;
  ds::NDArray* cost_matrix_ = nullptr;
  ds::NDArray* confusion_matrix_ = nullptr;

  // Training data
  std::vector<ds::NDArray*> training_;
  std::vector<ds::NDArray*> training_labels_;
  std::vector<std::vector<float>> training_vals_;
  std::vector<std::vector<float>> training_labels_vals_;

  // Test data
  std::vector<ds::NDArray*> testing_;
  std::vector<ds::NDArray*> testing_labels_;
  std::vector<std::vector<float>> testing_vals_;
  std::vector<std::vector<float>> testing_labels_vals_;

  // Builtin functions
  struct Builtins {
    Builtins(builtins::ActivationOptions activation_options) : activation(activation_options) {}
    builtins::Activation activation;
    builtins::Loss loss;
    builtins::Math math;
    builtins::System system;
    builtins::Message message;
  } builtins_;

  // Snippet codes
  struct Snippets {
    snippet::MatrixSnippet* matrix;
  } snippets_;

  // Initalize functions
  void InitImports();
  void InitDefinitions();
  // Model functions
  void AllocateLayers();
  void AllocateTraining();
  void AllocateTest();
  void MakeTrainingData(wabt::Var memory);
  void MakeTestingData(wabt::Var memory);
  void MakeWeightData(wabt::Var memory);
  // Generate neural network algorithms
  wabt::Var GenerateFeedForward();
  wabt::Var GenerateBackpropagation();
  wabt::Var GenerateConfusionMatrixFunction();
public:
  Model(ModelOptions options);
  ~Model();
  const wasmpp::ModuleManager& ModuleManager() const { return module_manager_; }
  void SetLayers(std::vector<Layer*> layers);
  void AddLayer(Layer* layer);
  bool RemoveLayer(uint32_t index);
  Layer* GetLayer(uint32_t index) const;

  void CompileLayers(uint32_t batch_size, builtins::LossFunction loss);
  void CompileTraining(uint32_t epoch, float learning_rate, const std::vector<std::vector<float>> &input,
                       const std::vector<std::vector<float>> &labels);
  void CompileTesting(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &labels);
  void CompileInitialization();
  bool Validate();
  const Builtins& Builtins() const { return builtins_; }
};

} // namespace arch
} // namespace nn

#endif
