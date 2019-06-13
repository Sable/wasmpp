#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/arch/layers/layer.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/math.h>
#include <src/nn-builder/src/builtins/system.h>
#include <src/nn-builder/src/builtins/message.h>
#include <src/nn-builder/src/data_structure/ndarray.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/src/snippet/analysis.h>
#include <src/nn-builder/src/arch/initializers.h>
#include <memory>
#include <utility>

namespace nn {
namespace arch {

struct ModelOptions {
  bool log_training_accuracy = false;
  bool log_training_error = false;
  bool log_training_time = false;
  bool log_training_confusion_matrix = false;
  bool log_testing_accuracy = false;
  bool log_testing_error = false;
  bool log_testing_time = false;
  bool log_testing_confusion_matrix = false;
  bool log_prediction_results = false;
  bool log_prediction_results_softmax = false;
  bool log_prediction_results_hardmax = false;
  bool log_prediction_time = false;
  bool use_simd = false;
  builtins::ActivationOptions activation_options;
  WeightDistributionOptions weights_options;
};

struct BuiltinFunctions {
  BuiltinFunctions(builtins::ActivationOptions activation_options) : activation(activation_options) {}
  builtins::Activation activation;
  builtins::Loss loss;
  builtins::Math math;
  builtins::System system;
  builtins::Message message;
};

struct SnippetCode {
  snippet::MatrixSnippet* matrix;
  snippet::AnalysisSnippet* analysis;
};

class Model {
private:
  ModelOptions options_;
  wasmpp::ModuleManager module_manager_;
  std::vector<layer::Layer*> layers_;
  builtins::LossFunction loss_;
  uint32_t training_batch_size_;
  uint32_t testing_batch_size_;
  uint32_t prediction_batch_size_;

  // Limits
  const uint32_t MAX_FLOAT_PER_DATA = 1048576; // 2^20

  // Model members
  wasmpp::Memory* learning_rate = nullptr;

  // Model functions
  wabt::Var forward_training_func_;
  wabt::Var forward_testing_func_;
  wabt::Var forward_prediction_func_;
  wabt::Var backward_func_;
  wabt::Var confusion_matrix_training_func_;
  wabt::Var confusion_matrix_testing_func_;
  wabt::Var count_correct_predictions_training_func_;
  wabt::Var count_correct_predictions_testing_func_;

  // Training data
  std::vector<ds::NDArray*> training_batch_;
  std::vector<ds::NDArray*> training_labels_batch_;
  std::vector<std::vector<float>> training_vals_;
  std::vector<std::vector<float>> training_labels_vals_;

  // Test data
  std::vector<ds::NDArray*> testing_batch_;
  std::vector<ds::NDArray*> testing_labels_batch_;
  std::vector<std::vector<float>> testing_vals_;
  std::vector<std::vector<float>> testing_labels_vals_;

  // Builtin functions
  BuiltinFunctions builtins_;

  // Snippet codes
  SnippetCode snippets_;

  // Initialize builtins
  void InitBuiltinImports();
  void InitBuiltinDefinitions();

  // Initialize snippets
  void InitSnippets();

  // Memory allocation
  void AllocateMembers();
  void AllocateLayers();
  void AllocateTraining();
  void AllocateTest();

  // Date generation
  void MakeData(wabt::Var memory, std::vector<std::vector<float>> data_vals,
                std::vector<std::vector<float>> labels_vals, std::vector<ds::NDArray*> data_batch,
                std::vector<ds::NDArray*> labels_batch, uint32_t batch_size);
  void MakeTrainingData(wabt::Var memory);
  void MakeTestingData(wabt::Var memory);
  void MakeLayersData(wabt::Var memory);

  // Generate neural network algorithms
  wabt::Var ForwardAlgorithmFunction(uint8_t mode_index);
  wabt::Var BackwardAlgorithmFunction();
  wabt::Var ConfusionMatrixFunction(uint8_t mode_index);
  wabt::Var CountCorrectPredictionsFunction(uint8_t mode_index);
public:
  Model(ModelOptions options);
  wasmpp::ModuleManager& ModuleManager() { return module_manager_; }
  std::vector<layer::Layer*> Layers() const { return layers_; }
  void SetLayers(std::vector<layer::Layer*> layers);

  // Compile functions
  void CompileLayers(uint32_t training_batch_size, uint32_t testing_batch_size, uint32_t prediction_batch_size,
                     builtins::LossFunction loss);
  void CompileTrainingFunction(uint32_t epoch, float learning_rate, const std::vector<std::vector<float>> &input,
                       const std::vector<std::vector<float>> &labels);
  void CompileTestingFunction(const std::vector<std::vector<float>> &input, const std::vector<std::vector<float>> &labels);
  void CompilePredictionFunctions();
  void CompileWeightsFunctions();
  void CompileInitialization();

  // Members accessors
  wabt::ExprList* SetLearningRate(wabt::ExprList* val);
  wabt::ExprList* GetLearningRate();

  bool Validate();
  const BuiltinFunctions& Builtins() const { return builtins_; }
  const SnippetCode& Snippets() const { return snippets_; }
  uint32_t TrainingBatchSize() const { return training_batch_size_; }
  uint32_t TestingBatchSize() const { return testing_batch_size_; }
  uint32_t PredictionBatchSize() const { return prediction_batch_size_; }
  const ModelOptions& Options() const { return options_; }
  const builtins::LossFunction& Loss() const { return loss_; }

  enum Mode : uint8_t {
    Training = 0,
    Testing,
    Prediction,

    // Keep those pointers at the end
    // because enums are used as indices
    FIRST_MODE = Training,
    LAST_MODE = Prediction
  };
};

} // namespace arch
} // namespace nn

#endif
