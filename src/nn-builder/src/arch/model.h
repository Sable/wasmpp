#ifndef NN_ARCH_MODEL_H_
#define NN_ARCH_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <src/nn-builder/src/arch/layers/layer.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/builtins/loss.h>
#include <src/nn-builder/src/builtins/math.h>
#include <src/nn-builder/src/builtins/system.h>
#include <src/nn-builder/src/data_structure/ndarray.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/src/snippet/analysis.h>
#include <src/nn-builder/src/arch/initializers.h>
#include <memory>
#include <utility>

namespace nn {
namespace arch {

// Specify optional features for the mode
// Each feature has its corresponding bytecode
// in model, and it will be injected depending
// on whether it's enabled or not
struct ModelBytecodeOptions {
  bool gen_training_accuracy            = false;
  bool gen_training_error               = false;
  bool gen_training_confusion_matrix    = false;
  bool gen_training_time                = true;
  bool gen_testing_accuracy             = false;
  bool gen_testing_error                = false;
  bool gen_testing_confusion_matrix     = false;
  bool gen_testing_time                 = true;
  bool gen_prediction_results           = false;
  bool gen_prediction_results_softmax   = false;
  bool gen_prediction_results_hardmax   = false;
  bool gen_prediction_time              = true;
  bool gen_forward                      = false;
  bool gen_backward                     = false;
  bool use_simd                         = false;
};

struct ModelOptions {
  ModelBytecodeOptions bytecode_options;
  builtins::ActivationOptions activation_options;
  WeightDistributionOptions weights_options;
};

struct BuiltinFunctions {
  BuiltinFunctions(builtins::ActivationOptions activation_options) : activation(activation_options) {}
  builtins::Activation activation;
  builtins::Loss loss;
  builtins::Math math;
  builtins::System system;
};

struct SnippetCode {
  snippet::MatrixSnippet* matrix;
  snippet::AnalysisSnippet* analysis;
};

#ifdef WABT_EXPERIMENTAL
struct NativeFunctions {
  wabt::Var dot_product;
  wabt::Var dot_product_rt;
  wabt::Var dot_product_lt;
};
#endif

// The fields in the below structs correspond
// to the forward and backward algorithm steps
// in a fully connected layer
// read the comment to associate each
// field to its function
// The fields are used for logging the time
// it takes to compute each step of the neural
// network algorithms
#define DEFINE_MEMBERS(name)                        \
  wasmpp::Memory* name = nullptr;                   \
  wabt::ExprList* Get##name();                      \
  wabt::ExprList* Set##name(wabt::ExprList* value);
#define DENSE_FORWARD_TIME_MEMBERS(V) \
  V(Time)                             \
  V(A_1)                              \
  V(A_2)                              \
  V(B)
struct DenseForwardTimeMembers {
  DENSE_FORWARD_TIME_MEMBERS(DEFINE_MEMBERS)
};

#define DENSE_BACKWARD_TIME_MEMBERS(V)  \
  V(Time)                               \
  V(A)                                  \
  V(B_1)                                \
  V(B_2)                                \
  V(C_1)                                \
  V(C_2)                                \
  V(D_1)                                \
  V(D_2)                                \
  V(E)                                  \
  V(F_1)                                \
  V(F_2)                                \
  V(G_1)                                \
  V(G_2)
struct DenseBackwardTimeMembers {
  DENSE_BACKWARD_TIME_MEMBERS(DEFINE_MEMBERS)
};
#undef DEFINE_MEMBERS

class Model {
private:
  ModelOptions options_;
  wasmpp::ModuleManager module_manager_;
  std::vector<layer::Layer*> layers_;
  builtins::LossFunction loss_;
  uint32_t training_batch_size_;
  uint32_t testing_batch_size_;
  uint32_t prediction_batch_size_;
  uint32_t training_batches_in_memory_;
  uint32_t testing_batches_in_memory_;

  // Model members
  wasmpp::Memory* learning_rate_ = nullptr;
  wasmpp::Memory* training_error_ = nullptr;
  wasmpp::Memory* training_hits_ = nullptr;
  wasmpp::Memory* training_time_ = nullptr;
  wasmpp::Memory* testing_hits_ = nullptr;
  wasmpp::Memory* testing_error_ = nullptr;
  wasmpp::Memory* testing_time_ = nullptr;
  wasmpp::Memory* prediction_time_ = nullptr;
  DenseForwardTimeMembers dense_forward_logging_members_;
  DenseBackwardTimeMembers dense_backward_logging_members_;

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
  wasmpp::Memory* training_data_batches_;
  wasmpp::Memory* training_labels_batches_;

  // Test data
  wasmpp::Memory* testing_data_batches_;
  wasmpp::Memory* testing_labels_batches_;

  // Builtin functions
  BuiltinFunctions builtins_;

  // Snippet codes
  SnippetCode snippets_;

  // Initialize builtins
  void InitBuiltinImports();
  void InitBuiltinDefinitions();

  // Initialize snippets
  void InitSnippets();

#ifdef WABT_EXPERIMENTAL
  // Native functions
  NativeFunctions natives_;

  // Initialize native imports
  void InitNativeImports();
#endif

  // Memory allocation
  void AllocateMembers();
  void AllocateLayers();
  void AllocateMemory();

  // Generate neural network algorithms
  wabt::Var ForwardAlgorithmFunction(uint8_t mode_index);
  wabt::Var BackwardAlgorithmFunction();
  wabt::Var ConfusionMatrixFunction(uint8_t mode_index);
  wabt::Var CountCorrectPredictionsFunction(uint8_t mode_index);

  // Make functions
  void MakeLayersFunctions();
  void MakeAlgorithmsFunctions();
  void MakeTrainingFunctions();
  void MakeTestingFunctions();
  void MakePredictionFunctions();
  void MakeData();

  // Helper
  void GetInputOutputSize(uint32_t *input_size, uint32_t *output_size);
public:
  Model(ModelOptions options);
  wasmpp::ModuleManager& ModuleManager() { return module_manager_; }
  std::vector<layer::Layer*> Layers() const { return layers_; }
  void SetLayers(std::vector<layer::Layer*> layers);

  // Build model
  void Build(uint32_t training_batch_size, uint32_t training_batches_in_memory,
                           uint32_t testing_batch_size, uint32_t testing_batches_in_memory,
                           uint32_t prediction_batch_size, builtins::LossFunction loss);

  // Members accessors
  wabt::ExprList* SetLearningRate(wabt::ExprList* val);
  wabt::ExprList* GetLearningRate();
  DenseForwardTimeMembers DenseForwardTime() const { return dense_forward_logging_members_; }
  DenseBackwardTimeMembers DenseBackwardTime() const { return dense_backward_logging_members_; }

  bool Validate();
  const BuiltinFunctions& Builtins() const { return builtins_; }
  const SnippetCode& Snippets() const { return snippets_; }
#ifdef WABT_EXPERIMENTAL
  const NativeFunctions& Natives() const { return natives_; }
#endif
  uint32_t TrainingBatchSize() const { return training_batch_size_; }
  uint32_t TrainingBatchesInMemory() const { return training_batches_in_memory_; }
  uint32_t TestingBatchSize() const { return testing_batch_size_; }
  uint32_t TestingBatchesInMemory() const { return testing_batches_in_memory_; }
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
