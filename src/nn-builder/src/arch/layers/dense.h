#ifndef NN_ARCH_LAYER_DENSE_H_
#define NN_ARCH_LAYER_DENSE_H_

#include <src/nn-builder/src/arch/layers/layer.h>

namespace nn {
namespace arch {
namespace layer {

class FullyConnectedLayer : public TypedLayer<FullyConnected> {
protected:
  // Number of nodes
  uint32_t nodes_;
  // Activation function
  builtins::ActivationFunction activation_func_;
  // Weight distribution initalization
  WeightDistributionType weight_type_;
  // Regularization
  const float KEEP_PROB_MAX = 1.0;
  const float KEEP_PROB_MIN = 0.0;
  float keep_prob_ = KEEP_PROB_MAX;
  // Feed-forward arrays
  ds::NDArray* W_ = nullptr;
  ds::NDArray* Z_ = nullptr;
  ds::NDArray* A_ = nullptr;
  ds::NDArray* b_ = nullptr;
  // Back-propagation arrays
  ds::NDArray* dW_ = nullptr;
  ds::NDArray* dZ_ = nullptr;
  ds::NDArray* dA_ = nullptr;
  ds::NDArray* db_ = nullptr;
  // Regularization
  ds::NDArray* inverted_dropout_ = nullptr;
  // Other
  ds::NDArray* A_hardmax_ = nullptr;
  ds::NDArray* confusion_matrix_ = nullptr;
public:
  FullyConnectedLayer(LayerPosition position, uint32_t nodes, builtins::ActivationFunction act_func) :
      TypedLayer(position), nodes_(nodes), activation_func_(act_func) {}
  uint32_t Nodes() const { return nodes_; }
  wabt::ExprList* Forward(bool is_training, wabt::Var input_begin, wabt::Var target_begin,
                          std::vector<wabt::Var> locals);
  wabt::ExprList* Backward(wabt::Var input_begin, std::vector<wabt::Var> locals);

  // Memory functions
  void AllocateMemory();
  void MakeData(wabt::Var memory);

  // Layer configuration
  virtual FullyConnectedLayer* KeepProb(float keep_prob);
  FullyConnectedLayer* WeightType(WeightDistributionType type);
};

class DenseHiddenLayer : public FullyConnectedLayer {
public:
  DenseHiddenLayer(uint32_t nodes, builtins::ActivationFunction act_func) :
      FullyConnectedLayer(Hidden, nodes, act_func) {}
};

class DenseOutputLayer : public FullyConnectedLayer {
public:
  DenseOutputLayer(uint32_t nodes, builtins::ActivationFunction act_func) :
      FullyConnectedLayer(Output, nodes, act_func) {}
  // Get arrays
  ds::NDArray* Predictions() const { return A_; }
  ds::NDArray* PredictionsHardmax() const { return A_hardmax_; }
  ds::NDArray* ConfusionMatrix() const { return confusion_matrix_; }
  // Error out on keep probability on the output layer
  FullyConnectedLayer* KeepProb(float keep_prob);
  // Perform hardmax on the predictions
  wabt::ExprList* HardmaxPredictions(std::vector<wabt::Var> locals);
  // Update confusion matrix
  wabt::ExprList* UpdateConfusionMatrix(wabt::Var target_begin, std::vector<wabt::Var> locals);
  // Count number of correct predictions
  wabt::ExprList* CountCorrectPredictions(wabt::Var target_begin, wabt::Var result, std::vector<wabt::Var> locals);
};

class DenseInputLayer : public FullyConnectedLayer {
public:
  // Input Dense Layer does not have an activation function
  // and it will not be used in the forward or
  // the backward algorithms
  DenseInputLayer(uint32_t nodes) :
      FullyConnectedLayer(Input, nodes, builtins::ActivationFunction()) {}
};

} // namespace layer
} // namespace arch
} // namespace nn

#endif