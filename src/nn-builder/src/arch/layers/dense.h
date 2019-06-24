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
  ds::NDArray* Z_[3]; // Training, Testing, Prediction
  ds::NDArray* A_[3]; // Training, Testing, Prediction
  ds::NDArray* b_ = nullptr;
  // Back-propagation arrays
  ds::NDArray* dW_ = nullptr;
  ds::NDArray* dZ_ = nullptr;
  ds::NDArray* dA_ = nullptr;
  ds::NDArray* db_ = nullptr;
  // Regularization
  ds::NDArray* inverted_dropout_ = nullptr;
public:
  FullyConnectedLayer(LayerPosition position, uint32_t nodes, builtins::ActivationFunction act_func) :
      TypedLayer(position), nodes_(nodes), activation_func_(act_func) {}
  uint32_t Nodes() const { return nodes_; }
  wabt::ExprList* Forward(uint8_t mode_index, wabt::Var input_begin, std::vector<wabt::Var> locals) override;
  wabt::ExprList* Backward(wabt::Var input_begin, wabt::Var taget_begin, std::vector<wabt::Var> locals) override;

  // Memory functions
  void AllocateMemory() override ;
  void MakeData(wabt::Var memory) override ;

  // Create functions
  void MakeFunctions() override;

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
  // Allocate memory
  void AllocateMemory() override ;
  // Get arrays
  ds::NDArray* Predictions(uint8_t mode_index) const;
  ds::NDArray* PredictionsHardmax(uint8_t mode_index) const;
  ds::NDArray* PredictionsSoftmax(uint8_t mode_index) const;
  ds::NDArray* ConfusionMatrix(uint8_t mode_index) const;
  // Error out on keep probability on the output layer
  FullyConnectedLayer* KeepProb(float keep_prob) override ;
  // Apply softmax to predictions
  DenseOutputLayer* Softmax(uint8_t mode_index);
  // Apply hardmax to predictions
  DenseOutputLayer* Hardmax(uint8_t mode_index);
  // Augment forward algorithm
  wabt::ExprList* Forward(uint8_t mode_index, wabt::Var input_begin, std::vector<wabt::Var> locals) final;
  void MakeFunctions() override ;

  // Compute cost value
  wabt::ExprList* ComputeCost(uint8_t mode_index, wabt::Var target_begin);
  // Update confusion matrix
  wabt::ExprList* UpdateConfusionMatrix(uint8_t mode_index, wabt::Var target_begin, std::vector<wabt::Var> locals);
  // Count number of correct predictions
  wabt::ExprList* CountCorrectPredictions(uint8_t mode_index, wabt::Var target_begin, wabt::Var result, std::vector<wabt::Var> locals);
private:
  bool softmax_[3] = {false, false, false}; // Training, Testing, Prediction
  bool hardmax_[3] = {false, false, false}; // Training, Testing, Prediction
  ds::NDArray* A_hardmax_[3]; // Training, Testing, Prediction
  ds::NDArray* A_softmax_[3]; // Training, Testing, Prediction
  ds::NDArray* confusion_matrix_[2]; // Training, Testing
};

class DenseInputLayer : public FullyConnectedLayer {
public:
  // Input Dense Layer does not have an activation function
  // and it will not be used in the forward or
  // the backward algorithms
  DenseInputLayer(uint32_t nodes) : FullyConnectedLayer(Input, nodes, builtins::ActivationFunction()) {}
  ds::NDArray* InputData(uint8_t mode_index) const;
};

} // namespace layer
} // namespace arch
} // namespace nn

#endif