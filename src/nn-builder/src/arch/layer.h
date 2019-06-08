#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/initializers.h>

namespace nn {
namespace arch {

enum LayerType {
  FullyConnected
};

enum LayerPosition {
  Input,
  Hidden,
  Output
};

class Layer {
private:
  LayerType type_;
  LayerPosition position_;
  Model* model_;
  uint32_t index_;
public:
  Layer(LayerType type, LayerPosition position) : type_(type), position_(position) {}
  LayerType Type() const  { return type_; }
  LayerPosition Position() const  { return position_; }
  void SetModel(Model* model) { model_ = model; }
  Model* NetworkModel() const { return model_; }
  void SetIndex(uint32_t index) { index_ = index; }
  uint32_t LayerIndex() const { return index_; }
  virtual wabt::ExprList* Forward(bool is_training, wabt::Var input_begin, wabt::Var target_begin,
                                  std::vector<wabt::Var> locals) = 0;
  virtual wabt::ExprList* Backward(wabt::Var input_begin, std::vector<wabt::Var> locals) = 0;
  virtual void AllocateMemory() = 0;
  virtual void MakeData(wabt::Var memory) = 0;
};

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer(LayerPosition position) : Layer(type, position) {}
};

template <typename T, typename... Args>
T* NewLayer(Args&&... args) {
  static_assert(std::is_base_of<Layer, T>::value);
  return new T(std::forward<Args>(args)...);
}

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
  FullyConnectedLayer* KeepProb(float keep_prob);
  FullyConnectedLayer* WeightType(WeightDistributionType type);
};

class HiddenDenseLayer : public FullyConnectedLayer {
public:
  HiddenDenseLayer(uint32_t nodes, builtins::ActivationFunction act_func) :
      FullyConnectedLayer(Hidden, nodes, act_func) {}
};

class OutputDenseLayer : public FullyConnectedLayer {
public:
  OutputDenseLayer(uint32_t nodes, builtins::ActivationFunction act_func) :
      FullyConnectedLayer(Output, nodes, act_func) {}
  ds::NDArray* Predictions() const { return A_; }
};

class InputDenseLayer : public FullyConnectedLayer {
public:
  InputDenseLayer(uint32_t nodes, builtins::ActivationFunction act_func) :
  FullyConnectedLayer(Input, nodes, act_func) {}
};

} // namespace arch
} // namespace nn

#endif
