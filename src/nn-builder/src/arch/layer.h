#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/initializers.h>

namespace nn {
namespace arch {

enum LayerType {
  FullyConnected
};

class Layer {
private:
  LayerType type_;
  uint32_t nodes_;
  builtins::ActivationFunction func_;
  WeightDistributionType weight_type_;
  float keep_prob_ = 1.0;
public:
  Layer(LayerType type, uint32_t nodes, builtins::ActivationFunction func, WeightDistributionType weight_type, float keep_prob) :
      type_(type), nodes_(nodes), func_(func), weight_type_(weight_type), keep_prob_(keep_prob) {}
  LayerType Type() const  { return type_; }
  uint32_t Nodes() const { return nodes_; }
  float KeepProb() const { assert(keep_prob_ >= 0.0 && keep_prob_ <= 1.0); return keep_prob_; }
  WeightDistributionType WeightType() const { return weight_type_; }
  builtins::ActivationFunction ActivationFunction() const { return func_; }
};

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer(uint32_t nodes, builtins::ActivationFunction func, WeightDistributionType weight_type, float keep_prob) :
      Layer(type, nodes, func, weight_type, keep_prob) {}
};

typedef TypedLayer<FullyConnected> FullyConnectedLayer;
} // namespace arch
} // namespace nn

#endif
