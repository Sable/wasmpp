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
public:
  Layer(LayerType type, uint32_t nodes, builtins::ActivationFunction func, WeightDistributionType weight_type) :
      type_(type), nodes_(nodes), func_(func), weight_type_(weight_type) {}
  LayerType Type() const  { return type_; }
  uint32_t Nodes() const { return nodes_; }
  WeightDistributionType WeightType() const { return weight_type_; }
  builtins::ActivationFunction ActivationFunction() const { return func_; }
};

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer(uint32_t nodes, builtins::ActivationFunction func, WeightDistributionType weight_type) :
      Layer(type, nodes, func, weight_type) {}
};

typedef TypedLayer<FullyConnected> FullyConnectedLayer;
} // namespace arch
} // namespace nn

#endif
