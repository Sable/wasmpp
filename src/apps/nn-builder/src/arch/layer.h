#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

#include <src/apps/nn-builder/src/builtins/activation.h>

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
public:
  Layer(LayerType type, uint32_t nodes, builtins::ActivationFunction func) : type_(type), nodes_(nodes), func_(func) {}
  LayerType Type() const  { return type_; }
  uint32_t Nodes() const { return nodes_; }
  builtins::ActivationFunction ActivationFunction() const { return func_; }
};

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer(uint32_t nodes, builtins::ActivationFunction func) : Layer(type, nodes, func) {}
};

typedef TypedLayer<FullyConnected> FullyConnectedLayer;
} // namespace arch
} // namespace nn

#endif
