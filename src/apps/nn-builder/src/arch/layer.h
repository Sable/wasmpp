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
public:
  Layer(LayerType type) : type_(type) {}
  LayerType Type() const  { return type_; }
};

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer() : Layer(type) {}
};

class FullyConnectedLayer : public TypedLayer<FullyConnected> {
private:
  uint32_t nodes_;
  builtins::ActivationFunction func_;
public:
  FullyConnectedLayer(uint32_t nodes, builtins::ActivationFunction func) : nodes_(nodes), func_(func) {}
  uint32_t Nodes() const { return nodes_; }
  builtins::ActivationFunction ActivationFunction() const { return func_; }
};

} // namespace arch
} // namespace nn

#endif
