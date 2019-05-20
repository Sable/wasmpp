#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

namespace nn {
namespace arch {

enum LayerType {
  Input,
  FullyConnected
};

class Layer {
private:
  LayerType type_;
public:
  Layer(LayerType type) : type_(type) {}
  LayerType Type() const  { return type_; }
};

typedef std::shared_ptr<Layer> layer_sptr;

template <LayerType type>
class TypedLayer : public Layer {
public:
  TypedLayer() : Layer(type) {}
};

class InputLayer : public TypedLayer<Input> {
private:
  uint32_t nodes_;
public:
  InputLayer(uint32_t nodes) : nodes_(nodes) {}
};

class FullyConnectedLayer : public TypedLayer<FullyConnected> {
private:
  uint32_t nodes_;
  wabt::Var activation_;
public:
  FullyConnectedLayer(uint32_t nodes, wabt::Var activation) : nodes_(nodes), activation_(activation) {}
  uint32_t Nodes() const { return nodes_; }
  wabt::Var Activation() const { return activation_; }
};

template <class T, typename... Args>
layer_sptr MakeLayer(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

} // namespace arch
} // namespace nn

#endif
