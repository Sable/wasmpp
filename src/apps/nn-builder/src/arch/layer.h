#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

namespace nn {
namespace arch {


enum LayerType {
  FULLY_CONNECTED
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

} // namespace arch
} // namespace nn

#endif
