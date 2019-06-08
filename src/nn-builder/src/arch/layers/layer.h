#ifndef NN_ARCH_LAYER_H_
#define NN_ARCH_LAYER_H_

#include <src/nn-builder/src/builtins/activation.h>
#include <src/nn-builder/src/arch/initializers.h>

namespace nn {
namespace arch {
namespace layer {

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
protected:
  Model* NetworkModel() const { return model_; }
  uint32_t LayerIndex() const { return index_; }
public:
  Layer(LayerType type, LayerPosition position) : type_(type), position_(position) {}
  LayerType Type() const  { return type_; }
  LayerPosition Position() const  { return position_; }
  void SetModel(Model* model) { model_ = model; }
  void SetIndex(uint32_t index) { index_ = index; }
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

} // namespace layer
} // namespace arch
} // namespace nn

#endif
