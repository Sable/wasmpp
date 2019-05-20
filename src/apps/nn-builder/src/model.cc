#include <src/apps/nn-builder/src/model.h>

namespace nn {

Model::Model() {
  wasmpp::ModuleManagerOptions options;
  options.system.EnableAll();
  options.memory.EnableAll();
  options.math.EnableAll();
  module_manager_ = wabt::MakeUnique<wasmpp::ModuleManager>(options);
}

void Model::AddLayer(nn::Layer layer) {
  layers_.push_back(layer);
}

bool Model::RemoveLayer(uint32_t index) {
  if(index >= layers_.size()) {
    return false;
  }
  layers_.erase(layers_.begin() + index);
  return true;
}

Layer Model::GetLayer(uint32_t index) const {
  if(index < layers_.size()) {
    return layers_[index];
  }
  assert(!"Index out of bound");
}

void Model::Setup() {
  // TODO
}

} // namespace nn
