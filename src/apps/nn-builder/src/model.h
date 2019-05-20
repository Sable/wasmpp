#ifndef WASMPP_MODEL_H_
#define WASMPP_MODEL_H_

#include <src/wasmpp/wasm-manager.h>
#include <memory>
#include <utility>

namespace nn {

struct Layer {
  enum Kind {
    FULLY_CONNECTED
  } kind;
  uint32_t nodes;
};

class Model {
private:
  std::unique_ptr<wasmpp::ModuleManager> module_manager_ = nullptr;
  std::vector<Layer> layers_;
public:
  Model();
  void SetLayers(std::vector<Layer> layers) { layers_ = std::move(layers); }
  void AddLayer(Layer layer);
  bool RemoveLayer(uint32_t index);
  Layer GetLayer(uint32_t index) const;
  void Setup();
};

} // namespace nn

#endif
