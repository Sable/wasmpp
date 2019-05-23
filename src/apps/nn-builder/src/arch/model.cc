#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/data_structure/ndarray.h>
#include <src/apps/nn-builder/src/math/matrix.h>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;

struct LayerMeta {
  LayerMeta(Layer* l):layer(l) {}
  Layer* layer = nullptr;
  // weights
  ds::NDArray* W = nullptr;
  // values
  ds::NDArray* Z = nullptr;
  // activations = g(Z)
  ds::NDArray* A = nullptr;
  // bias
  ds::NDArray* b = nullptr;
};

void Model::SetLayers(std::vector<nn::arch::Layer *> layers) {
  for(auto l : layers) {
    AddLayer(l);
  }
}

void Model::AddLayer(Layer* layer) {
  layers_.push_back(new LayerMeta(layer));
}

Model::~Model() {
  for(auto l=0; l < layers_.size(); ++l) {
    delete layers_[l];
  }
}

bool Model::RemoveLayer(uint32_t index) {
  if(index >= layers_.size()) {
    return false;
  }
  layers_.erase(layers_.begin() + index);
  return true;
}

Layer* Model::GetLayer(uint32_t index) const {
  if(index < layers_.size()) {
    return layers_[index]->layer;
  }
  ERROR_EXIT("Index out of bound");
}

void Model::InitImports() {
  builtins_.system.InitImports(this, &module_manager_, "System");
  builtins_.math.InitImports(this, &module_manager_, "Math");
  builtins_.activation.InitImports(this, &module_manager_, "Activation");
}

void Model::InitDefinitions() {
  builtins_.system.InitDefinitions(this, &module_manager_);
  builtins_.math.InitDefinitions(this, &module_manager_);
  builtins_.activation.InitDefinitions(this, &module_manager_);
}

void Model::SetupLayers() {
  const uint64_t val_type_size = TypeSize(Type::F64);
  for(auto l = 0; l < layers_.size(); ++l) {
    if(layers_[l]->layer->Type() == FullyConnected) {
      auto cur_layer = static_cast<FullyConnectedLayer*>(layers_[l]->layer);
      // values
      auto memZ = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
      layers_[l]->Z = new ds::NDArray(memZ, {cur_layer->Nodes(), 1}, val_type_size);
      // activations
      auto memA = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
      layers_[l]->A = new ds::NDArray(memA, {cur_layer->Nodes(), 1}, val_type_size);
      if(l > 0) {
        auto prev_layer = static_cast<FullyConnectedLayer*>(layers_[l-1]->layer);
        // weight
        auto memW = module_manager_.Memory().Allocate(cur_layer->Nodes() * prev_layer->Nodes() * val_type_size);
        layers_[l]->W = new ds::NDArray(memW, {cur_layer->Nodes(), prev_layer->Nodes()}, val_type_size);
        // bias
        auto memb = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
        layers_[l]->b = new ds::NDArray(memb, {cur_layer->Nodes(), 1}, val_type_size);
      }
    } else {
      assert(!"not implemented yet");
    }
  }
}

Var Model::GenerateFeedForward() {
  std::vector<Type> locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F64};
  return module_manager_.MakeFunction("feedforward", {}, locals,
          [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vi32_6 = locals[5];
    auto vf64_1 = locals[6];
    for(int l=1; l < layers_.size(); ++l) {
      // Z[l] = W . A
      auto mul = math::Multiply2DArrays<Type::F64>(f.Label(), layers_[l]->W, layers_[l-1]->A, layers_[l]->Z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6, vf64_1});
      // Z[l] = Z + b
      auto add = math::Add2DArrays<Type::F64>(f.Label(), layers_[l]->Z, layers_[l]->b, layers_[l]->Z, {vi32_1, vi32_2});
      // A[l] = g(Z[l])
      auto app = math::ApplyFx2DArrays<Type::F64>(f.Label(), layers_[l]->Z, builtins_.activation.Sigmoid().function,
          layers_[l]->A, {vi32_1, vi32_2});
      f.Insert(mul);
      f.Insert(add);
      f.Insert(app);
    }
  });
}

wabt::Var Model::GenerateBackpropagation() {

}

void Model::Setup() {
  SetupLayers();
  InitImports();
  module_manager_.MakeMemory(module_manager_.Memory().Pages());
  InitDefinitions();

  auto feedforward = GenerateFeedForward();
  auto backpropagation = GenerateBackpropagation();

  auto train = module_manager_.MakeFunction("train", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeCall(feedforward, {}));
  });

  auto main = module_manager_.MakeFunction("main", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeCall(train, {}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
