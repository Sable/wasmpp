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
  builtins.print_i32 = module_manager_.MakeFuncImport("System", "print", {{Type::I32}, {}});
  builtins.print_i64 = module_manager_.MakeFuncImport("System", "print", {{Type::I64}, {}});
  builtins.print_f32 = module_manager_.MakeFuncImport("System", "print", {{Type::F32}, {}});
  builtins.print_f64 = module_manager_.MakeFuncImport("System", "print", {{Type::F64}, {}});
  builtins.exp = module_manager_.MakeFuncImport("Math", "exp", {{Type::F64}, {Type::F64}});
}

void Model::InitDefinitions() {
  builtins.sigmoid = module_manager_.MakeFunction(nullptr, {{Type::F64}, {Type::F64}}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto denom = MakeBinary(Opcode::F64Add, MakeF64Const(1),
        MakeCall(builtins.exp, {MakeUnary(Opcode::F64Neg, MakeLocalGet(params[0]))}));
    auto div = MakeBinary(Opcode::F64Div, MakeF64Const(1), denom);
    f.Insert(div);
  });
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
  module_manager_.MakeMemory(module_manager_.Memory().Pages());
  return module_manager_.MakeFunction("feedforward", {}, locals,
          [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vi32_6 = locals[5];
    auto vtype_1 = locals[6];
    for(int l=1; l < layers_.size(); ++l) {
      auto mul = math::Multiply2DArrays<Type::F64>(f.Label(), layers_[l]->W, layers_[l-1]->A, layers_[l]->Z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6, vtype_1});
      auto add = math::Add2DArrays<Type::F64>(f.Label(), layers_[l]->Z, layers_[l]->b, layers_[l]->Z, {vi32_1, vi32_2});
      auto app = math::ApplyFx2DArrays<Type::F64>(f.Label(), layers_[l]->Z, builtins.sigmoid, layers_[l]->A,
          {vi32_1, vi32_2});
      f.Insert(mul);
      f.Insert(add);
      f.Insert(app);
    }
  });
}

void Model::Setup() {
  InitImports();
  InitDefinitions();
  SetupLayers();
  auto ff = GenerateFeedForward();

  auto train = module_manager_.MakeFunction("train", {}, {},
      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeCall(ff, {}));
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
