#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/data_structure/ndarray.h>
#include <src/apps/nn-builder/src/helper/matrix.h>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;

struct LayerMeta {
  LayerMeta(Layer* l):layer(l) {}
  Layer* layer = nullptr;

  // Feed-forward arrays
  // weights
  ds::NDArray* W = nullptr;
  // values
  ds::NDArray* z = nullptr;
  // activations = g(Z)
  ds::NDArray* a = nullptr;
  // bias
  ds::NDArray* b = nullptr;

  // Back-propagation arrays
  // weights
  ds::NDArray* dW = nullptr;
  // values
  ds::NDArray* dz = nullptr;
  // activations
  ds::NDArray* da = nullptr;
  // bias
  ds::NDArray* &db = da;

};

void Model::SetLayers(std::vector<nn::arch::Layer *> layers) {
  for(auto l : layers) {
    AddLayer(l);
  }
}

void Model::AddLayer(Layer* layer) {
  layers_.push_back(new LayerMeta(layer));
}

Model::Model() {
  InitImports();
  InitDefinitions();
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
  ERROR_UNLESS(layers_.size() > 2, "At least an input and output layer should be defined");
  const uint64_t val_type_size = TypeSize(Type::F64);
  for(auto l = 0; l < layers_.size(); ++l) {
    // FIXME For now only support fully connected layer
    assert(layers_[l]->layer->Type() == FullyConnected);
    auto cur_layer = layers_[l]->layer;
    // values
    auto memZ = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
    layers_[l]->z = new ds::NDArray(memZ, {cur_layer->Nodes(), 1}, val_type_size);
    // activations
    auto memA = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
    layers_[l]->a = new ds::NDArray(memA, {cur_layer->Nodes(), 1}, val_type_size);
    if(l > 0) {
      auto prev_layer = layers_[l-1]->layer;
      // weight
      auto memW = module_manager_.Memory().Allocate(cur_layer->Nodes() * prev_layer->Nodes() * val_type_size);
      layers_[l]->W = new ds::NDArray(memW, {cur_layer->Nodes(), prev_layer->Nodes()}, val_type_size);
      // bias
      auto memb = module_manager_.Memory().Allocate(cur_layer->Nodes() * val_type_size);
      layers_[l]->b = new ds::NDArray(memb, {cur_layer->Nodes(), 1}, val_type_size);
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
      auto mul = helper::MatrixDot(f.Label(), layers_[l]->W, layers_[l-1]->a, layers_[l]->z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6, vf64_1});
      // Z[l] = Z + b
      auto add = helper::MatrixAddition(f.Label(), layers_[l]->z, layers_[l]->b, layers_[l]->z, {vi32_1, vi32_2});
      // A[l] = g(Z[l])
      auto app = helper::MatrixActivation(f.Label(), layers_[l]->z, layers_[l]->layer->ActivationFunction(),
          layers_[l]->a, {vi32_1, vi32_2});
      f.Insert(mul);
      f.Insert(add);
      f.Insert(app);
    }
  });
}

wabt::Var Model::GenerateBackpropagation() {
  std::vector<Type> locals = {};
  return module_manager_.MakeFunction("backpropagation", {}, locals, [&](FuncBody f, std::vector<Var> params,
                                                                     std::vector<Var> locals) {

  });
}

void Model::Setup() {
  SetupLayers();
  auto memo = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memo);
}

wabt::Var Debug(ModuleManager* mm, Model* model) {
  std::vector<Type> func_locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F64};
  return mm->MakeFunction("", {{}, {}}, func_locals, [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto i32_1 = locals[0];
    auto i32_2 = locals[1];
    auto i32_3 = locals[2];
    auto i32_4 = locals[3];
    auto i32_5 = locals[4];
    auto i32_6 = locals[5];
    auto f64_1 = locals[6];

    uint32_t side = 3;
    uint32_t type_size = sizeof(double);
    wasmpp::Memory* A = mm->Memory().Allocate(side * side * type_size);
    ds::NDArray* A_ = new ds::NDArray(A, {side, side}, type_size);
    wasmpp::Memory* B = mm->Memory().Allocate(side * side * type_size);
    ds::NDArray* B_ = new ds::NDArray(B, {side, side}, type_size);
    wasmpp::Memory* C = mm->Memory().Allocate(side * side * type_size);
    ds::NDArray* C_ = new ds::NDArray(C, {side, side}, type_size);

    // Populate A
    uint32_t val = 1;
    for(int r=0; r < side; ++r) {
      for(int c=0; c < side; ++c) {
        f.Insert(MakeF64Store(MakeI32Const(A_->GetLinearIndex({r, c})), MakeF64Const(val++)));
      }
    }

    // Print A
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(A_->Memory()->Begin()),
        MakeI32Const(A_->Shape()[0]),
        MakeI32Const(A_->Shape()[1])
    }));

    // Populate B
    val = 9;
    for(int r=0; r < side; ++r) {
      for(int c=0; c < side; ++c) {
        f.Insert(MakeF64Store(MakeI32Const(B_->GetLinearIndex({r, c})), MakeF64Const(val--)));
      }
    }

    // Print B
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(B_->Memory()->Begin()),
        MakeI32Const(B_->Shape()[0]),
        MakeI32Const(B_->Shape()[1])
    }));

    f.Insert(helper::MatrixDot(f.Label(), A_, B_, C_, {i32_1, i32_2, i32_3, i32_4, i32_5, i32_6, f64_1}));
//    f.Insert(helper::MatrixActivation(f.Label(), A_, model->Builtins().activation.Sigmoid(), C_, {i32_1, i32_2}));
//    f.Insert(helper::MatrixAddition(f.Label(), A_, B_, C_, {i32_1, i32_2}));
//    f.Insert(helper::MatrixScalar(f.Label(), A_, MakeF64Const(0.01), C_, {i32_1, i32_2}));

    // Print C
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(C_->Memory()->Begin()),
        MakeI32Const(C_->Shape()[0]),
        MakeI32Const(C_->Shape()[1])
    }));
  });
}

void Model::Train(std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels) {
  assert(layers_.size() > 0);
  assert(layers_.front()->a != nullptr);
  assert(layers_.front()->a->Shape().size() == 2);
  ERROR_UNLESS(input.size() > 0, "training input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");
  auto train = module_manager_.MakeFunction("train", {}, {}, [&](FuncBody f, std::vector<Var> params,
      std::vector<Var> locals) {
    for(auto i=0; i < 1; i++) {
      ERROR_UNLESS(layers_.front()->a->Shape()[0] == input[i].size(), "input at index %d has wrong shape", i);
      ERROR_UNLESS(layers_.back()->a->Shape()[0] == labels[i].size(), "label at index %d has wrong shape", i);
      // TODO copy data to layer->front()->a
    }
    auto feedforward = GenerateFeedForward();
    auto backpropagation = GenerateBackpropagation();
    f.Insert(MakeCall(feedforward, {}));
    f.Insert(MakeCall(backpropagation, {}));

    f.Insert(MakeCall(Debug(&module_manager_, this), {}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
