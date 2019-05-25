#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/snippet/matrix.h>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;

struct LayerMeta {
  LayerMeta(Layer* l):layer(l) {}
  Layer* layer = nullptr;
  // Feed-forward arrays
  ds::NDArray* W = nullptr;
  ds::NDArray* Z = nullptr;
  ds::NDArray* A = nullptr;
  ds::NDArray* B = nullptr;
  // Back-propagation arrays
  ds::NDArray* dW = nullptr;
  ds::NDArray* dZ = nullptr;
  ds::NDArray* dA = nullptr;
  ds::NDArray* dB = nullptr;

};

void Model::SetLayers(std::vector<nn::arch::Layer *> layers) {
  for(auto l : layers) {
    AddLayer(l);
  }
}

void Model::AddLayer(Layer* layer) {
  layers_.push_back(new LayerMeta(layer));
}

Model::Model() : module_manager_({MemoryType::WASM32}) {
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
  builtins_.loss.InitImports(this, &module_manager_, "Loss");
}

void Model::InitDefinitions() {
  builtins_.system.InitDefinitions(this, &module_manager_);
  builtins_.math.InitDefinitions(this, &module_manager_);
  builtins_.activation.InitDefinitions(this, &module_manager_);
  builtins_.loss.InitDefinitions(this, &module_manager_);
}

#define ALLOCATE_MEMORY(array, rows, cols) \
    array = new ds::NDArray(module_manager_.Memory().Allocate((rows) * (cols) * TypeSize(Type::F64)), \
                            {rows, cols}, TypeSize(Type::F64));

void Model::SetupLayers(uint32_t batch_size) {
  ERROR_UNLESS(layers_.size() > 2, "At least an input and output layer should be defined");
  for(auto l = 0; l < layers_.size(); ++l) {
    // FIXME For now only support fully connected layer
    assert(layers_[l]->layer->Type() == FullyConnected);

    ALLOCATE_MEMORY(layers_[l]->Z, layers_[l]->layer->Nodes(), batch_size);
    ALLOCATE_MEMORY(layers_[l]->dZ, layers_[l]->layer->Nodes(), batch_size);
    ALLOCATE_MEMORY(layers_[l]->A, layers_[l]->layer->Nodes(), batch_size);
    ALLOCATE_MEMORY(layers_[l]->dA, layers_[l]->layer->Nodes(), batch_size);
    if(l > 0) {
      ALLOCATE_MEMORY(layers_[l]->W, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->dW, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->B, layers_[l]->layer->Nodes(), batch_size);
      ALLOCATE_MEMORY(layers_[l]->dB, layers_[l]->layer->Nodes(), batch_size);
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
      f.Insert(snippet::MatrixDot(f.Label(), layers_[l]->W, layers_[l-1]->A, layers_[l]->Z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6, vf64_1}));
      // Z[l] = Z + b
      f.Insert(snippet::MatrixAddition(f.Label(), layers_[l]->Z, layers_[l]->B, layers_[l]->Z, {vi32_1, vi32_2}));
      // A[l] = g(Z[l])
      f.Insert(snippet::MatrixActivation(f.Label(), layers_[l]->Z, layers_[l]->layer->ActivationFunction(),
          layers_[l]->A, {vi32_1, vi32_2}));
    }
  });
}

wabt::Var Model::GenerateBackpropagation() {
  std::vector<Type> locals = {};
  return module_manager_.MakeFunction("backpropagation", {}, locals, [&](FuncBody f, std::vector<Var> params,
                                                                     std::vector<Var> locals) {

  });
}

wabt::Var Debug(ModuleManager* mm, Model* model) {
  std::vector<Type> func_locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F64};
  return mm->MakeFunction(nullptr, {{}, {}}, func_locals, [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
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

//    f.Insert(snippet::MatrixDot(f.Label(), A_, B_, C_, {i32_1, i32_2, i32_3, i32_4, i32_5, i32_6, f64_1}));
//    f.Insert(snippet::MatrixActivation(f.Label(), A_, model->Builtins().activation.Sigmoid(), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixAddition(f.Label(), A_, B_, C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixScalar(f.Label(), A_, MakeF64Const(0.01), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixLoss(f.Label(), A_, B_, model->Builtins().loss.MeanSquaredError(), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixCopy(f.Label(), B_, C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixBiasBroadcast(f.Label(), C_, {i32_1, i32_2}));

    // Print C
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(C_->Memory()->Begin()),
        MakeI32Const(C_->Shape()[0]),
        MakeI32Const(C_->Shape()[1])
    }));
  });
}

void Model::Train(uint32_t batch_size, std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels){
  ERROR_UNLESS(batch_size >= 1, "batch size must be at least 1");
  ERROR_UNLESS(input.size() > 0, "training input cannot be empty");
  ERROR_UNLESS(batch_size <= input.size(), "batch size must be at most equal to the input size");
  ERROR_UNLESS(input.size() % batch_size == 0, "batch size must be a multiple of the input size");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");

  SetupLayers(batch_size);

  std::vector<ds::NDArray*> training_data_arrays;
  std::vector<ds::NDArray*> labels_arrays;

  auto feedforward = GenerateFeedForward();
  auto backpropagation = GenerateBackpropagation();

  // Allocate memory
  for(uint32_t b=0; b < input.size(); b += batch_size) {
    ds::NDArray* training_array = nullptr;
    ALLOCATE_MEMORY(training_array, input[0].size(), batch_size);
    training_data_arrays.push_back(training_array);
  }

  // Create required memory
  // Do not allocate anymore memory
  // after calling MakeMemory
  auto memo = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memo);

  // Create wasm data entries
  for(uint32_t i=0; i < training_data_arrays.size(); ++i) {
    std::vector<DataEntry> entries;
    auto offset = i * batch_size;
    for (uint32_t col = 0; col < training_data_arrays[i]->Shape()[0]; ++col) {
      for (uint32_t rel_row = 0; rel_row < training_data_arrays[i]->Shape()[1]; ++rel_row) {
        entries.push_back(DataEntry::MakeF64(input[offset+rel_row][col]));
      }
    }
    module_manager_.MakeData(memo, training_data_arrays[i]->Memory()->Begin(), entries);
  }

  std::vector<Type> locals_type = {Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    auto i32_1 = locals[0];
    auto i32_2 = locals[1];

    // Copy training data into the first layer
    f.Insert(snippet::MatrixCopy(f.Label(), training_data_arrays[0], layers_.front()->A, {i32_1, i32_2}));

    // Print input
    f.Insert(MakeCall(builtins_.system.PrintTableF64(), {
      MakeI32Const(layers_[1]->W->Memory()->Begin()),
      MakeI32Const(layers_[1]->W->Shape()[0]),
      MakeI32Const(layers_[1]->W->Shape()[1])
    }));

    f.Insert(MakeCall(feedforward, {}));
    f.Insert(MakeCall(backpropagation, {}));

    // Debugging
//    f.Insert(MakeCall(Debug(&module_manager_, this), {}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
