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
  ds::NDArray* T = nullptr;
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

void Model::AllocateLayers() {
  ERROR_UNLESS(layers_.size() > 2, "At least an input and output layer should be defined");
  for(auto l = 0; l < layers_.size(); ++l) {
    // FIXME For now only support fully connected layer
    assert(layers_[l]->layer->Type() == FullyConnected);

    ALLOCATE_MEMORY(layers_[l]->Z, layers_[l]->layer->Nodes(), batch_size_);
    ALLOCATE_MEMORY(layers_[l]->dZ, layers_[l]->layer->Nodes(), batch_size_);
    ALLOCATE_MEMORY(layers_[l]->A, layers_[l]->layer->Nodes(), batch_size_);
    ALLOCATE_MEMORY(layers_[l]->dA, layers_[l]->layer->Nodes(), batch_size_);
    if(l > 0) {
      ALLOCATE_MEMORY(layers_[l]->W, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->dW, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->B, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->dB, layers_[l]->layer->Nodes(), batch_size_);
    }
    if(l == layers_.size() - 1) {
      ALLOCATE_MEMORY(layers_[l]->T, layers_[l]->layer->Nodes(), batch_size_);
    }
  }
}

void Model::AllocateInput(std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels) {
  for(uint32_t b=0; b < input.size(); b += batch_size_) {
    // Training data
    ds::NDArray* training_array = nullptr;
    ALLOCATE_MEMORY(training_array, input[0].size(), batch_size_);
    training_.push_back(training_array);
    // Training labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MEMORY(labels_array, labels[0].size(), batch_size_);
    labels_.push_back(labels_array);
  }
}

std::vector<wasmpp::DataEntry> MakeTransposeData(ds::NDArray* array, std::vector<std::vector<double>> input) {
  std::vector<DataEntry> entries;
  for (uint32_t col = 0; col < array->Shape()[0]; ++col) {
    for (uint32_t row = 0; row < array->Shape()[1]; ++row) {
      entries.push_back(DataEntry::MakeF64(input[row][col]));
    }
  }
  return entries;
}

void Model::MakeInputData(std::vector<std::vector<double>> input, std::vector<std::vector<double>> labels) {
  assert(training_.size() == labels_.size());

  for(uint32_t i=0; i < training_.size(); ++i) {
    auto begin = input.begin() + (i * batch_size_);
    auto end = input.begin() + ((i+1) * batch_size_);
    // Training data
    std::vector<std::vector<double>> sub_input(begin, end);
    module_manager_.MakeData(memory_, training_[i]->Memory()->Begin(), MakeTransposeData(training_[i], sub_input));
    // Training labels
    std::vector<std::vector<double>> sub_labels(begin, end);
    module_manager_.MakeData(memory_, labels_[i]->Memory()->Begin(), MakeTransposeData(labels_[i], sub_labels));
  }
}

void Model::MakeWeightData() {
  for(int l=1; l < layers_.size(); ++l) {
    auto total = layers_[l]->W->Shape()[0] * layers_[l]->W->Shape()[1];
    std::vector<DataEntry> entries(total, DataEntry::MakeF64(0.1));
    module_manager_.MakeData(memory_, layers_[l]->W->Memory()->Begin(), entries);
  }
}

void Model::MakeBiasData() {
  for(int l=1; l < layers_.size(); ++l) {
    auto total = layers_[l]->B->Shape()[0] * layers_[l]->B->Shape()[1];
    std::vector<DataEntry> entries(total, DataEntry::MakeF64(0.2));
    module_manager_.MakeData(memory_, layers_[l]->B->Memory()->Begin(), entries);
  }
}

Var Model::GenerateFeedForward() {
  std::vector<Type> locals_types = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F64};
  return module_manager_.MakeFunction("feedforward", {}, locals_types, [&](FuncBody f, std::vector<Var> params,
                                                                           std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf64_1 = locals[5];

    for(int l=1; l < layers_.size(); ++l) {
      // Z[l] = W . A + b
      // {dot} : Z[l] = W . A
      // {add} : Z[l] = Z[l] + b
      auto dot = snippet::MatrixDot(f.Label(), layers_[l]->W, layers_[l-1]->A, layers_[l]->Z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf64_1});
      auto add = snippet::MatrixAddition(f.Label(), layers_[l]->Z, layers_[l]->B, layers_[l]->Z, {vi32_1, vi32_2});
      f.Insert(dot);
      f.Insert(add);

      // A[l] = g(Z[l])
      f.Insert(snippet::MatrixActivation(f.Label(), layers_[l]->Z, layers_[l]->layer->ActivationFunction(),
          layers_[l]->A, {vi32_1, vi32_2}, false));
    }
  });
}

wabt::Var Model::GenerateBackpropagation() {
  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F64};
  return module_manager_.MakeFunction("backpropagation", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                                              std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf64_1 = locals[5];

    // dA[L] = Loss(Target, Prediction)
    f.Insert(snippet::MatrixLoss(f.Label(), layers_.back()->A, layers_.back()->T, builtins_.loss.MeanSquaredError(),
                                 layers_.back()->dA, {vi32_1, vi32_2}));

    for(auto l = layers_.size()-1; l > 0; --l) {
       // dZ[l] = dA[l] * g'(Z[l])
       // {der} : dZ[l] = g'(Z[l])
       // {mul} : dZ[l] = dA[l] * dZ[l]
      auto der = snippet::MatrixActivation(f.Label(), layers_[l]->Z, layers_[l]->layer->ActivationFunction(),
          layers_[l]->A, {vi32_1, vi32_2}, true);
      auto mul = snippet::MatrixMultiplication(f.Label(), layers_[l]->dA, layers_[l]->dZ, layers_[l]->dZ,
          {vi32_1, vi32_2});
      f.Insert(der);
      f.Insert(mul);

      // dW[l] = (1/m) dZ[l] . A[l-1]^T
      // {dot} : dW[l] = dZ[l] . A[l-1]^T
      // {sca} : dW[l] = (1/m) dW[l]
    }
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

    uint32_t lhs_row = 3;
    uint32_t lhs_col = 4;
    uint32_t rhs_row = 3;
    uint32_t rhs_col = 5;
    uint32_t dst_row = lhs_col;
    uint32_t dst_col = rhs_col;
    ds::NDArray* A_ = nullptr;
    ds::NDArray* B_ = nullptr;
    ds::NDArray* C_ = nullptr;
    auto &module_manager_ = *mm;
    ALLOCATE_MEMORY(A_, lhs_row, lhs_col);
    ALLOCATE_MEMORY(B_, rhs_row, rhs_col);
    ALLOCATE_MEMORY(C_, dst_row, dst_col);

    // Populate A
    uint32_t val = 1;
    for(int r=0; r < lhs_row; ++r) {
      for(int c=0; c < lhs_col; ++c) {
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
    val = rhs_row * rhs_col;
    for(int r=0; r < rhs_row; ++r) {
      for(int c=0; c < rhs_col; ++c) {
        f.Insert(MakeF64Store(MakeI32Const(B_->GetLinearIndex({r, c})), MakeF64Const(val--)));
      }
    }

    // Print B
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(B_->Memory()->Begin()),
        MakeI32Const(B_->Shape()[0]),
        MakeI32Const(B_->Shape()[1])
    }));

//    f.Insert(snippet::MatrixDot(f.Label(), A_, B_, C_, {i32_1, i32_2, i32_3, i32_4, i32_5, f64_1}));
//    f.Insert(snippet::MatrixActivation(f.Label(), A_, model->Builtins().activation.Sigmoid(), C_, {i32_1, i32_2},false));
//    f.Insert(snippet::MatrixAddition(f.Label(), A_, B_, C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixScalar(f.Label(), A_, MakeF64Const(0.01), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixLoss(f.Label(), A_, B_, model->Builtins().loss.MeanSquaredError(), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixCopy(f.Label(), B_, C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixBiasBroadcast(f.Label(), C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixMultiplication(f.Label(), A_, B_, C_, {i32_1, i32_2}));
//    f.Insert(snippet::MatrixDotRT(f.Label(), A_, B_, C_, {i32_1, i32_2, i32_3, i32_4, i32_5, f64_1}));
    f.Insert(snippet::MatrixDotLT(f.Label(), A_, B_, C_, {i32_1, i32_2, i32_3, i32_4, i32_5, f64_1}));

    // Print C
    f.Insert(MakeCall(model->Builtins().system.PrintTableF64(), {
        MakeI32Const(C_->Memory()->Begin()),
        MakeI32Const(C_->Shape()[0]),
        MakeI32Const(C_->Shape()[1])
    }));
  });
}

void Model::Setup(uint32_t batch_size, std::vector<std::vector<double>> input,
                  std::vector<std::vector<double>> labels) {
  ERROR_UNLESS(training_.empty(), "cannot setup again the same model");
  ERROR_UNLESS(batch_size >= 1, "batch size must be at least 1");
  ERROR_UNLESS(input.size() > 0, "training input cannot be empty");
  ERROR_UNLESS(batch_size <= input.size(), "batch size must be at most equal to the input size");
  ERROR_UNLESS(input.size() % batch_size == 0, "batch size must be a multiple of the input size");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");

  batch_size_ = batch_size;
  AllocateLayers();
  AllocateInput(input, labels);
  memory_ = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memory_);
  MakeInputData(input, labels);
  MakeWeightData();
  MakeBiasData();
}

void Model::Train(){
  auto feedforward = GenerateFeedForward();
  auto backpropagation = GenerateBackpropagation();

  std::vector<Type> locals_type = {Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    auto i32_1 = locals[0];
    auto i32_2 = locals[1];

    // Copy training data into the first layer
    f.Insert(snippet::MatrixCopy(f.Label(), training_[0], layers_.front()->A, {i32_1, i32_2}));
    f.Insert(snippet::MatrixCopy(f.Label(), labels_[0], layers_.back()->T, {i32_1, i32_2}));

    // Apply neural network algorithms
    f.Insert(MakeCall(feedforward, {}));
    f.Insert(MakeCall(backpropagation, {}));

    // Print for debugging
//    f.Insert(MakeCall(builtins_.system.PrintTableF64(), {
//        MakeI32Const(layers_[2]->T->Memory()->Begin()),
//        MakeI32Const(layers_[2]->T->Shape()[0]),
//        MakeI32Const(layers_[2]->T->Shape()[1])
//    }));

    // Debugging
    f.Insert(MakeCall(Debug(&module_manager_, this), {}));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
