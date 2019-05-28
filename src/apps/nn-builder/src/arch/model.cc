#include <src/apps/nn-builder/src/arch/model.h>
#include <src/apps/nn-builder/src/snippet/matrix.h>
#include <src/wasmpp/wasm-instructions-gen.h>

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

#define PRINT_TABLE(table)                              \
  f.Insert(MakeCall(builtins_.system.PrintTableF64(), { \
      MakeI32Const((table)->Memory()->Begin()),         \
      MakeI32Const((table)->Shape()[0]),                \
      MakeI32Const((table)->Shape()[1])                 \
  }));

void Model::AllocateLayers() {
  ERROR_UNLESS(layers_.size() > 2, "At least an input and output layer should be defined");
  for(auto l = 0; l < layers_.size(); ++l) {
    // FIXME For now only support fully connected layer
    assert(layers_[l]->layer->Type() == FullyConnected);

    ALLOCATE_MEMORY(layers_[l]->A, layers_[l]->layer->Nodes(), batch_size_);
    if(l > 0) {
      ALLOCATE_MEMORY(layers_[l]->Z, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->dZ, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->dA, layers_[l]->layer->Nodes(), batch_size_);
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
    // Training data
    auto training_begin = input.begin() + (i * batch_size_);
    std::vector<std::vector<double>> sub_input(training_begin, training_begin + batch_size_);
    module_manager_.MakeData(memory_, training_[i]->Memory()->Begin(), MakeTransposeData(training_[i], sub_input));
    // Training labels
    auto labels_begin = labels.begin() + (i * batch_size_);
    std::vector<std::vector<double>> sub_labels(labels_begin, labels_begin + batch_size_);
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
    std::vector<DataEntry> entries(total, DataEntry::MakeF64(0.1));
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
      // Z[l] = W[l] . A[l-1] + B[l]
      // 1) Z[l] = W[l] . A[l-1]
      // 2) Z[l] = Z[l] + B[l]
      f.Insert(snippet::MatrixDot(f.Label(), layers_[l]->W, layers_[l-1]->A, layers_[l]->Z,
          {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf64_1}));
      f.Insert(snippet::MatrixAddition(f.Label(), layers_[l]->Z, layers_[l]->B, layers_[l]->Z, {vi32_1, vi32_2}));

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

    // dA[L] = Loss(T[L], A[L])
    f.Insert(snippet::MatrixLoss(f.Label(), layers_.back()->T, layers_.back()->A, builtins_.loss.MeanSquaredError(),
                                 layers_.back()->dA, {vi32_1, vi32_2}));

    for(auto l = layers_.size()-1; l > 0; --l) {
       // dZ[l] = dA[l] * g'(Z[l])
       // 1) dZ[l] = g'(Z[l])
       // 2) dZ[l] = dA[l] * dZ[l]
      f.Insert(snippet::MatrixActivation(f.Label(), layers_[l]->Z, layers_[l]->layer->ActivationFunction(),
          layers_[l]->dZ, {vi32_1, vi32_2}, true));
      f.Insert(snippet::MatrixMultiplication(f.Label(), layers_[l]->dA, layers_[l]->dZ, layers_[l]->dZ,
          {vi32_1, vi32_2}));

      // dW[l] = (1/m) dZ[l] . A[l-1]^T
      // 1) dW[l] = dZ[l] . A[l-1]^T
      // 2) dW[l] = (1/m) dW[l]
      f.Insert(snippet::MatrixDotRT(f.Label(), layers_[l]->dZ, layers_[l-1]->A, layers_[l]->dW,
                                    {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf64_1}));
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dW, MakeF64Const(1.0/batch_size_), layers_[l]->dW,
                                     {vi32_1, vi32_2}));

      // dB[l] = (1/m) dZ[l]
      // FIXME Sum dZ[l] horizontally, store it into first column of dB[l], then broadcast
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dZ, MakeF64Const(1.0/batch_size_), layers_[l]->dB,
                                     {vi32_1, vi32_2}));

      if(l-1 > 0) {
        // dA[l-1] = W[l]^T . dZ[l]
        f.Insert(snippet::MatrixDotLT(f.Label(), layers_[l]->W, layers_[l]->dZ, layers_[l-1]->dA,
                                      {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf64_1}));
      }

      // W[l] = W[l] - alpha * dW[l]
      // 1) dW[l] = alpha * dW[l]
      // 2) W[l] = W[l] - dW[l]
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dW, MakeF64Const(learning_rate_), layers_[l]->dW,
                                     {vi32_1, vi32_2}));
      f.Insert(snippet::MatrixSubtraction(f.Label(), layers_[l]->W, layers_[l]->dW, layers_[l]->W, {vi32_1, vi32_2}));
    }
  });
}

void Model::Setup(uint32_t epochs, uint32_t batch_size, double learning_rate, std::vector<std::vector<double>> input,
                  std::vector<std::vector<double>> labels) {
  ERROR_UNLESS(training_.empty(), "cannot setup again the same model");
  ERROR_UNLESS(batch_size >= 1, "batch size must be at least 1");
  ERROR_UNLESS(epochs >= 1, "epoch must be at least 1");
  ERROR_UNLESS(input.size() > 0, "training input cannot be empty");
  ERROR_UNLESS(batch_size <= input.size(), "batch size must be at most equal to the input size");
  ERROR_UNLESS(input.size() % batch_size == 0, "batch size must be a multiple of the input size");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");

  batch_size_ = batch_size;
  epochs_ = epochs;
  learning_rate_ = learning_rate;
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

  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    auto epoch = locals[0];
    auto i32_1 = locals[1];
    auto i32_2 = locals[2];

    f.Insert(GenerateRangeLoop(f.Label(), epoch, 0, epochs_, 1, [&](BlockBody* b) {
      for(int t=0; t < training_.size(); ++t) {
        // Copy training data into the first layer
        b->Insert(snippet::MatrixCopy(f.Label(), training_[t], layers_.front()->A, {i32_1, i32_2}));
        b->Insert(snippet::MatrixCopy(f.Label(), labels_[t], layers_.back()->T, {i32_1, i32_2}));

        // Apply neural network algorithms
        b->Insert(MakeCall(feedforward, {}));
        b->Insert(MakeCall(backpropagation, {}));
      }
    }));

    // Test on training data (for debugging)
    for(int t=0; t < training_.size(); ++t) {
      // Copy training data into the first layer
      f.Insert(snippet::MatrixCopy(f.Label(), training_[t], layers_.front()->A, {i32_1, i32_2}));
      f.Insert(snippet::MatrixCopy(f.Label(), labels_[t], layers_.back()->T, {i32_1, i32_2}));

      // Apply neural network algorithms
      f.Insert(MakeCall(feedforward, {}));

      PRINT_TABLE(layers_.back()->A)
    }
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
