#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/snippet/matrix.h>
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
  ds::NDArray* Cost = nullptr;
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

Model::Model(ModelOptions options) : options_(options), builtins_(options_.activation_options) {
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
  builtins_.message.InitImports(this, &module_manager_, "Message");
}

void Model::InitDefinitions() {
  builtins_.system.InitDefinitions(this, &module_manager_);
  builtins_.math.InitDefinitions(this, &module_manager_);
  builtins_.activation.InitDefinitions(this, &module_manager_);
  builtins_.loss.InitDefinitions(this, &module_manager_);
  builtins_.message.InitDefinitions(this, &module_manager_);
}

#define ALLOCATE_MEMORY(array, rows, cols) \
    array = new ds::NDArray(module_manager_.Memory().Allocate((rows) * (cols) * TypeSize(Type::F32)), \
                            {rows, cols}, TypeSize(Type::F32));

#define PRINT_TABLE(table)                              \
  f.Insert(MakeCall(builtins_.system.PrintTableF32(), { \
      MakeI32Const((table)->Memory()->Begin()),         \
      MakeI32Const((table)->Shape()[0]),                \
      MakeI32Const((table)->Shape()[1])                 \
  }));

void Model::AllocateLayers() {
  ERROR_UNLESS(layers_.size() >= 2, "At least an input and output layer should be defined");
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
      if(options_.log_training_error) {
        ALLOCATE_MEMORY(layers_[l]->Cost, layers_[l]->layer->Nodes(), batch_size_);
      }
    }
  }
}

void Model::AllocateInput(std::vector<std::vector<float>> input, std::vector<std::vector<float>> labels) {
  // Do not merge loops so that all
  // training data are consecutive in memory

  for(uint32_t b=0; b < input.size(); b += batch_size_) {
    // Training data
    ds::NDArray* training_array = nullptr;
    ALLOCATE_MEMORY(training_array, (uint32_t) input[0].size(), batch_size_);
    training_.push_back(training_array);
  }

  for(uint32_t b=0; b < input.size(); b += batch_size_) {
    // Training labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MEMORY(labels_array, (uint32_t) labels[0].size(), batch_size_);
    labels_.push_back(labels_array);
  }
}

std::vector<wasmpp::DataEntry> MakeTransposeData(ds::NDArray* array, std::vector<std::vector<float>> input) {
  std::vector<DataEntry> entries;
  for (uint32_t col = 0; col < array->Shape()[0]; ++col) {
    for (uint32_t row = 0; row < array->Shape()[1]; ++row) {
      entries.push_back(DataEntry::MakeF32(input[row][col]));
    }
  }
  return entries;
}

void Model::MakeInputData(std::vector<std::vector<float>> input, std::vector<std::vector<float>> labels) {
  assert(training_.size() == labels_.size());

  for(uint32_t i=0; i < training_.size(); ++i) {
    // Training data
    auto training_begin = input.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_input(training_begin, training_begin + batch_size_);
    module_manager_.MakeData(memory_, training_[i]->Memory()->Begin(), MakeTransposeData(training_[i], sub_input));
    // Training labels
    auto labels_begin = labels.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + batch_size_);
    module_manager_.MakeData(memory_, labels_[i]->Memory()->Begin(), MakeTransposeData(labels_[i], sub_labels));
  }
}

void Model::MakeWeightData() {
  for(int l=1; l < layers_.size(); ++l) {
    auto total = layers_[l]->W->Shape()[0] * layers_[l]->W->Shape()[1];
    std::vector<DataEntry> entries(total, DataEntry::MakeF32(0.1));
    module_manager_.MakeData(memory_, layers_[l]->W->Memory()->Begin(), entries);
  }
}

void Model::MakeBiasData() {
  for(int l=1; l < layers_.size(); ++l) {
    auto total = layers_[l]->B->Shape()[0] * layers_[l]->B->Shape()[1];
    std::vector<DataEntry> entries(total, DataEntry::MakeF32(0.2));
    module_manager_.MakeData(memory_, layers_[l]->B->Memory()->Begin(), entries);
  }
}

Var Model::GenerateFeedForward() {
  std::vector<Type> locals_types = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction("feedforward", {{Type::I32},{}}, locals_types, [&](FuncBody f, std::vector<Var> params,
                                                                           std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];

    auto input_begin = params[0];

    for(int l=1; l < layers_.size(); ++l) {
      // Z[l] = W[l] . A[l-1] + B[l]
      // 1) Z[l] = W[l] . A[l-1]
      // 2) Z[l] = Z[l] + B[l]
      f.Insert(snippet::MatrixDot(f.Label(), layers_[l]->W,
                                  (l == 1) ? snippet::Mat(layers_[0]->A, input_begin) : snippet::Mat(layers_[l-1]->A),
                                  layers_[l]->Z, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1}));
      f.Insert(snippet::MatrixAddition(f.Label(), layers_[l]->Z, layers_[l]->B, layers_[l]->Z, {vi32_1, vi32_2}));

      // A[l] = g(Z[l])
      f.Insert(snippet::MatrixActivation(f.Label(), snippet::Mat(layers_[l]->Z), layers_[l]->layer->ActivationFunction(),
          layers_[l]->A, {vi32_1, vi32_2}, false));
    }
  });
}

wabt::Var Model::GenerateBackpropagation() {
  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction("backpropagation", {{Type::I32, Type::I32},{}}, locals_type,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];

    auto input_begin = params[0];
    auto target_begin = params[1];

    // dA[L] = Loss(T[L], A[L])
    f.Insert(snippet::MatrixLoss(f.Label(), snippet::Mat(layers_.back()->T, target_begin),
                                 snippet::Mat(layers_.back()->A), loss_, layers_.back()->dA, {vi32_1, vi32_2}, true));

    for(auto l = layers_.size()-1; l > 0; --l) {
       // dZ[l] = dA[l] * g'(Z[l])
       // 1) dZ[l] = g'(Z[l])
       // 2) dZ[l] = dA[l] * dZ[l]
      f.Insert(snippet::MatrixActivation(f.Label(), snippet::Mat(layers_[l]->Z), layers_[l]->layer->ActivationFunction(),
          layers_[l]->dZ, {vi32_1, vi32_2}, true));
      f.Insert(snippet::MatrixMultiplication(f.Label(), layers_[l]->dA, layers_[l]->dZ, layers_[l]->dZ,
          {vi32_1, vi32_2}));

      // dW[l] = (1/m) dZ[l] . A[l-1]^T
      // 1) dW[l] = dZ[l] . A[l-1]^T
      // 2) dW[l] = (1/m) dW[l]
      f.Insert(snippet::MatrixDotRT(f.Label(), layers_[l]->dZ,
                                    (l == 1) ? snippet::Mat(layers_[0]->A, input_begin) : snippet::Mat(layers_[l-1]->A),
                                    layers_[l]->dW, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1}));
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dW, MakeF32Const(1.0f/batch_size_), layers_[l]->dW,
                                     {vi32_1, vi32_2}));

      // dB[l] = (1/m) dZ[l]
      // FIXME Sum dZ[l] horizontally, store it into first column of dB[l], then broadcast
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dZ, MakeF32Const(1.0f/batch_size_), layers_[l]->dB,
                                     {vi32_1, vi32_2}));

      if(l > 1) {
        // dA[l-1] = W[l]^T . dZ[l]
        f.Insert(snippet::MatrixDotLT(f.Label(), layers_[l]->W, layers_[l]->dZ, layers_[l-1]->dA,
                                      {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1}));
      }

      // W[l] = W[l] - alpha * dW[l]
      // 1) dW[l] = alpha * dW[l]
      // 2) W[l] = W[l] - dW[l]
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dW, MakeF32Const(learning_rate_), layers_[l]->dW,
                                     {vi32_1, vi32_2}));
      f.Insert(snippet::MatrixSubtraction(f.Label(), layers_[l]->W, layers_[l]->dW, layers_[l]->W, {vi32_1, vi32_2}));

      // B[l] = B[l] - alpha * dB[l]
      // 1) dB[l] = alpha * dB[l]
      // 2) B[l] = B[l] - dB[l]
      f.Insert(snippet::MatrixScalar(f.Label(), layers_[l]->dB, MakeF32Const(learning_rate_), layers_[l]->dB,
                                     {vi32_1, vi32_2}));
      f.Insert(snippet::MatrixSubtraction(f.Label(), layers_[l]->B, layers_[l]->dB, layers_[l]->B, {vi32_1, vi32_2}));
    }
  });
}

wabt::Var Model::GenerateCostFunction() {
  std::vector<Type> locals = {Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction("cost_function", {{Type::I32},{Type::F32}}, locals,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vf32_1 = locals[2];

    auto target_begin = params[0];

    // Compute training error
    f.Insert(snippet::MatrixLoss(f.Label(), snippet::Mat(layers_.back()->T, target_begin),
                                 snippet::Mat(layers_.back()->A), loss_, layers_.back()->Cost, {vi32_1, vi32_2}, false));
    f.Insert(snippet::MatrixMean(f.Label(), layers_.back()->Cost, {vi32_1}, vf32_1));
    f.Insert(MakeLocalGet(vf32_1));
  });
}

void Model::Setup(uint32_t epochs, uint32_t batch_size, float learning_rate, builtins::LossFunction loss,
                  std::vector<std::vector<float>> input, std::vector<std::vector<float>> labels) {
  ERROR_UNLESS(training_.empty(), "cannot setup again the same model");
  ERROR_UNLESS(batch_size >= 1, "batch size must be at least 1");
  ERROR_UNLESS(epochs >= 1, "epoch must be at least 1");
  ERROR_UNLESS(input.size() > 0, "training input cannot be empty");
  ERROR_UNLESS(batch_size <= input.size(), "batch size must be at most equal to the input size");
  ERROR_UNLESS(input.size() % batch_size == 0, "batch size must be a multiple of the input size");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");

  batch_size_ = batch_size;
  epochs_ = epochs;
  loss_ = loss;
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
  auto forward = GenerateFeedForward();
  auto backword = GenerateBackpropagation();
  Var cost_func;

  std::vector<Type> locals_type = {Type::F64, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    auto time = locals[0];
    auto cost_mean = locals[1];
    auto epoch = locals[2];
    auto train_addr = locals[3];
    auto label_addr = locals[4];
    auto vi32_1 = locals[5];
    auto vi32_2 = locals[6];

    auto train_begin = training_.front()->Memory()->Begin();
    auto train_end = training_.back()->Memory()->End();
    auto train_size = training_.front()->Memory()->Bytes();
    auto label_begin = labels_.front()->Memory()->Begin();
    auto label_size = labels_.front()->Memory()->Bytes();

    if(options_.log_training_error) {
      cost_func = GenerateCostFunction();
    }
    if(options_.log_training_time) {
      // Start training timer
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }
    f.Insert(GenerateRangeLoop(f.Label(), epoch, 0, epochs_, 1, {}, [&](BlockBody* b1) {
      b1->Insert(MakeLocalSet(label_addr, MakeI32Const(label_begin)));
      b1->Insert(GenerateRangeLoop(f.Label(), train_addr, train_begin, train_end, train_size, {}, [&](BlockBody* b2){
        // Apply neural network algorithms
        b2->Insert(MakeCall(forward, { MakeLocalGet(train_addr)}));
        b2->Insert(MakeCall(backword, {MakeLocalGet(train_addr), MakeLocalGet(label_addr)}));

        if(options_.log_training_error) {
          // Compute training error
          auto call_cost = MakeCall(cost_func, {MakeLocalGet(label_addr)});
          b2->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Add, call_cost));
        }

        b2->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
      }));

      if(options_.log_training_error) {
        // Log training error
        b1->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Div, MakeF32Const(training_.size())));
        b1->Insert(MakeCall(builtins_.message.LogTrainingError(), {MakeLocalGet(cost_mean)}));
      }
    }));

    if(options_.log_training_time) {
      // Log training time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogTrainingTime(), {MakeLocalGet(time)}));
    }

    // Test on training data (for debugging)
    for(int t=0; t < training_.size(); ++t) {
      f.Insert(MakeCall(forward, {MakeI32Const(training_[t]->Memory()->Begin())}));
      PRINT_TABLE(layers_.back()->A)
    }
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
