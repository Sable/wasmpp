#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/src/arch/initializers.h>
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
  ds::NDArray* b = nullptr;
  // Back-propagation arrays
  ds::NDArray* dW = nullptr;
  ds::NDArray* dZ = nullptr;
  ds::NDArray* dA = nullptr;
  ds::NDArray* db = nullptr;
  // Regularization
  ds::NDArray* inverted_dropout = nullptr;
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

  // Init snippets
  if(options.use_simd) {
    snippets_.matrix = new snippet::MatrixSnippetSimd(&module_manager_.Label());
  } else {
    snippets_.matrix = new snippet::MatrixSnippet(&module_manager_.Label());

  }
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

#define PRINT_TABLE(f, table)                              \
  (f).Insert(MakeCall(builtins_.system.PrintTableF32(), { \
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
    ALLOCATE_MEMORY(layers_[l]->inverted_dropout, layers_[l]->layer->Nodes(), batch_size_);
    if(l > 0) {
      ALLOCATE_MEMORY(layers_[l]->Z, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->dZ, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->dA, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(layers_[l]->W, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->dW, layers_[l]->layer->Nodes(), layers_[l-1]->layer->Nodes());
      ALLOCATE_MEMORY(layers_[l]->b, layers_[l]->layer->Nodes(), 1);
      ALLOCATE_MEMORY(layers_[l]->db, layers_[l]->layer->Nodes(), 1);
    }
    if(l == layers_.size() - 1) {
      ALLOCATE_MEMORY(true_matrix_, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(cost_matrix_, layers_[l]->layer->Nodes(), batch_size_);
      ALLOCATE_MEMORY(confusion_matrix_, layers_[l]->layer->Nodes(), layers_[l]->layer->Nodes());
    }
  }
}

void Model::AllocateTraining() {
  // Do not merge loops so that all
  // training data are consecutive in memory

  for(uint32_t b=0; b < training_vals_.size(); b += batch_size_) {
    // Training data
    ds::NDArray* training_array = nullptr;
    ALLOCATE_MEMORY(training_array, (uint32_t) training_vals_[0].size(), batch_size_);
    training_.push_back(training_array);
  }

  for(uint32_t b=0; b < training_labels_vals_.size(); b += batch_size_) {
    // Training labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MEMORY(labels_array, (uint32_t) training_labels_vals_[0].size(), batch_size_);
    training_labels_.push_back(labels_array);
  }
}

void Model::AllocateTest() {
  // Do not merge loops so that all
  // test data are consecutive in memory

  for(uint32_t b=0; b < testing_vals_.size(); b += batch_size_) {
    // Testing data
    ds::NDArray* testing_array = nullptr;
    ALLOCATE_MEMORY(testing_array, (uint32_t) testing_vals_[0].size(), batch_size_);
    testing_.push_back(testing_array);
  }

  for(uint32_t b=0; b < testing_labels_vals_.size(); b += batch_size_) {
    // Testing labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MEMORY(labels_array, (uint32_t) testing_labels_vals_[0].size(), batch_size_);
    testing_labels_.push_back(labels_array);
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

void Model::MakeTrainingData(wabt::Var memory) {
  for(uint32_t i=0; i < training_.size(); ++i) {
    // Training data
    auto training_begin = training_vals_.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_input(training_begin, training_begin + batch_size_);
    module_manager_.MakeData(memory, training_[i]->Memory()->Begin(), MakeTransposeData(training_[i], sub_input));
  }

  for(uint32_t i=0; i < training_labels_.size(); ++i) {
    // Training labels
    auto labels_begin = training_labels_vals_.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + batch_size_);
    module_manager_.MakeData(memory, training_labels_[i]->Memory()->Begin(),
                             MakeTransposeData(training_labels_[i], sub_labels));
  }
}

void Model::MakeTestingData(wabt::Var memory) {
  for(uint32_t i=0; i < testing_.size(); ++i) {
    // Testing data
    auto testing_begin = testing_vals_.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_input(testing_begin, testing_begin + batch_size_);
    module_manager_.MakeData(memory, testing_[i]->Memory()->Begin(), MakeTransposeData(testing_[i], sub_input));
  }

  for(uint32_t i=0; i < testing_labels_.size(); ++i) {
    // Testing labels
    auto labels_begin = testing_labels_vals_.begin() + (i * batch_size_);
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + batch_size_);
    module_manager_.MakeData(memory, testing_labels_[i]->Memory()->Begin(),
                             MakeTransposeData(testing_labels_[i], sub_labels));
  }
}

void Model::MakeWeightData(wabt::Var memory) {
  for(int l=1; l < layers_.size(); ++l) {
    auto weight_type = layers_[l]->layer->WeightType();
    if(l == layers_.size() - 1) {
      ERROR_UNLESS(weight_type >= FIRST_HIDDEN_OUTPUT && weight_type <= LAST_HIDDEN_OUTPUT,
                   "Wrong weight distribution for the output layer");
    }
    auto weight_size = layers_[l]->W->Shape()[0] * layers_[l]->W->Shape()[1];
    auto bias_size = layers_[l]->b->Shape()[0] * layers_[l]->b->Shape()[1];
    std::vector<DataEntry> weight_entries;
    std::vector<DataEntry> bias_entries;
    switch (weight_type) {
      case XavierUniform:
      case XavierNormal:
        weight_entries = XavierDistribution(weight_size, layers_[l-1]->A->Shape()[0], layers_[l+1]->A->Shape()[0],
            weight_type == XavierUniform, options_.weights_options.seed);
        bias_entries = XavierDistribution(bias_size, layers_[l-1]->A->Shape()[0], layers_[l+1]->A->Shape()[0],
            weight_type == XavierUniform, options_.weights_options.seed);
        break;
      case LeCunUniform:
      case LeCunNormal:
        weight_entries = LeCunDistribution(weight_size, layers_[l - 1]->A->Shape()[0], weight_type == LeCunUniform,
            options_.weights_options.seed);
        bias_entries = LeCunDistribution(bias_size, layers_[l - 1]->A->Shape()[0], weight_type == LeCunUniform,
            options_.weights_options.seed);
        break;
      case Gaussian:
        weight_entries = GaussianDistribution(weight_size, options_.weights_options.gaussian_mean,
            options_.weights_options.gaussian_std_dev, options_.weights_options.seed);
        bias_entries = GaussianDistribution(bias_size, options_.weights_options.gaussian_mean,
            options_.weights_options.gaussian_std_dev, options_.weights_options.seed);
        break;
      case Uniform:
        weight_entries = UniformDistribution(weight_size, options_.weights_options.uniform_low,
            options_.weights_options.uniform_high, options_.weights_options.seed);
        bias_entries = UniformDistribution(bias_size, options_.weights_options.uniform_low,
            options_.weights_options.uniform_high, options_.weights_options.seed);
        break;
      case Constant:
        weight_entries = ConstantDistribution(weight_size, options_.weights_options.constant_value);
        bias_entries = ConstantDistribution(bias_size, options_.weights_options.constant_value);
        break;
      default:
        assert(!"Weight distribution not implemented");
    }
    module_manager_.MakeData(memory, layers_[l]->W->Memory()->Begin(), weight_entries);
    module_manager_.MakeData(memory, layers_[l]->b->Memory()->Begin(), bias_entries);
  }
}

Var Model::GenerateFeedForward() {
  std::vector<Type> locals_types = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction("forward", {{Type::I32, Type::I32, Type::I32},{}}, locals_types,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];

    auto input_begin = params[0];
    auto target_begin = params[1];
    auto is_training = params[2];

    for(int l=1; l < layers_.size(); ++l) {

      // Check for dropout regularization
      if(layers_[l-1]->layer->KeepProb() != 1.0) {
        // Only dropout on training
        f.Insert(MakeIf(f.Label(), MakeBinary(Opcode::I32Eq, MakeLocalGet(is_training), MakeI32Const(1)), {},
                        [&](BlockBody true_block, Var true_label){
          // Generate a mask matrix
          true_block.Insert(MakeCall(builtins_.math.MaskMatrix(), {
              MakeI32Const(layers_[l-1]->inverted_dropout->Memory()->Begin()),
              MakeI32Const(layers_[l-1]->inverted_dropout->Memory()->End()),
              MakeF32Const(layers_[l-1]->layer->KeepProb())
          }));
          // A[l-1] = (1/keep_prob) * (A[l-1] * inverted_dropout[l-1])
          // 1) A[l-1] = A[l-1] * inverted_dropout[l-1]
          // 2) A[l-1] = A[l-1] * (1/keep_prob)
          true_block.Insert(snippets_.matrix->MatrixMultiplication(layers_[l-1]->A, layers_[l-1]->inverted_dropout,
                                                 layers_[l-1]->A, {vi32_1, vi32_2}));
          auto scalar = MakeBinary(Opcode::F32Div, MakeF32Const(1.0f), MakeF32Const(layers_[l-1]->layer->KeepProb()));
          true_block.Insert(snippets_.matrix->MatrixScalar(layers_[l-1]->A, scalar, layers_[l-1]->A,
                                                  {vi32_1, vi32_2, vf32_1}));
        }));
      }

      // Z[l] = W[l] . A[l-1] + b[l]
      // 1) Z[l] = W[l] . A[l-1]
      // 2) Z[l] = Z[l] + b[l]
      f.Insert(snippets_.matrix->MatrixDot(layers_[l]->W,
                                  (l == 1) ? snippet::RelocMat(layers_[0]->A, input_begin) :
                                  snippet::RelocMat(layers_[l-1]->A), layers_[l]->Z,
                                  {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1}));
      f.Insert(snippets_.matrix->MatrixVectorAddition(layers_[l]->Z, layers_[l]->b, layers_[l]->Z,
                                                      {vi32_1, vi32_2, vi32_3, vi32_4}));

      // A[l] = g(Z[l])
      f.Insert(snippets_.matrix->MatrixActivation(snippet::RelocMat(layers_[l]->Z), layers_[l]->layer->ActivationFunction(),
          layers_[l]->A, {vi32_1, vi32_2}, false));
    }

    // dA[L] = Loss(T[L], A[L])
    f.Insert(MakeCall(loss_.loss, {
      MakeLocalGet(target_begin),
      MakeI32Const(layers_.back()->A->Memory()->Begin()),
      MakeI32Const(layers_.back()->dA->Memory()->Begin()),
      MakeI32Const(layers_.back()->A->Shape()[0]),
      MakeI32Const(layers_.back()->A->Shape()[1])
    }));
  });
}

wabt::Var Model::GenerateBackpropagation() {
  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128};
  return module_manager_.MakeFunction("backward", {{Type::I32},{}}, locals_type,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];
    auto v128_1 = locals[6];

    auto input_begin = params[0];

    for(auto l = layers_.size()-1; l > 0; --l) {
       // dZ[l] = dA[l] * g'(Z[l])
       // 1) dZ[l] = g'(Z[l])
       // 2) dZ[l] = dA[l] * dZ[l]
      f.Insert(snippets_.matrix->MatrixActivation(snippet::RelocMat(layers_[l]->Z),
                                                  layers_[l]->layer->ActivationFunction(), layers_[l]->dZ,
                                                  {vi32_1, vi32_2}, true));
      f.Insert(snippets_.matrix->MatrixMultiplication(layers_[l]->dA, layers_[l]->dZ, layers_[l]->dZ,
                                                      {vi32_1, vi32_2}));

      // dW[l] = (1/m) dZ[l] . A[l-1]^T
      // 1) dW[l] = dZ[l] . A[l-1]^T
      // 2) dW[l] = (1/m) dW[l]
      f.Insert(snippets_.matrix->MatrixDotRT(layers_[l]->dZ,
                                             (l == 1) ? snippet::RelocMat(layers_[0]->A, input_begin) :
                                             snippet::RelocMat(layers_[l-1]->A), layers_[l]->dW,
                                             {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));
      f.Insert(snippets_.matrix->MatrixScalar(layers_[l]->dW, MakeF32Const(1.0f/batch_size_), layers_[l]->dW,
                                              {vi32_1, vi32_2, vf32_1}));

      // db[l] = (1/m) dZ[l]
      // 1) db[l] = SUM(dZ[l], row wise)
      // 2) db[l] = (1/m) db[l]
      f.Insert(snippets_.matrix->MatrixRowSum(layers_[l]->dZ, layers_[l]->db, {vi32_1, vi32_2, vi32_3, vf32_1, v128_1}));
      f.Insert(snippets_.matrix->MatrixScalar(layers_[l]->db, MakeF32Const(1.0f/batch_size_), layers_[l]->db,
                                              {vi32_1, vi32_2, vf32_1}));

      if(l > 1) {
        // dA[l-1] = W[l]^T . dZ[l]
        f.Insert(snippets_.matrix->MatrixDotLT(layers_[l]->W, layers_[l]->dZ, layers_[l-1]->dA,
                                               {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1}));
      }

      // W[l] = W[l] - alpha * dW[l]
      // 1) dW[l] = alpha * dW[l]
      // 2) W[l] = W[l] - dW[l]
      f.Insert(snippets_.matrix->MatrixScalar(layers_[l]->dW, MakeF32Const(learning_rate_), layers_[l]->dW,
                                              {vi32_1, vi32_2, vf32_1}));
      f.Insert(snippets_.matrix->MatrixSubtraction(layers_[l]->W, layers_[l]->dW, layers_[l]->W, {vi32_1, vi32_2}));

      // b[l] = b[l] - alpha * db[l]
      // 1) db[l] = alpha * db[l]
      // 2) b[l] = b[l] - db[l]
      f.Insert(snippets_.matrix->MatrixScalar(layers_[l]->db, MakeF32Const(learning_rate_), layers_[l]->db,
                                              {vi32_1, vi32_2, vf32_1}));
      f.Insert(snippets_.matrix->MatrixSubtraction(layers_[l]->b, layers_[l]->db, layers_[l]->b, {vi32_1, vi32_2}));
    }
  });
}

wabt::Var Model::GenerateConfusionMatrixFunction() {
  std::vector<Type> locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  return module_manager_.MakeFunction("confusion_matrix_function", {{Type::I32}, {}}, locals,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    auto col = locals[0];
    auto row = locals[1];
    auto rel_row = locals[2];
    auto vi32_1 = locals[3];
    auto vi32_2 = locals[4];
    auto vi32_3 = locals[5];
    auto vi32_4 = locals[6];
    auto x = vi32_1;
    auto y = vi32_2;
    auto offset = vi32_3;

    auto label_begin = params[0];

    auto A = layers_.back()->A;
    uint32_t type_size = TypeSize(Type::F32);
    uint32_t A_width = A->Shape()[1] * type_size;
    uint32_t A_height = A->Shape()[0] * A_width;

    f.Insert(snippets_.matrix->MatrixColumnArgmax(A, {vi32_1, vi32_2, vi32_3, vi32_4}));
    f.Insert(GenerateRangeLoop(f.Label(), col, 0, A_width, type_size, {}, [&](BlockBody* b1) {
      // Find 1 in both A and target
      b1->Insert(MakeLocalSet(rel_row, MakeI32Const(0)));
      b1->Insert(GenerateRangeLoop(f.Label(), row, 0, A_height, A_width, {}, [&](BlockBody* b2) {
        b2->Insert(MakeLocalSet(offset, MakeBinary(Opcode::I32Add, MakeLocalGet(col), MakeLocalGet(row))));
        auto label_cur = MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeLocalGet(label_begin));
        auto A_cur = MakeBinary(Opcode::I32Add, MakeLocalGet(offset), MakeI32Const(A->Memory()->Begin()));
        b2->Insert(MakeIf(f.Label(), MakeBinary(Opcode::F32Eq, MakeF32Load(label_cur), MakeF32Const(1)), {},
                          [&](BlockBody t, Var label) {
          t.Insert(MakeLocalSet(y, MakeLocalGet(rel_row)));
        }));
        b2->Insert(MakeIf(f.Label(), MakeBinary(Opcode::F32Eq, MakeF32Load(A_cur), MakeF32Const(1)), {},
                          [&](BlockBody t, Var label) {
          t.Insert(MakeLocalSet(x, MakeLocalGet(rel_row)));
        }));
        b2->Insert(GenerateCompoundAssignment(rel_row, Opcode::I32Add, MakeI32Const(type_size)));
      }));

      // Add 1 in confusion matrix
      auto cm_y = MakeBinary(Opcode::I32Mul, MakeLocalGet(y), MakeI32Const(confusion_matrix_->Shape()[0]));
      b1->Insert(MakeLocalSet(offset, MakeBinary(Opcode::I32Add, MakeLocalGet(x), cm_y)));
      b1->Insert(GenerateCompoundAssignment(offset, Opcode::I32Add, MakeI32Const(confusion_matrix_->Memory()->Begin())));
      b1->Insert(MakeF32Store(MakeLocalGet(offset), MakeBinary(Opcode::F32Add, MakeF32Load(MakeLocalGet(offset)),
                                                             MakeF32Const(1))));
    }));
  });
}

void Model::CompileInitialization() {
  Var memory = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memory);
  MakeTrainingData(memory);
  MakeTestingData(memory);
  MakeWeightData(memory);
}

void Model::CompileLayers(uint32_t batch_size, float learning_rate, nn::builtins::LossFunction loss) {
  ERROR_UNLESS(batch_size >= 1, "batch size must be at least 1");
  batch_size_ = batch_size;
  loss_ = loss;
  learning_rate_ = learning_rate;
  AllocateLayers();
  forward_ = GenerateFeedForward();
  backward_ = GenerateBackpropagation();
  confusion_matrix_func_ = GenerateConfusionMatrixFunction();
}

void Model::CompileTraining(uint32_t epochs, const std::vector<std::vector<float>> &input,
                            const std::vector<std::vector<float>> &labels) {
  ERROR_UNLESS(epochs >= 1, "epoch must be at least 1");
  ERROR_UNLESS(!input.empty(), "training input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");

  // FIXME Add zero padding for unaligned batches
  assert(batch_size_ == 1);

  training_vals_ = input;
  training_labels_vals_ = labels;
  AllocateTraining();

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
    auto label_begin = training_labels_.front()->Memory()->Begin();
    auto label_size = training_labels_.front()->Memory()->Bytes();

    if(options_.log_training_time) {
      // Start training timer
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }
    f.Insert(GenerateRangeLoop(f.Label(), epoch, 0, epochs, 1, {}, [&](BlockBody* b1) {
      b1->Insert(MakeLocalSet(label_addr, MakeI32Const(label_begin)));
      b1->Insert(MakeLocalSet(cost_mean, MakeF32Const(0)));
      b1->Insert(GenerateRangeLoop(f.Label(), train_addr, train_begin, train_end, train_size, {}, [&](BlockBody* b2){
        // Apply neural network algorithms
        b2->Insert(MakeCall(forward_, {
          MakeLocalGet(train_addr),
          MakeLocalGet(label_addr),
          MakeI32Const(1) // is_training = 1
        }));
        b2->Insert(MakeCall(backward_, {
          MakeLocalGet(train_addr)
        }));

        if(options_.log_training_error) {
          // Compute training error
          b2->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Add, MakeCall(loss_.cost, {
              MakeLocalGet(label_addr),
              MakeI32Const(layers_.back()->A->Memory()->Begin()),
              MakeI32Const(layers_.back()->A->Shape()[0]),
              MakeI32Const(layers_.back()->A->Shape()[1])
          })));
        }

        b2->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
      }));

      if(options_.log_training_error) {
        // Log training error
        b1->Insert(MakeCall(builtins_.message.LogTrainingError(), {
          MakeLocalGet(epoch),
          MakeBinary(Opcode::F32Div, MakeLocalGet(cost_mean), MakeF32Const(training_.size()))
        }));
      }
    }));

    if(options_.log_training_time) {
      // Log training time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogTrainingTime(), {MakeLocalGet(time)}));
    }
  });
}

void Model::CompileTesting(const std::vector<std::vector<float>> &input,
                           const std::vector<std::vector<float>> &labels) {
  ERROR_UNLESS(!input.empty(), "test input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "testing and labels size should match");

  // FIXME Add zero padding for unaligned batches
  assert(batch_size_ == 1);

  testing_vals_ = input;
  testing_labels_vals_ = labels;
  AllocateTest();

  std::vector<Type> locals_type = {Type::F64, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("test", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    auto time = locals[0];
    auto cost_mean = locals[1];
    auto test_addr = locals[2];
    auto label_addr = locals[3];
    auto vi32_1 = locals[4];
    auto vi32_2 = locals[5];

    auto test_begin = testing_.front()->Memory()->Begin();
    auto test_end = testing_.back()->Memory()->End();
    auto test_size = testing_.front()->Memory()->Bytes();
    auto label_begin = testing_labels_.front()->Memory()->Begin();
    auto label_size = testing_labels_.front()->Memory()->Bytes();

    if(options_.log_testing_time) {
      // Start testing timer
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }
    f.Insert(MakeLocalSet(label_addr, MakeI32Const(label_begin)));
    f.Insert(GenerateRangeLoop(f.Label(), test_addr, test_begin, test_end, test_size, {}, [&](BlockBody* b1){
      b1->Insert(MakeCall(forward_, {
        MakeLocalGet(test_addr),
        MakeLocalGet(label_addr),
        MakeI32Const(0) // is_training = 0
      }));

      if(options_.log_testing_error) {
        // Compute testing error
        b1->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Add, MakeCall(loss_.cost, {
            MakeLocalGet(label_addr),
            MakeI32Const(layers_.back()->A->Memory()->Begin()),
            MakeI32Const(layers_.back()->A->Shape()[0]),
            MakeI32Const(layers_.back()->A->Shape()[1])
        })));
      }

      if(options_.log_testing_confusion_matrix) {
        // Update confusion matrix
        // Note: Call Confusion matrix function
        // will alter the values of A[L]
        b1->Insert(MakeCall(confusion_matrix_func_, {MakeLocalGet(label_addr)}));
      }

      b1->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
    }));

    if(options_.log_testing_error) {
      // Log testing error
      f.Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Div, MakeF32Const(testing_.size())));
      f.Insert(MakeCall(builtins_.message.LogTestingError(), {MakeLocalGet(cost_mean)}));
    }

    if(options_.log_testing_time) {
      // Log testing time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogTestingTime(), {MakeLocalGet(time)}));
    }

    if(options_.log_testing_confusion_matrix) {
      // Log confusion matrix
      PRINT_TABLE(f, confusion_matrix_);
    }
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
