#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/arch/layers/dense.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/src/arch/initializers.h>
#include <src/wasmpp/wasm-instructions-gen.h>
#include <sstream>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;
using namespace layer;

#define V128_IF_SIMD(t) (options_.use_simd ? Type::V128 : (t))

#define DEFINE_TIME_MEMBERS(name)                                     \
ExprList* DenseForwardTimeMembers::Get##name() {                      \
  assert(name != nullptr);                                            \
  return MakeF64Load(MakeI32Const(name->Begin()));                    \
}                                                                     \
ExprList* DenseForwardTimeMembers::Set##name(wabt::ExprList *value) { \
  assert(name != nullptr);                                            \
  return MakeF64Store(MakeI32Const(name->Begin()), value);            \
}
DENSE_FORWARD_TIME_MEMBERS(DEFINE_TIME_MEMBERS)
#undef DEFINE_TIME_MEMBERS

#define DEFINE_TIME_MEMBERS(name)                                       \
ExprList* DenseBackwardTimeMembers::Get##name() {                       \
  assert(name != nullptr);                                              \
  return MakeF64Load(MakeI32Const(name->Begin()));                      \
}                                                                       \
ExprList* DenseBackwardTimeMembers::Set##name(wabt::ExprList *value) {  \
  assert(name != nullptr);                                              \
  return MakeF64Store(MakeI32Const(name->Begin()), value);              \
}
DENSE_BACKWARD_TIME_MEMBERS(DEFINE_TIME_MEMBERS)
#undef DEFINE_TIME_MEMBERS

wabt::ExprList* Model::SetLearningRate(wabt::ExprList *val) {
  assert(learning_rate_ != nullptr);
  return MakeF32Store(MakeI32Const(learning_rate_->Begin()), val);
}

wabt::ExprList* Model::GetLearningRate() {
  assert(learning_rate_ != nullptr);
  return MakeF32Load(MakeI32Const(learning_rate_->Begin()));
}

void Model::SetLayers(std::vector<Layer *> layers) {
  ERROR_UNLESS(layers.size() >= 2, "At least an input and output layer should be defined");
  for(uint32_t index = 0; index < layers.size(); index++) {
    if(index == 0) {
      ERROR_UNLESS(layers[index]->Position() == Input, "First layer must be an input layer");
    } else if(index == layers.size() - 1) {
      ERROR_UNLESS(layers[index]->Position() == Output, "Last layer must be an output layer");
    } else {
      ERROR_UNLESS(layers[index]->Position() == Hidden, "Middle layer must be a hidden layer");
    }
    layers[index]->SetIndex(index);
    layers[index]->SetModel(this);
    layers_.push_back(layers[index]);
  }

  // Create function to get the total number of layers function
  module_manager_.MakeFunction("total_layers", {{},{Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeI32Const((uint32_t)layers_.size()));
  });

  // Overwrite some configuration as needed
  if(layers.back()->Type() == FullyConnected) {
    auto out_layer = static_cast<DenseOutputLayer*>(layers.back());
    if(options_.log_training_confusion_matrix || options_.log_training_accuracy) {
      out_layer->Hardmax(Training);
    }
    if(options_.log_testing_confusion_matrix || options_.log_testing_accuracy) {
      out_layer->Hardmax(Testing);
    }
    if(options_.log_prediction_results_softmax) {
      out_layer->Softmax(Prediction);
    }
    if(options_.log_prediction_results_hardmax) {
      out_layer->Hardmax(Prediction);
    }
  } else {
    assert(!"Not implemented!");
  }
}

Model::Model(ModelOptions options) : options_(options), builtins_(options_.activation_options) {
  AllocateMembers();
#ifdef WABT_EXPERIMENTAL
  InitNativeImports();
#endif
  InitBuiltinImports();
  InitBuiltinDefinitions();
  InitSnippets();
}

void Model::InitBuiltinImports() {
  builtins_.system.InitImports(this, &module_manager_, "System");
  builtins_.math.InitImports(this, &module_manager_, "Math");
  builtins_.activation.InitImports(this, &module_manager_, "Activation");
  builtins_.loss.InitImports(this, &module_manager_, "Loss");
  builtins_.message.InitImports(this, &module_manager_, "Message");
}

void Model::InitBuiltinDefinitions() {
  // TODO Create simd option (same as snippets)
  builtins_.system.InitDefinitions(this, &module_manager_);
  builtins_.math.InitDefinitions(this, &module_manager_);
  builtins_.activation.InitDefinitions(this, &module_manager_);
  builtins_.loss.InitDefinitions(this, &module_manager_);
  builtins_.message.InitDefinitions(this, &module_manager_);
}

void Model::InitSnippets() {
  if(options_.use_simd) {
    snippets_.matrix = new snippet::MatrixSnippetSimd(&module_manager_.Label(), &builtins_);
    snippets_.analysis = new snippet::AnalysisSnippet(&module_manager_.Label(), &builtins_);
  } else {
    snippets_.matrix = new snippet::MatrixSnippet(&module_manager_.Label(), &builtins_);
    snippets_.analysis = new snippet::AnalysisSnippetSimd(&module_manager_.Label(), &builtins_);
  }
}

#ifdef WABT_EXPERIMENTAL
void Model::InitNativeImports() {
  std::vector<Type> params = {
      Type::I32, // lhs addr
      Type::I32, // rhs addr
      Type::I32, // dst addr
      Type::I32, // lhs_rows
      Type::I32, // lhs_cols
      Type::I32  // rhs_cols
  };
  natives_.dot_product = module_manager_.MakeNativeFunction("dot_product_f32", {params, {}});
  natives_.dot_product_rt = module_manager_.MakeNativeFunction("dot_product_rt_f32", {params, {}});
  natives_.dot_product_lt = module_manager_.MakeNativeFunction("dot_product_lt_f32", {params, {}});
}
#endif

#define ALLOCATE_MATRIX(array, rows, cols) \
    array = new ds::NDArray(module_manager_.Memory().Allocate((rows) * (cols) * TypeSize(Type::F32)), \
                            {rows, cols}, TypeSize(Type::F32));

void Model::AllocateMembers() {
  learning_rate_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  training_hits_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  training_error_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
#define ALLOCATE_TIME_MEMBERS(name) \
  dense_forward_logging_members_.name = module_manager_.Memory().Allocate(TypeSize(Type::F64));
  DENSE_FORWARD_TIME_MEMBERS(ALLOCATE_TIME_MEMBERS)
#undef ALLOCATE_TIME_MEMBERS

#define ALLOCATE_TIME_MEMBERS(name) \
  dense_backward_logging_members_.name = module_manager_.Memory().Allocate(TypeSize(Type::F64));
  DENSE_BACKWARD_TIME_MEMBERS(ALLOCATE_TIME_MEMBERS)
#undef ALLOCATE_TIME_MEMBERS
  }

void Model::AllocateLayers() {
  for(auto l = 0; l < layers_.size(); ++l) {
    layers_[l]->AllocateMemory();
  }
}

void Model::AllocateTraining() {
  // Do not merge loops so that all
  // training data are consecutive in memory

  for(uint32_t b=0; b < training_vals_.size(); b += TrainingBatchSize()) {
    // Training data
    ds::NDArray* training_array = nullptr;
    ALLOCATE_MATRIX(training_array, (uint32_t) training_vals_[0].size(), TrainingBatchSize());
    training_batch_.push_back(training_array);
  }

  for(uint32_t b=0; b < training_labels_vals_.size(); b += TrainingBatchSize()) {
    // Training labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MATRIX(labels_array, (uint32_t) training_labels_vals_[0].size(), TrainingBatchSize());
    training_labels_batch_.push_back(labels_array);
  }
}

void Model::AllocateTest() {
  // Do not merge loops so that all
  // test data are consecutive in memory

  for(uint32_t b=0; b < testing_vals_.size(); b += TestingBatchSize()) {
    // Testing data
    ds::NDArray* testing_array = nullptr;
    ALLOCATE_MATRIX(testing_array, (uint32_t) testing_vals_[0].size(), TestingBatchSize());
    testing_batch_.push_back(testing_array);
  }

  for(uint32_t b=0; b < testing_labels_vals_.size(); b += TestingBatchSize()) {
    // Testing labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MATRIX(labels_array, (uint32_t) testing_labels_vals_[0].size(), TestingBatchSize());
    testing_labels_batch_.push_back(labels_array);
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

void Model::MakeData(wabt::Var memory, std::vector<std::vector<float>> data_vals,
                     std::vector<std::vector<float>> labels_vals, std::vector<ds::NDArray*> data_batch,
                     std::vector<ds::NDArray*> labels_batch, uint32_t batch_size) {
  // Store all data in one vector
  std::vector<DataEntry> data_entries;
  for(uint32_t i=0; i < data_batch.size(); ++i) {
    auto training_begin = data_vals.begin() + (i * batch_size);
    std::vector<std::vector<float>> sub_input(training_begin, training_begin + batch_size);
    auto entry = MakeTransposeData(data_batch[i], sub_input);
    data_entries.insert(data_entries.end(), entry.begin(), entry.end());
  }
  // Split data into chunks
  auto total_data_chunks = 1 + ((data_entries.size() - 1) / MAX_FLOAT_PER_DATA);
  for(uint32_t i=0; i < total_data_chunks; i++) {
    auto begin = data_entries.begin() + (i * MAX_FLOAT_PER_DATA);
    auto end = begin + MAX_FLOAT_PER_DATA;
    if(end > data_entries.end()) {
      end = data_entries.end();
    }
    auto sub_data = std::vector<DataEntry>(begin, end);
    auto memory_begin = data_batch.front()->Memory()->Begin() + (i * MAX_FLOAT_PER_DATA * TypeSize(Type::F32));
    module_manager_.MakeData(memory, memory_begin, sub_data);
  }

  // Store all labels in one vector
  std::vector<DataEntry> labels_entries;
  for(uint32_t i=0; i < labels_batch.size(); ++i) {
    auto labels_begin = labels_vals.begin() + (i * batch_size);
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + batch_size);
    auto entry = MakeTransposeData(labels_batch[i], sub_labels);
    labels_entries.insert(labels_entries.end(), entry.begin(), entry.end());
  }
  // Split labels into chunks
  auto total_labels_chunks = 1 + ((labels_entries.size() - 1) / MAX_FLOAT_PER_DATA);
  for(uint32_t i=0; i < total_labels_chunks; i++) {
    auto begin = labels_entries.begin() + (i * MAX_FLOAT_PER_DATA);
    auto end = begin + MAX_FLOAT_PER_DATA;
    if(end > labels_entries.end()) {
      end = labels_entries.end();
    }
    auto sub_labels = std::vector<DataEntry>(begin, end);
    auto memory_begin = labels_batch.front()->Memory()->Begin() + (i * MAX_FLOAT_PER_DATA * TypeSize(Type::F32));
    module_manager_.MakeData(memory, memory_begin, sub_labels);
  }
}

void Model::MakeTrainingData(wabt::Var memory) {
  MakeData(memory, training_vals_, training_labels_vals_, training_batch_, training_labels_batch_, TrainingBatchSize());
}

void Model::MakeTestingData(wabt::Var memory) {
  MakeData(memory, testing_vals_, testing_labels_vals_, testing_batch_, testing_labels_batch_, TestingBatchSize());
}

void Model::MakeLayersData(wabt::Var memory) {
  for(int l=1; l < layers_.size(); ++l) {
    layers_[l]->MakeData(memory);
  }
}

Var Model::ForwardAlgorithmFunction(uint8_t mode_index) {
  std::vector<Type> locals_types = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::F32,
                                    V128_IF_SIMD(Type::I32)};
  return module_manager_.MakeFunction(nullptr, {{Type::I32},{}}, locals_types,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    assert(locals.size() == 8);
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];
    auto vf32_2 = locals[6];
    auto v128_1 = locals[7];

    assert(params.size() == 1);
    auto input_begin = params[0];

    for(int l=0; l < layers_.size(); ++l) {
      if(layers_[l]->Type() == FullyConnected) {
        if(layers_[l]->Position() == Output) {
          f.Insert(layers_[l]->Forward(mode_index, input_begin, {vi32_1,vi32_2,vi32_3,vi32_4,vi32_5,vf32_1,vf32_2, v128_1}));
        } else {
          f.Insert(layers_[l]->Forward(mode_index, input_begin, {vi32_1,vi32_2,vi32_3,vi32_4,vi32_5,vf32_1, v128_1}));
        }
      } else {
        assert(!"Not implemented!");
      }
    }
  });
}

wabt::Var Model::BackwardAlgorithmFunction() {
  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32,
                                   V128_IF_SIMD(Type::I32)};
  return module_manager_.MakeFunction(nullptr, {{Type::I32, Type::I32},{}}, locals_type,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    assert(locals.size() == 7);
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];
    auto v128_1 = locals[6];

    assert(params.size() == 2);
    auto input_begin = params[0];
    auto target_begin = params[1];

    for(int64_t l = layers_.size()-1; l >= 0; --l) {
      f.Insert(layers_[l]->Backward(input_begin, target_begin, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));
    }
  });
}

wabt::Var Model::ConfusionMatrixFunction(uint8_t mode_index) {
  std::vector<Type> locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  return module_manager_.MakeFunction(nullptr, {{Type::I32}, {}}, locals,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    assert(locals.size() == 6);
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vi32_6 = locals[5];

    assert(params.size() == 1);
    auto target_begin = params[0];

    assert(layers_.back()->Position() == Output);
    if(layers_.back()->Type() == FullyConnected) {
      auto out_layer = static_cast<DenseOutputLayer*>(layers_.back());
      // Update confusion matrix
      f.Insert(out_layer->UpdateConfusionMatrix(mode_index, target_begin, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6}));
    } else {
      assert(!"Not implemented!");
    }
  });
}

wabt::Var Model::CountCorrectPredictionsFunction(uint8_t mode_index) {
  std::vector<Type> locals = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction(nullptr, {{Type::I32}, {Type::F32}}, locals,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    assert(locals.size() == 6);
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto correct_count = locals[5];

    assert(params.size() == 1);
    auto target_begin = params[0];

    assert(layers_.back()->Position() == Output);
    if(layers_.back()->Type() == FullyConnected) {
      auto out_layer = static_cast<DenseOutputLayer*>(layers_.back());
      // Count correct prediction
      f.Insert(out_layer->CountCorrectPredictions(mode_index, target_begin, correct_count, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5}));
      // Return correct count
      f.Insert(MakeLocalGet(correct_count));
    } else {
      assert(!"Not implemented!");
    }
  });

}

void Model::CompileInitialization() {
  Var memory = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memory);
  MakeTrainingData(memory);
  MakeTestingData(memory);
  MakeLayersData(memory);
}

void Model::CompileLayers(uint32_t training_batch_size, uint32_t testing_batch_size, uint32_t prediction_batch_size,
                          nn::builtins::LossFunction loss) {
  ERROR_UNLESS(training_batch_size >= 1, "training batch size must be at least 1");
  ERROR_UNLESS(testing_batch_size >= 1, "testing batch size must be at least 1");
  ERROR_UNLESS(prediction_batch_size >= 1, "prediction batch size must be at least 1");
  training_batch_size_ = training_batch_size;
  testing_batch_size_ = testing_batch_size;
  prediction_batch_size_ = prediction_batch_size;
  loss_ = loss;

  // Allocate layers should be called before create the model
  // algorithms. Otherwise the addresses wouldn't be defined.
  AllocateLayers();

  // Create layers specific functions
  for(auto layer : layers_) {
    layer->MakeFunctions();
  }

  // Create algorithms and export them
  forward_training_func_ = ForwardAlgorithmFunction(Mode::Training);
  forward_testing_func_ = ForwardAlgorithmFunction(Mode::Testing);
  forward_prediction_func_ = ForwardAlgorithmFunction(Mode::Prediction);
  backward_func_ = BackwardAlgorithmFunction();
  confusion_matrix_training_func_ = ConfusionMatrixFunction(Mode::Training);
  confusion_matrix_testing_func_ = ConfusionMatrixFunction(Mode::Testing);
  count_correct_predictions_training_func_ = CountCorrectPredictionsFunction(Mode::Training);
  count_correct_predictions_testing_func_ = CountCorrectPredictionsFunction(Mode::Testing);
}

void Model::CompileTrainingFunctions(uint32_t epochs, float learning_rate, const std::vector<std::vector<float>> &input,
                            const std::vector<std::vector<float>> &labels) {
  ERROR_UNLESS(epochs >= 1, "epoch must be at least 1");
  ERROR_UNLESS(!input.empty(), "training input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");
  ERROR_UNLESS(input.size() % training_batch_size_ == 0, "Input must be a multiple of the training batch size");

  training_vals_ = input;
  training_labels_vals_ = labels;
  AllocateTraining();

  std::vector<Type> locals_type = {Type::F64, Type::F64, Type::F32, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    // Set learning rate
    f.Insert(SetLearningRate(MakeF32Const(learning_rate)));

    assert(locals.size() == 9);
    auto time_total = locals[0];
    auto time_epoch = locals[1];
    auto cost_mean = locals[2];
    auto hits = locals[3]; // TODO Change to Type::I32
    auto epoch = locals[4];
    auto train_addr = locals[5];
    auto label_addr = locals[6];
    auto vi32_1 = locals[7];
    auto vi32_2 = locals[8];

    auto train_begin = training_batch_.front()->Memory()->Begin();
    auto train_end = training_batch_.back()->Memory()->End();
    auto train_size = training_batch_.front()->Memory()->Bytes();
    auto label_begin = training_labels_batch_.front()->Memory()->Begin();
    auto label_size = training_labels_batch_.front()->Memory()->Bytes();

    f.Insert(GenerateRangeLoop(f.Label(), epoch, 0, epochs, 1, {}, [&](BlockBody* b1) {
      b1->Insert(MakeLocalSet(label_addr, MakeI32Const(label_begin)));
      b1->Insert(MakeLocalSet(cost_mean, MakeF32Const(0)));
      b1->Insert(MakeLocalSet(hits, MakeF32Const(0)));
      b1->Insert(GenerateRangeLoop(f.Label(), train_addr, train_begin, train_end, train_size, {}, [&](BlockBody* b2){
        // Forward algorithm
        b2->Insert(MakeCall(forward_training_func_, {
          MakeLocalGet(train_addr)
        }));

        // Backward algorithm
        b2->Insert(MakeCall(backward_func_, {
          MakeLocalGet(train_addr),
          MakeLocalGet(label_addr)
        }));

        if(options_.log_training_accuracy) {
          // Count number of correct results
          b2->Insert(GenerateCompoundAssignment(hits, Opcode::F32Add, MakeCall(count_correct_predictions_training_func_, {
              MakeLocalGet(label_addr)
          })));
        }

        if(options_.log_training_error) {
          // Compute training error
          assert(layers_.back()->Position() == Output);
          if(layers_.back()->Type() == FullyConnected) {
            auto cost = static_cast<DenseOutputLayer*>(layers_.back())->ComputeCost(Mode::Training, label_addr);
            b2->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Add, cost));
          } else {
            assert(!"Compute cost not implemented");
          }
        }

        if(options_.log_training_confusion_matrix) {
          // Update confusion matrix
          b2->Insert(MakeCall(confusion_matrix_training_func_, {MakeLocalGet(label_addr)}));
        }

        // Move to the next label batch in memory
        b2->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
      }));

      if(options_.log_training_error) {
        // Store total cost error in memory
        b1->Insert(MakeF32Store(MakeI32Const(training_error_->Begin()), MakeLocalGet(cost_mean)));
      }

      if(options_.log_training_accuracy) {
        // Store accuracy in memory
        b1->Insert(MakeF32Store(MakeI32Const(training_hits_->Begin()), MakeLocalGet(hits)));
      }
    }));
  });

  if(options_.log_training_accuracy) {
    // Create function to access the number training batches hits
    module_manager_.MakeFunction("training_batches_hits", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF32Load(MakeI32Const(training_hits_->Begin())));
    });
  }

  if(options_.log_training_accuracy) {
    // Create function to access the number training batches hits
    module_manager_.MakeFunction("training_batches_error", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF32Load(MakeI32Const(training_error_->Begin())));
    });
  }

  // Create forward log functions
  if(options_.log_forward) {
#define LOG_TIME_MEMBER(name)                                                                       \
    module_manager_.MakeFunction("log_forward_" #name, {{}, {Type::F64}}, {},                       \
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){ \
      f.Insert(dense_forward_logging_members_.Get##name());                                         \
    });
    DENSE_FORWARD_TIME_MEMBERS(LOG_TIME_MEMBER)
#undef LOG_TIME_MEMBER
  }

  // Create backward log functions
  if(options_.log_backward) {
#define LOG_TIME_MEMBER(name)           \
    module_manager_.MakeFunction("log_backward_" #name, {{}, {Type::F64}}, {},                      \
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){ \
      f.Insert(dense_backward_logging_members_.Get##name());                                        \
    });
    DENSE_BACKWARD_TIME_MEMBERS(LOG_TIME_MEMBER)
#undef LOG_TIME_MEMBER
  }
}

void Model::CompileTestingFunction(const std::vector<std::vector<float>> &input,
                           const std::vector<std::vector<float>> &labels) {
  ERROR_UNLESS(!input.empty(), "test input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "testing and labels size should match");
  ERROR_UNLESS(input.size() % testing_batch_size_ == 0, "Input must be a multiple of the testing batch size");

  testing_vals_ = input;
  testing_labels_vals_ = labels;
  AllocateTest();

  std::vector<Type> locals_type = {Type::F64, Type::F32, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("test", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {
    assert(locals.size() == 7);
    auto time = locals[0];
    auto cost_mean = locals[1];
    auto accuracy = locals[2];
    auto test_addr = locals[3];
    auto label_addr = locals[4];
    auto vi32_1 = locals[5];
    auto vi32_2 = locals[6];

    auto test_begin = testing_batch_.front()->Memory()->Begin();
    auto test_end = testing_batch_.back()->Memory()->End();
    auto test_size = testing_batch_.front()->Memory()->Bytes();
    auto label_begin = testing_labels_batch_.front()->Memory()->Begin();
    auto label_size = testing_labels_batch_.front()->Memory()->Bytes();

    if(options_.log_testing_time) {
      // Start testing timer
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }
    f.Insert(MakeLocalSet(label_addr, MakeI32Const(label_begin)));
    f.Insert(GenerateRangeLoop(f.Label(), test_addr, test_begin, test_end, test_size, {}, [&](BlockBody* b1){

      // Forward algorithm
      b1->Insert(MakeCall(forward_testing_func_, {
        MakeLocalGet(test_addr)
      }));

      if(options_.log_testing_accuracy) {
        // Count number of correct results
        b1->Insert(GenerateCompoundAssignment(accuracy, Opcode::F32Add, MakeCall(count_correct_predictions_testing_func_, {
            MakeLocalGet(label_addr)
        })));
      }

      if(options_.log_testing_error) {
        // Compute testing error
        assert(layers_.back()->Position() == Output);
        if(layers_.back()->Type() == FullyConnected) {
          auto cost = static_cast<DenseOutputLayer*>(layers_.back())->ComputeCost(Mode::Testing, label_addr);
          b1->Insert(GenerateCompoundAssignment(cost_mean, Opcode::F32Add, cost));
        } else {
          assert(!"Compute cost not implemented");
        }
      }

      if(options_.log_testing_confusion_matrix) {
        // Update confusion matrix
        b1->Insert(MakeCall(confusion_matrix_testing_func_, {MakeLocalGet(label_addr)}));
      }

      b1->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
    }));

    if(options_.log_testing_error) {
      // Log testing error
      f.Insert(MakeCall(builtins_.message.LogTestingError(), {
        MakeBinary(Opcode::F32Div, MakeLocalGet(cost_mean), MakeF32Const(testing_batch_.size()))
      }));
    }

    if(options_.log_testing_accuracy) {
      // Log testing accuracy
      f.Insert(MakeCall(builtins_.message.LogTestingAccuracy(), {
          MakeBinary(Opcode::F32Div, MakeLocalGet(accuracy), MakeF32Const(testing_vals_.size()))
      }));
    }

    if(options_.log_testing_time) {
      // Log testing time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogTestingTime(), {MakeLocalGet(time)}));
    }

    if(options_.log_testing_confusion_matrix) {
      assert(layers_.back()->Position() == Output);
      if(layers_.back()->Type() == FullyConnected) {
        auto out_layer = static_cast<DenseOutputLayer*>(layers_.back());
        // Log confusion matrix
        f.Insert(MakeCall(builtins_.system.PrintTableF32(), {
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Testing)->Begin()),
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Testing)->Shape()[0]),
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Testing)->Shape()[1])
        }));
      } else {
        assert(!"Not implemented");
      }
    }
  });
}

void Model::CompilePredictionFunctions() {
  // Create a function to export the prediction input offset
  auto prediction_input_offset = module_manager_.MakeFunction("prediction_input_offset", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    assert(layers_.front()->Position() == Input);
    if(layers_.front()->Type() == FullyConnected) {
      f.Insert(MakeI32Const(static_cast<DenseInputLayer*>(layers_.front())->InputData(Mode::Prediction)->Begin()));
    } else {
      assert(!"No implemented!");
    }
  });

  // Create a function to export the prediction batch size
  module_manager_.MakeFunction("prediction_batch_size", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(PredictionBatchSize()));
  });

  // TODO add options for hardmax and softmax
  // Create prediction function
  auto local_types = {Type::F64, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::F32};
  module_manager_.MakeFunction("predict", {}, local_types,
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    assert(locals.size() == 8);
    auto time = locals[0];
    auto vi32_1 = locals[1];
    auto vi32_2 = locals[2];
    auto vi32_3 = locals[3];
    auto vi32_4 = locals[4];
    auto vi32_5 = locals[5];
    auto vf32_1 = locals[6];
    auto vf32_2 = locals[7];

    if(options_.log_prediction_time) {
      // Start prediction timer
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }

    // Apply forward algorithm
    f.Insert(MakeCall(forward_prediction_func_, {
      MakeCall(prediction_input_offset, {})
    }));

    if(options_.log_prediction_time) {
      // Log prediction time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogPredictionTime(), {MakeLocalGet(time)}));
    }

    // Print predictions
    assert(layers_.back()->Position() == Output);
    if(layers_.back()->Type() == FullyConnected) {
      auto out_layer = static_cast<DenseOutputLayer*>(layers_.back());
      if(options_.log_prediction_results) {
        f.Insert(MakeCall(builtins_.system.PrintTableF32(), {
            MakeI32Const(out_layer->Predictions(Mode::Prediction)->Begin()),
            MakeI32Const(out_layer->Predictions(Mode::Prediction)->Shape()[0]),
            MakeI32Const(out_layer->Predictions(Mode::Prediction)->Shape()[1])
        }));
      }
      if(options_.log_prediction_results_softmax) {
        f.Insert(MakeCall(builtins_.system.PrintTableF32(), {
            MakeI32Const(out_layer->PredictionsSoftmax(Mode::Prediction)->Begin()),
            MakeI32Const(out_layer->PredictionsSoftmax(Mode::Prediction)->Shape()[0]),
            MakeI32Const(out_layer->PredictionsSoftmax(Mode::Prediction)->Shape()[1])
        }));
      }
      if(options_.log_prediction_results_hardmax) {
        f.Insert(MakeCall(builtins_.system.PrintTableF32(), {
            MakeI32Const(out_layer->PredictionsHardmax(Mode::Prediction)->Begin()),
            MakeI32Const(out_layer->PredictionsHardmax(Mode::Prediction)->Shape()[0]),
            MakeI32Const(out_layer->PredictionsHardmax(Mode::Prediction)->Shape()[1])
        }));
      }
    } else {
      assert(!"Not implemented!");
    }
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
