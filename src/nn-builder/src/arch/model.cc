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

#define V128_IF_SIMD(t) (options_.bytecode_options.use_simd ? Type::V128 : (t))

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

void Model::Build(uint32_t training_batch_size, uint32_t training_batches_in_memory, uint32_t testing_batch_size,
                       uint32_t testing_batches_in_memory, uint32_t prediction_batch_size,
                       nn::builtins::LossFunction loss) {
  ERROR_UNLESS(training_batch_size >= 1, "training batch size must be at least 1");
  ERROR_UNLESS(testing_batch_size >= 1, "testing batch size must be at least 1");
  ERROR_UNLESS(prediction_batch_size >= 1, "prediction batch size must be at least 1");
  ERROR_UNLESS(training_batches_in_memory >= 1, "training batches in memory must be at least 1");
  ERROR_UNLESS(testing_batches_in_memory >= 1, "testing batches in memory must be at least 1");
  training_batch_size_ = training_batch_size;
  training_batches_in_memory_ = training_batches_in_memory;
  testing_batch_size_ = testing_batch_size;
  testing_batches_in_memory_ = testing_batches_in_memory;
  prediction_batch_size_ = prediction_batch_size;
  loss_ = loss;

  // Validate all layers now that
  // all parameters has been set
  for(auto &layer : layers_) {
    layer->Validate();
  }

  AllocateMemory();
  MakeLayersFunctions();
  MakeAlgorithmsFunctions();
  MakeTrainingFunctions();
  MakeTestingFunctions();
  MakePredictionFunctions();
  MakeData();
}

void Model::InitBuiltinImports() {
  builtins_.system.InitImports(this, &module_manager_, "System");
  builtins_.math.InitImports(this, &module_manager_, "Math");
  builtins_.activation.InitImports(this, &module_manager_, "Activation");
  builtins_.loss.InitImports(this, &module_manager_, "Loss");
}

void Model::InitBuiltinDefinitions() {
  // TODO Create simd option (same as snippets)
  builtins_.system.InitDefinitions(this, &module_manager_);
  builtins_.math.InitDefinitions(this, &module_manager_);
  builtins_.activation.InitDefinitions(this, &module_manager_);
  builtins_.loss.InitDefinitions(this, &module_manager_);
}

void Model::InitSnippets() {
  if(options_.bytecode_options.use_simd) {
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

void Model::AllocateMemory() {
  AllocateMembers();
  AllocateLayers();
}

void Model::AllocateMembers() {
  learning_rate_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  training_hits_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  training_error_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  training_time_ = module_manager_.Memory().Allocate(TypeSize(Type::F64));
  testing_hits_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  testing_error_ = module_manager_.Memory().Allocate(TypeSize(Type::F32));
  testing_time_ = module_manager_.Memory().Allocate(TypeSize(Type::F64));
  prediction_time_ = module_manager_.Memory().Allocate(TypeSize(Type::F64));
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
  for (auto &layer : layers_) {
    layer->AllocateMemory();
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

void Model::MakeData() {
  Var memory = module_manager_.MakeMemory(module_manager_.Memory().Pages());
  module_manager_.MakeMemoryExport("memory", memory);
  for(int l=1; l < layers_.size(); ++l) {
    layers_[l]->MakeData(memory);
  }
}

void Model::MakeLayersFunctions() {
  for(auto layer : layers_) {
    layer->MakeFunctions();
  }
}

void Model::MakeAlgorithmsFunctions() {
  forward_training_func_            = ForwardAlgorithmFunction(Mode::Training);
  forward_testing_func_             = ForwardAlgorithmFunction(Mode::Testing);
  forward_prediction_func_          = ForwardAlgorithmFunction(Mode::Prediction);
  backward_func_                    = BackwardAlgorithmFunction();
  if(options_.bytecode_options.gen_training_confusion_matrix) {
    confusion_matrix_training_func_ = ConfusionMatrixFunction(Mode::Training);
  }
  if(options_.bytecode_options.gen_testing_confusion_matrix) {
    confusion_matrix_testing_func_ = ConfusionMatrixFunction(Mode::Testing);
  }
  if(options_.bytecode_options.gen_training_accuracy) {
    count_correct_predictions_training_func_ = CountCorrectPredictionsFunction(Mode::Training);
  }
  if(options_.bytecode_options.gen_testing_accuracy) {
    count_correct_predictions_testing_func_ = CountCorrectPredictionsFunction(Mode::Testing);
  }
}

void Model::GetInputOutputSize(uint32_t *input_size, uint32_t *output_size) {
  if(layers_.front()->Type() == FullyConnected)  {
    *input_size = static_cast<DenseInputLayer*>(layers_.front())->Nodes();
  } else {
    assert(!"Not implemented");
  }
  if(layers_.back()->Type() == FullyConnected)  {
    *output_size = static_cast<DenseOutputLayer*>(layers_.back())->Nodes();
  } else {
    assert(!"Not implemented");
  }
}

void Model::MakeTrainingFunctions() {

  // Get the number of input and output
  uint32_t input_size;
  uint32_t output_size;
  GetInputOutputSize(&input_size, &output_size);

  // Allocate memory for data
  uint32_t data_entry_bytes             = input_size * TypeSize(Type::F32);
  uint32_t data_batch_bytes             = data_entry_bytes * TrainingBatchSize();
  uint32_t data_batches_in_memory_bytes = data_batch_bytes * TrainingBatchesInMemory();
  training_data_batches_ = module_manager_.Memory().Allocate(data_batches_in_memory_bytes);

  // Allocate memory for labels
  uint32_t labels_entry_bytes             = output_size * TypeSize(Type::F32);
  uint32_t labels_batch_bytes             = labels_entry_bytes * TrainingBatchSize();
  uint32_t labels_batches_in_memory_bytes = labels_batch_bytes * TrainingBatchesInMemory();
  training_labels_batches_ = module_manager_.Memory().Allocate(labels_batches_in_memory_bytes);

  // Create function to get the offset for training data in memory
  module_manager_.MakeFunction("training_data_offset", {{},{Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(training_data_batches_->Begin()));
  });

  // Create function to get the offset for training labels in memory
  module_manager_.MakeFunction("training_labels_offset", {{},{Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(training_labels_batches_->Begin()));
  });

  // Create function to get the learning rate
  module_manager_.MakeFunction("get_learning_rate", {{},{Type::F32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(GetLearningRate());
  });

  // Create function to set the learning rate
  module_manager_.MakeFunction("set_learning_rate", {{Type::F32},{}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(SetLearningRate(MakeLocalGet(params[0])));
  });

  // Create training function
  std::vector<Type> locals_type = {Type::F64, Type::F64, Type::F32, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("train_batches_in_memory", {{Type::I32},{}}, locals_type,
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    assert(params.size() == 1);
    auto batches_to_train_on = params[0];

    assert(locals.size() == 9);
    auto total_time = locals[0];
    auto batch_time = locals[1];
    auto cost = locals[2];
    auto hits = locals[3];        // TODO Change to Type::I32
    auto counter = locals[4];
    auto train_addr = locals[5];
    auto label_addr = locals[6];
    auto vi32_1 = locals[7];
    auto vi32_2 = locals[8];

    // Set the training pointer to the address of the first training batch
    f.Insert(MakeLocalSet(train_addr, MakeI32Const(training_data_batches_->Begin())));
    // Set the labels pointer to the address of the first labels batch
    f.Insert(MakeLocalSet(label_addr, MakeI32Const(training_labels_batches_->Begin())));

    // Loop on batches in memory
    f.Insert(GenerateDoWhileLoop(f.Label(), counter, batches_to_train_on, 1, {}, [&](BlockBody* b1){

      // Start timer
      if(options_.bytecode_options.gen_training_time) {
        b1->Insert(MakeLocalSet(batch_time, MakeCall(builtins_.system.TimeF64(), {})));
      }

      // Forward algorithm
      b1->Insert(MakeCall(forward_training_func_, {
        MakeLocalGet(train_addr)
      }));

      // Backward algorithm
      b1->Insert(MakeCall(backward_func_, {
        MakeLocalGet(train_addr),
        MakeLocalGet(label_addr)
      }));

      // Update time
      if(options_.bytecode_options.gen_training_time) {
        b1->Insert(GenerateCompoundAssignment(total_time, Opcode::F64Add,
                                              MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}),
                                                         MakeLocalGet(batch_time))));
      }

      // Count number of correct results
      if(options_.bytecode_options.gen_training_accuracy) {
        b1->Insert(GenerateCompoundAssignment(hits, Opcode::F32Add, MakeCall(count_correct_predictions_training_func_, {
            MakeLocalGet(label_addr)
        })));
      }

      // Compute training error
      if(options_.bytecode_options.gen_training_error) {
        assert(layers_.back()->Position() == Output);
        if(layers_.back()->Type() == FullyConnected) {
          auto compute_cost = static_cast<DenseOutputLayer*>(layers_.back())->ComputeCost(Mode::Training, label_addr);
          b1->Insert(GenerateCompoundAssignment(cost, Opcode::F32Add, compute_cost));
        } else {
          assert(!"Compute cost not implemented");
        }
      }

      // Update confusion matrix
      if(options_.bytecode_options.gen_training_confusion_matrix) {
        b1->Insert(MakeCall(confusion_matrix_training_func_, {MakeLocalGet(label_addr)}));
      }

      // Move to the next data batch in memory
      b1->Insert(GenerateCompoundAssignment(train_addr, Opcode::I32Add, MakeI32Const(data_batch_bytes)));
      // Move to the next label batch in memory
      b1->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(labels_batch_bytes)));
    }));

    // Store time
    if(options_.bytecode_options.gen_training_time) {
      f.Insert(MakeF64Store(MakeI32Const(training_time_->Begin()), MakeLocalGet(total_time)));
    }

    // Store total cost error in memory
    if(options_.bytecode_options.gen_training_error) {
      f.Insert(MakeF32Store(MakeI32Const(training_error_->Begin()), MakeLocalGet(cost)));
    }

    // Store accuracy in memory
    if(options_.bytecode_options.gen_training_accuracy) {
      f.Insert(MakeF32Store(MakeI32Const(training_hits_->Begin()), MakeLocalGet(hits)));
    }
  });

  // Create function to access training batches hits
  if(options_.bytecode_options.gen_training_accuracy) {
    module_manager_.MakeFunction("training_batches_hits", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF32Load(MakeI32Const(training_hits_->Begin())));
    });
  }

  // Create function to access training batches error
  if(options_.bytecode_options.gen_training_error) {
    module_manager_.MakeFunction("training_batches_error", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF32Load(MakeI32Const(training_error_->Begin())));
    });
  }

  // Create function to access training time
  if(options_.bytecode_options.gen_training_error) {
    module_manager_.MakeFunction("training_time", {{},{Type::F64}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF64Load(MakeI32Const(training_time_->Begin())));
    });
  }

  // Create function to access training batch size
  module_manager_.MakeFunction("training_batch_size", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(TrainingBatchSize()));
  });

  // Create function to access training batches in memory
  module_manager_.MakeFunction("training_batches_in_memory", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(TrainingBatchesInMemory()));
  });

  // Create forward log functions
  if(options_.bytecode_options.gen_forward_profiling) {
#define LOG_TIME_MEMBER(name)                                                                       \
    module_manager_.MakeFunction("log_forward_" #name, {{}, {Type::F64}}, {},                       \
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){ \
      f.Insert(dense_forward_logging_members_.Get##name());                                         \
    });
    DENSE_FORWARD_TIME_MEMBERS(LOG_TIME_MEMBER)
#undef LOG_TIME_MEMBER
  }

  // Create backward log functions
  if(options_.bytecode_options.gen_backward_profiling) {
#define LOG_TIME_MEMBER(name)           \
    module_manager_.MakeFunction("log_backward_" #name, {{}, {Type::F64}}, {},                      \
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){ \
      f.Insert(dense_backward_logging_members_.Get##name());                                        \
    });
    DENSE_BACKWARD_TIME_MEMBERS(LOG_TIME_MEMBER)
#undef LOG_TIME_MEMBER
  }
}

void Model::MakeTestingFunctions() {

  // Get the number of input and output
  uint32_t input_size;
  uint32_t output_size;
  GetInputOutputSize(&input_size, &output_size);

  // Allocate memory for data
  uint32_t data_entry_bytes             = input_size * TypeSize(Type::F32);
  uint32_t data_batch_bytes             = data_entry_bytes * TestingBatchSize();
  uint32_t data_batches_in_memory_bytes = data_batch_bytes * TestingBatchesInMemory();
  testing_data_batches_ = module_manager_.Memory().Allocate(data_batches_in_memory_bytes);

  // Allocate memory for labels
  uint32_t labels_entry_bytes             = output_size * TypeSize(Type::F32);
  uint32_t labels_batch_bytes             = labels_entry_bytes * TestingBatchSize();
  uint32_t labels_batches_in_memory_bytes = labels_batch_bytes * TestingBatchesInMemory();
  testing_labels_batches_ = module_manager_.Memory().Allocate(labels_batches_in_memory_bytes);

  // Create function to get the offset for testing data in memory
  module_manager_.MakeFunction("testing_data_offset", {{},{Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(testing_data_batches_->Begin()));
  });

  // Create function to get the offset for testing labels in memory
  module_manager_.MakeFunction("testing_labels_offset", {{},{Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(testing_labels_batches_->Begin()));
  });

  // Create testing function
  std::vector<Type> locals_type = {Type::F64, Type::F64, Type::F32, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("test_batches_in_memory", {{Type::I32},{}}, locals_type,
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    assert(params.size() == 1);
    auto batches_to_test_on = params[0];

    assert(locals.size() == 9);
    auto total_time = locals[0];
    auto batch_time = locals[1];
    auto cost = locals[2];
    auto hits = locals[3];
    auto counter = locals[4];
    auto test_addr = locals[5];
    auto label_addr = locals[6];
    auto vi32_1 = locals[7];
    auto vi32_2 = locals[8];

    // Set the testing pointer to the address of the first testing batch
    f.Insert(MakeLocalSet(test_addr, MakeI32Const(testing_data_batches_->Begin())));
    // Set the labels pointer to the address of the first labels batch
    f.Insert(MakeLocalSet(label_addr, MakeI32Const(testing_labels_batches_->Begin())));
    // Loop on batches in memory
    f.Insert(GenerateDoWhileLoop(f.Label(), counter, batches_to_test_on, 1, {}, [&](BlockBody* b1){

      // Start timer
      if(options_.bytecode_options.gen_testing_time) {
        b1->Insert(MakeLocalSet(batch_time, MakeCall(builtins_.system.TimeF64(), {})));
      }

      // Forward algorithm
      b1->Insert(MakeCall(forward_testing_func_, {
        MakeLocalGet(test_addr)
      }));

      // Update time
      if(options_.bytecode_options.gen_testing_time) {
        b1->Insert(GenerateCompoundAssignment(total_time, Opcode::F64Add,
                                              MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}),
                                                         MakeLocalGet(batch_time))));
      }

      // Count number of correct results
      if(options_.bytecode_options.gen_testing_accuracy) {
        b1->Insert(GenerateCompoundAssignment(hits, Opcode::F32Add, MakeCall(count_correct_predictions_testing_func_, {
            MakeLocalGet(label_addr)
        })));
      }

      // Compute testing error
      if(options_.bytecode_options.gen_testing_error) {
        assert(layers_.back()->Position() == Output);
        if(layers_.back()->Type() == FullyConnected) {
          auto cost_call = static_cast<DenseOutputLayer*>(layers_.back())->ComputeCost(Mode::Testing, label_addr);
          b1->Insert(GenerateCompoundAssignment(cost, Opcode::F32Add, cost_call));
        } else {
          assert(!"Compute cost not implemented");
        }
      }

      // Update confusion matrix
      if(options_.bytecode_options.gen_testing_confusion_matrix) {
        b1->Insert(MakeCall(confusion_matrix_testing_func_, {MakeLocalGet(label_addr)}));
      }

      // Move to the next data batch in memory
      b1->Insert(GenerateCompoundAssignment(test_addr, Opcode::I32Add, MakeI32Const(data_batch_bytes)));
      // Move to the next label batch in memory
      b1->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(labels_batch_bytes)));
    }));

     // Store time
     if(options_.bytecode_options.gen_testing_time) {
       f.Insert(MakeF64Store(MakeI32Const(testing_time_->Begin()), MakeLocalGet(total_time)));
     }

    // Store total cost error in memory
    if(options_.bytecode_options.gen_testing_error) {
      f.Insert(MakeF32Store(MakeI32Const(testing_error_->Begin()), MakeLocalGet(cost)));
    }

    // Store number of hits in memory
    if(options_.bytecode_options.gen_testing_accuracy) {
      f.Insert(MakeF32Store(MakeI32Const(testing_hits_->Begin()), MakeLocalGet(hits)));
    }
  });

  // Create function to access testing time
  if(options_.bytecode_options.gen_testing_time) {
    module_manager_.MakeFunction("testing_time", {{},{Type::F64}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF64Load(MakeI32Const(testing_time_->Begin())));
    });
  }

  // Create function to access testing batches hits
  if(options_.bytecode_options.gen_testing_accuracy) {
    module_manager_.MakeFunction("testing_batches_hits", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF32Load(MakeI32Const(testing_hits_->Begin())));
    });
  }

  // Create function to access testing batches error
  if(options_.bytecode_options.gen_testing_error) {
    module_manager_.MakeFunction("testing_batches_error", {{},{Type::F32}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeF32Load(MakeI32Const(testing_error_->Begin())));
   });
  }

  // Create function to access testing batch size
  module_manager_.MakeFunction("testing_batch_size", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(TestingBatchSize()));
  });

  // Create function to access testing batches in memory
  module_manager_.MakeFunction("testing_batches_in_memory", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(TestingBatchesInMemory()));
  });
}

void Model::MakePredictionFunctions() {

  uint32_t input_addr;
  assert(layers_.front()->Position() == Input);
  if(layers_.front()->Type() == FullyConnected) {
    input_addr = static_cast<DenseInputLayer*>(layers_.front())->InputArray(Model::Mode::Prediction)->Begin();
  } else {
    assert(!"Not implemented");
  }

  // Create prediction function
  module_manager_.MakeFunction("predict_batch", {}, {Type::F64},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){

    assert(locals.size() == 1);
    auto time = locals[0];

    // Start timer
    if(options_.bytecode_options.gen_prediction_time) {
      f.Insert(MakeLocalSet(time, MakeCall(builtins_.system.TimeF64(), {})));
    }

    // Apply forward algorithm
    f.Insert(MakeCall(forward_prediction_func_, {
      MakeI32Const(input_addr)
    }));

    // Store time
    if(options_.bytecode_options.gen_prediction_time) {
      f.Insert(MakeF64Store(MakeI32Const(prediction_time_->Begin()),
                            MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
    }
  });

  // Create function to access prediction time
  if(options_.bytecode_options.gen_prediction_time) {
    module_manager_.MakeFunction("prediction_time", {{},{Type::F64}}, {},
                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
      f.Insert(MakeF64Load(MakeI32Const(prediction_time_->Begin())));
    });
  }

  // Create a function to get prediction batch size
  module_manager_.MakeFunction("prediction_batch_size", {{}, {Type::I32}}, {},
                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(PredictionBatchSize()));
  });
}

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
