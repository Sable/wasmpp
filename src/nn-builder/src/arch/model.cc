#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/arch/layers/dense.h>
#include <src/nn-builder/src/snippet/matrix.h>
#include <src/nn-builder/src/arch/initializers.h>
#include <src/wasmpp/wasm-instructions-gen.h>

namespace nn {
namespace arch {

using namespace wasmpp;
using namespace wabt;
using namespace layer;

wabt::ExprList* Model::SetLearningRate(wabt::ExprList *val) {
  assert(learning_rate != nullptr);
  return MakeF32Store(MakeI32Const(learning_rate->Begin()), val);
}

wabt::ExprList* Model::GetLearningRate() {
  assert(learning_rate != nullptr);
  return MakeF32Load(MakeI32Const(learning_rate->Begin()));
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
}

Model::Model(ModelOptions options) : options_(options), builtins_(options_.activation_options) {
  AllocateMembers();
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
    snippets_.matrix = new snippet::MatrixSnippetSimd(&module_manager_.Label());
    snippets_.analysis = new snippet::AnalysisSnippet(&module_manager_.Label());
  } else {
    snippets_.matrix = new snippet::MatrixSnippet(&module_manager_.Label());
    snippets_.analysis = new snippet::AnalysisSnippetSimd(&module_manager_.Label());
  }
}

#define ALLOCATE_MATRIX(array, rows, cols) \
    array = new ds::NDArray(module_manager_.Memory().Allocate((rows) * (cols) * TypeSize(Type::F32)), \
                            {rows, cols}, TypeSize(Type::F32));

void Model::AllocateMembers() {
  learning_rate = module_manager_.Memory().Allocate(TypeSize(Type::F32));
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
    training_.push_back(training_array);
  }

  for(uint32_t b=0; b < training_labels_vals_.size(); b += TrainingBatchSize()) {
    // Training labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MATRIX(labels_array, (uint32_t) training_labels_vals_[0].size(), TrainingBatchSize());
    training_labels_.push_back(labels_array);
  }
}

void Model::AllocateTest() {
  // Do not merge loops so that all
  // test data are consecutive in memory

  for(uint32_t b=0; b < testing_vals_.size(); b += TestingBatchSize()) {
    // Testing data
    ds::NDArray* testing_array = nullptr;
    ALLOCATE_MATRIX(testing_array, (uint32_t) testing_vals_[0].size(), TestingBatchSize());
    testing_.push_back(testing_array);
  }

  for(uint32_t b=0; b < testing_labels_vals_.size(); b += TestingBatchSize()) {
    // Testing labels
    ds::NDArray* labels_array = nullptr;
    ALLOCATE_MATRIX(labels_array, (uint32_t) testing_labels_vals_[0].size(), TestingBatchSize());
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
    auto training_begin = training_vals_.begin() + (i * TrainingBatchSize());
    std::vector<std::vector<float>> sub_input(training_begin, training_begin + TrainingBatchSize());
    module_manager_.MakeData(memory, training_[i]->Memory()->Begin(), MakeTransposeData(training_[i], sub_input));
  }

  for(uint32_t i=0; i < training_labels_.size(); ++i) {
    // Training labels
    auto labels_begin = training_labels_vals_.begin() + (i * TrainingBatchSize());
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + TrainingBatchSize());
    module_manager_.MakeData(memory, training_labels_[i]->Memory()->Begin(),
                             MakeTransposeData(training_labels_[i], sub_labels));
  }
}

void Model::MakeTestingData(wabt::Var memory) {
  for(uint32_t i=0; i < testing_.size(); ++i) {
    // Testing data
    auto testing_begin = testing_vals_.begin() + (i * TestingBatchSize());
    std::vector<std::vector<float>> sub_input(testing_begin, testing_begin + TestingBatchSize());
    module_manager_.MakeData(memory, testing_[i]->Memory()->Begin(), MakeTransposeData(testing_[i], sub_input));
  }

  for(uint32_t i=0; i < testing_labels_.size(); ++i) {
    // Testing labels
    auto labels_begin = testing_labels_vals_.begin() + (i * TestingBatchSize());
    std::vector<std::vector<float>> sub_labels(labels_begin, labels_begin + TestingBatchSize());
    module_manager_.MakeData(memory, testing_labels_[i]->Memory()->Begin(),
                             MakeTransposeData(testing_labels_[i], sub_labels));
  }
}

void Model::MakeLayersData(wabt::Var memory) {
  for(int l=1; l < layers_.size(); ++l) {
    layers_[l]->MakeData(memory);
  }
}

Var Model::ForwardAlgorithmFunction(uint8_t mode_index) {
  std::vector<Type> locals_types = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32};
  return module_manager_.MakeFunction(nullptr, {{Type::I32},{}}, locals_types,
                                      [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    assert(locals.size() == 6);
    auto vi32_1 = locals[0];
    auto vi32_2 = locals[1];
    auto vi32_3 = locals[2];
    auto vi32_4 = locals[3];
    auto vi32_5 = locals[4];
    auto vf32_1 = locals[5];

    assert(params.size() == 1);
    auto input_begin = params[0];

    for(int l=0; l < layers_.size(); ++l) {
      if(layers_[l]->Type() == FullyConnected) {
        f.Insert(layers_[l]->Forward(mode_index, input_begin, {vi32_1,vi32_2,vi32_3,vi32_4,vi32_5,vf32_1}));
      } else {
        assert(!"Not implemented!");
      }
    }
  });
}

wabt::Var Model::BackwardAlgorithmFunction() {
  std::vector<Type> locals_type = {Type::I32, Type::I32, Type::I32, Type::I32, Type::I32, Type::F32, Type::V128};
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
  AllocateLayers();
  forward_training_func_ = ForwardAlgorithmFunction(Mode::Training);
  forward_testing_func_ = ForwardAlgorithmFunction(Mode::Testing);
  forward_prediction_func_ = ForwardAlgorithmFunction(Mode::Prediction);
  backward_func_ = BackwardAlgorithmFunction();
  confusion_matrix_training_func_ = ConfusionMatrixFunction(Mode::Training);
  confusion_matrix_testing_func_ = ConfusionMatrixFunction(Mode::Testing);
  count_correct_predictions_training_func_ = CountCorrectPredictionsFunction(Mode::Training);
  count_correct_predictions_testing_func_ = CountCorrectPredictionsFunction(Mode::Testing);
}

void Model::CompileTraining(uint32_t epochs, float learning_rate, const std::vector<std::vector<float>> &input,
                            const std::vector<std::vector<float>> &labels) {
  ERROR_UNLESS(epochs >= 1, "epoch must be at least 1");
  ERROR_UNLESS(!input.empty(), "training input cannot be empty");
  ERROR_UNLESS(input.size() == labels.size(), "training and labels size should match");
  ERROR_UNLESS(input.size() % training_batch_size_ == 0, "Input must be a multiple of the training batch size");

  training_vals_ = input;
  training_labels_vals_ = labels;
  AllocateTraining();

  std::vector<Type> locals_type = {Type::F64, Type::F32, Type::F32, Type::I32, Type::I32, Type::I32, Type::I32, Type::I32};
  module_manager_.MakeFunction("train", {}, locals_type, [&](FuncBody f, std::vector<Var> params,
                                                             std::vector<Var> locals) {

    // Set learning rate
    f.Insert(SetLearningRate(MakeF32Const(learning_rate)));

    assert(locals.size() == 8);
    auto time = locals[0];
    auto cost_mean = locals[1];
    auto accuracy = locals[2];
    auto epoch = locals[3];
    auto train_addr = locals[4];
    auto label_addr = locals[5];
    auto vi32_1 = locals[6];
    auto vi32_2 = locals[7];

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
      b1->Insert(MakeLocalSet(accuracy, MakeF32Const(0)));
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
          b2->Insert(GenerateCompoundAssignment(accuracy, Opcode::F32Add, MakeCall(count_correct_predictions_training_func_, {
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

        b2->Insert(GenerateCompoundAssignment(label_addr, Opcode::I32Add, MakeI32Const(label_size)));
      }));

      if(options_.log_training_error) {
        // Log training error
        b1->Insert(MakeCall(builtins_.message.LogTrainingError(), {
          MakeLocalGet(epoch),
          MakeBinary(Opcode::F32Div, MakeLocalGet(cost_mean), MakeF32Const(training_.size()))
        }));
      }

      if(options_.log_training_accuracy) {
        // Log training accuracy
        b1->Insert(MakeCall(builtins_.message.LogTrainingAccuracy(), {
            MakeLocalGet(epoch),
            MakeBinary(Opcode::F32Div, MakeLocalGet(accuracy), MakeF32Const(training_vals_.size()))
        }));
      }
    }));

    if(options_.log_training_time) {
      // Log training time
      f.Insert(MakeLocalSet(time, MakeBinary(Opcode::F64Sub, MakeCall(builtins_.system.TimeF64(), {}), MakeLocalGet(time))));
      f.Insert(MakeCall(builtins_.message.LogTrainingTime(), {MakeLocalGet(time)}));
    }

    if(options_.log_training_confusion_matrix) {
      assert(layers_.back()->Position() == Output);
      if(layers_.back()->Type() == FullyConnected) {
        auto out_layer = static_cast<DenseOutputLayer*>(layers_.back());
        // Log confusion matrix
        f.Insert(MakeCall(builtins_.system.PrintTableF32(), {
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Training)->Begin()),
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Training)->Shape()[0]),
            MakeI32Const(out_layer->ConfusionMatrix(Mode::Training)->Shape()[1])
        }));
      } else {
        assert(!"Not implemented");
      }
    }
  });
}

void Model::CompileTesting(const std::vector<std::vector<float>> &input,
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
        MakeBinary(Opcode::F32Div, MakeLocalGet(cost_mean), MakeF32Const(testing_.size()))
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

bool Model::Validate() {
  return module_manager_.Validate();
}

} // namespace arch
} // namespace nn
