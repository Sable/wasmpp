#include <src/nn-builder/src/arch/layers/dense.h>
#include <src/nn-builder/src/arch/model.h>
#include <sstream>

namespace nn {
namespace arch {
namespace layer {

using namespace wasmpp;
using namespace wabt;

FullyConnectedLayer* FullyConnectedLayer::KeepProb(float keep_prob) {
  ERROR_UNLESS(keep_prob >= KEEP_PROB_MIN && keep_prob <= KEEP_PROB_MAX, "Keep probability must be between %f and %f",
               KEEP_PROB_MIN, KEEP_PROB_MAX);
  keep_prob_ = keep_prob;
  return this;
}

FullyConnectedLayer* FullyConnectedLayer::WeightType(nn::arch::WeightDistributionType type) {
  weight_type_ = type;
  return this;
}

#define START_TIME()                                                                                                  \
  if(mode_index == Model::Mode::Training && NetworkModel()->Options().bytecode_options.gen_forward_profiling) {       \
    Merge(e, NetworkModel()->DenseForwardTime().SetTime(MakeCall(NetworkModel()->Builtins().system.TimeF64(), {})));  \
  }

#define END_TIME(name)                                                                                          \
  if(mode_index == Model::Mode::Training && NetworkModel()->Options().bytecode_options.gen_forward_profiling) { \
    Merge(e, NetworkModel()->DenseForwardTime()                                                                 \
    .SetTime(MakeBinary(Opcode::F64Sub, MakeCall(NetworkModel()->Builtins().system.TimeF64(), {}),              \
                       NetworkModel()->DenseForwardTime().GetTime())));                                         \
    Merge(e, NetworkModel()->DenseForwardTime()                                                                 \
    .Set##name(MakeBinary(Opcode::F64Add, NetworkModel()->DenseForwardTime().GetTime(),                         \
                       NetworkModel()->DenseForwardTime().Get##name())));                                       \
  }


wabt::ExprList* FullyConnectedLayer::Forward(uint8_t mode_index, Var input_begin, std::vector<Var> locals) {
  assert(mode_index >= Model::Mode::FIRST_MODE && mode_index <= Model::Mode::LAST_MODE);
  assert(locals.size() == 7);
  auto vi32_1 = locals[0];
  auto vi32_2 = locals[1];
  auto vi32_3 = locals[2];
  auto vi32_4 = locals[3];
  auto vi32_5 = locals[4];
  auto vf32_1 = locals[5];
  auto v128_1 = locals[6];

  ExprList* e = new ExprList();
  if(Position() != Input) {
    assert(LayerIndex() > 0);
    auto prev_layer = NetworkModel()->Layers()[LayerIndex() - 1];
    if(prev_layer->Type() == FullyConnected) {
      auto prev_fc_layer = static_cast<FullyConnectedLayer*>(prev_layer);
      // A) Z[l] = W[l] . A[l-1] + b[l]
      //    1) Z[l] = W[l] . A[l-1]
      //    2) Z[l] = Z[l] + b[l]
      START_TIME()
#ifdef WABT_EXPERIMENTAL
      Merge(e, MakeNativeCall(NetworkModel()->Natives().dot_product, {
        MakeI32Const(W_->Begin()),
        (LayerIndex() == 1) ? MakeLocalGet(input_begin) : MakeI32Const(prev_fc_layer->A_[mode_index]->Begin()),
        MakeI32Const(Z_[mode_index]->Begin()),
        MakeI32Const(W_->Shape()[0]),
        MakeI32Const(W_->Shape()[1]),
        MakeI32Const(prev_fc_layer->A_[mode_index]->Shape()[1])
      }));
#else
      Merge(e, NetworkModel()->Snippets().matrix->MatrixDot(W_, (LayerIndex() == 1) ?
                                                                snippet::RelocMat(prev_fc_layer->A_[mode_index], input_begin) :
                                                                snippet::RelocMat(prev_fc_layer->A_[mode_index]), Z_[mode_index],
                                                            {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));
#endif
      END_TIME(A_1)
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixVectorAddition(Z_[mode_index], b_, Z_[mode_index],
                                                                             {vi32_1, vi32_2, vi32_3, vi32_4}));
      END_TIME(A_2)

      // B) A[l] = g(Z[l])
      START_TIME()
      // Special case for softmax
      if(activation_func_ == NetworkModel()->Builtins().activation.Softmax()) {
        Merge(e, MakeCall(activation_func_.function, {
          MakeI32Const(Z_[mode_index]->Begin()),
          MakeI32Const(A_[mode_index]->Begin()),
          MakeI32Const(Z_[mode_index]->Shape()[0]),
          MakeI32Const(Z_[mode_index]->Shape()[1])
        }));
      } else {
        Merge(e, NetworkModel()->Snippets().matrix->MatrixActivation(snippet::RelocMat(Z_[mode_index]), activation_func_, A_[mode_index],
                                                                     {vi32_1, vi32_2}, false));
      }
      END_TIME(B)
    } else {
      assert(!"Not implemented!");
    }
  } else {
    // Place a nop because an expression list
    // cannot be empty
    Merge(e, MakeNop());
  }

  // Check for dropout regularization
  // Only apply for training forward algorithm
  // Skip dropout for output
  if(Position() != Output && keep_prob_ != KEEP_PROB_MAX && mode_index == Model::Mode::Training) {
    assert(LayerIndex() < NetworkModel()->Layers().size() - 1);

    // Generate a mask matrix
    Merge(e, MakeCall(NetworkModel()->Builtins().math.MaskMatrix(), {
        MakeI32Const(inverted_dropout_->Memory()->Begin()),
        MakeI32Const(inverted_dropout_->Memory()->End()),
        MakeF32Const(keep_prob_)
    }));

    // A[l] = (1/keep_prob) * (A[l] * inverted_dropout[l])
    // 1) A[l] = A[l] * inverted_dropout[l]
    // 2) A[l] = A[l] * (1/keep_prob)
    Merge(e, NetworkModel()->Snippets().matrix->MatrixMultiplication(A_[mode_index], inverted_dropout_, A_[mode_index],
                                                                             {vi32_1, vi32_2}));
    auto scalar = MakeBinary(Opcode::F32Div, MakeF32Const(1.0f), MakeF32Const(keep_prob_));
    Merge(e, NetworkModel()->Snippets().matrix->MatrixScalar(A_[mode_index], scalar, A_[mode_index], {vi32_1, vi32_2, vf32_1}));
  }
  return e;
}

wabt::ExprList* DenseOutputLayer::ComputeCost(uint8_t mode_index, wabt::Var target_begin) {
  assert(mode_index == Model::Mode::Training || mode_index == Model::Mode::Testing);
  return MakeCall(NetworkModel()->Loss().J, {
      MakeLocalGet(target_begin),
      MakeI32Const(A_[mode_index]->Begin()),
      MakeI32Const(A_[mode_index]->Shape()[0]),
      MakeI32Const(A_[mode_index]->Shape()[1])
  });

  // Compute l1_loss
  // cost + (l1_decay/2m) SUM(ABS(W[l]))
  // 1) local = SUM(ABS(W[l]))
  // 2) local = (l1_decay/2m) local
  // 3) res   = res + Local


  // Compute l2 loss
  // cost + (l2_decay/m) SUM(W[l] * W[l])
  // 1) local = SUM(W[l] * W[l])
  // 2) local = (l2_decay/m) local
  // 3) res   = res + local
}

#undef START_TIME
#undef END_TIME

#define START_TIME() \
  if(NetworkModel()->Options().bytecode_options.gen_backward_profiling) {                                             \
    Merge(e, NetworkModel()->DenseBackwardTime().SetTime(MakeCall(NetworkModel()->Builtins().system.TimeF64(), {}))); \
  }

#define END_TIME(name)                                                                              \
  if(NetworkModel()->Options().bytecode_options.gen_backward_profiling) {                           \
    Merge(e, NetworkModel()->DenseBackwardTime()                                                    \
    .SetTime(MakeBinary(Opcode::F64Sub, MakeCall(NetworkModel()->Builtins().system.TimeF64(), {}),  \
                       NetworkModel()->DenseBackwardTime().GetTime())));                            \
    Merge(e, NetworkModel()->DenseBackwardTime()                                                    \
    .Set##name(MakeBinary(Opcode::F64Add, NetworkModel()->DenseBackwardTime().GetTime(),            \
                       NetworkModel()->DenseBackwardTime().Get##name())));                          \
  }

wabt::ExprList* FullyConnectedLayer::Backward(wabt::Var input_begin, wabt::Var target_begin,
                                              std::vector<wabt::Var> locals) {
  assert(locals.size() == 7);
  auto vi32_1 = locals[0];
  auto vi32_2 = locals[1];
  auto vi32_3 = locals[2];
  auto vi32_4 = locals[3];
  auto vi32_5 = locals[4];
  auto vf32_1 = locals[5];
  auto v128_1 = locals[6];

  ExprList* e = new ExprList();
  if(Position() != Input) {
    assert(LayerIndex() > 0);

    if(Position() == Output && NetworkModel()->Loss() != NetworkModel()->Builtins().loss.SoftmaxCrossEntropy()) {
      // A) dA[L] = dJ(T, A[L])
      // Note: For softmax, we should skip this step because
      // we can directly store the result in dZ[L]
      START_TIME()
      Merge(e, MakeCall(NetworkModel()->Loss().dJ, {
          MakeLocalGet(target_begin),
          MakeI32Const(A_[Model::Mode::Training]->Begin()),
          MakeI32Const(dA_->Begin()),
          MakeI32Const(A_[Model::Mode::Training]->Shape()[0]),
          MakeI32Const(A_[Model::Mode::Training]->Shape()[1])
      }));
      END_TIME(A)
    }

    auto prev_layer = NetworkModel()->Layers()[LayerIndex()-1];
    if(prev_layer->Type() == FullyConnected) {
      auto prev_fc_layer = static_cast<FullyConnectedLayer*>(prev_layer);

      // Special case for softmax
      if(Position() == Output && NetworkModel()->Loss() == NetworkModel()->Builtins().loss.SoftmaxCrossEntropy()) {
        // B_Softmax) dZ[L] = dJ(T, A[L])
        START_TIME()
        Merge(e, MakeCall(NetworkModel()->Loss().dJ, {
            MakeLocalGet(target_begin),
            MakeI32Const(A_[Model::Mode::Training]->Begin()),
            MakeI32Const(dZ_->Begin()),
            MakeI32Const(A_[Model::Mode::Training]->Shape()[0]),
            MakeI32Const(A_[Model::Mode::Training]->Shape()[1])
        }));
        END_TIME(B_Softmax)
      } else {
        // B) dZ[l] = dA[l] * g'(Z[l])
        //    1) dZ[l] = g'(Z[l])
        //    2) dZ[l] = dA[l] * dZ[l]
        START_TIME()
        Merge(e, NetworkModel()->Snippets().matrix->MatrixActivation(snippet::RelocMat(Z_[Model::Mode::Training]),
                                                                     activation_func_, dZ_, {vi32_1, vi32_2}, true));
        END_TIME(B_1)
        START_TIME()
        Merge(e, NetworkModel()->Snippets().matrix->MatrixMultiplication(dA_, dZ_, dZ_, {vi32_1, vi32_2}));
        END_TIME(B_2)
      }

      // C) dW[l] = (1/m) dZ[l] . A[l-1]^T + (l2_decay/m) W[l] + (l1_decay/m) sign(W[l])
      //          = (1/m) (dZ[l] . A[l-1]^T + l2_decay W[l]) + l1_decay sign(W[l]))
      //    1) dW[l] = dZ[l] . A[l-1]^T
      //    2) dW[l] = dW[l] + l2_decay W[l]
      //    3) dW[l] = dW[l] + l1_decay sign(W[l])
      //    4) dW[l] = (1/m) dW[l]
      START_TIME()
#ifdef WABT_EXPERIMENTAL
      Merge(e, MakeNativeCall(NetworkModel()->Natives().dot_product_rt, {
          MakeI32Const(dZ_->Begin()),
          (LayerIndex() == 1) ? MakeLocalGet(input_begin) : MakeI32Const(prev_fc_layer->A_[Model::Mode::Training]->Begin()),
          MakeI32Const(dW_->Begin()),
          MakeI32Const(dZ_->Shape()[0]),
          MakeI32Const(dZ_->Shape()[1]),
          MakeI32Const(prev_fc_layer->A_[Model::Mode::Training]->Shape()[0])
      }));
#else
      Merge(e, NetworkModel()->Snippets().matrix->MatrixDotRT(dZ_, (LayerIndex() == 1) ?
                                                                   snippet::RelocMat(prev_fc_layer->A_[Model::Mode::Training],
                                                                                     input_begin) :
                                                                   snippet::RelocMat(prev_fc_layer->A_[Model::Mode::Training]), dW_,
                                                              {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));
#endif
      END_TIME(C_1)
      START_TIME()
      if(NetworkModel()->TrainingBatchSize() > 1) {
        Merge(e, NetworkModel()->Snippets().matrix->MatrixScalar(dW_, MakeF32Const(1.0f / NetworkModel()->TrainingBatchSize()),
                                                                 dW_, {vi32_1, vi32_2, vf32_1}));
      }
      END_TIME(C_2)

      // D) db[l] = (1/m) dZ[l]
      //    1) db[l] = SUM(dZ[l], row wise)
      //    2) db[l] = (1/m) db[l]
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixHorizontalSum(dZ_, db_,
                                                                      {vi32_1, vi32_2, vi32_3, vf32_1, v128_1}));
      END_TIME(D_1)
      START_TIME()
      if(NetworkModel()->TrainingBatchSize() > 1) {
        Merge(e, NetworkModel()->Snippets().matrix->MatrixScalar(db_, MakeF32Const(1.0f/NetworkModel()->TrainingBatchSize()),
                                                                 db_, {vi32_1, vi32_2, vf32_1}));
      }
      END_TIME(D_2)

      if(LayerIndex() > 1) {
        // E) dA[l-1] = W[l]^T . dZ[l]
        START_TIME()
#ifdef WABT_EXPERIMENTAL
        Merge(e, MakeNativeCall(NetworkModel()->Natives().dot_product_lt, {
            MakeI32Const(W_->Begin()),
            MakeI32Const(dZ_->Begin()),
            MakeI32Const(prev_fc_layer->dA_->Begin()),
            MakeI32Const(W_->Shape()[1]),
            MakeI32Const(W_->Shape()[0]),
            MakeI32Const(dZ_->Shape()[1])
        }));
#else
        Merge(e, NetworkModel()->Snippets().matrix->MatrixDotLT(W_, dZ_, prev_fc_layer->dA_,
                                                                {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));
#endif
        END_TIME(E)
      }

      // F) W[l] = W[l] - alpha * dW[l]
      //    1) dW[l] = alpha * dW[l] // TODO Shorten this to sub_mul operation
      //    2) W[l] = W[l] - dW[l]
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixScalar(dW_, NetworkModel()->GetLearningRate(), dW_,
                                                               {vi32_1, vi32_2, vf32_1}));
      END_TIME(F_1)
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixSubtraction(W_, dW_, W_, {vi32_1, vi32_2}));
      END_TIME(F_2)

      // G) b[l] = b[l] - alpha * db[l]
      //    1) db[l] = alpha * db[l] // TODO Shorten this to sub_mu operation
      //    2) b[l] = b[l] - db[l]
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixScalar(db_, NetworkModel()->GetLearningRate(), db_,
                                                               {vi32_1, vi32_2, vf32_1}));
      END_TIME(G_1)
      START_TIME()
      Merge(e, NetworkModel()->Snippets().matrix->MatrixSubtraction(b_, db_, b_, {vi32_1, vi32_2}));
      END_TIME(G_2)
    } else {
      assert(!"Not implemented!");
    }
  } else {
    // Place a nop because an expression list
    // cannot be empty
    Merge(e, MakeNop());
  }
  return e;
}

#define ALLOCATE_MEMORY(array, rows, cols)                                                            \
  array = new ds::NDArray(                                                                            \
      NetworkModel()->ModuleManager().Memory().Allocate((rows) * (cols) * TypeSize(Type::F32)), \
      {rows, cols}, TypeSize(Type::F32));

void FullyConnectedLayer::AllocateMemory() {
  ALLOCATE_MEMORY(A_[Model::Model::Training], Nodes(), NetworkModel()->TrainingBatchSize());
  ALLOCATE_MEMORY(A_[Model::Model::Testing], Nodes(), NetworkModel()->TestingBatchSize());
  ALLOCATE_MEMORY(A_[Model::Model::Prediction], Nodes(), NetworkModel()->PredictionBatchSize());
  if(Position()!= Input) {
    assert(LayerIndex() > 0);
    ALLOCATE_MEMORY(Z_[Model::Mode::Training], Nodes(), NetworkModel()->TrainingBatchSize());
    ALLOCATE_MEMORY(Z_[Model::Mode::Testing], Nodes(), NetworkModel()->TestingBatchSize());
    ALLOCATE_MEMORY(Z_[Model::Mode::Prediction], Nodes(), NetworkModel()->PredictionBatchSize());
    ALLOCATE_MEMORY(dZ_, Nodes(), NetworkModel()->TrainingBatchSize());
    ALLOCATE_MEMORY(dA_, Nodes(), NetworkModel()->TrainingBatchSize());
    ALLOCATE_MEMORY(b_, Nodes(), 1);
    ALLOCATE_MEMORY(db_, Nodes(), 1);

    auto prev_layer = NetworkModel()->Layers()[LayerIndex() - 1];
    if(prev_layer->Type() == FullyConnected) {
      uint32_t prev_nodes = static_cast<FullyConnectedLayer*>(prev_layer)->Nodes();
      ALLOCATE_MEMORY(W_, Nodes(), prev_nodes);
      ALLOCATE_MEMORY(dW_, Nodes(), prev_nodes);
    } else {
      assert(!"Not implemented!");
    }
  }
  if(Position() != Output) {
    ALLOCATE_MEMORY(inverted_dropout_, Nodes(), NetworkModel()->TrainingBatchSize());
  }
}

void FullyConnectedLayer::MakeData(wabt::Var memory) {
  // Input layer has no data to initialize
  if(Position() == Input) {
    return;
  }
  assert(LayerIndex() > 0);
  if(Position() == Output) {
    assert(LayerIndex() == NetworkModel()->Layers().size() - 1);
    ERROR_UNLESS(weight_type_ >= FIRST_HIDDEN_OR_OUTPUT && weight_type_ <= LAST_HIDDEN_OR_OUTPUT,
                 "Wrong weight distribution for the output layer");
  }
  auto weight_size = W_->Shape()[0] * W_->Shape()[1];
  auto bias_size = b_->Shape()[0] * b_->Shape()[1];
  std::vector<DataEntry> weight_entries;
  std::vector<DataEntry> bias_entries;
  switch (weight_type_) {
    case XavierUniform:
    case XavierNormal: {
      uint32_t incoming;
      uint32_t outgoing;
      auto prev_layer = NetworkModel()->Layers()[LayerIndex()-1];
      auto next_layer = NetworkModel()->Layers()[LayerIndex()+1];
      if(prev_layer->Type() == FullyConnected) {
          incoming = static_cast<FullyConnectedLayer*>(prev_layer)->Nodes();
      } else {
        assert(!"Not implemented!");
      }
      if(next_layer->Type() == FullyConnected) {
        outgoing = static_cast<FullyConnectedLayer*>(next_layer)->Nodes();
      } else {
        assert(!"Not implemented!");
      }
      weight_entries = XavierDistribution(weight_size, incoming, outgoing, weight_type_ == XavierUniform,
          NetworkModel()->Options().weights_options.seed);
      bias_entries = XavierDistribution(bias_size, incoming, outgoing, weight_type_ == XavierUniform,
          NetworkModel()->Options().weights_options.seed);
      break;
    }
    case LeCunUniform:
    case LeCunNormal: {
      uint32_t incoming;
      auto prev_layer = NetworkModel()->Layers()[LayerIndex()-1];
      if(prev_layer->Type() == FullyConnected) {
        incoming = static_cast<FullyConnectedLayer*>(prev_layer)->Nodes();
      } else {
        assert(!"Not implemented!");
      }
      weight_entries = LeCunDistribution(weight_size, incoming, weight_type_ == LeCunUniform,
          NetworkModel()->Options().weights_options.seed);
      bias_entries = LeCunDistribution(bias_size, incoming, weight_type_ == LeCunUniform,
          NetworkModel()->Options().weights_options.seed);
      break;
    }
    case Gaussian:
      weight_entries = GaussianDistribution(weight_size, NetworkModel()->Options().weights_options.gaussian_mean,
                                            NetworkModel()->Options().weights_options.gaussian_std_dev,
                                            NetworkModel()->Options().weights_options.seed);
      bias_entries = GaussianDistribution(bias_size, NetworkModel()->Options().weights_options.gaussian_mean,
                                          NetworkModel()->Options().weights_options.gaussian_std_dev,
                                          NetworkModel()->Options().weights_options.seed);
      break;
    case Uniform:
      weight_entries = UniformDistribution(weight_size, NetworkModel()->Options().weights_options.uniform_low,
                                           NetworkModel()->Options().weights_options.uniform_high,
                                           NetworkModel()->Options().weights_options.seed);
      bias_entries = UniformDistribution(bias_size, NetworkModel()->Options().weights_options.uniform_low,
                                         NetworkModel()->Options().weights_options.uniform_high,
                                         NetworkModel()->Options().weights_options.seed);
      break;
    case Constant:
      weight_entries = ConstantDistribution(weight_size, NetworkModel()->Options().weights_options.constant_value);
      bias_entries = ConstantDistribution(bias_size, NetworkModel()->Options().weights_options.constant_value);
      break;
    default:
      assert(!"Weight distribution not implemented");
  }
  NetworkModel()->ModuleManager().MakeData(memory, W_->Memory()->Begin(), weight_entries);
  NetworkModel()->ModuleManager().MakeData(memory, b_->Memory()->Begin(), bias_entries);
}

void FullyConnectedLayer::MakeFunctions() {

  // Create a function to get the number of nodes in this layer
  std::stringstream ss;
  ss << "layer_" << LayerIndex() << "_size";
  NetworkModel()->ModuleManager().MakeFunction(ss.str().c_str(), {{},{Type::I32}},{},
                                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(Nodes()));
  });

  // Create functions to get the weights offset and size in memory
  if(Position() != Input) {
    // Weights offset function
    std::stringstream w_offset_func_name;
    w_offset_func_name << "layer_" << LayerIndex() << "_weight_offset";
    NetworkModel()->ModuleManager().MakeFunction(w_offset_func_name.str().c_str(), {{},{Type::I32}}, {},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(W_->Begin()));
    });

    // Weights size function
    std::stringstream w_size_func_name;
    w_size_func_name << "layer_" << LayerIndex() << "_weight_byte_size";
    NetworkModel()->ModuleManager().MakeFunction(w_size_func_name.str().c_str(), {{},{Type::I32}}, {},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(W_->Memory()->Bytes()));
    });

    // Bias offset function
    std::stringstream b_offset_func_name;
    b_offset_func_name << "layer_" << LayerIndex() << "_bias_offset";
    NetworkModel()->ModuleManager().MakeFunction(b_offset_func_name.str().c_str(), {{},{Type::I32}}, {},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(b_->Begin()));
    });

    // Bias size function
    std::stringstream b_size_func_name;
    b_size_func_name << "layer_" << LayerIndex() << "_bias_byte_size";
    NetworkModel()->ModuleManager().MakeFunction(b_size_func_name.str().c_str(), {{},{Type::I32}}, {},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(b_->Memory()->Bytes()));
    });
  }
}

FullyConnectedLayer * DenseOutputLayer::KeepProb(float keep_prob) {
  ERROR_EXIT("Dense output layer cannot have a keep probability value "
             "because dropout regularization does not apply to it");
  return this;
}

wabt::ExprList* DenseOutputLayer::Forward(uint8_t mode_index, wabt::Var input_begin, std::vector<wabt::Var> locals) {
  // Not need to assert the mode_index because it is done
  // when calling the parent function

  assert(locals.size() == 8);
  auto vi32_1 = locals[0];
  auto vi32_2 = locals[1];
  auto vi32_3 = locals[2];
  auto vi32_4 = locals[3];
  auto vi32_5 = locals[4];
  auto vf32_1 = locals[5];
  auto vf32_2 = locals[6];
  auto v128_1 = locals[7];

  ExprList* e = new ExprList();
  Merge(e, FullyConnectedLayer::Forward(mode_index, input_begin, {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vf32_1, v128_1}));

  // Apply hardmax
  if(ShouldHardmax(mode_index)) {
    Merge(e, NetworkModel()->Snippets().matrix->MatrixColumnHardmax(Predictions(mode_index),
                                                                    hardmax_[mode_index],
                                                                    {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5}));
  }
  return e;
}

void DenseHiddenLayer::Validate() {
  ERROR_UNLESS(activation_func_ != NetworkModel()->Builtins().activation.Softmax(),
               "Hidden layer cannot have softmax activation function");
}

void DenseOutputLayer::Validate() {
  ERROR_UNLESS(activation_func_ == NetworkModel()->Builtins().activation.Softmax()
               || activation_func_ == NetworkModel()->Builtins().activation.Sigmoid() ,
               "Output layer must sigmoid or softmax as activation function");
  if(activation_func_ == NetworkModel()->Builtins().activation.Softmax()) {
    ERROR_UNLESS(NetworkModel()->Loss() == NetworkModel()->Builtins().loss.SoftmaxCrossEntropy(),
                 "Loss function not compatible with softmax activation function");
  }
  if(activation_func_ == NetworkModel()->Builtins().activation.Sigmoid()) {
    ERROR_UNLESS(NetworkModel()->Loss() != NetworkModel()->Builtins().loss.SoftmaxCrossEntropy(),
                 "Loss function not compatible with sigmoid activation function");
  }
}

void DenseOutputLayer::MakeFunctions() {
  // Create functions defined in parent
  FullyConnectedLayer::MakeFunctions();

  // Create training confusion matrix functions
  if(NetworkModel()->Options().bytecode_options.gen_training_confusion_matrix) {
    NetworkModel()->ModuleManager().MakeFunction("training_confusion_matrix_offset", {{},{Type::I32}},{},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(confusion_matrix_[Model::Mode::Training]->Begin()));
    });
  }

  // Create testing confusion matrix functions
  if(NetworkModel()->Options().bytecode_options.gen_testing_confusion_matrix) {
    NetworkModel()->ModuleManager().MakeFunction("testing_confusion_matrix_offset", {{},{Type::I32}},{},
                                                 [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
      f.Insert(MakeI32Const(confusion_matrix_[Model::Mode::Testing]->Begin()));
    });
  }

  // Create function to get the prediction result offset
  NetworkModel()->ModuleManager().MakeFunction("prediction_result_offset", {{},{Type::I32}},{},
                                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    f.Insert(MakeI32Const(Predictions(Model::Mode::Prediction)->Begin()));
  });
}

wabt::ExprList* DenseOutputLayer::UpdateConfusionMatrix(uint8_t mode_index, wabt::Var target_begin, std::vector<wabt::Var> locals) {
  assert(mode_index == Model::Mode::Training || mode_index == Model::Mode::Testing);
  assert(hardmax_[mode_index] != nullptr);
  assert(locals.size() == 6);
  auto vi32_1 = locals[0];
  auto vi32_2 = locals[1];
  auto vi32_3 = locals[2];
  auto vi32_4 = locals[3];
  auto vi32_5 = locals[4];
  auto vi32_6 = locals[5];

  wabt::ExprList *e = new ExprList();
  // Second update confusion matrix
  Merge(e, NetworkModel()->Snippets().analysis
      ->ConfusionMatrixUpdate(confusion_matrix_[mode_index], hardmax_[mode_index],
                              snippet::RelocMat(Predictions(mode_index), target_begin),
                              {vi32_1, vi32_2, vi32_3, vi32_4, vi32_5, vi32_6}));
  return e;
}

wabt::ExprList* DenseOutputLayer::CountCorrectPredictions(uint8_t mode_index, Var target_begin, Var result,
                                                          std::vector<Var> locals) {
  assert(mode_index == Model::Mode::Training || mode_index == Model::Mode::Testing);
  assert(hardmax_[mode_index] != nullptr);
  assert(locals.size() == 5);
  auto vi32_1 = locals[0];
  auto vi32_2 = locals[1];
  auto vi32_3 = locals[2];
  auto vi32_4 = locals[3];
  auto vi32_5 = locals[4];

  wabt::ExprList* e = new ExprList();
  // Second count correct predictions
  Merge(e, NetworkModel()->Snippets().analysis
      ->CorrectPredictions(hardmax_[mode_index],
                           snippet::RelocMat(Predictions(mode_index), target_begin),
                           result, {vi32_1, vi32_2, vi32_3}));
  return e;
}

bool DenseOutputLayer::ShouldHardmax(uint8_t mode_index) const {
  if(mode_index == Model::Mode::Training) {
    return NetworkModel()->Options().bytecode_options.gen_training_confusion_matrix
           || NetworkModel()->Options().bytecode_options.gen_training_accuracy;
  }

  if(mode_index == Model::Mode::Testing) {
    return NetworkModel()->Options().bytecode_options.gen_testing_confusion_matrix
           || NetworkModel()->Options().bytecode_options.gen_testing_accuracy;
  }
  return false;
}

void DenseOutputLayer::AllocateMemory() {
  FullyConnectedLayer::AllocateMemory();
  if(ShouldHardmax(Model::Mode::Training)) {
    ALLOCATE_MEMORY(hardmax_[Model::Mode::Training], Nodes(), NetworkModel()->TrainingBatchSize());
  }
  if(ShouldHardmax(Model::Mode::Testing)) {
    ALLOCATE_MEMORY(hardmax_[Model::Mode::Testing], Nodes(), NetworkModel()->TestingBatchSize());
  }
  if(NetworkModel()->Options().bytecode_options.gen_training_confusion_matrix) {
    ALLOCATE_MEMORY(confusion_matrix_[Model::Mode::Training], Nodes(), Nodes());
  }
  if(NetworkModel()->Options().bytecode_options.gen_testing_confusion_matrix) {
    ALLOCATE_MEMORY(confusion_matrix_[Model::Mode::Testing], Nodes(), Nodes());
  }
}

ds::NDArray* DenseOutputLayer::Predictions(uint8_t mode_index) const {
  assert(mode_index >= Model::Mode::FIRST_MODE && mode_index <= Model::Mode::LAST_MODE);
  return A_[mode_index];
}

ds::NDArray* DenseInputLayer::InputArray(uint8_t mode_index) const {
  assert(mode_index >= Model::Mode::FIRST_MODE && mode_index <= Model::Mode::LAST_MODE);
  return A_[mode_index];
}

void DenseInputLayer::MakeFunctions() {
  // Create functions defined in parent
  FullyConnectedLayer::MakeFunctions();

  // Create a function to get offset of the prediction data
  NetworkModel()->ModuleManager().MakeFunction("prediction_data_offset", {{}, {Type::I32}}, {},
                                               [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals){
    f.Insert(MakeI32Const(A_[Model::Mode::Prediction]->Begin()));
  });
}

} // namespace layer
} // namespace arch
} // namespace nn