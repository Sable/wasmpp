#include <emscripten/bind.h>
#include <emscripten/emscripten.h>
#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/arch/layers/dense.h>
#include <cinttypes>

using namespace nn::arch;
using namespace nn::arch::layer;
using namespace nn::builtins;
using namespace emscripten;

class LayerDescriptor {
public:
  LayerDescriptor(LayerType type, LayerPosition position) : type_(type), position_(position) {}
  LayerType Type() const { return type_; }
  LayerPosition Position() const { return position_; }
private:
  LayerType type_;
  LayerPosition position_;
};

class DenseInputLayerDescriptor : public LayerDescriptor {
private:
  uint32_t nodes_;
  float keep_prob_ = 1.0;
public:
  DenseInputLayerDescriptor(uint32_t nodes) : LayerDescriptor(FullyConnected, Input), nodes_(nodes) {}
  uint32_t GetNodes() const { return nodes_; }
  void SetKeepProb(float keep_prob) { keep_prob_ = keep_prob; }
  float GetKeepProb() const { return keep_prob_; }
};

class DenseHiddenLayerDescriptor : public LayerDescriptor {
private:
  uint32_t nodes_;
  float keep_prob_ = 1.0;
  std::string weight_type_;
  std::string act_func_;
public:
  DenseHiddenLayerDescriptor(uint32_t nodes, std::string act_func) :
      LayerDescriptor(FullyConnected, Hidden), nodes_(nodes), act_func_(act_func) {}
  uint32_t GetNodes() const { return nodes_; }
  std::string GetActivationFunction() const { return act_func_; }
  void SetKeepProb(float keep_prob) { keep_prob_ = keep_prob; }
  float GetKeepProb() const { return keep_prob_; }
  void SetWeightType(std::string weight_type) {weight_type_ = weight_type; }
  std::string GetWeightType() const { return weight_type_; }
};

class DenseOutputLayerDescriptor : public LayerDescriptor {
private:
  uint32_t nodes_;
  std::string weight_type_;
  std::string act_func_;
public:
  DenseOutputLayerDescriptor(uint32_t nodes, std::string act_func) :
      LayerDescriptor(FullyConnected, Output), nodes_(nodes), act_func_(act_func) {}
  uint32_t GetNodes() const { return nodes_; }
  std::string GetActivationFunction() const { return act_func_; }
  void SetWeightType(std::string weight_type) {weight_type_ = weight_type; }
  std::string GetWeightType() const { return weight_type_; }
};

class ModelWrapper {
private:
  Model model_;
  std::vector<Layer*> layers_;
  ActivationFunction StringToActivationFunction(std::string act_func) {
    if(act_func == "relu"){
      return model_.Builtins().activation.ReLU();
    } else if(act_func == "leaky_relu") {
      return model_.Builtins().activation.LeakyReLU();
    } else if(act_func == "tanh") {
      return model_.Builtins().activation.Tanh();
    } else if(act_func == "elu") {
      return model_.Builtins().activation.ELU();
    } else if(act_func == "linear"){
      return model_.Builtins().activation.Linear();
    } else if(act_func == "softmax") {
      return model_.Builtins().activation.Softmax();
    }
    // Default "sigmoid"
    return model_.Builtins().activation.Sigmoid();
  }
  WeightDistributionType StringToWeightDistribution(std::string weight_type) {
    if(weight_type == "xavier_uniform") {
      return XavierUniform;
    } else if(weight_type == "xavier_normal") {
      return XavierNormal;
    } else if(weight_type == "lecun_uniform") {
      return LeCunUniform;
    } else if(weight_type == "lecun_normal") {
      return LeCunNormal;
    } else if(weight_type == "gaussian") {
      return Gaussian;
    } else if(weight_type == "uniform") {
      return Uniform;
    }
    return Constant;
  }
public:
  ModelWrapper(ModelOptions options) : model_(options) {}
  std::string ToWat(bool folded, bool inlined) {
    return model_.ModuleManager().ToWat(folded, inlined);
  }
  std::vector<uint8_t> ToWasm() {
    return model_.ModuleManager().ToWasm().data;
  }
  bool Validate() {
    return model_.Validate();
  }
  void Build(uint32_t training_batch_size, uint32_t training_batches_in_memory,
             uint32_t testing_batch_size, uint32_t testing_batches_in_memory,
             uint32_t prediction_batch_size, std::string loss,
             float l1_regularizer, float l2_regularizer) {
    // Set layers
    model_.SetLayers(layers_);

    // Default "mean-squared-error"
    LossFunction loss_func = model_.Builtins().loss.MeanSquaredError();
    if(loss == "sigmoid-cross-entropy") {
      loss_func = model_.Builtins().loss.SigmoidCrossEntropy();
    } else if( loss == "softmax-cross-entropy") {
      loss_func = model_.Builtins().loss.SoftmaxCrossEntropy();
    }

    // Build model
    model_.Build(training_batch_size, training_batches_in_memory, testing_batch_size, testing_batches_in_memory,
                 prediction_batch_size, loss_func, l1_regularizer, l2_regularizer);
  }
  void AddDenseInputLayer(DenseInputLayerDescriptor desc) {
    layers_.push_back(NewLayer<DenseInputLayer>(desc.GetNodes())
        ->KeepProb(desc.GetKeepProb()));
  }
  void AddDenseHiddenLayer(DenseHiddenLayerDescriptor desc) {
    layers_.push_back(NewLayer<DenseHiddenLayer>(desc.GetNodes(), StringToActivationFunction(desc.GetActivationFunction()))
        ->KeepProb(desc.GetKeepProb())
        ->WeightType(StringToWeightDistribution(desc.GetWeightType())));
  }
  void AddDenseOutputLayer(DenseOutputLayerDescriptor desc) {
    layers_.push_back(NewLayer<DenseOutputLayer>(desc.GetNodes(), StringToActivationFunction(desc.GetActivationFunction()))
        ->WeightType(StringToWeightDistribution(desc.GetWeightType())));
  }
};

EMSCRIPTEN_BINDINGS(model_options) {

#define WEIGHT_DISTRIBUTION_OPTIONS(name) \
  .property(#name, &WeightDistributionOptions::name)

    class_<WeightDistributionOptions>("WeightDistributionOptions")
        .constructor<>()
        WEIGHT_DISTRIBUTION_OPTIONS(gaussian_mean)
        WEIGHT_DISTRIBUTION_OPTIONS(gaussian_std_dev)
        WEIGHT_DISTRIBUTION_OPTIONS(uniform_low)
        WEIGHT_DISTRIBUTION_OPTIONS(uniform_high)
        WEIGHT_DISTRIBUTION_OPTIONS(constant_value)
        WEIGHT_DISTRIBUTION_OPTIONS(seed);

#define ACTIVATION_OPTIONS(name) \
  .property(#name, &ActivationOptions::name)

    class_<ActivationOptions>("ActivationOptions")
        .constructor<>()
        ACTIVATION_OPTIONS(linear_slope)
        ACTIVATION_OPTIONS(leaky_relu_slope)
        ACTIVATION_OPTIONS(elu_slope);

#define MODEL_BYTECODE_OPTIONS(name) \
  .property(#name, &ModelBytecodeOptions::name)

  class_<ModelBytecodeOptions>("ModelBytecodeOptions")
      .constructor<>()
      MODEL_BYTECODE_OPTIONS(gen_training_accuracy)
      MODEL_BYTECODE_OPTIONS(gen_training_error)
      MODEL_BYTECODE_OPTIONS(gen_training_confusion_matrix)
      MODEL_BYTECODE_OPTIONS(gen_testing_accuracy)
      MODEL_BYTECODE_OPTIONS(gen_testing_error)
      MODEL_BYTECODE_OPTIONS(gen_testing_confusion_matrix)
      MODEL_BYTECODE_OPTIONS(gen_forward_profiling)
      MODEL_BYTECODE_OPTIONS(gen_backward_profiling)
      MODEL_BYTECODE_OPTIONS(use_simd);

#define MODEL_OPTIONS(name) \
  .property(#name, &ModelOptions::name)

  class_<ModelOptions>("ModelOptions")
      .constructor<>()
      MODEL_OPTIONS(bytecode_options)
      MODEL_OPTIONS(activation_options)
      MODEL_OPTIONS(weights_options);

#define MODEL_WRAPPER(name) \
  .function(#name, &ModelWrapper::name)

  class_<ModelWrapper>("Model")
      .constructor<ModelOptions>()
      MODEL_WRAPPER(ToWasm)
      MODEL_WRAPPER(ToWat)
      MODEL_WRAPPER(Validate)
      MODEL_WRAPPER(AddDenseInputLayer)
      MODEL_WRAPPER(AddDenseHiddenLayer)
      MODEL_WRAPPER(AddDenseOutputLayer)
      MODEL_WRAPPER(Build);

#define DENSE_INPUT_LAYER(name) \
  .function(#name, &DenseInputLayerDescriptor::name)

  class_<DenseInputLayerDescriptor>("DenseInputLayerDescriptor")
      .constructor<uint32_t>()
      DENSE_INPUT_LAYER(GetNodes)
      DENSE_INPUT_LAYER(SetKeepProb)
      DENSE_INPUT_LAYER(GetKeepProb);

#define DENSE_HIDDEN_LAYER(name) \
  .function(#name, &DenseHiddenLayerDescriptor::name)

  class_<DenseHiddenLayerDescriptor>("DenseHiddenLayerDescriptor")
    .constructor<uint32_t, std::string>()
    DENSE_HIDDEN_LAYER(GetNodes)
    DENSE_HIDDEN_LAYER(GetActivationFunction)
    DENSE_HIDDEN_LAYER(SetWeightType)
    DENSE_HIDDEN_LAYER(GetWeightType)
    DENSE_HIDDEN_LAYER(SetKeepProb)
    DENSE_HIDDEN_LAYER(GetKeepProb);

#define DENSE_OUTPUT_LAYER(name) \
  .function(#name, &DenseOutputLayerDescriptor::name)

  class_<DenseOutputLayerDescriptor>("DenseOutputLayerDescriptor")
    .constructor<uint32_t, std::string>()
    DENSE_OUTPUT_LAYER(GetNodes)
    DENSE_OUTPUT_LAYER(GetActivationFunction)
    DENSE_OUTPUT_LAYER(SetWeightType)
    DENSE_OUTPUT_LAYER(GetWeightType);

  register_vector<uint8_t>("ByteArray");
}

