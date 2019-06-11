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
  void CompileInitialization() {
    return model_.CompileInitialization();
  }
  bool CompileLayers(uint32_t training_batch, uint32_t testing_batch, uint32_t prediction_batch, std::string loss) {
    // Default "mean-squared-error"
    LossFunction loss_func = model_.Builtins().loss.MeanSquaredError();
    if(loss == "cross-entropy") {
      loss_func = model_.Builtins().loss.CrossEntropy();
    }
    model_.CompileLayers(training_batch, testing_batch, prediction_batch, loss_func);
    return true;
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
  void SetLayers() {
    model_.SetLayers(layers_);
  }
  void CompileTrainingFunction(uint32_t epoch, float learning_rate, std::vector<std::vector<float>> input,
                       std::vector<std::vector<float>> labels) {
    model_.CompileTrainingFunction(epoch, learning_rate, input, labels);
  }
  void CompileTestingFunction(std::vector<std::vector<float>> input, std::vector<std::vector<float>> labels) {
    model_.CompileTestingFunction(input, labels);
  }
  void CompilePredictionFunctions() {
    model_.CompilePredictionFunctions();
  }
  void CompileWeightsFunctions() {
    model_.CompileWeightsFunctions();
  }
};

EMSCRIPTEN_BINDINGS(model_options) {

  class_<ModelOptions>("ModelOptions")
      .constructor<>()
      .property("log_training_accuracy", &ModelOptions::log_training_accuracy)
      .property("log_training_error", &ModelOptions::log_training_error)
      .property("log_training_time", &ModelOptions::log_training_time)
      .property("log_training_confusion_matrix", &ModelOptions::log_training_confusion_matrix)
      .property("log_testing_accuracy", &ModelOptions::log_testing_accuracy)
      .property("log_testing_error", &ModelOptions::log_testing_error)
      .property("log_testing_time", &ModelOptions::log_testing_time)
      .property("log_testing_confusion_matrix", &ModelOptions::log_testing_confusion_matrix)
      .property("log_prediction_results", &ModelOptions::log_prediction_results)
      .property("log_prediction_time", &ModelOptions::log_prediction_time)
      .property("use_simd", &ModelOptions::use_simd);

  class_<ModelWrapper>("Model")
      .constructor<ModelOptions>()
      .function("ToWasm", &ModelWrapper::ToWasm)
      .function("ToWat", &ModelWrapper::ToWat)
      .function("Validate", &ModelWrapper::Validate)
      .function("SetLayers", &ModelWrapper::SetLayers)
      .function("AddDenseInputLayer", &ModelWrapper::AddDenseInputLayer)
      .function("AddDenseHiddenLayer", &ModelWrapper::AddDenseHiddenLayer)
      .function("AddDenseOutputLayer", &ModelWrapper::AddDenseOutputLayer)
      .function("CompileTrainingFunction", &ModelWrapper::CompileTrainingFunction)
      .function("CompileTestingFunction", &ModelWrapper::CompileTestingFunction)
      .function("CompilePredictionFunctions", &ModelWrapper::CompilePredictionFunctions)
      .function("CompileLayers", &ModelWrapper::CompileLayers)
      .function("CompileWeightsFunctions", &ModelWrapper::CompileWeightsFunctions)
      .function("CompileInitialization", &ModelWrapper::CompileInitialization);

  class_<DenseInputLayerDescriptor>("DenseInputLayerDescriptor")
      .constructor<uint32_t>()
      .function("GetNodes", &DenseInputLayerDescriptor::GetNodes)
      .function("SetKeepProb", &DenseInputLayerDescriptor::SetKeepProb)
      .function("GetKeepProb", &DenseInputLayerDescriptor::GetKeepProb);

  class_<DenseHiddenLayerDescriptor>("DenseHiddenLayerDescriptor")
    .constructor<uint32_t, std::string>()
    .function("GetNodes", &DenseHiddenLayerDescriptor::GetNodes)
    .function("GetActivationFunction", &DenseHiddenLayerDescriptor::GetActivationFunction)
    .function("SetWeightType", &DenseHiddenLayerDescriptor::SetWeightType)
    .function("GetWeightType", &DenseHiddenLayerDescriptor::GetWeightType)
    .function("SetKeepProb", &DenseHiddenLayerDescriptor::SetKeepProb)
    .function("GetKeepProb", &DenseHiddenLayerDescriptor::GetKeepProb);

  class_<DenseOutputLayerDescriptor>("DenseOutputLayerDescriptor")
    .constructor<uint32_t, std::string>()
    .function("GetNodes", &DenseOutputLayerDescriptor::GetNodes)
    .function("GetActivationFunction", &DenseOutputLayerDescriptor::GetActivationFunction)
    .function("SetWeightType", &DenseOutputLayerDescriptor::SetWeightType)
    .function("GetWeightType", &DenseOutputLayerDescriptor::GetWeightType);

  register_vector<float>("F32Array");
  register_vector<std::vector<float>>("F32Matrix");
  register_vector<uint8_t>("ByteArray");
}

