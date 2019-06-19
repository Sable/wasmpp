#include <src/nn-builder/src/arch/model.h>
#include <src/nn-builder/src/arch/layers/dense.h>
#include <iostream>
#include <getopt.h>
#include <fstream>

using namespace nn;
using namespace nn::arch;
using namespace nn::arch::layer;
using namespace wabt;

bool FLAG_to_wasm = false;
bool FLAG_to_wat = false;
std::string mnist_train;
std::string output_file;

void PrintUsage() {
  std::cout
      << "Mnist - Train MNIST" << std::endl
      << "Usage: mnist [OPTION]... [FILE]..." << std::endl
      << "    -t, --mnist_train   mnist_train.csv" << std::endl
      << "    -w, --to-wasm       Print wasm" << std::endl
      << "    -W, --to-wat        Print wat" << std::endl
      << "    -o, --output        Output file" << std::endl
      << "    -h, --help          Display this help message" << std::endl;
}

void InitParams(int argc, char *argv[]) {

  struct option longOptions[] = {

      {"to-wasm", no_argument, 0, 'w'},
      {"to-wat", no_argument, 0, 'W'},
      {"mnist-train", required_argument, 0, 't'},
      {"output", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
  };

  int optionIndex = 0;
  int c;
  while ((c = getopt_long(argc, argv, "hwWo:t:", longOptions, &optionIndex)) != -1) {
    switch (c) {
      case 'w':
        FLAG_to_wasm = true;
        break;
      case 'W':
        FLAG_to_wat = true;
        break;
      case 'o':
        output_file = optarg;
        break;
      case 't':
        mnist_train = optarg;
        break;
      case 'h':
        PrintUsage();
        exit(0);
      default:
        break;
    }
  }
}

void LoadValues(std::vector<std::vector<float>> &train_data, std::vector<std::vector<float>> &train_labels,
                std::vector<std::vector<float>> &test_data, std::vector<std::vector<float>> &test_labels,
                uint32_t training_limit, uint32_t testing_limit) {
  std::ifstream mnist_train_file(mnist_train);
  if (mnist_train_file.is_open()) {
    std::string line;
    uint32_t line_num = 0;
    while (getline(mnist_train_file, line)) {
      std::vector<float> data;
      std::vector<float> label(10, 0);
      // Skip label line
      if(line_num++ > 0) {
        // Read label
        label[line[0] - '0'] = 1;
        // Read data
        float number = 0;
        for(size_t i=2; i < line.size(); i++) {
          if(line[i] >= '0' && line[i] <= '9') {
            number *= 10;
            number += line[i] - '0';
          } else {
            data.push_back(number / 255.0f);
            number = 0;
          }
        }
        if(train_data.size() < training_limit) {
          train_data.push_back(std::move(data));
          train_labels.push_back(std::move(label));
        } else if(test_data.size() < testing_limit) {
          test_data.push_back(std::move(data));
          test_labels.push_back(std::move(label));
        } else {
          break;
        }
      }
    }
    mnist_train_file.close();
  } else {
    ERROR_EXIT("Error opening file: '%s'", mnist_train.c_str());
  }
}

int main(int argc, char *argv[]) {
  InitParams(argc, argv);

  if((!FLAG_to_wat && !FLAG_to_wasm) || output_file.empty() || mnist_train.empty()) {
    PrintUsage();
    exit(0);
  }

  // Load csv file
  uint32_t training_limit = 6000;
  uint32_t testing_limit = 1000;
  std::vector<std::vector<float>> train_data;
  std::vector<std::vector<float>> train_labels;
  std::vector<std::vector<float>> test_data;
  std::vector<std::vector<float>> test_labels;
  LoadValues(train_data, train_labels, test_data, test_labels, training_limit, testing_limit);

  ModelOptions options;
  options.log_training_accuracy           = true;
  options.log_training_error              = true;
  options.log_training_time               = true;
  options.log_training_confusion_matrix   = true;
  options.log_testing_accuracy            = true;
  options.log_testing_error               = true;
  options.log_testing_time                = true;
  options.log_testing_confusion_matrix    = true;
  options.log_forward                     = true;
  options.log_backward                    = true;
  options.use_simd                        = true;
  Model model(options);
  model.SetLayers({
     NewLayer<DenseInputLayer>(784)->WeightType(XavierUniform)->KeepProb(1),
     NewLayer<DenseHiddenLayer>(100, model.Builtins().activation.Sigmoid())->WeightType(XavierUniform)->KeepProb(1),
     NewLayer<DenseOutputLayer>(10, model.Builtins().activation.Sigmoid())->WeightType(LeCunUniform)
  });

  uint32_t epoch = 10;
  uint32_t training_batch_size = 1;
  uint32_t testing_batch_size = 1;
  uint32_t prediction_batch_size = 1;
  float learning_rate = 0.02;
  auto loss = model.Builtins().loss.CrossEntropy();
  model.CompileLayers(training_batch_size, testing_batch_size, prediction_batch_size, loss);
  model.CompileTrainingFunction(epoch, learning_rate, train_data, train_labels);
  model.CompileTestingFunction(test_data, test_labels);
  model.CompilePredictionFunctions();
  model.CompileWeightsFunctions();
  model.CompileInitialization();

  assert(model.Validate());
  if(!output_file.empty()) {
    std::ofstream file;
    file.open(output_file);
    if(FLAG_to_wasm) {
      auto data = model.ModuleManager().ToWasm().data;
      file << std::string(data.begin(), data.end());
    } else if(FLAG_to_wat) {
      file << model.ModuleManager().ToWat(true, true);
    }
    file.close();
  }
  return 0;
}