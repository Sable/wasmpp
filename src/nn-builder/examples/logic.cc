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
std::string output_file;

void PrintUsage() {
  std::cout
      << "Logic - Train an OR function" << std::endl
      << "Usage: logic [OPTION]... [FILE]..." << std::endl
      << "    -w, --to-wasm    Print wasm" << std::endl
      << "    -W, --to-wat     Print wat" << std::endl
      << "    -o, --output     Output file" << std::endl
      << "    -h, --help       Display this help message" << std::endl;
}

void InitParams(int argc, char *argv[]) {

  struct option longOptions[] = {

      {"to-wasm", no_argument, 0, 'w'},
      {"to-wat", no_argument, 0, 'W'},
      {"output", required_argument, 0, 'o'},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}
  };

  int optionIndex = 0;
  int c;
  while ((c = getopt_long(argc, argv, "hwWo:", longOptions, &optionIndex)) != -1) {
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
      case 'h':
      default:
        break;
    }
  }
}

int main(int argc, char *argv[]) {
  InitParams(argc, argv);

  if(!FLAG_to_wat && !FLAG_to_wasm && output_file.empty()) {
    PrintUsage();
    exit(0);
  }

  ModelOptions options;
  options.log_training_accuracy = true;
  options.log_training_error = true;
  options.log_training_time = true;
  options.log_testing_accuracy;
  options.log_testing_error = true;
  options.log_testing_time = true;
  options.log_testing_confusion_matrix = true;
  Model model(options);
  model.SetLayers({
     NewLayer<DenseInputLayer>(2)->WeightType(XavierNormal),
     NewLayer<DenseHiddenLayer>(2, model.Builtins().activation.Sigmoid())->WeightType(XavierNormal),
     NewLayer<DenseOutputLayer>(2, model.Builtins().activation.Sigmoid())->WeightType(LeCunNormal),
  });

  // Train
  std::vector<std::vector<float>> train = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
  };
  // Try with one node
  std::vector<std::vector<float>> labels = {
    {1, 0},
    {0, 1},
    {0, 1},
    {0, 1}
  };
  uint32_t epoch = 3000;
  uint32_t training_batch_size = 1;
  uint32_t testing_batch_size = 1;
  uint32_t prediction_batch_size = 4;
  float learning_rate = 0.02;
  auto loss = model.Builtins().loss.MeanSquaredError();
  model.CompileLayers(training_batch_size, testing_batch_size, prediction_batch_size, loss);
  model.CompileTrainingFunction(epoch, learning_rate, train, labels);
  model.CompileTestingFunction(train, labels);
  model.CompilePredictionFunctions();
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