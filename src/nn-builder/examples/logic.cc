#include <src/nn-builder/src/arch/model.h>
#include <iostream>
#include <getopt.h>
#include <fstream>

using namespace nn;
using namespace nn::arch;
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
  options.log_training_error = true;
  options.log_training_time = true;
  options.log_testing_error = true;
  options.log_testing_time = true;
  Model model(options);
  model.SetLayers({
     new FullyConnectedLayer(2, model.Builtins().activation.Linear()),
     new FullyConnectedLayer(2, model.Builtins().activation.ELU()),
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid())
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
  uint32_t epoch = 100000;
  uint32_t batch = 1;
  float learning_rate = 0.01;
  auto loss = model.Builtins().loss.MeanSquaredError();
  model.CompileLayers(batch, learning_rate, loss);
  model.CompileTraining(epoch, train, labels);
  model.CompileTesting(train, labels);
  model.CompileDone();

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