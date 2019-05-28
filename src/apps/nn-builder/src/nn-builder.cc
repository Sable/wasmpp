#include <src/apps/nn-builder/src/arch/model.h>
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
      << "EasyCC C++ - An easy compiler compiler program" << std::endl
      << "Usage: easycc [OPTION]... [FILE]..." << std::endl
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
        PrintUsage();
        exit(0);
      default:
        break;
    }
  }
}

int main(int argc, char *argv[]) {
  InitParams(argc, argv);

  Model model;
  model.SetLayers({
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(1, model.Builtins().activation.Sigmoid())
  });

  // Train
  std::vector<std::vector<double>> train = {
    {0, 0},
    {0, 1},
    {1, 0},
    {1, 1}
  };
  // Try with one node
  std::vector<std::vector<double>> labels = {
    {0},
    {1},
    {1},
    {1}
  };
  uint32_t epoch = 300000;
  uint32_t batch = 1;
  double learning_rate = 0.01;
  model.Setup(epoch, batch, learning_rate, train, labels);
  model.Train();

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