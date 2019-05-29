#include <src/nn-builder/src/arch/model.h>
#include <iostream>
#include <getopt.h>
#include <fstream>

using namespace nn;
using namespace nn::arch;
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

int main(int argc, char *argv[]) {
  InitParams(argc, argv);

  if((!FLAG_to_wat && !FLAG_to_wasm) || output_file.empty() || mnist_train.empty()) {
    PrintUsage();
    exit(0);
  }

  // Load csv file
  int LIMIT = 10;
  std::vector<std::vector<double>> train_data;
  std::vector<std::vector<double>> train_labels;
  std::ifstream mnist_train_file(mnist_train);
  if (mnist_train_file.is_open()) {
    std::string line;
    uint32_t line_num = 0;
    while (getline(mnist_train_file, line) && train_data.size() < LIMIT) {
      std::vector<double> data;
      std::vector<double> label(10, 0);
      // Skip label line
      if(line_num++ > 0) {
        // Read label
        label[line[0] - '0'] = 1;
        // Read data
        double number = 0;
        for(size_t i=2; i < line.size(); i++) {
          if(line[i] >= '0' && line[i] <= '9') {
            number *= 10;
            number += line[i] - '0';
          } else {
            data.push_back(number);
            number = 0;
          }
        }
        train_data.push_back(std::move(data));
        train_labels.push_back(std::move(label));
      }
    }
    mnist_train_file.close();
//    for(int i=0; i < train_data[0].size(); i++) {
//      printf("%lf ", train_data[0][i]);
//    }
//    printf("\n");
  } else {
    ERROR_EXIT("Error opening file: '%s'", mnist_train.c_str());
  }

  Model model;
  model.SetLayers({
     new FullyConnectedLayer(784, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(100, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(10, model.Builtins().activation.Sigmoid())
  });

  uint32_t epoch = 100;
  uint32_t batch = 1;
  double learning_rate = 0.1;
  model.Setup(epoch, batch, learning_rate, train_data, train_labels);
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