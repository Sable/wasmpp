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
      << "Mnist - Train MNIST" << std::endl
      << "Usage: mnist [OPTION]... [FILE]..." << std::endl
      << "    -w, --to-wasm       Print wasm" << std::endl
      << "    -W, --to-wat        Print wat" << std::endl
      << "    -o, --output        Output file" << std::endl
      << "    -h, --help          Display this help message" << std::endl;
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

  if((!FLAG_to_wat && !FLAG_to_wasm) || output_file.empty()) {
    PrintUsage();
    exit(0);
  }

  ModelOptions options;
  options.bytecode_options.gen_training_accuracy           = true;
  options.bytecode_options.gen_training_error              = true;
  options.bytecode_options.gen_training_confusion_matrix   = true;
  options.bytecode_options.gen_testing_accuracy            = true;
  options.bytecode_options.gen_testing_error               = true;
  options.bytecode_options.gen_testing_confusion_matrix    = true;
  options.bytecode_options.gen_forward_profiling           = true;
  options.bytecode_options.gen_backward_profiling          = true;
  options.bytecode_options.use_simd                        = true;
  Model model(options);
  model.SetLayers({
     NewLayer<DenseInputLayer>(784)->WeightType(XavierUniform)->KeepProb(1),
     NewLayer<DenseHiddenLayer>(64, model.Builtins().activation.Sigmoid())->WeightType(XavierUniform)->KeepProb(1),
     NewLayer<DenseOutputLayer>(10, model.Builtins().activation.Sigmoid())->WeightType(LeCunUniform)
  });

  uint32_t training_batch_size = 1;
  uint32_t training_batches_in_memory = 1;
  uint32_t testing_batch_size = 1;
  uint32_t testing_batches_in_memory = 1;
  uint32_t prediction_batch_size = 1;
  auto loss = model.Builtins().loss.SigmoidCrossEntropy();
  model.Build(training_batch_size, training_batches_in_memory,
              testing_batch_size, testing_batches_in_memory,
              prediction_batch_size, loss);

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