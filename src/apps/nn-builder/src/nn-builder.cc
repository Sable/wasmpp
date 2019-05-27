#include <src/apps/nn-builder/src/arch/model.h>
#include <iostream>

using namespace nn;
using namespace nn::arch;
using namespace wabt;

int main() {
  Model model;
  model.SetLayers({
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(10, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid())
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
    {1, 0},
    {0, 1},
    {0, 1},
    {0, 1}
  };
  uint32_t batch = 1;
  double learning_rate = 0.1;
  model.Setup(batch, learning_rate, train, labels);
  model.Train();

  assert(model.Validate());
  std::cout << model.ModuleManager().ToWat(true, true);
  return 0;
}