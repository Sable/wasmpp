#include <src/apps/nn-builder/src/arch/model.h>
#include <iostream>

using namespace nn;
using namespace nn::arch;
using namespace wabt;

int main() {
  Model model;
  model.SetLayers({
     new FullyConnectedLayer(2, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(100, model.Builtins().activation.Sigmoid()),
     new FullyConnectedLayer(1, model.Builtins().activation.Sigmoid())
  });
  model.Setup();

  assert(model.Validate());
  std::cout << model.ModuleManager().ToWat(true, true);
  return 0;
}