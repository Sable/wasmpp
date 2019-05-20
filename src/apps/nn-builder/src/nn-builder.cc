#include <src/apps/nn-builder/src/arch/model.h>
#include <iostream>

using namespace nn;
using namespace nn::arch;
using namespace wabt;

int main() {
  Model model;
  model.SetLayers({
     MakeLayer<InputLayer>(784),
     MakeLayer<FullyConnectedLayer>(100, model.Builtins().sigmoid)
  });
  model.Setup();

  assert(model.Validate());
  std::cout << model.ModuleManager().ToWat(true, true);
  return 0;
}