#include <src/apps/nn-builder/src/arch/model.h>
#include <iostream>

using namespace nn;

int main() {
  arch::Model model;
  model.Setup();

  assert(model.Validate());
  std::cout << model.ModuleManager().ToWat(true, true);
  return 0;
}