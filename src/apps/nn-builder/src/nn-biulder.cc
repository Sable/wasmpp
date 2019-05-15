#include <src/wasmpp/wasm-builder.h>
#include <src/apps/nn-builder/src/nn-math.h>
#include <iostream>

int main() {
  wasmpp::ModuleBuilder mb;
  nn::MathExtension mm(&mb);
  mm.Sigmoid();
  mm.Exp();
  std::cout << mb.ToWat(true, true);
  return 0;
}