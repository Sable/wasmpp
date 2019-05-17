#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <iostream>

using namespace wasmpp;
using namespace wabt;

int main() {
  wasmpp::ModuleManagerOptions options;
  options.math.sigmoid = true;
  options.math.exp = true;
  wasmpp::ModuleManager mm(options);
  std::cout << mm.ToWat(true, true);
  return 0;
}