#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <iostream>

using namespace wasmpp;
using namespace wabt;

int main() {
  wasmpp::ModuleManager mm;
  std::cout << mm.ToWat(true, true);
  return 0;
}