#include <src/apps/nn-builder/src/nn-model.h>
#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <iostream>

using namespace wabt;

int main() {
  wasmpp::ModuleManagerOptions options;
  options.math.EnableAll();
  options.memory.EnableAll();
  wasmpp::ModuleManager mm(options);

  auto a1Mem = mm.Memory().Allocate(100);
  auto a1    = wasmpp::NDArray(a1Mem, {10, 10});

  auto a2Mem = mm.Memory().Allocate(100);
  auto a2    = wasmpp::NDArray(a2Mem, {10, 10});

  auto dstMem = mm.Memory().Allocate(100);
  auto dst    = wasmpp::NDArray(dstMem, {10, 10});

  mm.MakeMemory(mm.Memory().Pages());
  Type type = Type::F64;
//  mm.MakeFunction("example", {}, {Type::I32, Type::I32, Type::I32, type, Type::I32, Type::I32, Type::I32},
//                  [&](wasmpp::FuncBody f, std::vector<wabt::Var> params, std::vector<wabt::Var> locals) {
//    nn::compute::math::Multiply2DArrays<Type::F64>(&f, a1, a2, dst, locals);
//  });

  if(mm.Validate()) {
    std::cout << mm.ToWat(true, true);
  }
  return 0;
}