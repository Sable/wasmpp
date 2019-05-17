#include <src/apps/nn-builder/src/nn-model.h>
#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <iostream>

int main() {
  wasmpp::ModuleManagerOptions options;
  options.math.sigmoid = true;
  options.math.exp = true;
  wasmpp::ModuleManager mm(options);

  auto a1Mem = mm.Memory().Allocate(100);
  auto a1    = wasmpp::NDArray(a1Mem, {10, 10});

  auto a2Mem = mm.Memory().Allocate(100);
  auto a2    = wasmpp::NDArray(a2Mem, {10, 10});

  auto dstMem = mm.Memory().Allocate(100);
  auto dst    = wasmpp::NDArray(dstMem, {10, 10});

  mm.MakeFunction("example", {}, {wabt::Type::I32, wabt::Type::I32, wabt::Type::I32}, [&](wasmpp::FuncBody f, std::vector<wabt::Var> params,
                                         std::vector<wabt::Var> locals) {
    nn::compute::math::Multiply2DArrays(&mm, &f, a1, a2, dst, locals[0], locals[1], locals[2]);
  });

  std::cout << mm.ToWat(true, true);
  return 0;
}