#include <src/apps/nn-builder/src/nn-model.h>
#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/common.h>
#include <iostream>

using namespace wabt;
using namespace wasmpp;
using namespace std;

int main() {
  wasmpp::ModuleManagerOptions options;
  options.math.EnableAll();
  options.memory.EnableAll();
  options.system.EnableAll();
  wasmpp::ModuleManager mm(options);

  Type type = Type::F64;
  auto side = 3;
  auto a1Mem = mm.Memory().Allocate(side * side * TypeSize(type));
  auto a1    = NDArray(a1Mem, {side, side}, TypeSize(type));
  mm.MakeMemory(mm.Memory().Pages());

//  auto a2Mem = mm.Memory().Allocate(3 * 3);
//  auto a2    = NDArray(a2Mem, {3, 3});
//
//  auto dstMem = mm.Memory().Allocate(9);
//  auto dst    = NDArray(dstMem, {3, 3});

//  mm.MakeFunction("mat_mul", {}, {Type::I32, Type::I32, Type::I32, type, Type::I32, Type::I32, Type::I32},
//                  [&](wasmpp::FuncBody f, std::vector<wabt::Var> params, std::vector<wabt::Var> locals) {
//    nn::compute::math::Multiply2DArrays<Type::I32>(&f, a1, a2, dst, locals);
//  });

  mm.MakeFunction("main", {{},{}}, {}, [&](FuncBody f, vector<Var> params, vector<Var> locals) {
    auto z = 1;
    for(uint32_t r=0; r < side; r++) {
      for(uint32_t c=0; c < side; c++) {
        f.Insert(MakeF64Store(MakeI32Const(a1.GetLinearIndex({r, c})), MakeF64Const(z++)));
      }
    }

    for(int i=0; i < side*side; i++) {
      f.Insert(MakeCall(mm.builtins.system.PrintF64(), {MakeF64Load(MakeI32Const(i*TypeSize(type)))}));
    }
  });

  if(mm.Validate()) {
    std::cout << mm.ToWat(true, true);
  }
  return 0;
}