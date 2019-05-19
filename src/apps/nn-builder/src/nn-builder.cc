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

  const Type type = Type::F64;
  uint64_t side = 3;
  auto total_bytes = side * side * TypeSize(type);
  auto a1Mem = mm.Memory().Allocate(total_bytes);
  auto a1    = NDArray(a1Mem, {side, side}, TypeSize(type));

  auto a2Mem = mm.Memory().Allocate(total_bytes);
  auto a2    = NDArray(a2Mem, {side, side}, TypeSize(type));

  auto a3Mem = mm.Memory().Allocate(total_bytes);
  auto a3    = NDArray(a3Mem, {side, side}, TypeSize(type));

  auto memo = mm.MakeMemory(mm.Memory().Pages());

  auto mat_mul = mm.MakeFunction("mat_mul", {}, {Type::I32, Type::I32, Type::I32, type, Type::I32, Type::I32, Type::I32},
                  [&](wasmpp::FuncBody f, std::vector<wabt::Var> params, std::vector<wabt::Var> locals) {
    nn::compute::math::Multiply2DArrays<type>(&f, a1, a2, a3, locals);
  });

  mm.MakeFunction("main", {{},{}}, {}, [&](FuncBody f, vector<Var> params, vector<Var> locals) {
    auto z = 1;
    for(uint32_t r=0; r < side; r++) {
      for(uint32_t c=0; c < side; c++) {
        f.Insert(MakeF64Store(MakeI32Const(a1.GetLinearIndex({r, c})), MakeF64Const(z++)));
      }
    }

    z = 9;
    for(uint32_t r=0; r < side; r++) {
      for(uint32_t c=0; c < side; c++) {
        f.Insert(MakeF64Store(MakeI32Const(a2.GetLinearIndex({r, c})), MakeF64Const(z--)));
      }
    }

    f.Insert(MakeCall(mat_mul, {}));

    for(uint32_t r=0; r < side; r++) {
      for(uint32_t c=0; c < side; c++) {
        f.Insert(MakeCall(f.Builtins()->system.PrintF64(), {MakeF64Load(MakeI32Const(a3.GetLinearIndex({r, c})))}));
      }
    }
  });


  if(mm.Validate()) {
    std::cout << mm.ToWat(true, true);
  }
  return 0;
}