#include <src/apps/nn-builder/src/nn-model.h>
#include <src/apps/nn-builder/src/math/matrix.h>
#include <src/wasmpp/data_structure/ndarray.h>
#include <src/wasmpp/wasm-manager.h>
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

  auto a1Mem = mm.Memory().Allocate(9);
  auto a1    = wasmpp::NDArray(a1Mem, {3, 3});

  auto a2Mem = mm.Memory().Allocate(9);
  auto a2    = wasmpp::NDArray(a2Mem, {3, 3});

  auto dstMem = mm.Memory().Allocate(9);
  auto dst    = wasmpp::NDArray(dstMem, {3, 3});

  mm.MakeMemory(mm.Memory().Pages());
  Type type = Type::F64;
  mm.MakeFunction("mat_mul", {}, {Type::I32, Type::I32, Type::I32, type, Type::I32, Type::I32, Type::I32},
                  [&](wasmpp::FuncBody f, std::vector<wabt::Var> params, std::vector<wabt::Var> locals) {
    nn::compute::math::Multiply2DArrays<Type::F64>(&f, a1, a2, dst, locals);
  });

  mm.MakeFunction("main", {{},{Type::I32}}, {}, [&](FuncBody f, vector<Var> params, vector<Var> locals) {
//    f.Insert(MakeCall(mm.builtins.memory.FillI32(), {MakeI32Const(4), MakeI32Const(10), MakeI32Const(42)}));
    f.Insert(MakeI32Load(MakeI32Const(11*4)));
  });

  if(mm.Validate()) {
    std::cout << mm.ToWat(true, true);
  }
  return 0;
}