#include <src/wasmpp/wasm-manager.h>
#include <src/wasmpp/wasm-instructions.h>
#include <iostream>

using namespace wasmpp;
using namespace wabt;

int main() {
  wasmpp::ModuleManager mm;
  mm.CreateFunction("hello", {}, {Type::I32, Type::I32}, [&](FuncBody f, std::vector<Var> params, std::vector<Var> locals) {
    auto lhs = CreateLocalGet(locals[0]);
    auto rhs = CreateLocalGet(locals[1]);
    auto add = CreateBinary(Opcode::I32Add, &lhs, &rhs);
    f.Insert(&add);

    auto loop = CreateLoop(&mm, [&](BlockBody bb, Var label) {
      auto a = CreateLocalGet(locals[0]);
      bb.Insert(&a);
    });
    f.Insert(&loop);
  });
  std::cout << mm.ToWat(true, true);
  return 0;
}