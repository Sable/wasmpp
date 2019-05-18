#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

exprs_sptr GenerateRangeLoop(ContentManager* ctn,
                             wabt::Var var, uint32_t start, uint32_t end,
                             uint32_t inc, std::function<void(BlockBody*)> content) {
  exprs_sptr e = CreateExprList();
  Merge(e, MakeLocalSet(var, MakeI32Const(start)));
  auto loop = MakeLoop(ctn, [&](BlockBody b, wabt::Var label) {
    content(&b);
    auto tee_local = MakeLocalTree(var, MakeBinary(wabt::Opcode::I32Add, MakeLocalGet(var), MakeI32Const(inc)));
    auto cmp = MakeBinary(wabt::Opcode::I32Ne, tee_local, MakeI32Const(end));
    auto br_if = MakeBrIf(label, cmp);
    b.Insert(br_if);
  });
  Merge(e, loop);
  return e;
}

} // namespace wasmpp