#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager,
                             wabt::Var var, uint32_t start, uint32_t end,
                             uint32_t inc, std::function<void(BlockBody*)> content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeI32Const(start)));
  auto loop = MakeLoop(label_manager, [&](BlockBody b, wabt::Var label) {
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