#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

wabt::ExprList* GenerateGenericDoWhileLoop(LabelManager* label_manager, wabt::Var var, wabt::ExprList* end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  auto loop = MakeLoop(label_manager, sig, [&](BlockBody b, wabt::Var label) {
    content(&b);
    auto tee_local = MakeLocalTree(var, MakeBinary(wabt::Opcode::I32Add, MakeLocalGet(var), MakeI32Const(inc)));
    auto cmp = MakeBinary(wabt::Opcode::I32Ne, tee_local, end);
    auto br_if = MakeBrIf(label, cmp);
    b.Insert(br_if);
  });
  Merge(e, loop);
  return e;
}

wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, uint32_t end,
                             uint32_t inc, wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  ERROR_UNLESS(start < end, "Start must be smaller than end");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeI32Const(start)));
  Merge(e, GenerateGenericDoWhileLoop(label_manager, var, MakeI32Const(end), inc, sig, content));
  return e;
}

wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, wabt::Var end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateGenericDoWhileLoop(label_manager, begin, MakeLocalGet(end), inc, sig, content));
  return e;
}

wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, uint32_t end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateGenericDoWhileLoop(label_manager, begin, MakeI32Const(end), inc, sig, content));
  return e;
}

wabt::ExprList* GenerateCompoundAssignment(wabt::Var var, wabt::Opcode op, wabt::ExprList* operand) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeBinary(op, MakeLocalGet(var), operand)));
  return e;
}

} // namespace wasmpp