#include <src/wasmpp/wasm-instructions-gen.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

wabt::ExprList* GenerateGenericDoWhileLoop(LabelManager* label_manager, wabt::Var var, wabt::ExprList* end, wabt::ExprList* inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  auto loop = MakeLoop(label_manager, sig, [&](BlockBody b, wabt::Var label) {
    content(&b);
    auto tee_local = MakeLocalTree(var, MakeBinary(wabt::Opcode::I32Add, MakeLocalGet(var), inc));
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
  Merge(e, GenerateGenericDoWhileLoop(label_manager, var, MakeI32Const(end), MakeI32Const(inc), sig, content));
  return e;
}

wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, wabt::Var end,
                                  uint32_t inc, wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeI32Const(start)));
  Merge(e, GenerateGenericDoWhileLoop(label_manager, var, MakeLocalGet(end), MakeI32Const(inc), sig, content));
  return e;
}

wabt::ExprList* GenerateRangeLoop(LabelManager* label_manager, wabt::Var var, uint32_t start, wabt::Var end,
                                  wabt::Var inc, wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeI32Const(start)));
  Merge(e, GenerateGenericDoWhileLoop(label_manager, var, MakeLocalGet(end), MakeLocalGet(inc), sig, content));
  return e;
}

wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, wabt::Var end, uint32_t inc,
                                  wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateGenericDoWhileLoop(label_manager, begin, MakeLocalGet(end), MakeI32Const(inc), sig, content));
  return e;
}

wabt::ExprList* GenerateDoWhileLoop(LabelManager* label_manager, wabt::Var begin, uint32_t end, uint32_t inc,
                                    wabt::FuncSignature sig, std::function<void(BlockBody*)> content) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, GenerateGenericDoWhileLoop(label_manager, begin, MakeI32Const(end), MakeI32Const(inc), sig, content));
  return e;
}

wabt::ExprList* GenerateCompoundAssignment(wabt::Var var, wabt::Opcode op, wabt::ExprList* operand) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, MakeLocalSet(var, MakeBinary(op, MakeLocalGet(var), operand)));
  return e;
}

wabt::ExprList* GenerateF32X4HorizontalLTRSum(wabt::Var var) {
  wabt::ExprList* e = new wabt::ExprList();
  auto add_1 = MakeBinary(wabt::Opcode::F32Add,
      MakeF32X4ExtractLane(MakeLocalGet(var), 0), MakeF32X4ExtractLane(MakeLocalGet(var), 1));
  auto add_2 = MakeBinary(wabt::Opcode::F32Add, add_1, MakeF32X4ExtractLane(MakeLocalGet(var), 2));
  auto add_3 = MakeBinary(wabt::Opcode::F32Add, add_2, MakeF32X4ExtractLane(MakeLocalGet(var), 3));
  Merge(e, add_3);
  return e;
}
} // namespace wasmpp