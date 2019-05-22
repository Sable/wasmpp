#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

wabt::ExprList* ExprToExprList(std::unique_ptr<wabt::Expr> expr) {
  assert(expr != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  e->push_back(std::move(expr));
  return e;
}

void Merge(wabt::ExprList* e1, wabt::ExprList* e2) {
  assert(e1 != nullptr);
  assert(e2 != nullptr);
  if(e2->empty()) {
    assert(!"Cannot merge empty expression list. "
            "Maybe this expression list has already been merged/used?");
  }
  while(!e2->empty()) {
    e1->push_back(e2->extract_front());
  }
}

wabt::ExprList* MakeLoop(LabelManager* label_manager, std::function<void(BlockBody, wabt::Var)> content) {
  assert(label_manager != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  auto loop = wabt::MakeUnique<wabt::LoopExpr>();
  BlockBody block_body(label_manager, &loop->block.exprs);
  loop->block.label = label_manager->Next();
  content(block_body, wabt::Var(loop->block.label));
  e->push_back(std::move(loop));
  return e;
}

wabt::ExprList* MakeBlock(LabelManager* label_manager, std::function<void(BlockBody, wabt::Var)> content) {
  assert(label_manager != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  auto block = wabt::MakeUnique<wabt::BlockExpr>();
  BlockBody block_body(label_manager, &block->block.exprs);
  block->block.label = label_manager->Next();
  content(block_body, wabt::Var(block->block.label));
  e->push_back(std::move(block));
  return e;
}

wabt::ExprList* MakeUnary(wabt::Opcode opcode, wabt::ExprList* op) {
  assert(op != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, op);
  e->push_back(wabt::MakeUnique<wabt::UnaryExpr>(opcode));
  return e;
}

wabt::ExprList* MakeBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2) {
  assert(op1 != nullptr);
  assert(op2 != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, op1);
  Merge(e, op2);
  e->push_back(wabt::MakeUnique<wabt::BinaryExpr>(opcode));
  return e;
}

wabt::ExprList* MakeI32Const(uint32_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I32(val)));
}

wabt::ExprList* MakeI64Const(uint64_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I64(val)));
}

wabt::ExprList* MakeF32Const(float val) {
  uint32_t value_bits;
  memcpy(&value_bits, &val, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F32(value_bits)));
}

wabt::ExprList* MakeF64Const(double val) {
  uint64_t value_bits;
  memcpy(&value_bits, &val, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F64(value_bits)));
}

wabt::ExprList* MakeBr(wabt::Var label) {
  return ExprToExprList(wabt::MakeUnique<wabt::BrExpr>(label));
}

wabt::ExprList* MakeBrIf(wabt::Var label, wabt::ExprList* cond) {
  assert(cond != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, cond);
  e->push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

wabt::ExprList* MakeLocalGet(wabt::Var var) {
  return ExprToExprList(wabt::MakeUnique<wabt::LocalGetExpr>(var));
}

wabt::ExprList* MakeLocalSet(wabt::Var var, wabt::ExprList* val) {
  assert(val != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalSetExpr>(var));
  return e;
}

wabt::ExprList* MakeLocalTree(wabt::Var var, wabt::ExprList* val) {
  assert(val != nullptr);
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalTeeExpr>(var));
  return e;
}

wabt::ExprList* MakeCall(wabt::Var var, std::vector<wabt::ExprList*> args) {
  wabt::ExprList* e = new wabt::ExprList();
  for(auto arg : args) {
    assert(arg != nullptr);
    Merge(e, arg);
  }
  e->push_back(wabt::MakeUnique<wabt::CallExpr>(var));
  return e;
}

#define DEFINE_LOAD(opcode) \
wabt::ExprList* Make##opcode(wabt::ExprList* index, wabt::Address align, uint32_t offset) { \
  assert(index != nullptr);                                                                 \
  wabt::ExprList* e = new wabt::ExprList();                                                 \
  Merge(e, index);                                                                          \
  e->push_back(wabt::MakeUnique<wabt::LoadExpr>(wabt::Opcode::opcode, align, offset));      \
  return e;                                                                                 \
}
LOAD_INSTRUCTIONS_LIST(DEFINE_LOAD)
#undef DEFINE_LOAD

#define DEFINE_STORE(opcode) \
wabt::ExprList* Make##opcode(wabt::ExprList* index, wabt::ExprList* val, wabt::Address align, \
    uint32_t offset) {                                                                        \
  assert(index != nullptr);                                                                   \
  assert(val != nullptr);                                                                     \
  wabt::ExprList* e = new wabt::ExprList();                                                   \
  Merge(e, index);                                                                            \
  Merge(e, val);                                                                              \
  e->push_back(wabt::MakeUnique<wabt::StoreExpr>(wabt::Opcode::opcode, align, offset));       \
  return e;                                                                                   \
}
STORE_INSTRUCTIONS_LIST(DEFINE_STORE)
#undef DEFINE_STORE

} // namespace wasmpp
