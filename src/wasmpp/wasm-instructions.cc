#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

wabt::ExprList* ExprToExprList(std::unique_ptr<wabt::Expr> expr) {
  ERROR_UNLESS(expr != nullptr, "expr cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  e->push_back(std::move(expr));
  return e;
}

void Merge(wabt::ExprList* e1, wabt::ExprList* e2) {
  ERROR_UNLESS(e1 != nullptr, "e1 cannot be null");
  ERROR_UNLESS(e2 != nullptr, "e2 cannot be null");
  ERROR_UNLESS(!e2->empty(), "Cannot merge empty expression list. "
                             "Maybe this expression list has already been merged/used?");
  while(!e2->empty()) {
    e1->push_back(e2->extract_front());
  }
}

wabt::ExprList* MakeLoop(LabelManager* label_manager, wabt::FuncSignature sig,
                         std::function<void(BlockBody, wabt::Var)> content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  auto loop = wabt::MakeUnique<wabt::LoopExpr>();
  BlockBody block_body(label_manager, &loop->block.exprs);
  loop->block.label = label_manager->Next();
  loop->block.decl.sig = sig;
  content(block_body, wabt::Var(loop->block.label));
  e->push_back(std::move(loop));
  return e;
}

wabt::ExprList* MakeBlock(LabelManager* label_manager, wabt::FuncSignature sig, std::function<void(BlockBody, wabt::Var)> content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  auto block = wabt::MakeUnique<wabt::BlockExpr>();
  BlockBody block_body(label_manager, &block->block.exprs);
  block->block.label = label_manager->Next();
  block->block.decl.sig = sig;
  content(block_body, wabt::Var(block->block.label));
  e->push_back(std::move(block));
  return e;
}

wabt::ExprList* MakeIf(LabelManager* label_manager, wabt::ExprList* cond, wabt::FuncSignature sig,
                       std::function<void(BlockBody, wabt::Var)> true_content, std::function<void(BlockBody)> false_content) {
  ERROR_UNLESS(label_manager != nullptr, "label manager cannot be null");
  ERROR_UNLESS(cond != nullptr, "cond cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, cond);
  auto if_block = wabt::MakeUnique<wabt::IfExpr>();
  BlockBody true_body(label_manager, &if_block->true_.exprs);
  if_block->true_.label = label_manager->Next();
  if_block->true_.decl.sig = sig;
  BlockBody false_body(label_manager, &if_block->false_);
  true_content(true_body, wabt::Var(if_block->true_.label));
  if(false_content) {
    false_content(false_body);
  }
  e->push_back(std::move(if_block));
  return e;
}

wabt::ExprList* MakeUnary(wabt::Opcode opcode, wabt::ExprList* op) {
  ERROR_UNLESS(op != nullptr, "op cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, op);
  e->push_back(wabt::MakeUnique<wabt::UnaryExpr>(opcode));
  return e;
}

wabt::ExprList* MakeBinary(wabt::Opcode opcode, wabt::ExprList* op1, wabt::ExprList* op2) {
  ERROR_UNLESS(op1 != nullptr, "op1 cannot be null");
  ERROR_UNLESS(op2 != nullptr, "op2 cannot be null");
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

#define DEFINE_EXTRACT(name)                                                                    \
wabt::ExprList* Make##name(wabt::ExprList* operand, uint64_t index) {                           \
  ERROR_UNLESS(operand != nullptr, "operand cannot be null");                                   \
  wabt::ExprList* e = new wabt::ExprList();                                                     \
  Merge(e, operand);                                                                            \
  Merge(e, ExprToExprList(wabt::MakeUnique<wabt::SimdLaneOpExpr>(wabt::Opcode::name, index)));  \
  return e;                                                                                     \
}
EXTRACT_LANE_LIST(DEFINE_EXTRACT)
#undef DEFINE_EXTRACT

#define DEFINE_REPLACE(name)  \
wabt::ExprList* Make##name(wabt::ExprList* operand, wabt::ExprList* val, uint64_t index) {      \
  ERROR_UNLESS(operand != nullptr, "operand cannot be null");                                   \
  ERROR_UNLESS(val != nullptr, "val cannot be null");                                           \
  wabt::ExprList* e = new wabt::ExprList();                                                     \
  Merge(e, operand);                                                                            \
  Merge(e, val);                                                                                \
  Merge(e, ExprToExprList(wabt::MakeUnique<wabt::SimdLaneOpExpr>(wabt::Opcode::name, index)));  \
  return e;                                                                                     \
}
  REPLACE_LANE_LIST(DEFINE_REPLACE)
#undef DEFINE_REPLACE

wabt::ExprList* MakeBr(wabt::Var label) {
  return ExprToExprList(wabt::MakeUnique<wabt::BrExpr>(label));
}

wabt::ExprList* MakeBrIf(wabt::Var label, wabt::ExprList* cond) {
  ERROR_UNLESS(cond != nullptr, "cond cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, cond);
  e->push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

wabt::ExprList* MakeLocalGet(wabt::Var var) {
  return ExprToExprList(wabt::MakeUnique<wabt::LocalGetExpr>(var));
}

wabt::ExprList* MakeLocalSet(wabt::Var var, wabt::ExprList* val) {
  ERROR_UNLESS(val != nullptr, "val cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalSetExpr>(var));
  return e;
}

wabt::ExprList* MakeLocalTree(wabt::Var var, wabt::ExprList* val) {
  ERROR_UNLESS(val != nullptr, "val cannot be null");
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalTeeExpr>(var));
  return e;
}

wabt::ExprList* MakeCall(wabt::Var var, std::vector<wabt::ExprList*> args) {
  wabt::ExprList* e = new wabt::ExprList();
  for(auto arg : args) {
    Merge(e, arg);
  }
  e->push_back(wabt::MakeUnique<wabt::CallExpr>(var));
  return e;
}

wabt::ExprList* MakeDrop() {
  wabt::ExprList* e = new wabt::ExprList();
  e->push_back(wabt::MakeUnique<wabt::DropExpr>());
  return e;
}

wabt::ExprList* MakeNop() {
  wabt::ExprList* e = new wabt::ExprList();
  e->push_back(wabt::MakeUnique<wabt::NopExpr>());
  return e;
}

#define DEFINE_LOAD(opcode) \
wabt::ExprList* Make##opcode(wabt::ExprList* index, wabt::Address align, uint32_t offset) { \
  ERROR_UNLESS(index != nullptr, "index cannot be null");                                   \
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
  ERROR_UNLESS(index != nullptr, "index cannot be null");                                     \
  ERROR_UNLESS(val != nullptr, "val cannot be null");                                         \
  wabt::ExprList* e = new wabt::ExprList();                                                   \
  Merge(e, index);                                                                            \
  Merge(e, val);                                                                              \
  e->push_back(wabt::MakeUnique<wabt::StoreExpr>(wabt::Opcode::opcode, align, offset));       \
  return e;                                                                                   \
}
STORE_INSTRUCTIONS_LIST(DEFINE_STORE)
#undef DEFINE_STORE

#ifdef WABT_EXPERIMENTAL
wabt::ExprList* MakeNativeCall(wabt::Var var, std::vector<wabt::ExprList*> args) {
  wabt::ExprList* e = new wabt::ExprList();
  for(auto arg : args) {
    Merge(e, arg);
  }
  e->push_back(wabt::MakeUnique<wabt::CallNativeExpr>(var));
  return e;
}

wabt::ExprList* MakeOffset32(wabt::ExprList* base, wabt::ExprList* offset, wabt::ExprList* size) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, base);
  Merge(e, offset);
  Merge(e, size);
  e->push_back(wabt::MakeUnique<wabt::Offset32Expr>());
  return e;
}

wabt::ExprList* MakeDup(wabt::ExprList* expr) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, expr);
  e->push_back(wabt::MakeUnique<wabt::DuplicateExpr>());
  return e;
}

wabt::ExprList* MakeSwap(wabt::ExprList* expr1, wabt::ExprList* expr2) {
  wabt::ExprList* e = new wabt::ExprList();
  Merge(e, expr1);
  Merge(e, expr2);
  e->push_back(wabt::MakeUnique<wabt::SwapExpr>());
  return e;
}
#endif // WABT_EXPERIMENTAL

} // namespace wasmpp
