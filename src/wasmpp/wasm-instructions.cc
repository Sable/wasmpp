#include <src/wasmpp/wasm-instructions.h>

namespace wasmpp {

wabt::ExprList ExprToExprList(std::unique_ptr<wabt::Expr> expr) {
  wabt::ExprList e;
  e.push_back(std::move(expr));
  return e;
}

void Merge(wabt::ExprList* e1, wabt::ExprList* e2) {
  while(e2->size() > 0) {
    e1->push_back(e2->extract_front());
  }
}

wabt::ExprList CreateLoop(ModuleManager* mm, std::function<void(BlockBody, wabt::Var)> content) {
  wabt::ExprList e;
  auto loop = wabt::MakeUnique<wabt::LoopExpr>();
  loop->block.label = mm->GenerateUid();
  BlockBody block_body;
  block_body.exprList = &loop->block.exprs;
  content(block_body, wabt::Var(loop->block.label));
  e.push_back(std::move(loop));
  return e;
}

wabt::ExprList CreateBlock(ModuleManager* mm, std::function<void(wabt::ExprList*, wabt::Var)> content) {
  wabt::ExprList e;
  auto block = wabt::MakeUnique<wabt::LoopExpr>();
  block->block.label = mm->GenerateUid();
  content(&block->block.exprs, wabt::Var(block->block.label));
  e.push_back(std::move(block));
  return e;
}

wabt::ExprList CreateUnary(wabt::Opcode opcode, wabt::ExprList* op) {
  wabt::ExprList e;
  Merge(&e, op);
  e.push_back(wabt::MakeUnique<wabt::UnaryExpr>(opcode));
  return e;
}

wabt::ExprList CreateBinary(wabt::Opcode opcode, wabt::ExprList* op1,
                                           wabt::ExprList* op2) {
  wabt::ExprList e;
  Merge(&e, op1);
  Merge(&e, op2);
  e.push_back(wabt::MakeUnique<wabt::BinaryExpr>(opcode));
  return e;
}

wabt::ExprList CreateI32Const(uint32_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I32(val)));
}

wabt::ExprList CreateI64Const(uint64_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I64(val)));
}

wabt::ExprList CreateF32Const(float val) {
  uint32_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F32(value)));
}

wabt::ExprList CreateF64Const(double val) {
  uint64_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F64(value)));
}

wabt::ExprList CreateBr(wabt::Var label) {
  return ExprToExprList(wabt::MakeUnique<wabt::BrExpr>(label));
}

wabt::ExprList CreateBrIf(wabt::Var label, wabt::ExprList* cond) {
  wabt::ExprList e;
  Merge(&e, cond);
  e.push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

wabt::ExprList CreateLocalGet(wabt::Var var) {
  return ExprToExprList(wabt::MakeUnique<wabt::LocalGetExpr>(var));
}

wabt::ExprList CreateLocalSet(wabt::Var var, wabt::ExprList* val) {
  wabt::ExprList e;
  Merge(&e, val);
  e.push_back(wabt::MakeUnique<wabt::LocalSetExpr>(var));
  return e;
}

wabt::ExprList CreateLocalTree(wabt::Var var, wabt::ExprList* val) {
  wabt::ExprList e;
  Merge(&e, val);
  e.push_back(wabt::MakeUnique<wabt::LocalTeeExpr>(var));
  return e;
}

wabt::ExprList CreateCall(wabt::Var var, std::vector<wabt::ExprList*> args) {
  wabt::ExprList e;
  for(auto arg : args) {
    Merge(&e, arg);
  }
  e.push_back(wabt::MakeUnique<wabt::CallExpr>(var));
  return e;
}

#define DEFINE_LOAD(opcode) \
wabt::ExprList Create##opcode(wabt::ExprList *index, wabt::Address align, uint32_t offset) { \
  wabt::ExprList e;                                                                                         \
  Merge(&e, index);                                                                                         \
  e.push_back(wabt::MakeUnique<wabt::LoadExpr>(wabt::Opcode::opcode, align, offset));                       \
  return e;                                                                                                 \
}
LOAD_INSTRUCTIONS_LIST(DEFINE_LOAD)
#undef DEFINE_LOAD

#define DEFINE_STORE(opcode) \
wabt::ExprList Create##opcode(wabt::ExprList *index, wabt::ExprList* val,        \
    wabt::Address align, uint32_t offset) {                                                     \
  wabt::ExprList e;                                                                             \
  Merge(&e, index);                                                                             \
  Merge(&e, val);                                                                               \
  e.push_back(wabt::MakeUnique<wabt::StoreExpr>(wabt::Opcode::opcode, align, offset));          \
  return e;                                                                                     \
}
STORE_INSTRUCTIONS_LIST(DEFINE_STORE)
#undef DEFINE_STORE

} // namespace wasmpp