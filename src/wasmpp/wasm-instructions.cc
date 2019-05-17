#include <src/wasmpp/wasm-instructions.h>
#include <src/wasmpp/wasm-manager.h>

namespace wasmpp {

exprs_sptr CreateExprList() {
  return std::make_shared<wabt::ExprList>();
}

exprs_sptr ExprToExprList(std::unique_ptr<wabt::Expr> expr) {
  assert(expr != nullptr);
  exprs_sptr e = CreateExprList();
  e->push_back(std::move(expr));
  return e;
}

void Merge(exprs_sptr e1, exprs_sptr e2) {
  assert(e1 != nullptr);
  assert(e2 != nullptr);
  while(e2->size() > 0) {
    e1->push_back(e2->extract_front());
  }
}

exprs_sptr MakeLoop(ModuleManager* mm, std::function<void(BlockBody, wabt::Var)> content) {
  exprs_sptr e = CreateExprList();
  auto loop = wabt::MakeUnique<wabt::LoopExpr>();
  loop->block.label = mm->GenerateUid();
  BlockBody block_body;
  block_body.expr_list= &loop->block.exprs;
  content(block_body, wabt::Var(loop->block.label));
  e->push_back(std::move(loop));
  return e;
}

exprs_sptr MakeBlock(ModuleManager* mm, std::function<void(BlockBody, wabt::Var)> content) {
  exprs_sptr e = CreateExprList();
  auto block = wabt::MakeUnique<wabt::BlockExpr>();
  block->block.label = mm->GenerateUid();
  BlockBody block_body;
  block_body.expr_list = &block->block.exprs;
  content(block_body, wabt::Var(block->block.label));
  e->push_back(std::move(block));
  return e;
}

exprs_sptr MakeUnary(wabt::Opcode opcode, exprs_sptr op) {
  assert(op != nullptr);
  exprs_sptr e = CreateExprList();
  Merge(e, op);
  e->push_back(wabt::MakeUnique<wabt::UnaryExpr>(opcode));
  return e;
}

exprs_sptr MakeBinary(wabt::Opcode opcode, exprs_sptr op1, exprs_sptr op2) {
  assert(op1 != nullptr);
  assert(op2 != nullptr);
  exprs_sptr e = CreateExprList();
  Merge(e, op1);
  Merge(e, op2);
  e->push_back(wabt::MakeUnique<wabt::BinaryExpr>(opcode));
  return e;
}

exprs_sptr MakeI32Const(uint32_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I32(val)));
}

exprs_sptr MakeI64Const(uint64_t val) {
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::I64(val)));
}

exprs_sptr MakeF32Const(float val) {
  uint32_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F32(value)));
}

exprs_sptr MakeF64Const(double val) {
  uint64_t value;
  memcpy(&val, &value, sizeof(val));
  return ExprToExprList(wabt::MakeUnique<wabt::ConstExpr>(wabt::Const::F64(value)));
}

exprs_sptr MakeBr(wabt::Var label) {
  return ExprToExprList(wabt::MakeUnique<wabt::BrExpr>(label));
}

exprs_sptr MakeBrIf(wabt::Var label, exprs_sptr cond) {
  assert(cond != nullptr);
  exprs_sptr e = CreateExprList();
  Merge(e, cond);
  e->push_back(wabt::MakeUnique<wabt::BrIfExpr>(label));
  return e;
}

exprs_sptr MakeLocalGet(wabt::Var var) {
  return ExprToExprList(wabt::MakeUnique<wabt::LocalGetExpr>(var));
}

exprs_sptr MakeLocalSet(wabt::Var var, exprs_sptr val) {
  assert(val != nullptr);
  exprs_sptr e = CreateExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalSetExpr>(var));
  return e;
}

exprs_sptr MakeLocalTree(wabt::Var var, exprs_sptr val) {
  assert(val != nullptr);
  exprs_sptr e = CreateExprList();
  Merge(e, val);
  e->push_back(wabt::MakeUnique<wabt::LocalTeeExpr>(var));
  return e;
}

exprs_sptr MakeCall(wabt::Var var, std::vector<exprs_sptr> args) {
  exprs_sptr e = CreateExprList();
  for(auto arg : args) {
    assert(arg != nullptr);
    Merge(e, arg);
  }
  e->push_back(wabt::MakeUnique<wabt::CallExpr>(var));
  return e;
}

#define DEFINE_LOAD(opcode) \
exprs_sptr Make##opcode(exprs_sptr index, wabt::Address align, uint32_t offset) {         \
  assert(index != nullptr);                                                               \
  exprs_sptr e = CreateExprList();                                                        \
  Merge(e, index);                                                                        \
  e->push_back(wabt::MakeUnique<wabt::LoadExpr>(wabt::Opcode::opcode, align, offset));    \
  return e;                                                                               \
}
LOAD_INSTRUCTIONS_LIST(DEFINE_LOAD)
#undef DEFINE_LOAD

#define DEFINE_STORE(opcode) \
exprs_sptr Make##opcode(exprs_sptr index, exprs_sptr val, wabt::Address align, uint32_t offset) { \
  assert(index != nullptr);                                                                       \
  assert(val != nullptr);                                                                         \
  exprs_sptr e = CreateExprList();                                                                \
  Merge(e, index);                                                                                \
  Merge(e, val);                                                                                  \
  e->push_back(wabt::MakeUnique<wabt::StoreExpr>(wabt::Opcode::opcode, align, offset));           \
  return e;                                                                                       \
}
STORE_INSTRUCTIONS_LIST(DEFINE_STORE)
#undef DEFINE_STORE

} // namespace wasmpp
